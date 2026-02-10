"""
Webhook delivery service with retry logic
"""

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp

from ara.api.webhooks.manager import webhook_manager
from ara.api.webhooks.models import (
    WebhookDelivery,
    WebhookEvent,
    WebhookEventType,
    WebhookResponse,
)

logger = logging.getLogger(__name__)


class WebhookDeliveryService:
    """
    Service for delivering webhooks with retry logic

    Features:
    - Asynchronous webhook delivery
    - Automatic retries with exponential backoff
    - Signature verification support
    - Delivery logging and monitoring
    """

    def __init__(self, max_retries: int = 3, timeout: int = 30, retry_delays: List[int] = None):
        """
        Initialize webhook delivery service

        Args:
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            retry_delays: List of delays between retries (seconds)
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delays = retry_delays or [5, 30, 300]  # 5s, 30s, 5min

        # Delivery queue: {delivery_id: delivery_data}
        self.delivery_queue: Dict[str, Dict] = {}

        # Delivery history: {delivery_id: delivery_record}
        self.delivery_history: Dict[str, Dict] = {}

        # Background task for processing retries
        self._retry_task: Optional[asyncio.Task] = None

    async def trigger_event(self, event_type: WebhookEventType, data: Dict[str, Any]):
        """
        Trigger an event and deliver to all subscribed webhooks

        Args:
            event_type: Type of event
            data: Event data
        """
        # Create event
        event = WebhookEvent(
            event_id=f"evt_{uuid.uuid4().hex[:16]}",
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
        )

        # Get webhooks subscribed to this event
        webhooks = webhook_manager.get_webhooks_for_event(event_type)

        if not webhooks:
            logger.debug(f"No webhooks subscribed to event: {event_type}")
            return

        logger.info(f"Triggering event {event_type} for {len(webhooks)} webhooks")

        # Deliver to all webhooks
        tasks = []
        for webhook in webhooks:
            task = self.deliver_webhook(webhook, event)
            tasks.append(task)

        # Execute deliveries concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def deliver_webhook(
        self, webhook: WebhookResponse, event: WebhookEvent, attempt: int = 1
    ) -> WebhookDelivery:
        """
        Deliver webhook to endpoint

        Args:
            webhook: Webhook configuration
            event: Event to deliver
            attempt: Attempt number

        Returns:
            Delivery record
        """
        delivery_id = f"del_{uuid.uuid4().hex[:16]}"

        # Create delivery record
        delivery = {
            "delivery_id": delivery_id,
            "webhook_id": webhook.id,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "status": "pending",
            "attempt": attempt,
            "max_attempts": self.max_retries,
            "response_code": None,
            "response_body": None,
            "error_message": None,
            "created_at": datetime.now(),
            "delivered_at": None,
            "next_retry_at": None,
        }

        # Store in history
        self.delivery_history[delivery_id] = delivery

        try:
            # Prepare payload
            payload = event.dict()

            # Generate signature if secret is configured
            headers = {"Content-Type": "application/json"}
            secret = webhook_manager.get_webhook_secret(webhook.id)
            if secret:
                signature = self._generate_signature(payload, secret)
                headers["X-Webhook-Signature"] = signature

            # Send webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    delivery["response_code"] = response.status
                    delivery["response_body"] = await response.text()

                    if 200 <= response.status < 300:
                        # Success
                        delivery["status"] = "success"
                        delivery["delivered_at"] = datetime.now()

                        webhook_manager.update_delivery_stats(
                            webhook.id, success=True, status="success"
                        )

                        logger.info(
                            f"Webhook delivered successfully: {webhook.id} "
                            f"(event: {event.event_type})"
                        )
                    else:
                        # HTTP error
                        raise Exception(f"HTTP {response.status}: {delivery['response_body']}")

        except Exception as e:
            # Delivery failed
            delivery["status"] = "failed"
            delivery["error_message"] = str(e)

            logger.error(
                f"Webhook delivery failed: {webhook.id} "
                f"(event: {event.event_type}, attempt: {attempt}): {e}"
            )

            # Schedule retry if attempts remaining
            if attempt < self.max_retries:
                delay = self._get_retry_delay(attempt)
                delivery["next_retry_at"] = datetime.now() + timedelta(seconds=delay)

                # Queue for retry
                self.delivery_queue[delivery_id] = {
                    "webhook": webhook,
                    "event": event,
                    "attempt": attempt + 1,
                    "retry_at": delivery["next_retry_at"],
                }

                logger.info(
                    f"Webhook retry scheduled: {webhook.id} "
                    f"(attempt {attempt + 1}/{self.max_retries} in {delay}s)"
                )

                # Start retry processor if not running
                if self._retry_task is None or self._retry_task.done():
                    self._retry_task = asyncio.create_task(self._process_retries())
            else:
                # Max retries reached
                webhook_manager.update_delivery_stats(webhook.id, success=False, status="failed")

                logger.error(
                    f"Webhook delivery failed permanently: {webhook.id} (event: {event.event_type})"
                )

        return WebhookDelivery(**delivery)

    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """
        Generate HMAC signature for webhook payload

        Args:
            payload: Webhook payload
            secret: Webhook secret

        Returns:
            Hex-encoded signature
        """
        payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        signature = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        return f"sha256={signature}"

    def _get_retry_delay(self, attempt: int) -> int:
        """
        Get retry delay for attempt number

        Args:
            attempt: Attempt number (1-indexed)

        Returns:
            Delay in seconds
        """
        if attempt - 1 < len(self.retry_delays):
            return self.retry_delays[attempt - 1]
        return self.retry_delays[-1]

    async def _process_retries(self):
        """
        Background task to process retry queue
        """
        while True:
            try:
                await asyncio.sleep(1)

                # Check for deliveries ready to retry
                now = datetime.now()
                ready_deliveries = []

                for delivery_id, delivery_data in list(self.delivery_queue.items()):
                    if delivery_data["retry_at"] <= now:
                        ready_deliveries.append((delivery_id, delivery_data))

                # Process ready deliveries
                for delivery_id, delivery_data in ready_deliveries:
                    # Remove from queue
                    del self.delivery_queue[delivery_id]

                    # Retry delivery
                    await self.deliver_webhook(
                        delivery_data["webhook"],
                        delivery_data["event"],
                        delivery_data["attempt"],
                    )

                # Exit if queue is empty
                if not self.delivery_queue:
                    break

            except Exception as e:
                logger.error(f"Error in retry processor: {e}")

    def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """
        Get delivery record by ID

        Args:
            delivery_id: Delivery identifier

        Returns:
            Delivery record or None
        """
        delivery = self.delivery_history.get(delivery_id)
        if delivery:
            return WebhookDelivery(**delivery)
        return None

    def list_deliveries(
        self,
        webhook_id: Optional[str] = None,
        event_type: Optional[WebhookEventType] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[WebhookDelivery]:
        """
        List delivery records

        Args:
            webhook_id: Optional filter by webhook ID
            event_type: Optional filter by event type
            status: Optional filter by status
            limit: Maximum number of records

        Returns:
            List of delivery records
        """
        deliveries = list(self.delivery_history.values())

        # Apply filters
        if webhook_id:
            deliveries = [d for d in deliveries if d["webhook_id"] == webhook_id]

        if event_type:
            deliveries = [d for d in deliveries if d["event_type"] == event_type]

        if status:
            deliveries = [d for d in deliveries if d["status"] == status]

        # Sort by creation time (newest first)
        deliveries.sort(key=lambda d: d["created_at"], reverse=True)

        # Limit results
        deliveries = deliveries[:limit]

        return [WebhookDelivery(**d) for d in deliveries]


# Global webhook delivery service instance
webhook_delivery_service = WebhookDeliveryService()
