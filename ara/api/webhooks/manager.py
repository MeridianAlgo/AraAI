"""
Webhook manager for registration and management
"""

from typing import Dict, List, Optional, Set
import uuid
from datetime import datetime
import logging

from ara.api.webhooks.models import (
    WebhookCreate,
    WebhookUpdate,
    WebhookResponse,
    WebhookEventType,
)

logger = logging.getLogger(__name__)


class WebhookManager:
    """
    Manages webhook registrations and subscriptions

    Features:
    - Webhook registration and management
    - Event subscription management
    - Webhook statistics tracking
    - In-memory storage (can be extended to use database)
    """

    def __init__(self):
        """Initialize webhook manager"""
        # Webhooks storage: {webhook_id: webhook_data}
        self.webhooks: Dict[str, Dict] = {}

        # Event subscriptions: {event_type: set(webhook_ids)}
        self.subscriptions: Dict[WebhookEventType, Set[str]] = {
            event_type: set() for event_type in WebhookEventType
        }

        # User webhooks: {user_id: set(webhook_ids)}
        self.user_webhooks: Dict[str, Set[str]] = {}

    def create_webhook(
        self, webhook_data: WebhookCreate, user_id: Optional[str] = None
    ) -> WebhookResponse:
        """
        Create a new webhook

        Args:
            webhook_data: Webhook creation data
            user_id: Optional user ID

        Returns:
            Created webhook information
        """
        # Generate webhook ID
        webhook_id = f"wh_{uuid.uuid4().hex[:16]}"

        # Create webhook record
        now = datetime.now()
        webhook = {
            "id": webhook_id,
            "url": str(webhook_data.url),
            "events": webhook_data.events,
            "secret": webhook_data.secret,
            "description": webhook_data.description,
            "active": webhook_data.active,
            "created_at": now,
            "updated_at": now,
            "user_id": user_id,
            "delivery_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "last_delivery_at": None,
            "last_delivery_status": None,
        }

        # Store webhook
        self.webhooks[webhook_id] = webhook

        # Register event subscriptions
        for event_type in webhook_data.events:
            self.subscriptions[event_type].add(webhook_id)

        # Track user webhooks
        if user_id:
            if user_id not in self.user_webhooks:
                self.user_webhooks[user_id] = set()
            self.user_webhooks[user_id].add(webhook_id)

        logger.info(f"Webhook created: {webhook_id} for events {webhook_data.events}")

        return WebhookResponse(**webhook)

    def get_webhook(self, webhook_id: str) -> Optional[WebhookResponse]:
        """
        Get webhook by ID

        Args:
            webhook_id: Webhook identifier

        Returns:
            Webhook information or None if not found
        """
        webhook = self.webhooks.get(webhook_id)
        if webhook:
            return WebhookResponse(**webhook)
        return None

    def list_webhooks(
        self, user_id: Optional[str] = None, active_only: bool = False
    ) -> List[WebhookResponse]:
        """
        List webhooks

        Args:
            user_id: Optional filter by user ID
            active_only: Only return active webhooks

        Returns:
            List of webhooks
        """
        webhooks = []

        # Filter by user if specified
        if user_id:
            webhook_ids = self.user_webhooks.get(user_id, set())
            webhooks = [self.webhooks[wh_id] for wh_id in webhook_ids if wh_id in self.webhooks]
        else:
            webhooks = list(self.webhooks.values())

        # Filter by active status
        if active_only:
            webhooks = [wh for wh in webhooks if wh["active"]]

        return [WebhookResponse(**wh) for wh in webhooks]

    def update_webhook(
        self, webhook_id: str, webhook_data: WebhookUpdate
    ) -> Optional[WebhookResponse]:
        """
        Update webhook

        Args:
            webhook_id: Webhook identifier
            webhook_data: Update data

        Returns:
            Updated webhook information or None if not found
        """
        if webhook_id not in self.webhooks:
            return None

        webhook = self.webhooks[webhook_id]

        # Update fields
        if webhook_data.url is not None:
            webhook["url"] = str(webhook_data.url)

        if webhook_data.events is not None:
            # Remove old subscriptions
            for event_type in webhook["events"]:
                self.subscriptions[event_type].discard(webhook_id)

            # Add new subscriptions
            webhook["events"] = webhook_data.events
            for event_type in webhook_data.events:
                self.subscriptions[event_type].add(webhook_id)

        if webhook_data.secret is not None:
            webhook["secret"] = webhook_data.secret

        if webhook_data.description is not None:
            webhook["description"] = webhook_data.description

        if webhook_data.active is not None:
            webhook["active"] = webhook_data.active

        webhook["updated_at"] = datetime.now()

        logger.info(f"Webhook updated: {webhook_id}")

        return WebhookResponse(**webhook)

    def delete_webhook(self, webhook_id: str) -> bool:
        """
        Delete webhook

        Args:
            webhook_id: Webhook identifier

        Returns:
            True if deleted, False if not found
        """
        if webhook_id not in self.webhooks:
            return False

        webhook = self.webhooks[webhook_id]

        # Remove event subscriptions
        for event_type in webhook["events"]:
            self.subscriptions[event_type].discard(webhook_id)

        # Remove from user webhooks
        user_id = webhook.get("user_id")
        if user_id and user_id in self.user_webhooks:
            self.user_webhooks[user_id].discard(webhook_id)

        # Delete webhook
        del self.webhooks[webhook_id]

        logger.info(f"Webhook deleted: {webhook_id}")

        return True

    def get_webhooks_for_event(self, event_type: WebhookEventType) -> List[WebhookResponse]:
        """
        Get all active webhooks subscribed to an event type

        Args:
            event_type: Event type

        Returns:
            List of webhooks subscribed to this event
        """
        webhook_ids = self.subscriptions.get(event_type, set())

        webhooks = []
        for webhook_id in webhook_ids:
            if webhook_id in self.webhooks:
                webhook = self.webhooks[webhook_id]
                # Only include active webhooks
                if webhook["active"]:
                    webhooks.append(WebhookResponse(**webhook))

        return webhooks

    def update_delivery_stats(self, webhook_id: str, success: bool, status: str):
        """
        Update webhook delivery statistics

        Args:
            webhook_id: Webhook identifier
            success: Whether delivery was successful
            status: Delivery status
        """
        if webhook_id not in self.webhooks:
            return

        webhook = self.webhooks[webhook_id]
        webhook["delivery_count"] += 1

        if success:
            webhook["success_count"] += 1
        else:
            webhook["failure_count"] += 1

        webhook["last_delivery_at"] = datetime.now()
        webhook["last_delivery_status"] = status

    def get_webhook_secret(self, webhook_id: str) -> Optional[str]:
        """
        Get webhook secret for signature verification

        Args:
            webhook_id: Webhook identifier

        Returns:
            Webhook secret or None
        """
        webhook = self.webhooks.get(webhook_id)
        if webhook:
            return webhook.get("secret")
        return None


# Global webhook manager instance
webhook_manager = WebhookManager()
