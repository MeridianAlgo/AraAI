"""
Webhook management API routes
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ara.api.auth.dependencies import get_current_user
from ara.api.webhooks.delivery import webhook_delivery_service
from ara.api.webhooks.manager import webhook_manager
from ara.api.webhooks.models import (
    WebhookCreate,
    WebhookDeliveryListResponse,
    WebhookEventType,
    WebhookListResponse,
    WebhookResponse,
    WebhookUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


@router.post(
    "",
    response_model=WebhookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create webhook",
    description="Register a new webhook for event notifications",
)
async def create_webhook(webhook_data: WebhookCreate, user_id: str = Depends(get_current_user)):
    """
    Create a new webhook

    **Required fields**:
    - `url`: Webhook callback URL (must be HTTPS in production)
    - `events`: List of event types to subscribe to

    **Optional fields**:
    - `secret`: Secret key for signature verification (recommended)
    - `description`: Description of the webhook
    - `active`: Whether the webhook is active (default: true)

    **Event Types**:
    - `prediction_complete`: Triggered when a prediction is completed
    - `model_trained`: Triggered when a model training is completed
    - `model_deployed`: Triggered when a model is deployed
    - `backtest_complete`: Triggered when a backtest is completed
    - `alert_triggered`: Triggered when an alert condition is met
    - `data_quality_issue`: Triggered when data quality issues are detected
    - `system_error`: Triggered on system errors

    **Signature Verification**:
    If a secret is provided, all webhook deliveries will include an
    `X-Webhook-Signature` header with an HMAC-SHA256 signature of the payload.

    Example signature verification (Python):
    ```python
    import hmac
    import hashlib
    import json

    def verify_signature(payload, signature, secret):
        expected = hmac.new(
            secret.encode(),
            json.dumps(payload, sort_keys=True).encode(),
            hashlib.sha256
        ).hexdigest()
        return signature == f"sha256={expected}"
    ```
    """
    try:
        webhook = webhook_manager.create_webhook(webhook_data, user_id)
        return webhook
    except Exception as e:
        logger.error(f"Error creating webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create webhook: {str(e)}",
        )


@router.get(
    "",
    response_model=WebhookListResponse,
    summary="List webhooks",
    description="Get all webhooks for the authenticated user",
)
async def list_webhooks(
    active_only: bool = Query(False, description="Only return active webhooks"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    user_id: str = Depends(get_current_user),
):
    """
    List all webhooks for the authenticated user

    Returns paginated list of webhooks with delivery statistics.
    """
    try:
        webhooks = webhook_manager.list_webhooks(user_id, active_only)

        # Pagination
        total = len(webhooks)
        start = (page - 1) * page_size
        end = start + page_size
        webhooks_page = webhooks[start:end]

        return WebhookListResponse(
            webhooks=webhooks_page, total=total, page=page, page_size=page_size
        )
    except Exception as e:
        logger.error(f"Error listing webhooks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list webhooks: {str(e)}",
        )


@router.get(
    "/{webhook_id}",
    response_model=WebhookResponse,
    summary="Get webhook",
    description="Get webhook details by ID",
)
async def get_webhook(webhook_id: str, user_id: str = Depends(get_current_user)):
    """
    Get webhook details by ID

    Returns webhook configuration and delivery statistics.
    """
    webhook = webhook_manager.get_webhook(webhook_id)

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    # Verify ownership
    if webhook.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    return webhook


@router.patch(
    "/{webhook_id}",
    response_model=WebhookResponse,
    summary="Update webhook",
    description="Update webhook configuration",
)
async def update_webhook(
    webhook_id: str,
    webhook_data: WebhookUpdate,
    user_id: str = Depends(get_current_user),
):
    """
    Update webhook configuration

    All fields are optional. Only provided fields will be updated.
    """
    # Verify webhook exists and user owns it
    webhook = webhook_manager.get_webhook(webhook_id)

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    if webhook.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # Update webhook
    try:
        updated_webhook = webhook_manager.update_webhook(webhook_id, webhook_data)
        return updated_webhook
    except Exception as e:
        logger.error(f"Error updating webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update webhook: {str(e)}",
        )


@router.delete(
    "/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete webhook",
    description="Delete a webhook",
)
async def delete_webhook(webhook_id: str, user_id: str = Depends(get_current_user)):
    """
    Delete a webhook

    This will permanently delete the webhook and stop all future deliveries.
    """
    # Verify webhook exists and user owns it
    webhook = webhook_manager.get_webhook(webhook_id)

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    if webhook.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # Delete webhook
    try:
        webhook_manager.delete_webhook(webhook_id)
    except Exception as e:
        logger.error(f"Error deleting webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete webhook: {str(e)}",
        )


@router.get(
    "/{webhook_id}/deliveries",
    response_model=WebhookDeliveryListResponse,
    summary="List webhook deliveries",
    description="Get delivery history for a webhook",
)
async def list_webhook_deliveries(
    webhook_id: str,
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Page size"),
    user_id: str = Depends(get_current_user),
):
    """
    Get delivery history for a webhook

    Returns paginated list of delivery attempts with status and response details.
    """
    # Verify webhook exists and user owns it
    webhook = webhook_manager.get_webhook(webhook_id)

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    if webhook.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # Get deliveries
    try:
        deliveries = webhook_delivery_service.list_deliveries(
            webhook_id=webhook_id,
            status=status_filter,
            limit=1000,  # Get more for pagination
        )

        # Pagination
        total = len(deliveries)
        start = (page - 1) * page_size
        end = start + page_size
        deliveries_page = deliveries[start:end]

        return WebhookDeliveryListResponse(
            deliveries=deliveries_page, total=total, page=page, page_size=page_size
        )
    except Exception as e:
        logger.error(f"Error listing deliveries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list deliveries: {str(e)}",
        )


@router.post(
    "/test",
    status_code=status.HTTP_200_OK,
    summary="Test webhook",
    description="Send a test event to a webhook",
)
async def test_webhook(
    webhook_id: str = Query(..., description="Webhook ID to test"),
    user_id: str = Depends(get_current_user),
):
    """
    Send a test event to a webhook

    This will trigger a test delivery to verify the webhook is configured correctly.
    """
    # Verify webhook exists and user owns it
    webhook = webhook_manager.get_webhook(webhook_id)

    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    if webhook.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    # Send test event
    try:
        await webhook_delivery_service.trigger_event(
            event_type=WebhookEventType.SYSTEM_ERROR,  # Use as test event
            data={
                "test": True,
                "message": "This is a test webhook delivery",
                "webhook_id": webhook_id,
                "timestamp": "2024-01-01T00:00:00Z",
            },
        )

        return {"message": "Test webhook sent successfully", "webhook_id": webhook_id}
    except Exception as e:
        logger.error(f"Error testing webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test webhook: {str(e)}",
        )
