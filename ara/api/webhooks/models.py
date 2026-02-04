"""
Webhook data models
"""

from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class WebhookEventType(str, Enum):
    """Webhook event types"""

    PREDICTION_COMPLETE = "prediction_complete"
    MODEL_TRAINED = "model_trained"
    MODEL_DEPLOYED = "model_deployed"
    BACKTEST_COMPLETE = "backtest_complete"
    ALERT_TRIGGERED = "alert_triggered"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    SYSTEM_ERROR = "system_error"


class WebhookCreate(BaseModel):
    """Request model for creating a webhook"""

    url: HttpUrl = Field(..., description="Webhook callback URL")
    events: List[WebhookEventType] = Field(
        ..., description="List of event types to subscribe to", min_items=1
    )
    secret: Optional[str] = Field(
        None,
        description="Secret key for signature verification",
        min_length=16,
        max_length=128,
    )
    description: Optional[str] = Field(
        None, description="Optional description of the webhook", max_length=500
    )
    active: bool = Field(True, description="Whether the webhook is active")

    @validator("events")
    def validate_events(cls, v):
        """Ensure events list is not empty and contains valid types"""
        if not v:
            raise ValueError("At least one event type must be specified")
        return v


class WebhookUpdate(BaseModel):
    """Request model for updating a webhook"""

    url: Optional[HttpUrl] = Field(None, description="Webhook callback URL")
    events: Optional[List[WebhookEventType]] = Field(
        None, description="List of event types to subscribe to"
    )
    secret: Optional[str] = Field(
        None,
        description="Secret key for signature verification",
        min_length=16,
        max_length=128,
    )
    description: Optional[str] = Field(
        None, description="Optional description of the webhook", max_length=500
    )
    active: Optional[bool] = Field(None, description="Whether the webhook is active")


class WebhookResponse(BaseModel):
    """Response model for webhook information"""

    id: str = Field(..., description="Webhook ID")
    url: str = Field(..., description="Webhook callback URL")
    events: List[WebhookEventType] = Field(..., description="Subscribed event types")
    description: Optional[str] = Field(None, description="Webhook description")
    active: bool = Field(..., description="Whether the webhook is active")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    user_id: Optional[str] = Field(None, description="Owner user ID")
    delivery_count: int = Field(0, description="Total delivery attempts")
    success_count: int = Field(0, description="Successful deliveries")
    failure_count: int = Field(0, description="Failed deliveries")
    last_delivery_at: Optional[datetime] = Field(None, description="Last delivery timestamp")
    last_delivery_status: Optional[str] = Field(None, description="Last delivery status")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "wh_1234567890",
                "url": "https://example.com/webhook",
                "events": ["prediction_complete", "model_trained"],
                "description": "Production webhook for predictions",
                "active": True,
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "user_id": "user_123",
                "delivery_count": 100,
                "success_count": 98,
                "failure_count": 2,
                "last_delivery_at": "2024-01-01T12:00:00Z",
                "last_delivery_status": "success",
            }
        }


class WebhookEvent(BaseModel):
    """Webhook event payload"""

    event_id: str = Field(..., description="Unique event ID")
    event_type: WebhookEventType = Field(..., description="Event type")
    timestamp: datetime = Field(..., description="Event timestamp")
    data: Dict[str, Any] = Field(..., description="Event data")

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_1234567890",
                "event_type": "prediction_complete",
                "timestamp": "2024-01-01T12:00:00Z",
                "data": {
                    "symbol": "AAPL",
                    "prediction_id": "pred_123",
                    "confidence": 0.85,
                    "predicted_price": 150.25,
                },
            }
        }


class WebhookDelivery(BaseModel):
    """Webhook delivery record"""

    delivery_id: str = Field(..., description="Delivery ID")
    webhook_id: str = Field(..., description="Webhook ID")
    event_id: str = Field(..., description="Event ID")
    event_type: WebhookEventType = Field(..., description="Event type")
    status: str = Field(..., description="Delivery status (pending, success, failed)")
    attempt: int = Field(..., description="Attempt number")
    max_attempts: int = Field(3, description="Maximum retry attempts")
    response_code: Optional[int] = Field(None, description="HTTP response code")
    response_body: Optional[str] = Field(None, description="Response body")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Creation timestamp")
    delivered_at: Optional[datetime] = Field(None, description="Delivery timestamp")
    next_retry_at: Optional[datetime] = Field(None, description="Next retry timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "delivery_id": "del_1234567890",
                "webhook_id": "wh_1234567890",
                "event_id": "evt_1234567890",
                "event_type": "prediction_complete",
                "status": "success",
                "attempt": 1,
                "max_attempts": 3,
                "response_code": 200,
                "response_body": '{"status": "received"}',
                "error_message": None,
                "created_at": "2024-01-01T12:00:00Z",
                "delivered_at": "2024-01-01T12:00:01Z",
                "next_retry_at": None,
            }
        }


class WebhookListResponse(BaseModel):
    """Response model for listing webhooks"""

    webhooks: List[WebhookResponse] = Field(..., description="List of webhooks")
    total: int = Field(..., description="Total number of webhooks")
    page: int = Field(1, description="Current page")
    page_size: int = Field(10, description="Page size")


class WebhookDeliveryListResponse(BaseModel):
    """Response model for listing webhook deliveries"""

    deliveries: List[WebhookDelivery] = Field(..., description="List of deliveries")
    total: int = Field(..., description="Total number of deliveries")
    page: int = Field(1, description="Current page")
    page_size: int = Field(10, description="Page size")
