# Webhook System

This module provides a webhook system for receiving HTTP callbacks when events occur in the ARA AI prediction system.

## Features

- **Event subscriptions**: Subscribe to specific event types
- **Automatic retries**: Failed deliveries are automatically retried with exponential backoff
- **Signature verification**: Secure webhooks with HMAC-SHA256 signatures
- **Delivery tracking**: Monitor delivery success/failure rates
- **Multiple webhooks**: Register multiple webhooks for different purposes

## Webhook Events

### Available Event Types

1. **`prediction_complete`**: Triggered when a prediction is completed
   ```json
   {
       "event_id": "evt_1234567890",
       "event_type": "prediction_complete",
       "timestamp": "2024-01-01T12:00:00Z",
       "data": {
           "symbol": "AAPL",
           "prediction_id": "pred_123",
           "confidence": 0.85,
           "predicted_price": 150.25,
           "prediction_horizon": 5
       }
   }
   ```

2. **`model_trained`**: Triggered when a model training is completed
   ```json
   {
       "event_id": "evt_1234567890",
       "event_type": "model_trained",
       "timestamp": "2024-01-01T12:00:00Z",
       "data": {
           "model_id": "model_123",
           "model_type": "transformer",
           "symbol": "AAPL",
           "accuracy": 0.82,
           "training_duration": 3600
       }
   }
   ```

3. **`model_deployed`**: Triggered when a model is deployed
4. **`backtest_complete`**: Triggered when a backtest is completed
5. **`alert_triggered`**: Triggered when an alert condition is met
6. **`data_quality_issue`**: Triggered when data quality issues are detected
7. **`system_error`**: Triggered on system errors

## API Endpoints

### Create Webhook

**POST** `/api/v1/webhooks`

Create a new webhook subscription.

**Request Body**:
```json
{
    "url": "https://example.com/webhook",
    "events": ["prediction_complete", "model_trained"],
    "secret": "your-secret-key-min-16-chars",
    "description": "Production webhook for predictions",
    "active": true
}
```

**Response**:
```json
{
    "id": "wh_1234567890",
    "url": "https://example.com/webhook",
    "events": ["prediction_complete", "model_trained"],
    "description": "Production webhook for predictions",
    "active": true,
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z",
    "user_id": "user_123",
    "delivery_count": 0,
    "success_count": 0,
    "failure_count": 0
}
```

### List Webhooks

**GET** `/api/v1/webhooks?active_only=true&page=1&page_size=10`

List all webhooks for the authenticated user.

**Response**:
```json
{
    "webhooks": [...],
    "total": 5,
    "page": 1,
    "page_size": 10
}
```

### Get Webhook

**GET** `/api/v1/webhooks/{webhook_id}`

Get webhook details by ID.

### Update Webhook

**PATCH** `/api/v1/webhooks/{webhook_id}`

Update webhook configuration.

**Request Body** (all fields optional):
```json
{
    "url": "https://example.com/new-webhook",
    "events": ["prediction_complete"],
    "active": false
}
```

### Delete Webhook

**DELETE** `/api/v1/webhooks/{webhook_id}`

Delete a webhook permanently.

### List Deliveries

**GET** `/api/v1/webhooks/{webhook_id}/deliveries?status=failed&page=1&page_size=10`

Get delivery history for a webhook.

**Response**:
```json
{
    "deliveries": [
        {
            "delivery_id": "del_1234567890",
            "webhook_id": "wh_1234567890",
            "event_id": "evt_1234567890",
            "event_type": "prediction_complete",
            "status": "success",
            "attempt": 1,
            "max_attempts": 3,
            "response_code": 200,
            "response_body": "{\"status\": \"received\"}",
            "created_at": "2024-01-01T12:00:00Z",
            "delivered_at": "2024-01-01T12:00:01Z"
        }
    ],
    "total": 100,
    "page": 1,
    "page_size": 10
}
```

### Test Webhook

**POST** `/api/v1/webhooks/test?webhook_id=wh_1234567890`

Send a test event to verify webhook configuration.

## Signature Verification

All webhook deliveries include an `X-Webhook-Signature` header with an HMAC-SHA256 signature of the payload (if a secret is configured).

### Verify Signature (Python)

```python
import hmac
import hashlib
import json

def verify_webhook_signature(payload, signature, secret):
    """
    Verify webhook signature
    
    Args:
        payload: Request body (dict)
        signature: X-Webhook-Signature header value
        secret: Webhook secret
        
    Returns:
        True if signature is valid
    """
    # Compute expected signature
    payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
    expected = hmac.new(
        secret.encode('utf-8'),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()
    
    # Compare signatures
    expected_sig = f"sha256={expected}"
    return hmac.compare_digest(expected_sig, signature)

# Usage in Flask/FastAPI
@app.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    signature = request.headers.get("X-Webhook-Signature")
    
    if not verify_webhook_signature(payload, signature, "your-secret"):
        return {"error": "Invalid signature"}, 401
    
    # Process webhook
    event_type = payload["event_type"]
    data = payload["data"]
    
    # Handle event...
    
    return {"status": "received"}
```

### Verify Signature (Node.js)

```javascript
const crypto = require('crypto');

function verifyWebhookSignature(payload, signature, secret) {
    // Compute expected signature
    const payloadString = JSON.stringify(payload, Object.keys(payload).sort());
    const hmac = crypto.createHmac('sha256', secret);
    hmac.update(payloadString);
    const expected = `sha256=${hmac.digest('hex')}`;
    
    // Compare signatures (timing-safe)
    return crypto.timingSafeEqual(
        Buffer.from(expected),
        Buffer.from(signature)
    );
}

// Usage in Express
app.post('/webhook', (req, res) => {
    const payload = req.body;
    const signature = req.headers['x-webhook-signature'];
    
    if (!verifyWebhookSignature(payload, signature, 'your-secret')) {
        return res.status(401).json({ error: 'Invalid signature' });
    }
    
    // Process webhook
    const { event_type, data } = payload;
    
    // Handle event...
    
    res.json({ status: 'received' });
});
```

## Retry Logic

Failed webhook deliveries are automatically retried with exponential backoff:

- **Attempt 1**: Immediate delivery
- **Attempt 2**: 5 seconds after failure
- **Attempt 3**: 30 seconds after failure
- **Attempt 4**: 5 minutes after failure (final attempt)

After 3 failed attempts, the webhook delivery is marked as permanently failed.

## Best Practices

### 1. Respond Quickly

Your webhook endpoint should respond within 30 seconds. For long-running tasks, acknowledge receipt immediately and process asynchronously:

```python
@app.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    
    # Acknowledge receipt immediately
    asyncio.create_task(process_webhook_async(payload))
    
    return {"status": "received"}

async def process_webhook_async(payload):
    # Long-running processing...
    pass
```

### 2. Implement Idempotency

Webhooks may be delivered multiple times. Use the `event_id` to detect duplicates:

```python
processed_events = set()

@app.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    event_id = payload["event_id"]
    
    # Check if already processed
    if event_id in processed_events:
        return {"status": "already_processed"}
    
    # Process event
    process_event(payload)
    
    # Mark as processed
    processed_events.add(event_id)
    
    return {"status": "received"}
```

### 3. Use HTTPS

Always use HTTPS URLs for webhook endpoints in production to ensure secure delivery.

### 4. Verify Signatures

Always verify webhook signatures to ensure requests are from ARA AI:

```python
@app.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    signature = request.headers.get("X-Webhook-Signature")
    
    # Verify signature
    if not verify_webhook_signature(payload, signature, SECRET):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Process webhook...
```

### 5. Handle Errors Gracefully

Return appropriate HTTP status codes:

- **200-299**: Success (webhook will not be retried)
- **400-499**: Client error (webhook will not be retried)
- **500-599**: Server error (webhook will be retried)

```python
@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        payload = await request.json()
        process_event(payload)
        return {"status": "received"}
    except ValidationError:
        # Client error - don't retry
        return {"error": "Invalid payload"}, 400
    except Exception:
        # Server error - will retry
        return {"error": "Processing failed"}, 500
```

### 6. Monitor Delivery Success

Regularly check webhook delivery statistics:

```bash
curl -X GET "https://api.ara-ai.com/api/v1/webhooks/wh_123/deliveries" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Example: Complete Webhook Handler

```python
from fastapi import FastAPI, Request, HTTPException
import hmac
import hashlib
import json
from typing import Dict, Any

app = FastAPI()

# Store processed events (use Redis in production)
processed_events = set()

WEBHOOK_SECRET = "your-secret-key"

def verify_signature(payload: Dict[str, Any], signature: str) -> bool:
    """Verify webhook signature"""
    payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
    expected = hmac.new(
        WEBHOOK_SECRET.encode('utf-8'),
        payload_bytes,
        hashlib.sha256
    ).hexdigest()
    return signature == f"sha256={expected}"

@app.post("/webhook")
async def handle_webhook(request: Request):
    """Handle ARA AI webhook"""
    # Get payload and signature
    payload = await request.json()
    signature = request.headers.get("X-Webhook-Signature", "")
    
    # Verify signature
    if not verify_signature(payload, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Check for duplicate
    event_id = payload["event_id"]
    if event_id in processed_events:
        return {"status": "already_processed"}
    
    # Handle event
    event_type = payload["event_type"]
    data = payload["data"]
    
    try:
        if event_type == "prediction_complete":
            handle_prediction(data)
        elif event_type == "model_trained":
            handle_model_trained(data)
        elif event_type == "alert_triggered":
            handle_alert(data)
        else:
            print(f"Unknown event type: {event_type}")
        
        # Mark as processed
        processed_events.add(event_id)
        
        return {"status": "received"}
    
    except Exception as e:
        # Return 500 to trigger retry
        raise HTTPException(status_code=500, detail=str(e))

def handle_prediction(data: Dict[str, Any]):
    """Handle prediction complete event"""
    symbol = data["symbol"]
    predicted_price = data["predicted_price"]
    confidence = data["confidence"]
    
    print(f"Prediction for {symbol}: ${predicted_price} (confidence: {confidence})")
    # Process prediction...

def handle_model_trained(data: Dict[str, Any]):
    """Handle model trained event"""
    model_id = data["model_id"]
    accuracy = data["accuracy"]
    
    print(f"Model {model_id} trained with accuracy: {accuracy}")
    # Process model training completion...

def handle_alert(data: Dict[str, Any]):
    """Handle alert triggered event"""
    alert_type = data["type"]
    message = data["message"]
    
    print(f"Alert: {alert_type} - {message}")
    # Send notification...
```

## Triggering Events (Internal Use)

To trigger webhook events from your application code:

```python
from ara.api.webhooks.delivery import webhook_delivery_service
from ara.api.webhooks.models import WebhookEventType

# Trigger prediction complete event
await webhook_delivery_service.trigger_event(
    event_type=WebhookEventType.PREDICTION_COMPLETE,
    data={
        "symbol": "AAPL",
        "prediction_id": "pred_123",
        "confidence": 0.85,
        "predicted_price": 150.25,
        "prediction_horizon": 5
    }
)

# Trigger model trained event
await webhook_delivery_service.trigger_event(
    event_type=WebhookEventType.MODEL_TRAINED,
    data={
        "model_id": "model_123",
        "model_type": "transformer",
        "symbol": "AAPL",
        "accuracy": 0.82,
        "training_duration": 3600
    }
)
```

## See Also

- [WebSocket System](../websocket/README.md) - For real-time updates
- [API Documentation](../README.md) - REST API reference
- [Authentication](../auth/README.md) - Authentication guide
