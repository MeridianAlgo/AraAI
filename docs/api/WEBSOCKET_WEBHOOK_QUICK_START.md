# WebSocket & Webhook Quick Start Guide

Get started with real-time updates and event callbacks in 5 minutes.

## WebSocket Quick Start

### 1. Connect to WebSocket

**JavaScript Example**:
```javascript
// Connect to predictions stream
const ws = new WebSocket('ws://localhost:8000/ws/predictions/AAPL?token=YOUR_TOKEN');

ws.onopen = () => console.log('Connected!');

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    
    // Respond to heartbeat
    if (msg.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));
    }
    
    // Handle prediction update
    if (msg.type === 'prediction_update') {
        console.log('New prediction:', msg.data);
    }
};
```

**Python Example**:
```python
import asyncio
import websockets
import json

async def connect():
    uri = "ws://localhost:8000/ws/predictions/AAPL?token=YOUR_TOKEN"
    
    async with websockets.connect(uri) as ws:
        async for message in ws:
            data = json.loads(message)
            
            if data['type'] == 'ping':
                await ws.send(json.dumps({'type': 'pong'}))
            elif data['type'] == 'prediction_update':
                print('Prediction:', data['data'])

asyncio.run(connect())
```

### 2. Available Endpoints

- **Predictions**: `ws://host/ws/predictions/{symbol}`
- **Market Data**: `ws://host/ws/market-data/{symbol}`
- **Alerts**: `ws://host/ws/alerts`

## Webhook Quick Start

### 1. Create Webhook

```bash
curl -X POST "http://localhost:8000/api/v1/webhooks" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-domain.com/webhook",
    "events": ["prediction_complete", "model_trained"],
    "secret": "your-secret-key-min-16-chars",
    "description": "My webhook"
  }'
```

### 2. Handle Webhook Events

**Python (FastAPI)**:
```python
from fastapi import FastAPI, Request
import hmac
import hashlib
import json

app = FastAPI()

@app.post("/webhook")
async def handle_webhook(request: Request):
    # Get payload
    payload = await request.json()
    signature = request.headers.get("X-Webhook-Signature")
    
    # Verify signature
    secret = "your-secret-key-min-16-chars"
    payload_str = json.dumps(payload, sort_keys=True)
    expected = f"sha256={hmac.new(secret.encode(), payload_str.encode(), hashlib.sha256).hexdigest()}"
    
    if signature != expected:
        return {"error": "Invalid signature"}, 401
    
    # Handle event
    event_type = payload["event_type"]
    data = payload["data"]
    
    if event_type == "prediction_complete":
        print(f"Prediction: {data['symbol']} -> ${data['predicted_price']}")
    
    return {"status": "received"}
```

**Node.js (Express)**:
```javascript
const express = require('express');
const crypto = require('crypto');

const app = express();
app.use(express.json());

app.post('/webhook', (req, res) => {
    const payload = req.body;
    const signature = req.headers['x-webhook-signature'];
    
    // Verify signature
    const secret = 'your-secret-key-min-16-chars';
    const payloadStr = JSON.stringify(payload, Object.keys(payload).sort());
    const expected = `sha256=${crypto.createHmac('sha256', secret).update(payloadStr).digest('hex')}`;
    
    if (signature !== expected) {
        return res.status(401).json({ error: 'Invalid signature' });
    }
    
    // Handle event
    const { event_type, data } = payload;
    
    if (event_type === 'prediction_complete') {
        console.log(`Prediction: ${data.symbol} -> $${data.predicted_price}`);
    }
    
    res.json({ status: 'received' });
});

app.listen(3000);
```

### 3. Test Webhook

```bash
curl -X POST "http://localhost:8000/api/v1/webhooks/test?webhook_id=wh_123" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Common Use Cases

### 1. Real-Time Dashboard

```javascript
// Connect to multiple streams
const predictionWs = new WebSocket('ws://localhost:8000/ws/predictions/AAPL');
const marketWs = new WebSocket('ws://localhost:8000/ws/market-data/AAPL');
const alertsWs = new WebSocket('ws://localhost:8000/ws/alerts');

// Update dashboard on prediction
predictionWs.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'prediction_update') {
        updatePredictionChart(msg.data);
    }
};

// Update price on market data
marketWs.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'market_data_update') {
        updatePriceDisplay(msg.data);
    }
};

// Show alerts
alertsWs.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'alert') {
        showNotification(msg.data);
    }
};
```

### 2. Automated Trading Bot

```python
import asyncio
import websockets
import json

async def trading_bot():
    uri = "ws://localhost:8000/ws/predictions/AAPL?token=YOUR_TOKEN"
    
    async with websockets.connect(uri) as ws:
        async for message in ws:
            data = json.loads(message)
            
            if data['type'] == 'prediction_update':
                prediction = data['data']
                
                # Trading logic
                if prediction['confidence'] > 0.8:
                    if prediction['predicted_return'] > 0.05:
                        execute_buy_order('AAPL', quantity=100)
                    elif prediction['predicted_return'] < -0.05:
                        execute_sell_order('AAPL', quantity=100)

asyncio.run(trading_bot())
```

### 3. Notification System

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/webhook")
async def handle_webhook(request: Request):
    payload = await request.json()
    
    if payload["event_type"] == "alert_triggered":
        alert = payload["data"]
        
        # Send email notification
        send_email(
            to="trader@example.com",
            subject=f"Alert: {alert['symbol']}",
            body=alert['message']
        )
        
        # Send SMS
        send_sms(
            to="+1234567890",
            message=f"{alert['symbol']}: {alert['message']}"
        )
    
    return {"status": "received"}
```

## Troubleshooting

### WebSocket Connection Issues

**Problem**: Connection closes immediately

**Solution**: Check authentication token and ensure it's valid

```javascript
ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = (event) => {
    console.log('Close code:', event.code);
    console.log('Close reason:', event.reason);
};
```

### Webhook Not Receiving Events

**Problem**: Webhook endpoint not receiving events

**Solutions**:
1. Check webhook is active: `GET /api/v1/webhooks/{webhook_id}`
2. Verify URL is accessible from the internet
3. Check delivery history: `GET /api/v1/webhooks/{webhook_id}/deliveries`
4. Test webhook: `POST /api/v1/webhooks/test?webhook_id={webhook_id}`

### Signature Verification Fails

**Problem**: Signature verification always fails

**Solutions**:
1. Ensure you're using the correct secret
2. Verify payload is serialized with sorted keys: `json.dumps(payload, sort_keys=True)`
3. Check signature format: `sha256={hex_digest}`

## Next Steps

- Read the [WebSocket Documentation](websocket/README.md)
- Read the [Webhook Documentation](webhooks/README.md)
- Explore [API Examples](../examples/)
- Check [Authentication Guide](auth/README.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/your-repo/issues
- Documentation: https://docs.ara-ai.com
- Email: support@ara-ai.com
