# WebSocket Real-Time Updates

This module provides WebSocket endpoints for real-time updates in the ARA AI prediction system.

## Features

- **Real-time prediction updates**: Get notified when new predictions are available
- **Market data streaming**: Stream live price and volume data
- **Alert notifications**: Receive instant alerts for configured conditions
- **Connection management**: Automatic heartbeat and reconnection handling
- **Authentication**: Optional JWT or API key authentication

## WebSocket Endpoints

### 1. Predictions WebSocket

**Endpoint**: `ws://host:port/ws/predictions/{symbol}?token=YOUR_TOKEN`

Receive real-time prediction updates for a specific symbol.

**Connection Example** (JavaScript):
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/predictions/AAPL?token=YOUR_TOKEN');

ws.onopen = () => {
    console.log('Connected to predictions stream');
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
    
    if (message.type === 'ping') {
        // Respond to heartbeat
        ws.send(JSON.stringify({ type: 'pong' }));
    } else if (message.type === 'prediction_update') {
        // Handle prediction update
        console.log('New prediction:', message.data);
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Disconnected from predictions stream');
};
```

**Message Types (Server → Client)**:
- `connected`: Connection confirmation
- `ping`: Heartbeat ping (respond with `pong`)
- `prediction_update`: New prediction available
- `error`: Error message

**Message Types (Client → Server)**:
- `pong`: Heartbeat response
- `subscribe`: Change symbol subscription
- `request_prediction`: Request immediate prediction

### 2. Market Data WebSocket

**Endpoint**: `ws://host:port/ws/market-data/{symbol}?token=YOUR_TOKEN`

Stream real-time market data for a specific symbol.

**Connection Example** (Python):
```python
import asyncio
import websockets
import json

async def stream_market_data():
    uri = "ws://localhost:8000/ws/market-data/AAPL?token=YOUR_TOKEN"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to market data stream")
        
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'ping':
                # Respond to heartbeat
                await websocket.send(json.dumps({'type': 'pong'}))
            elif data['type'] == 'market_data_update':
                # Handle market data
                print(f"Price update: {data['data']}")

asyncio.run(stream_market_data())
```

**Message Types (Server → Client)**:
- `connected`: Connection confirmation
- `ping`: Heartbeat ping
- `market_data_update`: New market data
- `error`: Error message

### 3. Alerts WebSocket

**Endpoint**: `ws://host:port/ws/alerts?token=YOUR_TOKEN`

Receive real-time alert notifications.

**Connection Example** (JavaScript):
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/alerts?token=YOUR_TOKEN');

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    if (message.type === 'alert') {
        // Handle alert
        console.log('Alert:', message.data);
        showNotification(message.data);
    } else if (message.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));
    }
};
```

**Message Types (Server → Client)**:
- `connected`: Connection confirmation
- `ping`: Heartbeat ping
- `alert`: New alert notification
- `error`: Error message

## Connection Management

### Heartbeat

The server sends periodic `ping` messages (default: every 30 seconds). Clients should respond with `pong` messages to keep the connection alive.

```javascript
ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    if (message.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong' }));
    }
};
```

### Reconnection

Implement automatic reconnection logic in your client:

```javascript
function connectWebSocket() {
    const ws = new WebSocket('ws://localhost:8000/ws/predictions/AAPL');
    
    ws.onclose = () => {
        console.log('Connection closed, reconnecting in 5 seconds...');
        setTimeout(connectWebSocket, 5000);
    };
    
    return ws;
}

let ws = connectWebSocket();
```

## Authentication

WebSocket connections support optional authentication via query parameter:

```
ws://host:port/ws/predictions/AAPL?token=YOUR_JWT_TOKEN
```

Or with API key:

```
ws://host:port/ws/predictions/AAPL?token=YOUR_API_KEY
```

## Broadcasting Updates

To broadcast updates to connected clients (from your application code):

```python
from ara.api.websocket.handlers import (
    broadcast_prediction_update,
    broadcast_market_data,
    send_alert
)

# Broadcast prediction update
await broadcast_prediction_update(
    symbol="AAPL",
    prediction_data={
        "predicted_price": 150.25,
        "confidence": 0.85,
        "timestamp": "2024-01-01T12:00:00Z"
    }
)

# Broadcast market data
await broadcast_market_data(
    symbol="AAPL",
    market_data={
        "price": 149.50,
        "volume": 1000000,
        "timestamp": "2024-01-01T12:00:00Z"
    }
)

# Send alert
await send_alert(
    alert_data={
        "type": "price_alert",
        "symbol": "AAPL",
        "message": "Price exceeded $150",
        "severity": "high"
    }
)
```

## Connection Statistics

Get connection statistics:

```python
from ara.api.websocket.connection_manager import connection_manager

# Get total connections
total = connection_manager.get_connection_count()

# Get connections for specific channel
predictions_count = connection_manager.get_connection_count("predictions")

# Get all active channels
channels = connection_manager.get_channels()
```

## Error Handling

Handle errors gracefully:

```javascript
ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    
    if (message.type === 'error') {
        console.error('Server error:', message.message);
        // Handle error appropriately
    }
};
```

## Best Practices

1. **Always respond to pings**: Keep the connection alive by responding to heartbeat pings
2. **Implement reconnection**: Handle disconnections gracefully with automatic reconnection
3. **Handle errors**: Implement proper error handling for network issues
4. **Use authentication**: Protect sensitive data with JWT or API key authentication
5. **Limit subscriptions**: Don't open too many concurrent connections
6. **Clean up**: Close connections when no longer needed

## Example: Complete Client

```javascript
class ARAWebSocketClient {
    constructor(endpoint, symbol, token) {
        this.endpoint = endpoint;
        this.symbol = symbol;
        this.token = token;
        this.ws = null;
        this.reconnectDelay = 5000;
    }
    
    connect() {
        const url = `${this.endpoint}/${this.symbol}?token=${this.token}`;
        this.ws = new WebSocket(url);
        
        this.ws.onopen = () => {
            console.log('Connected');
        };
        
        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };
        
        this.ws.onerror = (error) => {
            console.error('Error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected, reconnecting...');
            setTimeout(() => this.connect(), this.reconnectDelay);
        };
    }
    
    handleMessage(message) {
        switch (message.type) {
            case 'ping':
                this.ws.send(JSON.stringify({ type: 'pong' }));
                break;
            case 'prediction_update':
                this.onPredictionUpdate(message.data);
                break;
            case 'error':
                console.error('Server error:', message.message);
                break;
        }
    }
    
    onPredictionUpdate(data) {
        // Override this method to handle predictions
        console.log('Prediction update:', data);
    }
    
    subscribe(newSymbol) {
        this.symbol = newSymbol;
        this.ws.send(JSON.stringify({
            type: 'subscribe',
            symbol: newSymbol
        }));
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Usage
const client = new ARAWebSocketClient(
    'ws://localhost:8000/ws/predictions',
    'AAPL',
    'YOUR_TOKEN'
);

client.onPredictionUpdate = (data) => {
    console.log('New prediction:', data);
    // Update UI, etc.
};

client.connect();
```

## See Also

- [Webhook System](../webhooks/README.md) - For HTTP callbacks
- [API Documentation](../README.md) - REST API reference
- [Authentication](../auth/README.md) - Authentication guide
