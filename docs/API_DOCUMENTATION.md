# ARA AI API Documentation

Complete API reference for the ARA AI Prediction System.

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Rate Limits](#rate-limits)
- [Endpoints](#endpoints)
  - [Authentication](#authentication-endpoints)
  - [Predictions](#prediction-endpoints)
  - [Backtesting](#backtesting-endpoints)
  - [Portfolio](#portfolio-endpoints)
  - [Models](#model-endpoints)
  - [Market Analysis](#market-analysis-endpoints)
  - [WebSocket](#websocket-endpoints)
  - [Webhooks](#webhook-endpoints)
- [Code Examples](#code-examples)
- [Error Handling](#error-handling)
- [Changelog](#changelog)

## Overview

The ARA AI API provides programmatic access to world-class financial predictions for stocks, cryptocurrencies, and forex.

**Base URL**: `http://localhost:8000` (development) or `https://api.ara-ai.com` (production)

**API Version**: v1

**Interactive Documentation**: 
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI Spec: `http://localhost:8000/openapi.json`

## Authentication

All API endpoints (except `/health` and `/`) require authentication.

### Authentication Methods

#### 1. JWT Token (Recommended for web applications)

```bash
# Login to get JWT token
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your-username",
    "password": "your-password"
  }'

# Use token in subsequent requests
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 7}'
```

#### 2. API Key (Recommended for server-to-server)

```bash
# Create API key (requires JWT token)
curl -X POST "http://localhost:8000/api/v1/auth/api-keys" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My API Key",
    "tier": "pro"
  }'

# Use API key in requests
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 7}'
```

### Registration

```bash
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your-username",
    "email": "your-email@example.com",
    "password": "your-secure-password",
    "tier": "free"
  }'
```

## Rate Limits

Rate limits vary by subscription tier:

| Tier | Requests/Hour | Requests/Day |
|------|---------------|--------------|
| Free | 100 | 1,000 |
| Pro | 1,000 | 10,000 |
| Enterprise | 10,000 | 100,000 |

### Rate Limit Headers

Every response includes rate limit information:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1699564800
```

When rate limit is exceeded, you'll receive a `429 Too Many Requests` response:

```json
{
  "error": "RateLimitExceeded",
  "message": "Rate limit exceeded. Try again in 3600 seconds.",
  "details": {
    "limit": 1000,
    "reset_at": "2024-11-15T12:00:00Z"
  }
}
```

## Endpoints

### Authentication Endpoints

#### POST /api/v1/auth/register

Register a new user account.

**Request Body**:
```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "tier": "free|pro|enterprise"
}
```

**Response** (201 Created):
```json
{
  "user_id": "uuid",
  "username": "string",
  "email": "string",
  "tier": "free",
  "created_at": "2024-11-15T10:00:00Z"
}
```

#### POST /api/v1/auth/login

Login and receive JWT token.

**Request Body**:
```json
{
  "username": "string",
  "password": "string"
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "user_id": "uuid",
    "username": "string",
    "tier": "pro"
  }
}
```

#### POST /api/v1/auth/api-keys

Create a new API key.

**Headers**: `Authorization: Bearer <token>`

**Request Body**:
```json
{
  "name": "string",
  "tier": "free|pro|enterprise"
}
```

**Response** (201 Created):
```json
{
  "key_id": "uuid",
  "api_key": "ara_live_1234567890abcdef",
  "name": "string",
  "tier": "pro",
  "created_at": "2024-11-15T10:00:00Z"
}
```

⚠️ **Important**: Save the `api_key` value - it's only shown once!

#### GET /api/v1/auth/api-keys

List all API keys for the authenticated user.

**Headers**: `Authorization: Bearer <token>`

**Response** (200 OK):
```json
{
  "api_keys": [
    {
      "key_id": "uuid",
      "name": "string",
      "tier": "pro",
      "created_at": "2024-11-15T10:00:00Z",
      "last_used": "2024-11-15T11:30:00Z"
    }
  ]
}
```

### Prediction Endpoints

#### POST /api/v1/predict

Generate price prediction for a single asset.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Request Body**:
```json
{
  "symbol": "AAPL",
  "days": 7,
  "analysis_level": "standard"
}
```

**Parameters**:
- `symbol` (required): Asset symbol (e.g., "AAPL", "BTC-USD", "EURUSD")
- `days` (optional): Prediction horizon in days (1-30, default: 5)
- `analysis_level` (optional): "quick", "standard", or "comprehensive" (default: "standard")

**Response** (200 OK):
```json
{
  "symbol": "AAPL",
  "asset_type": "stock",
  "current_price": 175.50,
  "predictions": [
    {
      "day": 1,
      "date": "2024-11-16",
      "predicted_price": 176.20,
      "predicted_return": 0.004,
      "confidence": 0.85,
      "lower_bound": 174.50,
      "upper_bound": 177.90,
      "contributing_factors": [
        {
          "name": "RSI",
          "value": 65.3,
          "contribution": 0.15,
          "description": "Momentum indicator showing bullish strength"
        }
      ]
    }
  ],
  "confidence": {
    "overall": 0.85,
    "model_agreement": 0.90,
    "data_quality": 0.95,
    "regime_stability": 0.80,
    "historical_accuracy": 0.78
  },
  "explanations": {
    "top_factors": [
      {
        "name": "RSI",
        "value": 65.3,
        "contribution": 0.15,
        "description": "Momentum indicator showing bullish strength"
      }
    ],
    "natural_language": "The prediction is driven primarily by strong momentum indicators and positive sentiment."
  },
  "regime": {
    "current_regime": "bull",
    "confidence": 0.82,
    "duration_in_regime": 15
  },
  "timestamp": "2024-11-15T10:00:00Z",
  "model_version": "v2.0.0"
}
```

#### POST /api/v1/predict/batch

Generate predictions for multiple assets.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Request Body**:
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "days": 5,
  "analysis_level": "quick"
}
```

**Response** (200 OK):
```json
{
  "predictions": [
    {
      "symbol": "AAPL",
      "predictions": [...],
      "confidence": {...}
    }
  ],
  "batch_id": "uuid",
  "total_symbols": 3,
  "successful": 3,
  "failed": 0,
  "timestamp": "2024-11-15T10:00:00Z"
}
```

#### GET /api/v1/predictions/{prediction_id}

Retrieve a previously generated prediction.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Response** (200 OK): Same as POST /api/v1/predict

### Backtesting Endpoints

#### POST /api/v1/backtest

Run backtest on historical data.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Request Body**:
```json
{
  "symbol": "AAPL",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "strategy": "buy_and_hold",
  "initial_capital": 10000
}
```

**Response** (200 OK):
```json
{
  "symbol": "AAPL",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "total_predictions": 252,
  "accuracy": 0.76,
  "precision": 0.78,
  "recall": 0.74,
  "f1_score": 0.76,
  "directional_accuracy": 0.72,
  "mae": 2.45,
  "rmse": 3.21,
  "sharpe_ratio": 1.85,
  "sortino_ratio": 2.10,
  "max_drawdown": -0.15,
  "win_rate": 0.68,
  "profit_factor": 1.92,
  "final_value": 12450.00,
  "total_return": 0.245
}
```

### Portfolio Endpoints

#### POST /api/v1/portfolio/optimize

Optimize portfolio allocation.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Request Body**:
```json
{
  "assets": ["AAPL", "MSFT", "GOOGL", "BTC-USD"],
  "risk_tolerance": "moderate",
  "constraints": {
    "max_position_size": 0.30,
    "min_position_size": 0.05
  },
  "optimization_method": "sharpe"
}
```

**Response** (200 OK):
```json
{
  "assets": ["AAPL", "MSFT", "GOOGL", "BTC-USD"],
  "optimal_weights": {
    "AAPL": 0.25,
    "MSFT": 0.30,
    "GOOGL": 0.25,
    "BTC-USD": 0.20
  },
  "expected_return": 0.15,
  "expected_volatility": 0.18,
  "sharpe_ratio": 0.83,
  "var_95": -0.025,
  "cvar_95": -0.035
}
```

#### POST /api/v1/portfolio/analyze

Analyze existing portfolio.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Request Body**:
```json
{
  "positions": {
    "AAPL": 0.30,
    "MSFT": 0.30,
    "GOOGL": 0.40
  }
}
```

**Response** (200 OK):
```json
{
  "total_value": 100000,
  "expected_return": 0.12,
  "volatility": 0.20,
  "sharpe_ratio": 0.60,
  "var_95": -0.030,
  "diversification_ratio": 0.75,
  "risk_contribution": {
    "AAPL": 0.28,
    "MSFT": 0.32,
    "GOOGL": 0.40
  }
}
```

### Model Endpoints

#### GET /api/v1/models/status

Get status of all models.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Response** (200 OK):
```json
{
  "models": [
    {
      "name": "transformer",
      "version": "v2.0.0",
      "status": "active",
      "accuracy": 0.78,
      "last_trained": "2024-11-10T10:00:00Z"
    }
  ],
  "total_models": 12,
  "active_models": 12
}
```

#### POST /api/v1/models/train

Train or retrain a model.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Request Body**:
```json
{
  "symbol": "AAPL",
  "model_type": "transformer",
  "data_period": "5y"
}
```

**Response** (202 Accepted):
```json
{
  "job_id": "uuid",
  "status": "queued",
  "estimated_time": 3600,
  "message": "Training job queued. Check status at /api/v1/models/jobs/{job_id}"
}
```

### Market Analysis Endpoints

#### GET /api/v1/market/regime

Get current market regime.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Query Parameters**:
- `symbol` (required): Asset symbol

**Response** (200 OK):
```json
{
  "symbol": "AAPL",
  "current_regime": "bull",
  "confidence": 0.82,
  "transition_probabilities": {
    "bull": 0.70,
    "bear": 0.10,
    "sideways": 0.15,
    "high_volatility": 0.05
  },
  "duration_in_regime": 15,
  "expected_duration": 30
}
```

#### GET /api/v1/market/sentiment

Get sentiment analysis.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Query Parameters**:
- `symbol` (required): Asset symbol
- `sources` (optional): Comma-separated list (e.g., "twitter,reddit,news")

**Response** (200 OK):
```json
{
  "symbol": "AAPL",
  "overall_sentiment": 0.65,
  "sentiment_by_source": {
    "twitter": 0.70,
    "reddit": 0.60,
    "news": 0.65
  },
  "sentiment_momentum": 0.05,
  "volume": 15420,
  "timestamp": "2024-11-15T10:00:00Z"
}
```

### WebSocket Endpoints

#### WS /ws/predictions/{symbol}

Stream real-time predictions.

**Query Parameters**:
- `token` (required): JWT token for authentication

**Example**:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/predictions/AAPL?token=YOUR_JWT_TOKEN');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New prediction:', data);
};
```

**Message Format**:
```json
{
  "type": "prediction_update",
  "symbol": "AAPL",
  "predicted_price": 176.50,
  "confidence": 0.85,
  "timestamp": "2024-11-15T10:00:00Z"
}
```

#### WS /ws/market-data/{symbol}

Stream real-time market data.

**Query Parameters**:
- `token` (required): JWT token for authentication

**Message Format**:
```json
{
  "type": "price_update",
  "symbol": "AAPL",
  "price": 175.50,
  "volume": 1000000,
  "timestamp": "2024-11-15T10:00:00Z"
}
```

### Webhook Endpoints

#### POST /api/v1/webhooks

Register a webhook.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Request Body**:
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["prediction_complete", "model_trained"],
  "secret": "your-webhook-secret"
}
```

**Response** (201 Created):
```json
{
  "webhook_id": "uuid",
  "url": "https://your-server.com/webhook",
  "events": ["prediction_complete", "model_trained"],
  "created_at": "2024-11-15T10:00:00Z"
}
```

#### GET /api/v1/webhooks

List all webhooks.

**Headers**: `Authorization: Bearer <token>` or `X-API-Key: <key>`

**Response** (200 OK):
```json
{
  "webhooks": [
    {
      "webhook_id": "uuid",
      "url": "https://your-server.com/webhook",
      "events": ["prediction_complete"],
      "status": "active",
      "created_at": "2024-11-15T10:00:00Z"
    }
  ]
}
```

## Code Examples

### Python

#### Basic Prediction

```python
import requests

# Setup
API_KEY = "your-api-key-here"
BASE_URL = "http://localhost:8000"
headers = {"X-API-Key": API_KEY}

# Make prediction
response = requests.post(
    f"{BASE_URL}/api/v1/predict",
    headers=headers,
    json={
        "symbol": "AAPL",
        "days": 7,
        "analysis_level": "standard"
    }
)

result = response.json()
print(f"Current price: ${result['current_price']:.2f}")
print(f"Predicted price (7 days): ${result['predictions'][-1]['predicted_price']:.2f}")
print(f"Confidence: {result['confidence']['overall']:.2%}")
```

#### Batch Predictions

```python
import requests

headers = {"X-API-Key": "your-api-key-here"}

response = requests.post(
    "http://localhost:8000/api/v1/predict/batch",
    headers=headers,
    json={
        "symbols": ["AAPL", "MSFT", "GOOGL", "BTC-USD"],
        "days": 5
    }
)

results = response.json()
for prediction in results["predictions"]:
    symbol = prediction["symbol"]
    price = prediction["predictions"][-1]["predicted_price"]
    confidence = prediction["confidence"]["overall"]
    print(f"{symbol}: ${price:.2f} (confidence: {confidence:.2%})")
```

#### Portfolio Optimization

```python
import requests

headers = {"X-API-Key": "your-api-key-here"}

response = requests.post(
    "http://localhost:8000/api/v1/portfolio/optimize",
    headers=headers,
    json={
        "assets": ["AAPL", "MSFT", "GOOGL", "BTC-USD"],
        "risk_tolerance": "moderate",
        "optimization_method": "sharpe"
    }
)

result = response.json()
print("Optimal Portfolio Allocation:")
for asset, weight in result["optimal_weights"].items():
    print(f"  {asset}: {weight:.2%}")
print(f"\nExpected Return: {result['expected_return']:.2%}")
print(f"Expected Volatility: {result['expected_volatility']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
```

### JavaScript/TypeScript

#### Basic Prediction

```javascript
const API_KEY = 'your-api-key-here';
const BASE_URL = 'http://localhost:8000';

async function getPrediction(symbol, days = 7) {
  const response = await fetch(`${BASE_URL}/api/v1/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': API_KEY
    },
    body: JSON.stringify({
      symbol,
      days,
      analysis_level: 'standard'
    })
  });

  const result = await response.json();
  return result;
}

// Usage
getPrediction('AAPL', 7).then(result => {
  console.log(`Current price: $${result.current_price}`);
  console.log(`Predicted price: $${result.predictions[result.predictions.length - 1].predicted_price}`);
  console.log(`Confidence: ${(result.confidence.overall * 100).toFixed(1)}%`);
});
```

#### WebSocket Connection

```javascript
const token = 'your-jwt-token';
const ws = new WebSocket(`ws://localhost:8000/ws/predictions/AAPL?token=${token}`);

ws.onopen = () => {
  console.log('Connected to prediction stream');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New prediction:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from prediction stream');
};
```

### cURL

#### Basic Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "symbol": "AAPL",
    "days": 7,
    "analysis_level": "standard"
  }'
```

#### Batch Predictions

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "days": 5
  }'
```

#### Backtest

```bash
curl -X POST "http://localhost:8000/api/v1/backtest" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "strategy": "buy_and_hold"
  }'
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "field": "Additional context"
  },
  "timestamp": "2024-11-15T10:00:00Z"
}
```

### Common Error Codes

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | ValidationError | Invalid request parameters |
| 401 | AuthenticationError | Missing or invalid authentication |
| 403 | AuthorizationError | Insufficient permissions |
| 404 | NotFoundError | Resource not found |
| 429 | RateLimitExceeded | Rate limit exceeded |
| 500 | InternalServerError | Server error |
| 503 | ServiceUnavailable | Service temporarily unavailable |

### Error Handling Examples

#### Python

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/api/v1/predict",
        headers={"X-API-Key": "your-api-key"},
        json={"symbol": "AAPL", "days": 7}
    )
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    error = e.response.json()
    print(f"Error: {error['error']}")
    print(f"Message: {error['message']}")
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

#### JavaScript

```javascript
async function getPrediction(symbol) {
  try {
    const response = await fetch('http://localhost:8000/api/v1/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'your-api-key'
      },
      body: JSON.stringify({ symbol, days: 7 })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`${error.error}: ${error.message}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Prediction failed:', error.message);
    throw error;
  }
}
```

## Changelog

### Version 1.0.0 (2024-11-15)

**Initial Release**

- ✨ Multi-asset predictions (stocks, crypto, forex)
- ✨ Advanced ML models (Transformer, CNN-LSTM, Ensemble)
- ✨ Real-time data integration
- ✨ Comprehensive backtesting
- ✨ Portfolio optimization
- ✨ Market regime detection
- ✨ Sentiment analysis
- ✨ Explainable AI features
- ✨ WebSocket support for real-time updates
- ✨ Webhook callbacks
- ✨ JWT and API key authentication
- ✨ Rate limiting by tier
- ✨ Interactive API documentation

---

**Need Help?**

- 📖 [Full Documentation](https://github.com/yourusername/ara-ai)
- 🐛 [Report Issues](https://github.com/yourusername/ara-ai/issues)
- 💬 [Community Forum](https://github.com/yourusername/ara-ai/discussions)
- 📧 [Email Support](mailto:support@ara-ai.com)
