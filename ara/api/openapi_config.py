"""
OpenAPI/Swagger configuration and customization
"""

from typing import Dict, Any


# API metadata
API_METADATA = {
    "title": "ARA AI Prediction API",
    "description": """
# ARA AI Prediction API

World-class financial prediction system for stocks, cryptocurrencies, and forex.

## Features

- **Multi-Asset Support**: Stocks, cryptocurrencies, and forex predictions
- **Advanced ML Models**: Transformer, CNN-LSTM, and ensemble models
- **Real-Time Data**: Sub-second latency for market data
- **Comprehensive Analysis**: 100+ technical indicators, sentiment analysis, on-chain metrics
- **Risk Management**: Portfolio optimization, VaR, Sharpe ratio calculations
- **Backtesting**: Validate predictions against historical data
- **Explainable AI**: SHAP values and feature importance for every prediction

## Authentication

All API endpoints (except `/health` and `/`) require authentication using either:

1. **JWT Token**: Include in `Authorization` header as `Bearer <token>`
2. **API Key**: Include in `X-API-Key` header

Get your API key by registering at `/api/v1/auth/register` and logging in at `/api/v1/auth/login`.

## Rate Limits

Rate limits vary by tier:

- **Free**: 100 requests/hour
- **Pro**: 1,000 requests/hour  
- **Enterprise**: 10,000 requests/hour

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time when limit resets (Unix timestamp)

## Versioning

The API uses URL versioning (e.g., `/api/v1/`). Current version: **v1**

## Support

- Documentation: https://github.com/yourusername/ara-ai
- Issues: https://github.com/yourusername/ara-ai/issues
- Email: support@ara-ai.com
    """,
    "version": "1.0.0",
    "contact": {
        "name": "ARA AI Support",
        "url": "https://github.com/yourusername/ara-ai",
        "email": "support@ara-ai.com",
    },
    "license": {"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    "servers": [
        {"url": "http://localhost:8000", "description": "Development server"},
        {"url": "https://api.ara-ai.com", "description": "Production server"},
    ],
}


# Tag metadata for grouping endpoints
TAGS_METADATA = [
    {"name": "health", "description": "Health check and service status endpoints"},
    {
        "name": "auth",
        "description": "Authentication and authorization endpoints. Register, login, and manage API keys.",
    },
    {
        "name": "predictions",
        "description": """
Prediction endpoints for generating price forecasts.

Supports stocks, cryptocurrencies, and forex with multiple analysis levels:
- **quick**: Fast predictions with basic analysis
- **standard**: Balanced speed and detail (default)
- **comprehensive**: Full analysis with all features
        """,
    },
    {
        "name": "backtesting",
        "description": "Backtest predictions against historical data to validate model performance.",
    },
    {
        "name": "portfolio",
        "description": "Portfolio optimization and analysis endpoints. Calculate optimal allocations, risk metrics, and rebalancing strategies.",
    },
    {
        "name": "models",
        "description": "Model management endpoints. Check status, train models, compare performance, and deploy new versions.",
    },
    {
        "name": "market",
        "description": "Market analysis endpoints. Get regime detection, sentiment analysis, correlations, and technical indicators.",
    },
    {
        "name": "websocket",
        "description": "WebSocket endpoints for real-time streaming data and predictions.",
    },
    {
        "name": "webhooks",
        "description": "Webhook management for receiving callbacks on prediction completion and other events.",
    },
]


# Security schemes
SECURITY_SCHEMES = {
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT token obtained from `/api/v1/auth/login`",
    },
    "apiKeyAuth": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key obtained from `/api/v1/auth/api-keys`",
    },
}


# Code examples for different languages
CODE_EXAMPLES = {
    "python": {
        "predict": """
import requests

# Using API key
headers = {
    "X-API-Key": "your-api-key-here"
}

# Make prediction request
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    headers=headers,
    json={
        "symbol": "AAPL",
        "days": 7,
        "analysis_level": "standard"
    }
)

result = response.json()
print(f"Predicted price in 7 days: ${result['predictions'][-1]['predicted_price']:.2f}")
print(f"Confidence: {result['confidence']['overall']:.2%}")
        """,
        "batch_predict": """
import requests

headers = {"X-API-Key": "your-api-key-here"}

response = requests.post(
    "http://localhost:8000/api/v1/predict/batch",
    headers=headers,
    json={
        "symbols": ["AAPL", "MSFT", "GOOGL"],
        "days": 5
    }
)

results = response.json()
for result in results["predictions"]:
    symbol = result["symbol"]
    price = result["predictions"][-1]["predicted_price"]
    print(f"{symbol}: ${price:.2f}")
        """,
        "backtest": """
import requests

headers = {"X-API-Key": "your-api-key-here"}

response = requests.post(
    "http://localhost:8000/api/v1/backtest",
    headers=headers,
    json={
        "symbol": "AAPL",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "strategy": "buy_and_hold"
    }
)

result = response.json()
print(f"Accuracy: {result['accuracy']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        """,
    },
    "javascript": {
        "predict": """
// Using fetch API
const response = await fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-api-key-here'
  },
  body: JSON.stringify({
    symbol: 'AAPL',
    days: 7,
    analysis_level: 'standard'
  })
});

const result = await response.json();
console.log(`Predicted price: $${result.predictions[result.predictions.length - 1].predicted_price}`);
console.log(`Confidence: ${(result.confidence.overall * 100).toFixed(1)}%`);
        """,
        "websocket": """
// WebSocket connection for real-time predictions
const ws = new WebSocket('ws://localhost:8000/ws/predictions/AAPL?token=your-jwt-token');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New prediction:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
        """,
    },
    "curl": {
        "predict": """
curl -X POST "http://localhost:8000/api/v1/predict" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-api-key-here" \\
  -d '{
    "symbol": "AAPL",
    "days": 7,
    "analysis_level": "standard"
  }'
        """,
        "batch_predict": """
curl -X POST "http://localhost:8000/api/v1/predict/batch" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-api-key-here" \\
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "days": 5
  }'
        """,
        "backtest": """
curl -X POST "http://localhost:8000/api/v1/backtest" \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-api-key-here" \\
  -d '{
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "strategy": "buy_and_hold"
  }'
        """,
    },
}


def get_openapi_config() -> Dict[str, Any]:
    """
    Get complete OpenAPI configuration

    Returns:
        OpenAPI configuration dictionary
    """
    return {
        **API_METADATA,
        "tags": TAGS_METADATA,
        "components": {"securitySchemes": SECURITY_SCHEMES},
    }
