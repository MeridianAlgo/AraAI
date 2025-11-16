# ARA AI REST API

World-class financial prediction API built with FastAPI.

## Features

- **Predictions**: Generate price predictions for stocks, cryptocurrencies, and forex
- **Backtesting**: Run comprehensive backtests on historical data
- **Portfolio Management**: Optimize and analyze portfolios
- **Model Management**: Train, compare, and deploy ML models
- **Market Analysis**: Get market regime, sentiment, correlations, and indicators

## Quick Start

### Installation

Install required dependencies:

```bash
pip install fastapi uvicorn pydantic python-multipart
```

### Running the API

Start the development server:

```bash
uvicorn ara.api.app:app --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

### Testing

Run basic tests:

```bash
python -m ara.api.test_api
```

## API Endpoints

### Core Prediction Endpoints

#### POST /api/v1/predict
Generate prediction for a single asset.

**Request:**
```json
{
  "symbol": "AAPL",
  "days": 5,
  "asset_type": "stock",
  "analysis_level": "standard",
  "include_explanations": true
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "asset_type": "stock",
  "current_price": 175.50,
  "predictions": [
    {
      "day": 1,
      "date": "2024-01-15T00:00:00",
      "predicted_price": 176.20,
      "predicted_return": 0.40,
      "confidence": 0.85,
      "lower_bound": 174.50,
      "upper_bound": 177.90
    }
  ],
  "confidence": {
    "overall": 0.85,
    "model_agreement": 0.85,
    "data_quality": 0.90,
    "regime_stability": 0.80,
    "historical_accuracy": 0.75
  },
  "timestamp": "2024-01-14T12:00:00",
  "model_version": "4.0.0",
  "request_id": "abc-123"
}
```

#### POST /api/v1/predict/batch
Generate predictions for multiple assets (max 100).

**Request:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "days": 5,
  "analysis_level": "basic"
}
```

#### GET /api/v1/predictions/{id}
Get prediction status by ID.

### Backtesting Endpoints

#### POST /api/v1/backtest
Run backtest on historical data (async operation).

**Request:**
```json
{
  "symbol": "AAPL",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "initial_capital": 10000,
  "strategy": "buy_and_hold"
}
```

#### GET /api/v1/backtest/{job_id}
Get backtest job status and results.

### Portfolio Endpoints

#### POST /api/v1/portfolio/optimize
Optimize portfolio allocation.

**Request:**
```json
{
  "assets": ["AAPL", "MSFT", "GOOGL", "BTC-USD"],
  "risk_tolerance": "moderate",
  "optimization_method": "mpt"
}
```

**Response:**
```json
{
  "assets": ["AAPL", "MSFT", "GOOGL", "BTC-USD"],
  "optimal_weights": {
    "AAPL": 0.30,
    "MSFT": 0.25,
    "GOOGL": 0.25,
    "BTC-USD": 0.20
  },
  "expected_return": 0.15,
  "expected_volatility": 0.20,
  "sharpe_ratio": 0.75,
  "var_95": 0.05,
  "cvar_95": 0.07
}
```

#### GET /api/v1/portfolio/analyze
Analyze portfolio metrics.

#### POST /api/v1/portfolio/rebalance
Calculate rebalancing trades.

### Model Management Endpoints

#### GET /api/v1/models/status
Get status of all models.

#### POST /api/v1/models/train
Train model for a symbol (async operation).

**Request:**
```json
{
  "symbol": "AAPL",
  "data_period": "2y",
  "force_retrain": false
}
```

#### GET /api/v1/models/compare
Compare model performance.

#### POST /api/v1/models/deploy
Deploy model to production.

#### DELETE /api/v1/models/{model_id}
Delete a model.

### Market Analysis Endpoints

#### GET /api/v1/market/regime?symbol=AAPL
Get market regime for an asset.

**Response:**
```json
{
  "symbol": "AAPL",
  "current_regime": "bull",
  "confidence": 0.85,
  "transition_probabilities": {
    "bull": 0.70,
    "bear": 0.10,
    "sideways": 0.15,
    "high_volatility": 0.05
  },
  "duration_in_regime": 45,
  "expected_duration": 60
}
```

#### GET /api/v1/market/sentiment?symbol=AAPL&sources=twitter,reddit,news
Get market sentiment from multiple sources.

#### GET /api/v1/market/correlations?assets=AAPL,MSFT,GOOGL
Calculate correlations between assets.

#### GET /api/v1/market/indicators?symbol=AAPL
Get technical indicators for an asset.

### Health & Info Endpoints

#### GET /health
Health check endpoint.

#### GET /
API information and available endpoints.

## Authentication

Currently, the API accepts all requests. Authentication will be implemented in task 17.

To prepare for authentication, include the `X-API-Key` header in your requests:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/predict
```

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "additional": "context"
  },
  "timestamp": "2024-01-14T12:00:00",
  "request_id": "abc-123"
}
```

Common error codes:
- **400**: Bad Request (validation errors, invalid parameters)
- **404**: Not Found (resource doesn't exist)
- **500**: Internal Server Error (unexpected errors)

## Rate Limiting

Rate limiting will be implemented in task 17. Current implementation has no rate limits.

## Caching

Response caching is implemented for prediction endpoints with a 60-second TTL. Identical requests within the cache window will return cached results.

## Async Operations

Long-running operations (backtesting, model training) are handled asynchronously:

1. Submit request → Receive job ID
2. Poll status endpoint with job ID
3. Retrieve results when status is "completed"

Example:

```python
import requests
import time

# Submit backtest
response = requests.post("http://localhost:8000/api/v1/backtest", json={
    "symbol": "AAPL",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31"
})
job_id = response.json()["job_id"]

# Poll for completion
while True:
    status = requests.get(f"http://localhost:8000/api/v1/backtest/{job_id}")
    if status.json()["status"] == "completed":
        result = status.json()["result"]
        break
    time.sleep(5)
```

## Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Generate prediction
response = requests.post(f"{BASE_URL}/api/v1/predict", json={
    "symbol": "AAPL",
    "days": 5,
    "include_explanations": True
})

prediction = response.json()
print(f"Current price: ${prediction['current_price']}")
print(f"Predicted price (day 5): ${prediction['predictions'][4]['predicted_price']}")
print(f"Confidence: {prediction['confidence']['overall']}")
```

## JavaScript Client Example

```javascript
// Generate prediction
const response = await fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    symbol: 'AAPL',
    days: 5,
    include_explanations: true
  })
});

const prediction = await response.json();
console.log(`Current price: $${prediction.current_price}`);
console.log(`Predicted price (day 5): $${prediction.predictions[4].predicted_price}`);
console.log(`Confidence: ${prediction.confidence.overall}`);
```

## Production Deployment

For production deployment:

1. Use a production ASGI server (Gunicorn + Uvicorn workers)
2. Enable HTTPS/TLS
3. Configure CORS properly
4. Implement authentication and rate limiting
5. Set up monitoring and logging
6. Use Redis for caching and job storage
7. Use PostgreSQL for persistent storage

Example production command:

```bash
gunicorn ara.api.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## Next Steps

- Task 17: Implement authentication and authorization
- Task 18: Implement WebSocket endpoints for real-time updates
- Task 19: Create comprehensive API documentation

## Support

For issues or questions, please refer to the main ARA AI documentation.
