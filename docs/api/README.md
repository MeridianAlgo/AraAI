# ARA AI API Documentation

Welcome to the ARA AI API documentation! This directory contains comprehensive guides and references for integrating with the ARA AI Prediction System.

## 📚 Documentation Index

### Getting Started

1. **[Quick Start Guide](../QUICK_START.md)** - Get up and running in 5 minutes
2. **[Installation Guide](../INSTALLATION.md)** - Detailed installation instructions
3. **[Authentication Guide](../AUTHENTICATION_GUIDE.md)** - Learn how to authenticate with the API

### API Reference

4. **[API Documentation](../API_DOCUMENTATION.md)** - Complete API endpoint reference
5. **[Code Examples](../API_CODE_EXAMPLES.md)** - Examples in Python, JavaScript, cURL, Go, Java, and C#
6. **[API Changelog](../API_CHANGELOG.md)** - Version history and breaking changes

### Interactive Documentation

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Spec**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

## 🚀 Quick Example

### Python

```python
import requests

# Setup
API_KEY = "your-api-key-here"
headers = {"X-API-Key": API_KEY}

# Make prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    headers=headers,
    json={"symbol": "AAPL", "days": 7}
)

result = response.json()
print(f"Predicted price: ${result['predictions'][-1]['predicted_price']:.2f}")
print(f"Confidence: {result['confidence']['overall']:.2%}")
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your-api-key-here',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ symbol: 'AAPL', days: 7 })
});

const result = await response.json();
console.log(`Predicted price: $${result.predictions[result.predictions.length - 1].predicted_price}`);
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 7}'
```

## 🔑 Authentication

The API supports two authentication methods:

1. **JWT Tokens** - Best for web applications
   ```
   Authorization: Bearer <token>
   ```

2. **API Keys** - Best for server-to-server
   ```
   X-API-Key: <key>
   ```

See the [Authentication Guide](../AUTHENTICATION_GUIDE.md) for detailed instructions.

## 📊 Available Endpoints

### Predictions
- `POST /api/v1/predict` - Single asset prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/predictions/{id}` - Retrieve prediction

### Backtesting
- `POST /api/v1/backtest` - Run historical backtest

### Portfolio
- `POST /api/v1/portfolio/optimize` - Optimize allocation
- `POST /api/v1/portfolio/analyze` - Analyze portfolio
- `POST /api/v1/portfolio/rebalance` - Get rebalancing recommendations

### Models
- `GET /api/v1/models/status` - Model health check
- `POST /api/v1/models/train` - Train/retrain models
- `GET /api/v1/models/compare` - Compare model performance

### Market Analysis
- `GET /api/v1/market/regime` - Market regime detection
- `GET /api/v1/market/sentiment` - Sentiment analysis
- `GET /api/v1/market/correlations` - Correlation analysis
- `GET /api/v1/market/indicators` - Technical indicators

### WebSocket
- `WS /ws/predictions/{symbol}` - Real-time predictions
- `WS /ws/market-data/{symbol}` - Real-time market data
- `WS /ws/alerts` - Alert notifications

### Webhooks
- `POST /api/v1/webhooks` - Register webhook
- `GET /api/v1/webhooks` - List webhooks
- `DELETE /api/v1/webhooks/{id}` - Delete webhook

## 🎯 Features

- **Multi-Asset Support**: Stocks, cryptocurrencies, and forex
- **Advanced ML Models**: Transformer, CNN-LSTM, and ensemble models
- **Real-Time Data**: Sub-second latency for market data
- **Comprehensive Analysis**: 100+ technical indicators, sentiment analysis
- **Risk Management**: Portfolio optimization, VaR, Sharpe ratio
- **Backtesting**: Validate predictions against historical data
- **Explainable AI**: SHAP values and feature importance

## 📈 Rate Limits

| Tier | Requests/Hour | Requests/Day |
|------|---------------|--------------|
| Free | 100 | 1,000 |
| Pro | 1,000 | 10,000 |
| Enterprise | 10,000 | 100,000 |

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Maximum requests per window
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Time when limit resets

## 🔧 SDKs and Libraries

### Official
- Python: Built-in client in examples
- JavaScript/TypeScript: Examples provided

### Community
- Go: Example provided
- Java: Example provided
- C#: Example provided
- Ruby: Coming soon
- PHP: Coming soon

## 🐛 Troubleshooting

### Common Issues

**401 Unauthorized**
- Check your API key or JWT token
- Ensure correct header format: `X-API-Key` or `Authorization: Bearer`

**429 Rate Limit Exceeded**
- Wait for rate limit to reset (check `X-RateLimit-Reset` header)
- Upgrade to higher tier
- Implement exponential backoff

**500 Internal Server Error**
- Check API status at `/health`
- Review request payload for errors
- Contact support if issue persists

See [Troubleshooting Guide](../TROUBLESHOOTING.md) for more help.

## 📞 Support

- 📖 [Full Documentation](../DOCUMENTATION_INDEX.md)
- 🐛 [Report Issues](https://github.com/yourusername/ara-ai/issues)
- 💬 [Community Forum](https://github.com/yourusername/ara-ai/discussions)
- 📧 [Email Support](mailto:support@ara-ai.com)

## 📝 License

This API is part of the ARA AI project, licensed under the MIT License.

---

**Ready to get started?** Check out the [Quick Start Guide](../QUICK_START.md) or dive into the [API Documentation](../API_DOCUMENTATION.md)!
