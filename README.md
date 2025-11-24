# 🚀 ARA AI - Financial Prediction System

**World-class AI-powered financial prediction platform with ensemble ML models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-180%2F180%20passing-brightgreen.svg)](TEST_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **DISCLAIMER**: This software is for educational and research purposes only. NOT financial advice. You are solely responsible for your investment decisions.

---

## ✨ Features

- **Multiple ML Models**: Transformer, CNN-LSTM, Ensemble systems
- **Multi-Asset Support**: Stocks, Forex, Crypto, DeFi
- **Technical Analysis**: 44+ indicators (RSI, MACD, Bollinger Bands, etc.)
- **Sentiment Analysis**: Twitter, Reddit, News, FinBERT
- **Risk Management**: Portfolio optimization, risk calculators
- **Backtesting Engine**: Validate strategies with historical data
- **REST API**: FastAPI with WebSocket support
- **Real-time Monitoring**: Prometheus metrics, Grafana dashboards
- **Production Ready**: Authentication, rate limiting, security features

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/meridianalgo/AraAI.git
cd AraAI

# Install dependencies
pip install -r requirements.txt
```

### Make Predictions

```bash
# Stock predictions
python scripts/ara.py AAPL --days 5
python scripts/ara.py TSLA --days 7

# Forex predictions
python scripts/ara_forex.py EURUSD --days 3

# CSV data predictions
python scripts/ara_csv.py data.csv
```

### Start API Server

```bash
python scripts/run_api.py

# Access at:
# - API: http://localhost:8000
# - Docs: http://localhost:8000/docs
```

---

## 📊 Usage Examples

### Python API

```python
from meridianalgo.unified_ml import UnifiedStockML

# Initialize
ml = UnifiedStockML()

# Make prediction
result = ml.predict('AAPL', days=5)
print(result)
```

### Using ARA Package

```python
from ara.data.base_provider import BaseDataProvider
from ara.features.calculator import IndicatorCalculator
from ara.models.ensemble import EnhancedEnsemble

# Fetch data
provider = BaseDataProvider()
data = provider.fetch_historical('AAPL', period='1y')

# Calculate indicators
calc = IndicatorCalculator()
features = calc.calculate(data, ['rsi', 'macd', 'bb'])

# Make predictions
model = EnhancedEnsemble()
predictions = model.predict(features)
```

### REST API

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 5}'
```

---

## 🏗️ Architecture

### Core Components

```
ara/
├── api/              # FastAPI application
├── models/           # ML models (Transformer, CNN-LSTM, Ensemble)
├── data/             # Data providers (stocks, crypto, forex)
├── features/         # Technical indicators
├── risk/             # Portfolio optimization & risk management
├── backtesting/      # Strategy validation
├── sentiment/        # Sentiment analysis
├── visualization/    # Charts and reports
├── security/         # Authentication & security
└── monitoring/       # Metrics & observability

meridianalgo/
├── unified_ml.py     # Unified ML system
├── torch_ensemble.py # PyTorch ensemble models
└── ai_analysis.py    # AI-powered analysis
```

### ML Models

1. **Transformer Models** - Time series prediction with attention
2. **CNN-LSTM** - Hybrid convolutional-recurrent networks
3. **Ensemble Systems** - 9-model ensemble (XGBoost, LightGBM, RF, etc.)
4. **Regime Detection** - Market regime identification
5. **Adaptive Models** - Self-adjusting to market conditions

---

## 🔧 Configuration

### Config File

Edit `ara/config/config.example.yaml` and save as `config.yaml`:

```yaml
# Data providers
data:
  default_provider: yfinance
  cache_enabled: true
  cache_ttl: 3600

# ML models
models:
  use_gpu: false
  ensemble_weights: auto
  retraining_interval: 90

# API settings
api:
  host: 0.0.0.0
  port: 8000
  rate_limit: 100
```

### Environment Variables

```bash
# Optional API keys
export ALPHA_VANTAGE_API_KEY=your_key
export NEWS_API_KEY=your_key

# Database (optional)
export DATABASE_URL=postgresql://...

# Redis (optional)
export REDIS_URL=redis://localhost:6379
```

---

## 📈 Features Deep Dive

### Technical Indicators (44+)

- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, CCI, ROC
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: OBV, MFI, VWAP, A/D Line
- **Pattern Recognition**: Head & Shoulders, Triangles, Wedges

### Sentiment Analysis

- **Twitter**: Real-time tweet analysis
- **Reddit**: r/wallstreetbets, r/stocks sentiment
- **News**: Financial news sentiment scoring
- **FinBERT**: Transformer-based financial sentiment

### Risk Management

- **Portfolio Optimization**: Modern Portfolio Theory, Black-Litterman
- **Risk Metrics**: VaR, CVaR, Sharpe Ratio, Sortino Ratio
- **Constraint Management**: Position limits, sector exposure
- **Rebalancing**: Automated portfolio rebalancing

### Backtesting

- **Strategy Validation**: Test strategies on historical data
- **Performance Metrics**: Returns, drawdown, win rate
- **Walk-Forward Analysis**: Out-of-sample testing
- **Monte Carlo Simulation**: Risk assessment

---

## 🔐 Security Features

- **Authentication**: JWT tokens, API keys
- **Rate Limiting**: Prevent abuse
- **Input Sanitization**: SQL injection, XSS protection
- **Encryption**: AES-256 for sensitive data
- **Audit Logging**: Track all security events
- **Adversarial Protection**: Detect malicious inputs

---

## 📊 Monitoring & Observability

- **Prometheus Metrics**: Request rates, latencies, errors
- **Distributed Tracing**: OpenTelemetry integration
- **Error Tracking**: Sentry integration
- **Grafana Dashboards**: Pre-built visualization dashboards
- **Health Checks**: Liveness and readiness probes
- **Alerts**: Automated alerting for anomalies

---

## 🧪 Testing

### Run All Tests

```bash
# Comprehensive module tests
python test_all_modules.py

# Unit tests
pytest tests/

# Integration tests
pytest tests/test_integration.py

# Performance benchmarks
pytest tests/performance/
```

### Test Results

- **180/180 modules** passing (100%)
- **0 import errors**
- **0 syntax errors**

See [TEST_REPORT.md](TEST_REPORT.md) for details.

---

## 🚢 Deployment

### Docker

```bash
# Build image
docker build -t ara-ai .

# Run container
docker run -p 8000:8000 ara-ai

# Docker Compose
docker-compose up -d
```

### Kubernetes

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods -l app=ara-ai
```

### Production Checklist

- [ ] Set environment variables
- [ ] Configure database connection
- [ ] Set up Redis for caching
- [ ] Enable HTTPS/TLS
- [ ] Configure rate limiting
- [ ] Set up monitoring
- [ ] Configure backups
- [ ] Review security settings

---

## 📚 API Documentation

### Endpoints

#### Predictions
- `POST /api/v1/predict` - Make single prediction
- `POST /api/v1/predict/batch` - Batch predictions
- `GET /api/v1/predictions/{id}` - Get prediction status

#### Market Data
- `GET /api/v1/market/{symbol}` - Get market data
- `GET /api/v1/market/{symbol}/indicators` - Calculate indicators
- `GET /api/v1/market/{symbol}/sentiment` - Get sentiment

#### Portfolio
- `POST /api/v1/portfolio/optimize` - Optimize portfolio
- `GET /api/v1/portfolio/risk` - Calculate risk metrics
- `POST /api/v1/portfolio/backtest` - Run backtest

#### Authentication
- `POST /api/v1/auth/register` - Register user
- `POST /api/v1/auth/login` - Login
- `POST /api/v1/auth/refresh` - Refresh token

### Interactive Docs

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## 🛠️ Development

### Project Structure

```
AraAI/
├── ara/                  # Main package
├── meridianalgo/        # ML algorithms
├── scripts/             # Utility scripts
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Example code
├── deployment/          # Deployment configs
├── datasets/            # Sample data
├── models/              # Trained models
├── requirements.txt     # Dependencies
├── docker-compose.yml   # Docker setup
└── README.md           # This file
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Quality

```bash
# Format code
black ara/ meridianalgo/

# Lint code
flake8 ara/ meridianalgo/

# Type checking
mypy ara/ meridianalgo/
```

---

## 📖 Additional Resources

- **System Status**: [SYSTEM_STATUS.md](SYSTEM_STATUS.md)
- **Test Report**: [TEST_REPORT.md](TEST_REPORT.md)
- **Quick Start**: [QUICK_START.md](QUICK_START.md)
- **License**: [LICENSE](LICENSE)

---

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/AraAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/AraAI/discussions)
- **Email**: support@example.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This software is provided for educational and research purposes only. It is NOT financial advice and should NOT be used for actual trading or investment decisions. Past performance does not guarantee future results. You are solely responsible for your investment decisions and any financial losses.

---

## 🌟 Acknowledgments

- Built with FastAPI, PyTorch, scikit-learn, and pandas
- Inspired by modern quantitative finance research
- Community contributions and feedback

---

**Made with ❤️ by the ARA AI Team at MeridianAlgo**

*Last Updated: November 17, 2025*
