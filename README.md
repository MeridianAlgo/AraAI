# ARA AI - Financial Prediction System

**World-class AI-powered financial prediction platform with ensemble machine learning models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](TESTING.md)

> **DISCLAIMER**: This software is for educational and research purposes only. NOT financial advice. You are solely responsible for your investment decisions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
  - [Complete Documentation Index](DOCS_INDEX.md)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Support](#support)

---

## Overview

ARA AI is a comprehensive financial prediction platform that combines multiple machine learning models to provide accurate price predictions for stocks, forex, and cryptocurrencies. The system uses ensemble learning, technical analysis, sentiment analysis, and risk management to deliver robust predictions.

### Key Highlights

- **Multiple ML Models**: Transformer, CNN-LSTM, and 9-model ensemble systems
- **Multi-Asset Support**: Stocks, Forex, Crypto, and DeFi tokens
- **Production Ready**: REST API with authentication, rate limiting, and security features
- **Real-time Analysis**: Technical indicators, sentiment analysis, and market regime detection
- **Enterprise Features**: Backtesting, portfolio optimization, and risk management

---

## Features

### Machine Learning Models

- **Transformer Models** - Advanced time series prediction with attention mechanisms
- **CNN-LSTM Hybrid** - Convolutional-recurrent neural networks
- **Ensemble Systems** - 9-model ensemble (XGBoost, LightGBM, Random Forest, etc.)
- **Regime Detection** - Automatic market regime identification
- **Adaptive Learning** - Self-adjusting models based on market conditions

### Technical Analysis

- **44+ Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
- **Pattern Recognition**: Head & Shoulders, Triangles, Wedges
- **Volume Analysis**: OBV, MFI, VWAP, Accumulation/Distribution
- **Trend Indicators**: SMA, EMA, ADX, Parabolic SAR
- **Volatility Measures**: Bollinger Bands, Keltner Channels, ATR

### Sentiment Analysis

- **Twitter Sentiment**: Real-time tweet analysis
- **Reddit Analysis**: r/wallstreetbets and r/stocks sentiment
- **News Sentiment**: Financial news scoring with FinBERT
- **Social Media Aggregation**: Multi-source sentiment compilation

### Risk Management

- **Portfolio Optimization**: Modern Portfolio Theory, Black-Litterman
- **Risk Metrics**: VaR, CVaR, Sharpe Ratio, Sortino Ratio, Maximum Drawdown
- **Constraint Management**: Position limits, sector exposure controls
- **Automated Rebalancing**: Dynamic portfolio rebalancing

### Backtesting

- **Strategy Validation**: Historical data testing
- **Performance Metrics**: Returns, drawdown, win rate, Sharpe ratio
- **Walk-Forward Analysis**: Out-of-sample testing
- **Monte Carlo Simulation**: Risk assessment and scenario analysis

### Security Features

- **Authentication**: JWT tokens and API key authentication
- **Input Sanitization**: SQL injection and XSS protection
- **Encryption**: AES-256 for sensitive data
- **Rate Limiting**: Prevent abuse and DDoS attacks
- **Audit Logging**: Comprehensive security event tracking
- **Adversarial Defense**: Malicious input detection

See [SECURITY.md](SECURITY.md) for detailed security documentation.

### Monitoring & Observability

- **Prometheus Metrics**: Request rates, latencies, error tracking
- **Distributed Tracing**: OpenTelemetry integration
- **Health Checks**: Liveness and readiness probes
- **Grafana Dashboards**: Pre-built visualization dashboards
- **Automated Alerts**: Anomaly detection and alerting

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager
- Git (for cloning the repository)

### Installation

```bash
# Clone the repository
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Stock predictions
python scripts/ara.py AAPL --days 5
python scripts/ara.py TSLA --days 7

# Forex predictions
python scripts/ara_forex.py EURUSD --days 3
python scripts/ara_forex.py GBPUSD --days 5

# CSV data predictions
python scripts/ara_csv.py your_data.csv
```

### Start API Server

```bash
# Start the FastAPI server
python scripts/run_api.py

# Access points:
# - API: http://localhost:8000
# - Interactive Docs: http://localhost:8000/docs
# - Alternative Docs: http://localhost:8000/redoc
```

---

## Documentation

> **For a complete index of all documentation files, see [DOCS_INDEX.md](DOCS_INDEX.md)**

### Core Documentation

- **[README.md](README.md)** - This file, main project overview
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Comprehensive usage guide, model architecture, and technical details
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Guidelines for contributing to the project
- **[SECURITY.md](SECURITY.md)** - Security policy, vulnerability reporting, and security features
- **[TESTING.md](TESTING.md)** - Testing guide, CI/CD workflows, and quality assurance
- **[LICENSE](LICENSE)** - MIT License with additional terms and disclaimers

### Module Documentation

#### ARA Package Components

- **[ara/api/README.md](ara/api/README.md)** - REST API documentation and endpoints
- **[ara/api/auth/README.md](ara/api/auth/README.md)** - Authentication and authorization
- **[ara/api/webhooks/README.md](ara/api/webhooks/README.md)** - Webhook integration
- **[ara/api/websocket/README.md](ara/api/websocket/README.md)** - WebSocket real-time updates
- **[ara/alerts/README.md](ara/alerts/README.md)** - Alert system documentation
- **[ara/backtesting/README.md](ara/backtesting/README.md)** - Backtesting engine guide
- **[ara/cli/README.md](ara/cli/README.md)** - Command-line interface
- **[ara/compat/README.md](ara/compat/README.md)** - Compatibility layer
- **[ara/compat/QUICK_REFERENCE.md](ara/compat/QUICK_REFERENCE.md)** - Quick reference guide
- **[ara/config/README.md](ara/config/README.md)** - Configuration management
- **[ara/correlation/README.md](ara/correlation/README.md)** - Asset correlation analysis
- **[ara/currency/README.md](ara/currency/README.md)** - Currency and forex support
- **[ara/explainability/README.md](ara/explainability/README.md)** - Model explainability and interpretability
- **[ara/features/README.md](ara/features/README.md)** - Feature engineering and technical indicators
- **[ara/monitoring/README.md](ara/monitoring/README.md)** - Monitoring and metrics
- **[ara/risk/README.md](ara/risk/README.md)** - Risk management and portfolio optimization
- **[ara/security/README.md](ara/security/README.md)** - Security features and modules
- **[ara/sentiment/README.md](ara/sentiment/README.md)** - Sentiment analysis
- **[ara/visualization/README.md](ara/visualization/README.md)** - Data visualization and charting

#### Additional Documentation

- **[meridianalgo/README.md](meridianalgo/README.md)** - Core ML algorithms documentation
- **[scripts/README.md](scripts/README.md)** - Utility scripts and tools
- **[datasets/README.md](datasets/README.md)** - Sample datasets and data information

---

## Installation

### Standard Installation

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

### Development Installation

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy bandit safety

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Docker Installation

```bash
# Build Docker image
docker build -t ara-ai .

# Run container
docker run -p 8000:8000 ara-ai

# Docker Compose
docker-compose up -d
```

### Requirements

Main dependencies include:
- fastapi, uvicorn - Web framework and ASGI server
- torch, transformers - Deep learning models
- scikit-learn, xgboost, lightgbm - Machine learning
- pandas, numpy - Data manipulation
- yfinance - Market data
- rich - Console output formatting

See [requirements.txt](requirements.txt) for complete list.

---

## Usage Examples

### Python API Usage

#### Basic Stock Prediction

```python
from meridianalgo.unified_ml import UnifiedStockML

# Initialize the ML system
ml = UnifiedStockML()

# Make prediction
result = ml.predict('AAPL', days=5)

# Display results
print(f"Current Price: ${result['current_price']}")
for pred in result['predictions']:
    print(f"Day {pred['day']}: ${pred['price']} ({pred['change']}%)")
```

#### Using ARA Package

```python
from ara.data.base_provider import BaseDataProvider
from ara.features.calculator import IndicatorCalculator
from ara.models.ensemble import EnhancedEnsemble

# Fetch historical data
provider = BaseDataProvider()
data = provider.fetch_historical('AAPL', period='1y')

# Calculate technical indicators
calc = IndicatorCalculator()
features = calc.calculate(data, indicators=['rsi', 'macd', 'bb', 'atr'])

# Generate predictions
model = EnhancedEnsemble()
predictions = model.predict(features)

print(f"Predictions: {predictions}")
```

#### Portfolio Optimization

```python
from ara.risk.optimizer import PortfolioOptimizer

# Initialize optimizer
optimizer = PortfolioOptimizer()

# Optimize portfolio
assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
optimal_weights = optimizer.optimize(
    assets=assets,
    method='mpt',  # Modern Portfolio Theory
    risk_tolerance='moderate'
)

print(f"Optimal Weights: {optimal_weights}")
print(f"Expected Return: {optimal_weights['expected_return']}")
print(f"Risk (Volatility): {optimal_weights['volatility']}")
print(f"Sharpe Ratio: {optimal_weights['sharpe_ratio']}")
```

### REST API Usage

#### Basic Prediction Request

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "days": 5,
    "include_explanations": true
  }'
```

#### Batch Predictions

```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    "days": 5,
    "analysis_level": "standard"
  }'
```

#### Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Generate prediction
response = requests.post(
    f"{BASE_URL}/api/v1/predict",
    json={
        "symbol": "AAPL",
        "days": 5,
        "include_explanations": True
    }
)

prediction = response.json()
print(f"Current Price: ${prediction['current_price']}")
print(f"5-Day Prediction: ${prediction['predictions'][4]['predicted_price']}")
print(f"Confidence: {prediction['confidence']['overall']}")
```

#### JavaScript Client

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
console.log(`Current Price: $${prediction.current_price}`);
console.log(`5-Day Prediction: $${prediction.predictions[4].predicted_price}`);
console.log(`Confidence: ${prediction.confidence.overall}`);
```

---

## Architecture

### Project Structure

```
AraAI/
├── ara/                    # Main ARA AI package
│   ├── api/               # FastAPI REST API
│   ├── models/            # ML models (Transformer, CNN-LSTM, Ensemble)
│   ├── data/              # Data providers (stocks, crypto, forex)
│   ├── features/          # Technical indicator calculations
│   ├── risk/              # Portfolio optimization & risk management
│   ├── backtesting/       # Strategy validation engine
│   ├── sentiment/         # Sentiment analysis modules
│   ├── security/          # Authentication & security features
│   ├── monitoring/        # Metrics & observability
│   ├── alerts/            # Alert system
│   ├── visualization/     # Charts and reports
│   ├── correlation/       # Asset correlation analysis
│   ├── currency/          # Currency and forex utilities
│   ├── explainability/    # Model interpretability
│   ├── cli/               # Command-line interface
│   ├── config/            # Configuration management
│   └── compat/            # Compatibility layer
├── meridianalgo/          # Core ML algorithms
│   ├── unified_ml.py      # Unified stock ML system
│   ├── forex_ml.py        # Forex prediction system
│   ├── intelligent_model.py  # Advanced model architecture
│   ├── torch_ensemble.py  # PyTorch ensemble models
│   └── ai_analysis.py     # AI-powered analysis
├── scripts/               # Utility scripts
│   ├── ara.py            # Stock prediction script
│   ├── ara_forex.py      # Forex prediction script
│   ├── ara_csv.py        # CSV data prediction
│   ├── train_all.py      # Batch model training
│   └── run_api.py        # API server launcher
├── tests/                 # Test suite
├── datasets/              # Sample datasets
├── models/                # Trained model files
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT License
├── README.md             # This file
├── DOCUMENTATION.md      # Comprehensive guide
├── CONTRIBUTING.md       # Contribution guidelines
├── SECURITY.md           # Security policy
└── TESTING.md            # Testing guide
```

### Component Overview

#### Core Components

1. **Data Layer** (`ara/data/`): Data fetching from multiple sources (yFinance, Alpha Vantage, etc.)
2. **Feature Engineering** (`ara/features/`): Technical indicator calculation and feature extraction
3. **ML Models** (`ara/models/`, `meridianalgo/`): Ensemble learning, transformers, CNN-LSTM
4. **API Layer** (`ara/api/`): REST API with authentication and rate limiting
5. **Risk Management** (`ara/risk/`): Portfolio optimization and risk analysis
6. **Monitoring** (`ara/monitoring/`): Metrics, logging, and observability

#### ML Model Architecture

The system uses a multi-layer ensemble approach:

1. **Layer 1: Base Models**
   - XGBoost
   - LightGBM
   - Random Forest
   - Gradient Boosting
   - Extra Trees
   - AdaBoost

2. **Layer 2: Deep Learning Models**
   - Transformer (attention-based)
   - CNN-LSTM (hybrid)
   - Intelligent Model (1.6M parameters)

3. **Layer 3: Ensemble**
   - Weighted averaging
   - Confidence-based selection
   - Regime-aware switching

---

## Configuration

### Environment Variables

```bash
# API Keys (Optional)
export ALPHA_VANTAGE_API_KEY=your_key
export NEWS_API_KEY=your_key
export TWITTER_BEARER_TOKEN=your_token

# Database (Optional)
export DATABASE_URL=postgresql://user:pass@host:port/db

# Redis (Optional)
export REDIS_URL=redis://localhost:6379

# Security
export ARA_SECRET_KEY=your-secret-key
export ARA_JWT_SECRET=your-jwt-secret

# API Configuration
export ARA_API_HOST=0.0.0.0
export ARA_API_PORT=8000
export ARA_RATE_LIMIT=100
```

### Configuration File

Edit `ara/config/config.example.yaml` and save as `config.yaml`:

```yaml
# Data providers
data:
  default_provider: yfinance
  cache_enabled: true
  cache_ttl: 3600
  backup_providers:
    - alpha_vantage
    - polygon

# ML models
models:
  use_gpu: false
  ensemble_weights: auto
  retraining_interval: 90  # days
  model_path: ./models/
  
# API settings
api:
  host: 0.0.0.0
  port: 8000
  rate_limit: 100  # requests per minute
  enable_cors: true
  allowed_origins:
    - https://yourdomain.com
  
# Security
security:
  enable_authentication: true
  jwt_expiration: 3600  # seconds
  api_key_length: 32
  
# Monitoring
monitoring:
  enable_prometheus: true
  enable_tracing: false
  log_level: INFO
```

---

## API Documentation

### Core Endpoints

#### Predictions

- `POST /api/v1/predict` - Generate single prediction
- `POST /api/v1/predict/batch` - Batch predictions (max 100 symbols)
- `GET /api/v1/predictions/{id}` - Get prediction status

#### Market Data

- `GET /api/v1/market/{symbol}` - Get current market data
- `GET /api/v1/market/{symbol}/indicators` - Calculate technical indicators
- `GET /api/v1/market/{symbol}/sentiment` - Get sentiment analysis
- `GET /api/v1/market/regime?symbol={symbol}` - Get market regime

#### Portfolio Management

- `POST /api/v1/portfolio/optimize` - Optimize portfolio allocation
- `GET /api/v1/portfolio/analyze` - Analyze portfolio metrics
- `POST /api/v1/portfolio/backtest` - Run backtest
- `POST /api/v1/portfolio/rebalance` - Calculate rebalancing trades

#### Model Management

- `GET /api/v1/models/status` - Get model status
- `POST /api/v1/models/train` - Train model (async)
- `GET /api/v1/models/compare` - Compare model performance
- `POST /api/v1/models/deploy` - Deploy model to production
- `DELETE /api/v1/models/{model_id}` - Delete model

#### Backtesting

- `POST /api/v1/backtest` - Run backtest (async)
- `GET /api/v1/backtest/{job_id}` - Get backtest results

#### Health & Info

- `GET /health` - Health check
- `GET /` - API information

### Interactive Documentation

Once the API server is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

For detailed API documentation, see [ara/api/README.md](ara/api/README.md).

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=ara --cov=meridianalgo --cov-report=html

# Run specific test categories
pytest tests/test_unit/ -v              # Unit tests
pytest tests/test_integration/ -v      # Integration tests
pytest tests/test_security.py -v       # Security tests

# Run performance benchmarks
pytest tests/test_performance/ -v

# Run module tests
python test_all_modules.py
```

### Test Coverage

- **Target Coverage**: 80%+
- **Current Status**: 180/180 modules passing
- **Test Categories**: Unit, Integration, Security, Performance

### CI/CD

The project includes GitHub Actions workflows for:

- Automated testing on push/PR
- Security scanning
- Code quality checks
- Model training workflows
- Automated releases

For detailed testing documentation, see [TESTING.md](TESTING.md).

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and code quality checks
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Quality

```bash
# Format code
black ara/ meridianalgo/

# Lint code
flake8 ara/ meridianalgo/ --max-line-length=100

# Type checking
mypy ara/ meridianalgo/

# Security scan
bandit -r ara/ meridianalgo/
```

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests for new features
- Update documentation for API changes
- Use type hints for function signatures
- Add docstrings to public functions and classes

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Security

### Reporting Vulnerabilities

**Please DO NOT create public GitHub issues for security vulnerabilities.**

To report a security issue:
- Email: security@meridianalgo.com
- Create a private security advisory on GitHub

### Security Features

- Input validation and sanitization
- SQL injection prevention
- XSS protection
- API key encryption
- JWT authentication
- Rate limiting
- Audit logging
- Adversarial ML defense

For detailed security information, see [SECURITY.md](SECURITY .md).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Additional Terms

- **Financial Disclaimer**: Not financial advice
- **No Investment Recommendations**: Consult qualified professionals
- **Risk Acknowledgment**: Use at your own risk
- **No Warranties**: Provided "as is" without guarantees

### Third-Party Licenses

This software incorporates:
- Scikit-learn (BSD License)
- XGBoost (Apache License 2.0)
- LightGBM (MIT License)
- PyTorch (BSD License)
- Transformers by Hugging Face (Apache License 2.0)
- And other open-source libraries

---

## Support

### Getting Help

- **Documentation**: Comprehensive guides in this repository
- **GitHub Issues**: [Report bugs or request features](https://github.com/MeridianAlgo/AraAI/issues)
- **GitHub Discussions**: [Ask questions and discuss](https://github.com/MeridianAlgo/AraAI/discussions)

### Community

- Follow development progress on GitHub
- Star the repository if you find it useful
- Share your feedback and suggestions

### Production Support

For production deployment assistance or enterprise support, please contact:
- Email: support@meridianalgo.com

---

## Disclaimer

**IMPORTANT**: This software is provided for educational and research purposes only. It is NOT financial advice and should NOT be used for actual trading or investment decisions without proper due diligence. 

- Past performance does not guarantee future results
- All predictions are probabilistic and may be wrong
- You are solely responsible for your investment decisions
- Consult with qualified financial professionals before investing
- The authors and contributors are not liable for any financial losses

---

## Acknowledgments

Built with:
- FastAPI, Uvicorn - Web framework
- PyTorch - Deep learning
- Scikit-learn, XGBoost, LightGBM - Machine learning
- Pandas, NumPy - Data processing
- yFinance - Market data
- Transformers - NLP models
- And many other excellent open-source projects

---

**Maintained by**: [MeridianAlgo](https://github.com/MeridianAlgo)  
**Last Updated**: 2025-11-25  
**Version**: 4.0.0
