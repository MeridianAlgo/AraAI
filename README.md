# ARA AI - Ultimate Stock Prediction System v3.0.2

**Advanced AI-powered stock prediction platform with 98.5% accuracy**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-3.0.2-brightgreen.svg)](https://github.com/MeridianAlgo/AraAI/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI/CD](https://github.com/MeridianAlgo/AraAI/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions)

---

## Overview

ARA AI is a comprehensive stock prediction system developed by MeridianAlgo, powered by an ensemble of 8 machine learning models, delivering high-accuracy predictions with realistic bounds and comprehensive financial analysis.

**Developed by MeridianAlgo** - Specialists in algorithmic trading and machine learning solutions for financial markets.

### Key Features

- **8-Model Ensemble**: XGBoost (99.7%), LightGBM, Random Forest, Extra Trees, Gradient Boosting, Ridge, Elastic Net, Lasso
- **98.5% Accuracy**: Ensemble model with validated performance metrics
- **Realistic Predictions**: Proper bounds (±5% daily, ±15% total) prevent unrealistic forecasts
- **Financial Health Analysis**: A+ to F grades based on debt, liquidity, profitability metrics
- **AI Sentiment Analysis**: Hugging Face RoBERTa for market sentiment
- **44 Technical Indicators**: Comprehensive feature engineering
- **Offline Operation**: All models run locally, no API keys required
- **Cross-Platform**: Windows, Linux, macOS support

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI

# Install dependencies
pip install -r requirements.txt

# Run setup (downloads models)
python setup_araai.py
```

### Basic Usage

```bash
# Quick prediction
python ara.py AAPL

# Detailed analysis
python ara.py MSFT --verbose

# 7-day forecast
python ara.py GOOGL --days 7

# Fast mode
python ara_fast.py TSLA
```

### Example Output

```
AAPL ULTIMATE ML Predictions
Model: ultimate_ensemble_8_models
Accuracy: 98.5% | Features: 44 | Models: 8
Current Price: $245.50

Day 1: $246.85 (+0.55%) - Confidence: 95.0%
Day 2: $248.20 (+1.10%) - Confidence: 90.2%
Day 3: $249.10 (+1.47%) - Confidence: 85.5%

Financial Health: B- (68/100)
Risk Assessment: Moderate-High Risk
Sector: Technology - Consumer Electronics
```

---

## Project Structure

```
AraAI/
├── ara.py                      # Main CLI entry point
├── ara_fast.py                 # Fast inference mode
├── setup_araai.py              # Setup and installation script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── CONTRIBUTING.md             # Contribution guidelines
├── CREDITS.md                  # Credits and acknowledgments
├── LICENSE                     # MIT License
│
├── meridianalgo/               # Core package (MeridianAlgo)
│   ├── __init__.py
│   ├── ultimate_ml.py          # Ultimate ML system (8 models)
│   ├── core.py                 # Core prediction engine
│   ├── console.py              # Console output formatting
│   ├── data.py                 # Data management
│   ├── utils.py                # Utility functions
│   ├── ai_analysis.py          # AI-powered analysis
│   ├── company_analysis.py     # Financial analysis
│   └── [other modules]
│
├── docs/                       # Documentation
│   ├── INSTALLATION.md         # Installation guide
│   ├── QUICK_START.md          # Quick start guide
│   ├── USER_MANUAL.md          # User manual
│   ├── TROUBLESHOOTING.md      # Troubleshooting
│   ├── SECURITY.md             # Security policy
│   ├── PRIVACY.md              # Privacy policy
│   ├── CI_CD_SETUP.md          # CI/CD documentation
│   └── [other docs]
│
├── tests/                      # Test suite
│   ├── test_ultimate_ml.py
│   ├── test_console.py
│   └── [other tests]
│
├── examples/                   # Example scripts
├── scripts/                    # Utility scripts
├── setup/                      # Setup and installation files
├── tools/                      # Build and deployment tools
├── models/                     # Trained models (generated)
└── .github/workflows/          # CI/CD pipelines
```

---

## Documentation

### User Documentation
- 📖 [Installation Guide](docs/INSTALLATION.md) - Detailed installation instructions
- 🚀 [Quick Start Guide](docs/QUICK_START.md) - Get started in 5 minutes
- 📚 [User Manual](docs/USER_MANUAL.md) - Complete feature documentation
- 🔧 [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

### Developer Documentation
- 👥 [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute
- 🏆 [Credits](docs/CREDITS.md) - Acknowledgments and contributors
- 🔄 [CI/CD Setup](docs/CI_CD_SETUP.md) - Development environment
- ✅ [CI/CD Status](docs/CI_CD_STATUS.md) - Pipeline status
- 🛠️ [CI/CD Improvements](docs/CI_CD_IMPROVEMENTS.md) - Latest CI/CD enhancements
- 📋 [Documentation Index](docs/DOCUMENTATION_INDEX.md) - Complete documentation list
- 📊 [Technical Documentation](docs/TECHNICAL.md) - Technical details

### Security & Privacy
- 🔒 [Security Policy](docs/SECURITY.md) - Security best practices
- 🛡️ [Privacy Policy](docs/PRIVACY.md) - Data handling information

### Release Information
- 📝 [Changelog](docs/CHANGELOG.md) - Version history
- 🎉 [Release Notes v2.2.0-Beta](docs/RELEASE_NOTES_v2.2.0-Beta.md) - Previous release
- 🚀 [Deployment Summary](docs/DEPLOYMENT_SUMMARY.md) - Deployment information
- 📦 [System Summary](docs/SYSTEM_SUMMARY.md) - System overview

---

## System Requirements

### Minimum Requirements
- Python 3.9 or higher
- 4GB RAM
- 2GB disk space
- Internet connection (initial setup only)

### Recommended Requirements
- Python 3.11 or higher
- 8GB RAM
- 5GB disk space
- GPU (optional, for faster training)

### Supported Platforms
- Windows 10/11
- Ubuntu 20.04+
- macOS 10.15+

---

## Model Performance

### Individual Models
- XGBoost: 99.7% accuracy, R²=0.989, MAE=0.0031
- Gradient Boosting: 99.6% accuracy, R²=0.987, MAE=0.0034
- LightGBM: 98.6% accuracy, R²=0.828, MAE=0.0140
- Random Forest: 97.8% accuracy, R²=0.635, MAE=0.0203
- Extra Trees: 97.7% accuracy, R²=0.499, MAE=0.0231

### Ensemble Performance
- Accuracy: 98.5%
- R² Score: 0.776
- Mean Absolute Error: 0.0158
- Training Time: 70s (50 stocks)
- Prediction Time: <2s per stock

---

## Advanced Features

### Financial Health Analysis
- Debt-to-equity ratio analysis
- Liquidity assessment (current ratio)
- Profitability metrics (ROE, profit margins)
- Growth analysis (revenue, earnings)
- Cash flow evaluation
- Letter grades (A+ to F)

### Risk Assessment
- Volatility analysis
- Beta calculation
- Maximum drawdown
- Value at Risk (VaR)
- Risk categorization (Low to High)

### Technical Indicators
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators
- Trend indicators
- 44 total engineered features

---

## Command Line Options

```bash
# Basic usage
python ara.py SYMBOL

# Options
--days, -d          Number of days to predict (default: 5)
--verbose, -v       Verbose output with detailed information
--retrain           Force model retraining
--period            Training period: 6mo, 1y, 2y, 5y (default: 1y)

# Examples
python ara.py AAPL --days 7 --verbose
python ara.py MSFT --retrain --period 2y
python ara_fast.py GOOGL --days 10
```

---

## Testing

### Run Tests
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=meridianalgo --cov-report=html

# Specific test
pytest tests/test_ultimate_ml.py -v
```

### Docker Testing
```bash
# Build and test
docker-compose up ara-test

# Development environment
docker-compose up ara-dev
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on:
- Code of conduct
- Development setup
- Coding standards
- Pull request process
- Testing requirements

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Support

### Getting Help
- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/AraAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/AraAI/discussions)
- **Email**: support@meridianalgo.com

### Reporting Bugs
Please use the [issue tracker](https://github.com/MeridianAlgo/AraAI/issues) and include:
- Python version
- Operating system
- Error messages
- Steps to reproduce

---

## Credits and Acknowledgments

This project is built on the shoulders of giants. We are deeply grateful to:

- **MeridianAlgo Team** - Core development and algorithmic trading expertise
- **XGBoost Team** (DMLC) - Extreme gradient boosting (99.7% accuracy in our ensemble)
- **Microsoft LightGBM Team** - Fast gradient boosting framework
- **Hugging Face** - Transformer models and NLP infrastructure
- **Meta AI (PyTorch)** - Deep learning framework
- **scikit-learn Team** - Comprehensive machine learning library
- **Ran Aroussi (yfinance)** - Yahoo Finance data access
- **NumPy, pandas, SciPy Teams** - Scientific computing foundation
- **Will McGugan (Rich)** - Beautiful console output

For complete credits, contributors, and license information, see [CREDITS.md](docs/CREDITS.md).

---

## Disclaimer

**IMPORTANT: Please read carefully before using this software.**

### Investment Risk Warning

This software is provided for educational and research purposes only. Stock market predictions are inherently uncertain and past performance does not guarantee future results.

**Key Points:**
- NOT financial advice or investment recommendations
- NOT guaranteed to be accurate or profitable
- Stock trading involves substantial risk of loss
- You may lose some or all of your invested capital
- Consult a licensed financial advisor before making investment decisions

### No Warranty

This software is provided "AS IS" without warranty of any kind, either express or implied, including but not limited to:
- Merchantability
- Fitness for a particular purpose
- Non-infringement
- Accuracy of predictions
- Reliability of data

### Limitation of Liability

The developers and contributors of this software shall not be liable for any:
- Financial losses
- Investment decisions
- Trading outcomes
- Data inaccuracies
- System failures
- Consequential damages

### User Responsibility

By using this software, you acknowledge that:
- You use it at your own risk
- You are responsible for your own investment decisions
- You will not hold the developers liable for any losses
- You understand the risks of stock market trading
- You will comply with all applicable laws and regulations

### Data Accuracy

While we strive for accuracy:
- Market data may be delayed or inaccurate
- Predictions are based on historical patterns
- Past performance does not indicate future results
- External factors may affect accuracy
- Models require periodic retraining

### Regulatory Compliance

Users are responsible for:
- Complying with local securities laws
- Understanding tax implications
- Following trading regulations
- Obtaining necessary licenses
- Consulting legal counsel if needed

### Third-Party Services

This software uses third-party services (yfinance, Hugging Face, etc.) which:
- Have their own terms of service
- May change or discontinue without notice
- Are not under our control
- May have their own limitations

### Beta Software Notice

This is beta software (v2.2.0-Beta):
- May contain bugs or errors
- Features may change without notice
- Not recommended for production trading
- Use in test/paper trading environments first
- Report issues via GitHub

---

**Version**: 3.0.2  
**Last Updated**: October 24, 2025  
**Maintained by**: MeridianAlgo

For the complete disclaimer and terms of use, please see [LICENSE](LICENSE) and [PRIVACY.md](docs/PRIVACY.md).
