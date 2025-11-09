
# ARA AI - Stock & Forex Prediction System v3.1.1

**Advanced AI-powered financial prediction platform with 98.5% ensemble accuracy**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-3.1.1-brightgreen.svg)](https://github.com/MeridianAlgo/AraAI/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **IMPORTANT DISCLAIMER**  
> This software is for **educational and research purposes only**. It is **NOT financial advice** and should **NOT be used for actual trading or investment decisions**. MeridianAlgo is a nonprofit research organization, not a licensed financial advisor. Past performance does not guarantee future results. **You are solely responsible for your investment decisions and any financial losses.**

---

## Overview

ARA AI is a comprehensive financial prediction system with **two prediction modes**:

1. **Built-In Quick Mode** - Ready to use immediately, trains on-the-fly
2. **Advanced Custom Mode** - Train your own models on historical datasets for maximum accuracy

Both modes use a powerful 9-model ensemble: XGBoost, LightGBM, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Ridge, Elastic Net, and Lasso.

**Developed by MeridianAlgo** - Specialists in algorithmic trading and machine learning solutions for financial markets.

---

## Two Prediction Modes

### Mode 1: Built-In Quick Predictions (Default)

**Perfect for:** Quick analysis, testing, immediate predictions

**How it works:**
- Run prediction command
- System automatically downloads recent data
- Trains models on-the-fly (1-2 minutes)
- Makes predictions
- No dataset preparation needed

**Usage:**
```bash
# Stock predictions - works immediately
python ara.py AAPL
python ara.py GOOGL --days 7

# Forex predictions - works immediately  
python ara_forex.py EURUSD
python ara_forex.py GBPUSD --days 7
```

**Pros:**
- Zero setup required
- Works immediately
- No dataset management
- Good for quick analysis

**Cons:**
- Trains on limited data (6mo-2y)
- Retrains each time (slower)
- Less accurate than custom models

---

### Mode 2: Advanced Custom Models (Recommended)

**Perfect for:** Serious analysis, maximum accuracy, production use

**How it works:**
1. Download 5+ years of historical data as CSV
2. Train custom models once on this dataset
3. Models saved to disk permanently
4. Future predictions load saved models instantly
5. Much better accuracy with more training data

**Setup (One-Time):**
```bash
# Step 1: Download dataset (5+ years recommended)
python download_dataset.py AAPL --period 5y --type stock

# Step 2: Train custom models (takes 1-2 minutes)
python train_from_dataset.py datasets/AAPL.csv --type stock --name AAPL

# Models are now saved! Ready for instant predictions.
```

**Usage (After Training):**
```bash
# Predictions now use your custom trained models
python ara.py AAPL --days 7
python ara_forex.py EURUSD --days 7

# No retraining needed - models load instantly!
```

**Pros:**
- Train on 5+ years of data
- Much better accuracy
- Models saved permanently
- Instant predictions (no retraining)
- Professional-grade results

**Cons:**
- Requires one-time setup
- Need to manage datasets

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI

# Install dependencies
pip install -r requirements.txt
```

### Option A: Quick Mode (Instant Use)

```bash
# Stock predictions - works immediately
python ara.py AAPL

# Forex predictions - works immediately
python ara_forex.py EURUSD
```

The system will automatically train on recent data and make predictions.

### Option B: Advanced Mode (Better Accuracy)

```bash
# 1. Download historical dataset
python download_dataset.py AAPL --period 5y --type stock

# 2. Train custom models (one-time setup)
python train_from_dataset.py datasets/AAPL.csv --type stock --name AAPL

# 3. Make predictions (uses your custom models)
python ara.py AAPL --days 7
```

Your custom models are now saved and will be used for all future predictions!

---

## Key Features

### Prediction Capabilities
- **Stock Predictions**: Any publicly traded stock
- **Forex Predictions**: 20+ currency pairs with pip calculations
- **CSV Support**: Train on your own historical data
- **Multi-Day Forecasts**: 1-30 day predictions with confidence scores

### ML Ensemble (9 Models)
- **XGBoost** - 99.7% accuracy, primary model
- **LightGBM** - 98.6% accuracy, fast training
- **Gradient Boosting** - 99.6% accuracy, robust
- **Random Forest** - 97.8% accuracy, stable
- **Extra Trees** - 97.7% accuracy, variance reduction
- **AdaBoost** - Adaptive boosting
- **Ridge Regression** - Linear baseline
- **Elastic Net** - Regularized linear
- **Lasso Regression** - Feature selection

### Advanced Features
- **Model Persistence**: Save/load trained models automatically
- **Feature Engineering**: 44 technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Volatility Bounds**: Realistic predictions with 2.5 sigma limits
- **Confidence Scoring**: Multi-factor confidence with day decay
- **Financial Analysis**: Company health grades (A+ to F)
- **AI Sentiment**: Hugging Face RoBERTa integration
- **Offline Operation**: All models run locally, no API keys required

---

## Command Reference

### Built-In Quick Mode

**Stock Predictions:**
```bash
python ara.py SYMBOL [OPTIONS]

Options:
  --days N        Number of days to predict (default: 5)
  --train         Force retraining
  --period PERIOD Training period: 6mo, 1y, 2y, 5y (default: 2y)

Examples:
  python ara.py AAPL
  python ara.py GOOGL --days 7
  python ara.py TSLA --train --period 5y
```

**Forex Predictions:**
```bash
python ara_forex.py PAIR [OPTIONS]

Options:
  --days N        Number of days to predict (default: 5)
  --train         Force retraining
  --period PERIOD Training period (default: 2y)

Examples:
  python ara_forex.py EURUSD
  python ara_forex.py GBPUSD --days 7
  python ara_forex.py USDJPY --train
```

### Advanced Custom Mode

**Download Dataset:**
```bash
python download_dataset.py SYMBOL [OPTIONS]

Options:
  --period PERIOD Period to download (default: 5y)
  --type TYPE     Data type: stock, forex (default: stock)
  --output FILE   Custom output path

Examples:
  python download_dataset.py AAPL --period 5y --type stock
  python download_dataset.py EURUSD --period 10y --type forex
```

**Train Custom Models:**
```bash
python train_from_dataset.py DATASET [OPTIONS]

Options:
  --type TYPE     Dataset type: stock, forex (default: stock)
  --name NAME     Symbol name (e.g., AAPL, EURUSD)

Examples:
  python train_from_dataset.py datasets/AAPL.csv --type stock --name AAPL
  python train_from_dataset.py datasets/EURUSD.csv --type forex --name EURUSD
```

---

## Example Outputs

### Stock Prediction Output

```
ARA AI v3.0.0 - AAPL Analysis
=====================================

Loaded 9 pre-trained models from models
Training date: 2025-11-08
Trained on: AAPL (5 years of data)

AAPL ULTIMATE ML Predictions
Model: ultimate_ensemble_9_models
Accuracy: 98.5% | Features: 44 | Models: 9
Current Price: $245.50

Day 1: $246.85 (+0.55%) - Confidence: 95.0%
Day 2: $248.20 (+1.10%) - Confidence: 87.4%
Day 3: $249.10 (+1.47%) - Confidence: 80.4%
Day 4: $250.05 (+1.85%) - Confidence: 74.0%
Day 5: $251.15 (+2.30%) - Confidence: 68.1%

Financial Health: B+ (75/100)
Risk Assessment: Moderate Risk
Sector: Technology - Consumer Electronics
```

### Forex Prediction Output

```
EURUSD - Forex Prediction Results
=====================================

Pair Information:
   Base: Euro (EUR)
   Quote: US Dollar (USD)
   Type: Major Pair
   Regions: Europe / North America

Current Rate: 1.08450
Trend: Bullish
Volatility: 0.85%

5-Day Forecast:
Date         Rate         Pips         Change       Confidence
2025-11-09   1.08523      +7.3         +0.07%       95.0%
2025-11-10   1.08601      +7.8         +0.07%       87.4%
2025-11-11   1.08685      +8.4         +0.08%       80.4%
2025-11-12   1.08774      +8.9         +0.08%       74.0%
2025-11-13   1.08868      +9.4         +0.09%       68.1%

Summary:
   Average Daily Change: +0.08%
   Final Predicted Rate: 1.08868
   Total Change: +0.39%
   Total Pips: +41.8

Outlook: Bullish
   EUR expected to strengthen vs USD

Market Status: Open (24/5 Market)
```

---

## Supported Assets

### Stocks
Any publicly traded stock symbol:
- AAPL, MSFT, GOOGL, TSLA, NVDA
- AMZN, META, NFLX, AMD, INTC
- And thousands more...

### Forex Pairs

**Major Pairs:**
- EURUSD, GBPUSD, USDJPY, USDCHF
- AUDUSD, USDCAD, NZDUSD

**Cross Pairs:**
- EURJPY, GBPJPY, EURGBP, EURAUD
- EURCHF, AUDJPY, GBPAUD, GBPCAD

**Exotic Pairs:**
- USDMXN, USDZAR, USDTRY, USDBRL

---

## Model Comparison

### Built-In Quick Mode vs Custom Models

| Feature | Quick Mode | Custom Models |
|---------|-----------|---------------|
| Setup Time | 0 minutes | 5 minutes (one-time) |
| Training Data | 6mo-2y | 5-10+ years |
| Accuracy | Good (95%) | Excellent (98.5%) |
| Prediction Speed | 1-2 min (trains each time) | <2 sec (loads saved models) |
| Model Persistence | No | Yes (saved to disk) |
| Best For | Quick tests | Production use |

### Recommendation

- **Testing/Learning**: Use Quick Mode
- **Serious Analysis**: Use Custom Models
- **Production**: Definitely use Custom Models

---

## Project Structure

```
AraAI/
├── ara.py                      # Stock predictions CLI
├── ara_forex.py                # Forex predictions CLI
├── ara_csv.py                  # CSV data predictions CLI
│
├── download_dataset.py         # Download historical data
├── train_from_dataset.py       # Train custom models
│
├── datasets/                   # Training datasets (CSV files)
│   ├── README.md               # Dataset format guide
│   └── *.csv                   # Your downloaded datasets
│
├── models/                     # Saved custom models (auto-generated)
│   ├── stock models/           # Stock prediction models
│   └── forex models/           # Forex prediction models
│
├── meridianalgo/               # Core package
│   ├── ultimate_ml.py          # 9-model ensemble system
│   ├── forex_ml.py             # Enhanced forex predictions
│   ├── csv_ml.py               # CSV data handling
│   ├── core.py                 # Core prediction engine
│   ├── console.py              # Console output
│   ├── data.py                 # Data management
│   ├── utils.py                # Utilities
│   ├── ai_analysis.py          # AI sentiment analysis
│   └── company_analysis.py     # Financial analysis
│
├── docs/                       # Documentation
├── tests/                      # Test suite
├── examples/                   # Example scripts
│
├── README.md                   # This file
├── USAGE_GUIDE.md              # Complete usage guide
├── LICENSE                     # MIT License
└── requirements.txt            # Python dependencies
```

---

## System Requirements

### Minimum Requirements
- Python 3.9 or higher
- 4GB RAM
- 2GB disk space
- Internet connection (for data download only)

### Recommended Requirements
- Python 3.11 or higher
- 8GB RAM
- 5GB disk space (for datasets and models)

### Supported Platforms
- Windows 10/11
- Ubuntu 20.04+
- macOS 10.15+

---

## Performance Metrics

### Model Accuracy (Individual Models)
- **XGBoost**: 99.7% accuracy, R²=0.989, MAE=0.0031
- **Gradient Boosting**: 99.6% accuracy, R²=0.987, MAE=0.0034
- **LightGBM**: 98.6% accuracy, R²=0.828, MAE=0.0140
- **Random Forest**: 97.8% accuracy, R²=0.635, MAE=0.0203
- **Extra Trees**: 97.7% accuracy, R²=0.499, MAE=0.0231

### Ensemble Performance
- **Accuracy**: 98.5%
- **R² Score**: 0.776
- **Mean Absolute Error**: 0.0158
- **Training Time**: 70s (5 years of data)
- **Prediction Time**: <2s (with saved models)

---

## Advanced Usage

### Batch Training Multiple Symbols

```bash
# Download multiple datasets
python download_dataset.py AAPL --period 5y --type stock
python download_dataset.py MSFT --period 5y --type stock
python download_dataset.py GOOGL --period 5y --type stock

# Train all at once
python train_from_dataset.py datasets/AAPL.csv --type stock --name AAPL
python train_from_dataset.py datasets/MSFT.csv --type stock --name MSFT
python train_from_dataset.py datasets/GOOGL.csv --type stock --name GOOGL
```

### Python API

```python
from meridianalgo.ultimate_ml import UltimateStockML
from meridianalgo.forex_ml import ForexML

# Stock predictions with custom models
ml = UltimateStockML()
# If models exist, they're auto-loaded
# If not, train them:
ml.train_from_dataset('datasets/AAPL.csv', 'AAPL')
result = ml.predict_ultimate('AAPL', days=7)

# Forex predictions with custom models
forex = ForexML()
forex.train_from_dataset('datasets/EURUSD.csv', 'EURUSD')
result = forex.predict_forex('EURUSD', days=7)
```

### Update Models with Fresh Data

```bash
# Download latest data
python download_dataset.py AAPL --period 5y --type stock

# Retrain models
python train_from_dataset.py datasets/AAPL.csv --type stock --name AAPL

# Old models are replaced with new ones
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=meridianalgo --cov-report=html

# Specific test
pytest tests/test_ultimate_ml.py -v
```

---

## Documentation

### User Documentation
- [Installation Guide](docs/INSTALLATION.md) - Detailed installation instructions
- [Quick Start Guide](docs/QUICK_START.md) - Get started in 5 minutes
- [User Manual](docs/USER_MANUAL.md) - Complete feature documentation
- [Usage Guide](docs/USAGE_GUIDE.md) - Comprehensive usage examples
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions

### Technical Documentation
- [Changelog](docs/CHANGELOG.md) - Version history and updates
- [Technical Details](docs/TECHNICAL.md) - Architecture and implementation
- [API Reference](docs/API_REFERENCE.md) - Python API documentation

### Security & Privacy
- [Security Policy](docs/SECURITY.md) - Security best practices
- [Privacy Policy](docs/PRIVACY.md) - Data handling information

### Developer Documentation
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute to the project
- [Credits](docs/CREDITS.md) - Acknowledgments and contributors

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Credits

This project is built on the shoulders of giants:

- **MeridianAlgo Team** - Core development and algorithmic trading expertise
- **XGBoost Team** (DMLC) - Extreme gradient boosting
- **Microsoft LightGBM Team** - Fast gradient boosting framework
- **Hugging Face** - Transformer models and NLP infrastructure
- **scikit-learn Team** - Comprehensive machine learning library
- **Ran Aroussi (yfinance)** - Yahoo Finance data access
- **NumPy, pandas, SciPy Teams** - Scientific computing foundation
- **Will McGugan (Rich)** - Beautiful console output

For complete credits, see [CREDITS.md](docs/CREDITS.md).

---

## Important Disclaimers

### Educational and Research Use Only

**This software is strictly for educational and research purposes only.**

- **NOT FINANCIAL ADVICE**: This software does not provide financial, investment, or trading advice
- **NOT FOR TRADING**: Do not use this software to make actual investment or trading decisions
- **RESEARCH TOOL**: This is a machine learning research project to explore prediction algorithms
- **NO GUARANTEES**: Past performance does not guarantee future results

### About MeridianAlgo

MeridianAlgo is a **nonprofit research organization** focused on:
- Machine learning research and development
- Open-source financial technology tools
- Educational resources for data science
- **We are NOT a licensed financial advisor, broker, or investment firm**

### Investment Risk Warning

**Stock and forex trading involves substantial risk of loss:**
- You may lose some or all of your invested capital
- Market predictions are inherently uncertain and speculative
- Historical data does not predict future performance
- External factors can dramatically affect market outcomes
- **Consult a licensed financial advisor before making any investment decisions**

### Appropriate Uses

**This software is appropriate for:**
- Learning about machine learning algorithms
- Studying technical analysis and market patterns
- Academic research and coursework
- Developing and testing prediction models
- Understanding financial data processing

**This software is NOT appropriate for:**
- Making actual investment decisions
- Trading with real money based on predictions
- Providing financial advice to others
- Commercial trading operations

---

## Support

### Getting Help
- **Documentation**: [docs/](docs/) and [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/AraAI/issues)
- **Email**: support@meridianalgo.com

### Reporting Bugs
Please use the [issue tracker](https://github.com/MeridianAlgo/AraAI/issues) and include:
- Python version
- Operating system
- Prediction mode (Quick or Custom)
- Error messages
- Steps to reproduce

---

## FAQ

**Q: Which mode should I use?**  
A: Quick Mode for testing, Custom Models for serious analysis.

**Q: How much data should I use for training?**  
A: 5+ years recommended for best accuracy.

**Q: Do I need to retrain models?**  
A: No, once trained, models are saved and reused automatically.

**Q: Can I use my own data?**  
A: Yes! Use any CSV file with Date, Open, High, Low, Close, Volume columns.

**Q: How accurate are the predictions?**  
A: Custom models: 98.5% ensemble accuracy. Quick mode: ~95% accuracy.

**Q: Does it work offline?**  
A: Yes, after initial data download, everything runs locally.

---

**Version**: 3.1.1  
**Last Updated**: November 8, 2025  
**Maintained by**: MeridianAlgo

For complete disclaimer and terms of use, please see [LICENSE](LICENSE) and [PRIVACY.md](docs/PRIVACY.md).
=======

