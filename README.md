# ARA AI - Financial Prediction System

**AI-powered financial prediction platform with continuous ensemble machine learning model training**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Continuous Training](https://github.com/MeridianAlgo/AraAI/actions/workflows/daily-training.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/daily-training.yml)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-yellow)](https://huggingface.co/MeridianAlgo/ARA.AI)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-FFCC33?logo=weightsandbiases&logoColor=black)](https://wandb.ai)

> **DISCLAIMER**: This software is for educational and research purposes only. NOT financial advice. You are solely responsible for your investment decisions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Continuous Training](#continuous-training)
- [Architecture](#architecture)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

ARA AI is a financial prediction platform that combines ensemble machine learning models to provide price predictions for stocks and forex. The system uses:

- **Ensemble Learning**: XGBoost, LightGBM, Random Forest, Transformers, CNN-LSTM
- **Continuous Training**: Models retrain every 2 hours with latest market data
- **Experiment Tracking**: Weights & Biases integration for monitoring
- **Model Storage**: Hugging Face Hub for versioning and distribution

### Key Features

- **Automated Training**: 12 training cycles daily (every 2 hours)
- **Random Stock Selection**: 5 different stocks per cycle from 6,800+ tickers
- **Robust Forex Pairs**: EURUSD, GBPUSD, USDJPY
- **Incremental Learning**: Models improve continuously with new data
- **Model Versioning**: All models stored on Hugging Face Hub

---

## Features

### Machine Learning

- **Ensemble Models**: Combines 9+ different algorithms
- **Transformer Architecture**: Attention-based time series prediction
- **CNN-LSTM Hybrid**: Convolutional-recurrent neural networks
- **Incremental Training**: Continuous model improvement
- **Adaptive Learning**: Models adjust to market conditions

### Technical Analysis

- **44+ Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
- **Pattern Recognition**: Head & Shoulders, Triangles, Wedges
- **Volume Analysis**: OBV, MFI, VWAP
- **Trend Indicators**: SMA, EMA, ADX, Parabolic SAR
- **Volatility Measures**: Bollinger Bands, Keltner Channels, ATR

### Data & Training

- **2 Years Historical Data**: Comprehensive training dataset
- **Real-time Updates**: Latest market data fetched before each training cycle
- **Multi-asset Support**: Stocks and forex pairs
- **Automatic Scheduling**: GitHub Actions handles all training

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Download Models

```python
from huggingface_hub import hf_hub_download

# Download a stock model
model_path = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/stock_AAPL.pt"
)

# Download a forex model
forex_path = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/forex_EURUSD.pt"
)
```

---

## Installation

### Standard Installation

```bash
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -r requirements.txt
pip install pytest black flake8 mypy
```

### Docker

```bash
docker build -t ara-ai .
docker run -p 8000:8000 ara-ai
```

### Requirements

Main dependencies:
- **torch, transformers** - Deep learning
- **scikit-learn, xgboost, lightgbm** - Machine learning
- **pandas, numpy** - Data manipulation
- **yfinance** - Market data
- **wandb** - Experiment tracking
- **huggingface_hub** - Model storage

See [requirements.txt](requirements.txt) for complete list.

---

## Usage

### Load and Use Stock Models

```python
from meridianalgo.unified_ml import UnifiedStockML
from huggingface_hub import hf_hub_download

# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/stock_AAPL.pt"
)

# Load and predict
ml = UnifiedStockML(model_path=model_path)
prediction = ml.predict('AAPL', days=5)

print(f"Current Price: ${prediction['current_price']}")
print(f"5-Day Prediction: ${prediction['predictions'][4]['price']}")
```

### Load and Use Forex Models

```python
from meridianalgo.forex_ml import ForexML
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/forex_EURUSD.pt"
)

# Load and predict
forex_ml = ForexML(model_path=model_path)
prediction = forex_ml.predict('EURUSD', days=5)

print(f"Prediction: {prediction}")
```

### Train Locally

```bash
# Train a stock model
python scripts/train_model.py \
  --symbol AAPL \
  --db-file training.db \
  --output models/stock_AAPL.pt \
  --epochs 100 \
  --use-all-data

# Train a forex model
python scripts/train_forex_model.py \
  --pair EURUSD \
  --db-file training.db \
  --output models/forex_EURUSD.pt \
  --epochs 100 \
  --use-all-data
```

---

## Continuous Training

### How It Works

Models are automatically trained every 2 hours via GitHub Actions:

1. **Pull**: Download existing models from Hugging Face
2. **Select**: Choose 5 random stocks + 3 forex pairs
3. **Fetch**: Get latest 2 years of market data
4. **Train**: Incrementally train models (50 epochs)
5. **Push**: Upload updated models to Hugging Face
6. **Track**: Log metrics to Weights & Biases

### Training Schedule

- **Frequency**: Every 2 hours (12 cycles daily)
- **Stock Models**: 5 random stocks per cycle (60 different stocks daily)
- **Forex Models**: EURUSD, GBPUSD, USDJPY (36 training sessions daily)
- **Total**: 96 model updates per day

### Estimated Training Time

- Per stock: 5-10 minutes
- Per forex pair: 5-10 minutes
- Total per cycle: 30-60 minutes
- Buffer: 60+ minutes before next cycle

### Setup

1. Add `HF_TOKEN` to GitHub repository secrets (required)
   - Go to Settings  Secrets and variables  Actions
   - Add new secret with your Hugging Face API token

2. Add `WANDB_API_KEY` to GitHub repository secrets (optional)
   - For experiment tracking on Weights & Biases

### View Training

- **Models**: https://huggingface.co/MeridianAlgo/ARA.AI
- **Experiments**: https://wandb.ai/your-username/ara-ai
- **Workflow Runs**: GitHub Actions tab in repository

---

## Architecture

### Project Structure

```
AraAI/
 .github/workflows/
    daily-training.yml      # Continuous training workflow
 meridianalgo/               # Core ML algorithms
    unified_ml.py           # Stock prediction system
    forex_ml.py             # Forex prediction system
    torch_ensemble.py       # PyTorch ensemble models
    ...
 scripts/                    # Training scripts
    train_model.py          # Stock model training
    train_forex_model.py    # Forex model training
    fetch_training_data.py  # Data fetching
    store_training_data.py  # Data storage
    select_random_tickers.py # Random ticker selection
 models/                     # Trained model files
    README.md               # Model documentation
 ara/                        # ARA package (optional)
 requirements.txt            # Python dependencies
 LICENSE                     # MIT License
 README.md                   # This file
```

### Core Scripts

- **train_model.py**: Train stock prediction models
- **train_forex_model.py**: Train forex prediction models
- **fetch_training_data.py**: Fetch market data from Yahoo Finance
- **store_training_data.py**: Store data in SQLite database
- **select_random_tickers.py**: Select random stocks from all_tickers.txt

### Model Architecture

The system uses a multi-layer ensemble:

1. **Base Models**
   - XGBoost
   - LightGBM
   - Random Forest
   - Gradient Boosting
   - Extra Trees
   - AdaBoost

2. **Deep Learning**
   - Transformer (attention-based)
   - CNN-LSTM (hybrid)

3. **Ensemble**
   - Weighted averaging
   - Confidence-based selection

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Quality

```bash
# Format code
black scripts/ meridianalgo/

# Lint code
flake8 scripts/ meridianalgo/ --max-line-length=100

# Type checking
mypy scripts/ meridianalgo/
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

### Disclaimer

**IMPORTANT**: This software is for educational and research purposes only.

- NOT financial advice
- Past performance  future results
- All predictions are probabilistic
- You are solely responsible for investment decisions
- Consult qualified financial professionals
- Authors are not liable for financial losses

### Third-Party Licenses

- Scikit-learn (BSD License)
- XGBoost (Apache License 2.0)
- LightGBM (MIT License)
- PyTorch (BSD License)
- Transformers by Hugging Face (Apache License 2.0)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/AraAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/AraAI/discussions)
- **Models**: [Hugging Face Hub](https://huggingface.co/MeridianAlgo/ARA.AI)

---

**Maintained by**: [MeridianAlgo](https://github.com/MeridianAlgo)  
**Last Updated**: 2025-12-22  
**Version**: 5.2.0
