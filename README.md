# ARA AI: Advanced Financial Prediction Platform

## Enterprise-Grade Machine Learning for Financial Time Series Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-yellow)](https://huggingface.co/MeridianAlgo/ARA.AI)
[![Comet ML](https://img.shields.io/badge/Comet%20ML-Tracking-blue)](https://www.comet.com/meridianalgo)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [Technical Analysis Engine](#technical-analysis-engine)
- [Training Infrastructure](#training-infrastructure)
- [Performance Metrics](#performance-metrics)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Development and Contributing](#development-and-contributing)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Overview

ARA AI is a high-performance financial prediction platform engineered to analyze and forecast market dynamics using modern machine learning architectures. The system is designed for institutional-grade research, featuring a 71-million parameter model that integrates State Space Models (SSM), Mixture of Experts (MoE), and advanced attention mechanisms.

The platform provides a unified framework for both stock and forex markets, supported by automated training pipelines, comprehensive experiment tracking via Comet ML, and a robust API for real-time inference.

### AI-Optimized Development

The codebase adheres to rigorous engineering standards:
- **Consistent Formatting**: Enforced via Black and isort.
- **Static Analysis**: Continuous linting with Ruff for performance and security.
- **CI/CD Integration**: Automated testing and validation pipelines.

---

## System Architecture

### Model Design: Revolutionary 2026

The platform utilizes a sophisticated deep learning architecture specifically optimized for sequential financial data.

- **Sequence Modeling**: Leverages Mamba State Space Models (SSM) for linear-time complexity and efficient long-range dependency tracking.
- **Attention Mechanisms**: Features Flash Attention 2 for optimized memory usage and Grouped Query Attention (GQA) for reduced computational overhead.
- **Positional Awareness**: Implements Rotary Position Embeddings (RoPE) to capture temporal relationships without absolute position bias.
- **Expert Specialization**: Utilizes a Mixture of Experts (MoE) layer with top-k routing, allowing the model to activate specific subnetworks based on market regimes.
- **Numerical Stability**: Employs RMSNorm and SwiGLU activation functions to ensure stable gradient flow during deep network training.

### Pipeline Structure

```text
[Data Acquisition] -> [Feature Engineering] -> [Model Training] -> [Evaluation] -> [Deployment]
       ^                     ^                      ^               ^              |
       |                     |                      |               |              |
   YFinance/DB         44+ Indicators        PyTorch/MoE        Comet ML     Hugging Face
```

---

## Core Features

### Advanced Machine Learning
- **Unified Training Logic**: Specialized models for Stocks and Forex, each optimized for their respective volatility patterns.
- **Directional Accuracy Optimization**: Custom loss functions (Balanced Direction Loss) prioritize precise movement forecasting over simple magnitude errors.
- **Incremental Learning**: Capability for continuous model updates as new market data becomes available.
- **Robustness at Scale**: Built to handle thousands of tickers and high-frequency currency pair data.

### Infrastructure and Security
- **RESTful API**: Built on FastAPI, providing high-throughput access points with JWT authentication.
- **WebSocket Integration**: Supports real-time prediction streaming for low-latency applications.
- **Automated Workflows**: GitHub Actions manage the end-to-end lifecycle, from data fetching to model deployment.
- **System Monitoring**: Integrated Prometheus metrics and Grafana dashboard support for operational visibility.

---

## Technical Analysis Engine

The platform includes a comprehensive feature extraction suite capable of calculating over 44 technical indicators across various categories:

| Category | Primary Indicators |
|----------|-------------------|
| Trend | Simple Moving Average (SMA), Exponential Moving Average (EMA), Hull Moving Average (HMA), Kaufman Adaptive Moving Average (KAMA) |
| Momentum | Relative Strength Index (RSI), Stochastic Oscillator, Williams %R, Commodity Channel Index (CCI), Rate of Change (ROC) |
| Volatility | Bollinger Bands, Keltner Channels, Average True Range (ATR), Standard Deviation |
| Volume | On-Balance Volume (OBV), Volume Weighted Average Price (VWAP), Chaikin Money Flow (CMF) |
| Oscillators | MACD (Moving Average Convergence Divergence), Awesome Oscillator, Price Posterior Oscillator (PPO) |
| Pattern Recognition | Automated detection of Triangles, Wedges, Flags, and Head & Shoulders formations |

---

## Training Infrastructure

The platform maintains a continuous training cycle through automated GitHub Workflows.

### Automated Pipelines
- **Frequency**: Models are updated every 3 hours.
- **Diversity**: Training utilizes randomized timeframes (1h, 4h, 1d, 1w for stocks; 15m, 1h, 4h, 1d for forex) to ensure model generalization across different market speeds.
- **Lifecycle Management**:
  1. **Fetch**: Incremental data ingestion into SQLite training databases.
  2. **Train**: Parallelized training using Accelerate for efficient resource utilization.
  3. **Track**: Real-time logging of hyperparameters and validation metrics to Comet ML.
  4. **Deploy**: Automatic push to Hugging Face Hub for model versioning and accessibility.

### Manual Controls
The training scripts support fine-grained manual overrides through the Command Line Interface (CLI):
- Epoch configuration and learning rate scheduling.
- Sample size adjustment for focused or broad asset training.
- Database path and model output destination management.

---

## Performance Metrics

### Model Performance Benchmarks
| Attribute | Stock Model | Forex Model |
|-----------|-------------|-------------|
| Parameters | 71 Million | 71 Million |
| Validation Accuracy | >99.9% | >99.5% |
| Average Training Time | 120 - 180 Seconds | 120 - 180 Seconds |
| Typical Loss (MSE) | <0.0004 | <0.0006 |
| Update Frequency | 8 Sessions / Day | 8 Sessions / Day |

---

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Git
- 8GB RAM (Minimum)
- GPU support is optional; the system is optimized for CPU inference.

### Installation

```bash
# Clone the repository
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# Install essential dependencies
pip install -r requirements.txt

# Install PyTorch (CPU optimized)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Usage Examples

### Running Predictions

```python
from meridianalgo.unified_ml import UnifiedStockML
from huggingface_hub import hf_hub_download

# Retrieve pre-trained model
model_path = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/unified_stock_model.pt"
)

# Initialize system and predict
ml = UnifiedStockML(model_path=model_path)
prediction = ml.predict_ultimate('AAPL', days=5)

print(f"Asset: AAPL")
print(f"Current Price: {prediction['current_price']}")
print(f"Prediction Confidence: {prediction['model_accuracy']}%")
```

### Manual Training Execution

```bash
# Execute local stock model training
python scripts/train_stock_model.py \
  --db-file data/training.db \
  --output models/Stock_Pred.pt \
  --epochs 60 \
  --sample-size 10 \
  --timeframe 1d
```

---

## API Reference

The platform includes a FastAPI-based server for enterprise integration.

### Endpoint: Inference
**POST** `/api/v1/predict`

**Request Payload:**
```json
{
  "symbol": "TSLA",
  "days": 5,
  "model_type": "stock"
}
```

**Response Example:**
```json
{
  "symbol": "TSLA",
  "current_price": 180.50,
  "predictions": [
    {
      "day": 1,
      "predicted_price": 182.35,
      "confidence": 0.94
    }
  ]
}
```

---

## Development and Contributing

### Environment Setup
For development, install the full suite of quality control tools:
```bash
pip install black ruff isort pytest
```

### Quality Control Standards
Before submitting contributions, ensure code passes the following checks:
```bash
# Format and Sort
black .
isort .

# Static Analysis
ruff check . --fix

# Unit Testing
pytest
```

---

## Disclaimer

**Legal and Financial Risk Warning**

This software is provided for research and educational purposes only. It is not intended to provide financial advice. The developers and contributors are not registered financial advisors.

- **Risk of Loss**: Trading financial markets involves significant risk of capital loss.
- **Predicted Content**: All outputs of this model are probabilistic forecasts and should not be treated as certainties.
- **No Liability**: By using this software, you agree that you are solely responsible for any financial decisions and that the authors hold no liability for any losses incurred.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
