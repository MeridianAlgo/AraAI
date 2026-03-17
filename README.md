# Meridian.AI

### Real-Time Financial Prediction Engine

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-4.1.0-green.svg)
[![Meridian.AI for Forex](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-forex.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-forex.yml)
[![Meridian.AI for Stocks](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-stocks.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-stocks.yml)
[![Meridian.AI Lint](https://github.com/MeridianAlgo/AraAI/actions/workflows/lint.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/lint.yml)

---

Meridian.AI is a 45M-parameter deep learning system that predicts price movements for **any stock or forex pair** in real time. It combines state-of-the-art sequence modeling (Mamba-2 SSM), sparse expert routing (MoE with SwiGLU), and efficient attention (GQA + Flash Attention 2) into a single unified model — continuously trained and deployed via GitHub Actions every 2 hours.

**Models are hosted on Hugging Face:** [MeridianAlgo/ARA.AI](https://huggingface.co/MeridianAlgo/ARA.AI)

---

## Architecture

Meridian.AI v4.1 is built on a hybrid transformer-SSM backbone designed for financial time series:

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Sequence Modeling | Mamba-2 SSM with selective scan | Linear-time long-range dependencies |
| Attention | Grouped Query Attention (GQA) + QK-Norm | Efficient multi-head attention with reduced KV cache |
| Position Encoding | Rotary Position Embeddings (RoPE) | Relative temporal awareness without absolute bias |
| Expert Routing | Mixture of Experts (4 experts, top-2) | Regime-specific specialization |
| Activations | SwiGLU (gated linear units) | Improved gradient flow over GELU/ReLU |
| Normalization | RMSNorm + Layer Scale | Training stability at scale |
| Regularization | Stochastic Depth (drop path) | Better generalization, prevents overfitting |
| Training | Mixed precision (FP16/BF16) via Accelerate | 2x throughput on compatible hardware |
| Loss | BalancedDirectionLoss (Huber + BCE) | Jointly optimizes price regression and direction accuracy |

### Model Specifications

| Spec | Value |
|------|-------|
| Parameters | ~45 Million |
| Hidden Dimension | 384 |
| Layers | 6 |
| Attention Heads | 6 (2 KV heads) |
| Experts | 4 (top-2 routing) |
| Prediction Heads | 4 |
| Input Features | 44 technical indicators |
| Sequence Length | 30 timesteps |

---

## How It Works

```
Market Data (any ticker/pair)
        |
        v
  44 Technical Indicators
  (RSI, MACD, Bollinger, ATR, OBV, VWAP, etc.)
        |
        v
  Mamba-2 SSM Blocks (temporal patterns)
        |
        v
  GQA + Flash Attention 2 (cross-timestep relationships)
        |
        v
  MoE Layer (4 SwiGLU experts, top-2 routing)
        |
        v
  Multi-Head Prediction (4 heads, aggregated)
        |
        v
  Price Forecast + Direction Signal
```

The model processes 30 timesteps of 44 features each, routing through Mamba SSM blocks for sequential pattern recognition, then through attention and expert layers for regime-specific predictions. The output is an aggregated forecast from 4 prediction heads.

---

## Quick Start

```bash
# Clone
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI

# Install
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Run a Prediction

```python
from meridianalgo.unified_ml import UnifiedStockML
from huggingface_hub import hf_hub_download

# Download latest model
model_path = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/Meridian.AI_Stocks.pt"
)

# Predict any stock
ml = UnifiedStockML(model_path=model_path)
result = ml.predict_ultimate("AAPL", days=5)
print(result)
```

```python
from meridianalgo.forex_ml import ForexML
from huggingface_hub import hf_hub_download

# Download latest forex model
model_path = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/Meridian.AI_Forex.pt"
)

# Predict any forex pair
ml = ForexML(model_path=model_path)
result = ml.predict("EURUSD=X", days=5)
print(result)
```

---

## Training Pipeline

Models are trained automatically via GitHub Actions on a 2-hour cycle:

1. **Fetch** — Pull market data across **3 timeframes** (max daily, 2yr hourly, 5yr weekly) per symbol into SQLite — 96 stocks, 22 forex pairs
2. **Train** — 10 epochs with cosine warm restarts, gradient accumulation (effective batch 256), data augmentation, EMA weight averaging
3. **Track** — Metrics logged to Comet ML (loss curves, direction accuracy, EMA val loss)
4. **Deploy** — Push updated `.pt` checkpoint to Hugging Face Hub

Version gating ensures the pipeline never loads stale pre-v4.1 checkpoints — every run either resumes from a valid v4.1 model or trains fresh.

### Training Techniques

- **Multi-Timeframe Data**: Fetches daily + hourly + weekly per symbol (3x more training data)
- **Data Augmentation**: Gaussian noise injection (0.5%) + random timestep masking (5%)
- **Gradient Accumulation**: Simulates batch size 256 from smaller micro-batches
- **EMA Model Averaging**: Exponential moving average of weights (decay 0.999) — smoother, better-generalizing model
- **Cosine Warm Restarts**: LR schedule with periodic restarts to escape local minima
- **BalancedDirectionLoss**: 60% Huber regression + 40% balanced BCE with class-weighted direction classification
- **Early Stopping**: Patience of 15 epochs on EMA validation loss
- **All 44 Real Features**: No zero-padding — RSI, Stochastic RSI, MACD histogram, Bollinger %B, OBV, Williams %R, CCI, ADX, Keltner Channels, Z-score, and more

---

## Technical Indicators

44 real features extracted from raw OHLCV data (no zero-padding):

| Category | Indicators |
|----------|-----------|
| Price | Returns, Log Returns, Volatility, ATR |
| Trend | SMA (5/10/20/50/200), EMA (5/10/20/50/200) |
| Momentum | RSI, Fast RSI, Stochastic RSI, Momentum, ROC, Williams %R |
| Oscillators | MACD, MACD Signal, MACD Histogram, Stochastic K/D, CCI |
| Volatility | Bollinger Bands (Upper/Lower/Width/%B), Keltner Channels (Upper/Lower/%K) |
| Volume | Volume SMA, Volume Ratio, OBV (normalized) |
| Trend Strength | ADX, +DI, -DI, Price vs SMA50/SMA200, ATR% |
| Mean Reversion | Z-Score (20d), Distance from 52-week High |

---

## Project Structure

```
meridianalgo/
  revolutionary_model.py   # v4.1 model architecture (Mamba-2, GQA, MoE)
  large_torch_model.py     # Training loop, data loading, inference
  direction_loss.py        # BalancedDirectionLoss (Huber + BCE)
  unified_ml.py            # Stock prediction API
  forex_ml.py              # Forex prediction API
  indicators.py            # 44 technical indicators
scripts/
  train_stock_model.py     # Stock training entrypoint
  train_forex_model.py     # Forex training entrypoint
  fetch_and_store_data.py  # Market data ingestion
  push_elite_models.py     # HF Hub deployment
.github/workflows/
  meridian-stocks.yml      # Automated stock training (every 2h)
  meridian-forex.yml       # Automated forex training (every 2h)
  lint.yml                 # Code quality checks
```

---

## Disclaimer

**This software is for research and educational purposes only. It is not financial advice.**

Trading financial instruments carries significant risk. All predictions are probabilistic forecasts based on historical data — past performance does not guarantee future results. Markets can behave unpredictably during black swan events, liquidity crises, or structural shifts.

You should never trade with money you cannot afford to lose. Any trading decisions you make are yours alone. MeridianAlgo and its contributors are not liable for any financial losses.

This software is provided "as is" without warranties of any kind. By using it, you agree to hold MeridianAlgo and all contributors harmless from any claims arising from your use of the platform. Users are responsible for compliance with all applicable financial regulations.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Made with love by [MeridianAlgo](https://github.com/MeridianAlgo)
