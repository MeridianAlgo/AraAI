# Meridian.AI

### Real-Time Financial Prediction Engine

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-4.1.0-green.svg)
[![Meridian.AI for Forex](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-forex.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-forex.yml)
[![Meridian.AI for Stocks](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-stocks.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-stocks.yml)
[![Meridian.AI Lint](https://github.com/MeridianAlgo/AraAI/actions/workflows/lint.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/lint.yml)

---

Meridian.AI is a 280M-parameter deep learning system that predicts price movements for **any stock or forex pair** in real time. It combines state-of-the-art sequence modeling (Mamba-2 SSM), sparse expert routing (MoE with SwiGLU), and efficient attention (GQA + Flash Attention 2) into a single unified model — continuously trained and deployed via GitHub Actions every 2 hours.

**Models are hosted on Hugging Face:** [MeridianAlgo/ARA.AI](https://huggingface.co/MeridianAlgo/ARA.AI)

---

## Architecture

Meridian.AI v4.1 is built on a hybrid transformer-SSM backbone designed for financial time series:

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Sequence Modeling | Mamba-2 SSM with selective scan | Linear-time long-range dependencies |
| Attention | Grouped Query Attention (GQA) + QK-Norm | Efficient multi-head attention with reduced KV cache |
| Position Encoding | Rotary Position Embeddings (RoPE) | Relative temporal awareness without absolute bias |
| Expert Routing | Mixture of Experts (8 experts, top-2) | Regime-specific specialization |
| Activations | SwiGLU (gated linear units) | Improved gradient flow over GELU/ReLU |
| Normalization | RMSNorm + Layer Scale | Training stability at scale |
| Regularization | Stochastic Depth (drop path) | Better generalization, prevents overfitting |
| Training | Mixed precision (FP16/BF16) via Accelerate | 2x throughput on compatible hardware |
| Loss | BalancedDirectionLoss (Huber + BCE) | Jointly optimizes price regression and direction accuracy |

### Model Specifications

| Spec | Value |
|------|-------|
| Parameters | ~280 Million |
| Hidden Dimension | 768 |
| Layers | 8 |
| Attention Heads | 12 (3 KV heads) |
| Experts | 8 (top-2 routing) |
| Prediction Heads | 8 |
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
  MoE Layer (8 SwiGLU experts, top-2 routing)
        |
        v
  Multi-Head Prediction (8 heads, aggregated)
        |
        v
  Price Forecast + Direction Signal
```

The model processes 30 timesteps of 44 features each, routing through Mamba SSM blocks for sequential pattern recognition, then through attention and expert layers for regime-specific predictions. The output is an aggregated forecast from 8 prediction heads.

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

1. **Fetch** — Pull latest market data into SQLite (150 stocks, 30 forex pairs)
2. **Train** — 20 epochs with cosine LR decay + linear warmup, mixed precision
3. **Track** — Metrics logged to Comet ML (loss curves, direction accuracy)
4. **Deploy** — Push updated `.pt` checkpoint to Hugging Face Hub

Version gating ensures the pipeline never loads stale pre-v4.1 checkpoints — every run either resumes from a valid v4.1 model or trains fresh.

### Training Features

- **BalancedDirectionLoss**: 60% Huber regression + 40% BCE direction classification with scaled logits
- **LR Schedule**: Linear warmup (10% of epochs) + cosine annealing to zero
- **Early Stopping**: Patience of 15 epochs on validation loss
- **Experiment Tracking**: Full Comet ML integration for loss, accuracy, and hyperparameters

---

## Technical Indicators

44 features extracted from raw OHLCV data:

| Category | Indicators |
|----------|-----------|
| Trend | SMA, EMA, HMA, KAMA, ZLEMA, T3 |
| Momentum | RSI, Stochastic, Williams %R, CCI, ROC, MFI |
| Volatility | Bollinger Bands, Keltner Channels, ATR, StdDev |
| Volume | OBV, VWAP, Chaikin Money Flow, Volume Profile |
| Oscillators | MACD, Awesome Oscillator, PPO, Ultimate Oscillator |
| Patterns | Head & Shoulders, Triangles, Wedges, Flags |

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
