# Meridian.AI

### Real-Time Financial Prediction Engine

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/version-5.0.0-green.svg)
[![Meridian.AI for Forex](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-forex.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-forex.yml)
[![Meridian.AI for Stocks](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-stocks.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/meridian-stocks.yml)
[![Meridian.AI Lint](https://github.com/MeridianAlgo/AraAI/actions/workflows/lint.yml/badge.svg)](https://github.com/MeridianAlgo/AraAI/actions/workflows/lint.yml)

---

Meridian.AI is a deep learning system that predicts price movements for **any stock or forex pair** in real time. It combines sequence modeling (optional Mamba SSM), sparse expert routing (MoE with SwiGLU), and efficient attention (GQA + RoPE) into a single unified model — continuously trained and deployed via GitHub Actions **every hour**.

**Models are hosted on Hugging Face:** [meridianal/ARA.AI](https://huggingface.co/meridianal/ARA.AI)

---

## What Changed in v5.0

v5.0 fixed seven bugs that were silently corrupting training in all prior versions. Models trained before v5.0 never actually validated against held-out data, never properly recorded direction accuracy, and may have diverged due to numerical issues. Every checkpoint produced by v5.0 is verifiably correct.

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | **Validation ran after `break`** | Validation was unreachable — `best_val_loss` was always `None` | Moved validation block before the break condition |
| 2 | **Double normalisation** | Features normalised twice (once in loader, once in training loop) | Removed redundant second normalisation pass |
| 3 | **FP16 on CPU** | `autocast(dtype=float16)` raises on CPU, crashing training | Switched to `bfloat16` on CPU, `float16` only on CUDA |
| 4 | **NaN patience loop** | Early stopping fired on `NaN` loss before model could recover | Added `math.isfinite` guard; `NaN` epochs no longer count toward patience |
| 5 | **Feature saturation** | Raw indicator values hit `±1e6`, driving gradients to zero | Added `torch.clamp(..., -10, 10)` after feature extraction |
| 6 | **Slow Mamba scan** | Python loop over sequence length (~18min/epoch); hidden state leaked across samples | Replaced with batched matrix ops; `h` reset per batch |
| 7 | **Hardcoded `use_mamba=True`** | Checkpoint always saved `use_mamba=True` regardless of actual config | Reads `use_mamba` from the live model layer before saving |

Additionally, v5.0 renames the architecture from the internal working name to **MeridianModel** and switches CI from every 6 hours to **every hour**.

---

## Architecture

MeridianModel v5.0 is a hybrid transformer backbone designed for financial time series. The CPU default config (~11M params) is tuned to complete training within the 45-minute GitHub Actions window. A larger GPU config is available for offline training.

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Attention | Grouped Query Attention (GQA) + QK-Norm | Efficient multi-head attention with reduced KV cache |
| Position Encoding | Rotary Position Embeddings (RoPE) | Relative temporal awareness without absolute bias |
| Expert Routing | Mixture of Experts (SwiGLU, top-2) | Regime-specific specialization |
| Activations | SwiGLU (gated linear units) | Improved gradient flow over GELU/ReLU |
| Normalization | RMSNorm + Layer Scale | Training stability |
| Regularization | Stochastic Depth (drop path) | Better generalization |
| Optional SSM | Mamba block (vectorised scan) | Long-range sequential dependencies |
| Loss | BalancedDirectionLoss (Huber + BCE) | Jointly optimizes price regression and direction accuracy |

### Model Specifications

| Spec | CPU Default (v5.0) | GPU / Large |
|------|--------------------|-------------|
| Parameters | ~11M | ~45M |
| Hidden Dimension | 256 | 384 |
| Layers | 6 | 6 |
| Attention Heads | 4 (2 KV heads) | 6 (2 KV heads) |
| Experts | 4 (top-2 routing) | 4 (top-2 routing) |
| Prediction Heads | 4 | 4 |
| Mamba SSM | Disabled (CPU) | Optional |
| Input Features | 44 technical indicators | 44 technical indicators |
| Sequence Length | 30 timesteps | 30 timesteps |

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
  GQA + RoPE Attention (cross-timestep relationships)
        |
        v
  MoE Layer (SwiGLU experts, top-2 routing)
        |
        v
  Multi-Head Prediction (4 heads, aggregated)
        |
        v
  Price Forecast + Direction Signal
```

The model processes 30 timesteps of 44 features each. Features are clamped to `[-10, 10]` after normalisation to prevent gradient saturation. The output is an aggregated forecast from 4 prediction heads with a joint Huber + BCE direction loss.

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
    repo_id="meridianal/ARA.AI",
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
    repo_id="meridianal/ARA.AI",
    filename="models/Meridian.AI_Forex.pt"
)

# Predict any forex pair
ml = ForexML(model_path=model_path)
result = ml.predict("EURUSD=X", days=5)
print(result)
```

---

## Training Pipeline

Models train automatically via GitHub Actions on an **hourly cycle** (stocks at :00, forex at :30). Concurrent runs queue rather than cancel — a run in progress is never interrupted.

1. **Fetch** — Pull market data across 3 timeframes (daily, 2yr hourly, 5yr weekly) per symbol into SQLite — 96 stocks, 22 forex pairs
2. **Train** — Up to 999 epochs within a 45-minute budget; cosine warm restarts with 2-epoch linear warmup, gradient accumulation (effective batch 256), data augmentation, EMA weight averaging
3. **Track** — Metrics logged to Comet ML (loss curves, direction accuracy, EMA val loss)
4. **Deploy** — Push updated `.pt` checkpoint to Hugging Face Hub

Checkpoints are version-gated: the pipeline loads an existing v4.1+ model to continue training, or trains fresh if no valid checkpoint exists.

### Training Techniques

| Technique | Detail |
|-----------|--------|
| LR Warmup | 2-epoch linear ramp (0.1× → 1×) before cosine annealing |
| Cosine Warm Restarts | Periodic LR restarts to escape local minima |
| EMA Weight Averaging | Decay 0.999 — smoother, better-generalizing weights |
| Gradient Clipping | Max norm 1.0 — prevents exploding gradients |
| Gradient Accumulation | Simulates batch size 256 from smaller micro-batches |
| Data Augmentation | Gaussian noise (0.5%) + random timestep masking (5%) |
| BalancedDirectionLoss | 60% Huber regression + 40% weighted BCE direction |
| Early Stopping | Patience 15 epochs on EMA validation loss |
| Feature Clamping | `[-10, 10]` clamp after normalisation — prevents saturation |
| Mixed Precision | bfloat16 on CPU, float16 on CUDA |

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

## Checkpoint Format

Every `.pt` file saved by v5.0 contains the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `model_state_dict` | `dict` | PyTorch model weights |
| `model_type` | `str` | `"stock"` or `"forex"` |
| `architecture` | `str` | `"MeridianModel-2026"` |
| `version` | `str` | `"5.0"` |
| `input_size` | `int` | `44` (feature count) |
| `seq_len` | `int` | `30` (lookback window) |
| `dim` | `int` | Hidden dimension |
| `num_layers` | `int` | Transformer depth |
| `num_heads` | `int` | Attention heads |
| `num_kv_heads` | `int` | KV heads (GQA) |
| `num_experts` | `int` | MoE expert count |
| `num_prediction_heads` | `int` | Output prediction heads |
| `dropout` | `float` | Dropout rate |
| `use_mamba` | `bool` | Whether Mamba SSM is active |
| `mamba_state_dim` | `int` | Mamba hidden state size |
| `scaler_mean` | `Tensor` | Feature normalisation mean |
| `scaler_std` | `Tensor` | Feature normalisation std |
| `metadata` | `dict` | `best_val_loss`, `direction_accuracy`, `target_min/max`, `training_history` |

Old `"RevolutionaryFinancialModel-2026"` checkpoints (pre-v5.0) are still loadable — the loader accepts both architecture strings.

---

## Project Structure

```
meridianalgo/
  meridian_model.py        # MeridianModel v5.0 architecture (GQA, MoE, optional Mamba)
  revolutionary_model.py   # Backward-compat shim — re-exports from meridian_model.py
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
  meridian-stocks.yml      # Automated stock training (every hour, :00)
  meridian-forex.yml       # Automated forex training (every hour, :30)
  lint.yml                 # Code quality checks
tests/
  conftest.py              # Shared fixtures (model checkpoints)
  test_checkpoint_health.py      # Checkpoint metadata health checks
  test_model_inference.py        # Forward pass and state dict tests
  test_directional_signal.py     # Live directional accuracy on real market data
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
