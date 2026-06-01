---
library_name: pytorch
license: mit
tags:
- finance
- trading
- time-series
- transformer
- moe
- grouped-query-attention
- stock-prediction
- forex-prediction
---

# Meridian.AI — Financial Prediction Models

## Overview

Meridian.AI is a deep-learning system that forecasts next-day price movement
for stocks and forex pairs. It reads recent market history, turns it into 44
technical indicators, and outputs a next-day return estimate plus a direction
signal (up/down).

Version 6.0 is a deliberate reset. Earlier versions reported very high
direction accuracy in training but performed near chance live — they had
collapsed onto a near-constant downward prediction. The root cause was a
contaminated training pipeline, not the architecture. v6.0 rebuilds the data
path from the ground up and shrinks the network so it has to learn real signal
instead of memorizing noise. See **What changed in v6.0** below.

The models retrain automatically every hour via GitHub Actions and publish
each fresh checkpoint here. You do not need a GPU.

## What changed in v6.0

| Area | Before (≤ v5.x) | Now (v6.0) |
|------|-----------------|------------|
| Training data | Daily **+ hourly + weekly** bars mixed in one table | **Daily only** — one consistent prediction target |
| Windowing | One flat array concatenated across all symbols (windows spanned symbol boundaries) | **Per-symbol** windowing, then sorted by date |
| Targets | Raw next-step return, clip ±1.0 (allowed impossible +100% targets) | Next-day return, clip **±0.25 (stocks) / ±0.10 (forex)** |
| Feature scaler | Fit on the whole dataset (val leaked into train) | Fit on the **train split only** |
| Price adjustment | Splits/dividends showed up as fake returns | `auto_adjust=True` — no fake split-day moves |
| Capacity | ~11M params (collapsed to a constant) | **~430K params** (forced to extract signal) |
| Push safety | None | **Sanity gate** blocks degenerate models from publishing |

## Honest performance

This is a next-day directional model. Measured live on held-out daily data
(8 large-cap stocks, 240 predictions):

- Directional accuracy: **~57%**, vs a ~55% always-up baseline.
- Mean prediction: small and positive (~+0.7%), in line with real daily drift.
- No directional collapse (it predicts both up and down).

That is a small but real edge for **one day ahead**. It is not a multi-day or
week-ahead forecaster — daily price direction is close to efficient, error
compounds quickly past one step, and any tool claiming reliable week-ahead
price prediction from OHLCV alone is overfitting. Use this for a next-day
directional tilt, not as a crystal ball.

## Repository layout

```
meridianal/ARA.AI/
├── models/
│   ├── Meridian.AI_Stocks.pt    ← current v6 stock checkpoint
│   └── Meridian.AI_Forex.pt     ← current v6 forex checkpoint
└── legacy/
    ├── Meridian.AI_Stocks_v5.2.2.pt   ← archived pre-v6 (biased) checkpoint
    └── Meridian.AI_Forex_v5.2.2.pt    ← archived pre-v6 (biased) checkpoint
```

The loader only accepts checkpoints at **version 6.0 or newer**. The v5.x
checkpoints in `legacy/` have an incompatible (larger) architecture and are
kept only for reference — do not use them for live prediction.

## Architecture

The model is **MeridianModel**: a compact transformer adapted for financial
time series. Each block contains:

1. **RMSNorm** — pre-norm before attention
2. **Grouped Query Attention (GQA)** — fewer KV heads, QK-Norm for stability, RoPE positions
3. **Optional Mamba SSM** — vectorised selective scan (off by default on CPU)
4. **Layer Scale** — per-block learnable scalar for stable training at depth
5. **Stochastic Depth** — drop-path regularisation
6. **RMSNorm** — pre-norm before the MoE
7. **Mixture of Experts** — SwiGLU experts, top-2 routing

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Attention | GQA + QK-Norm | Reduced KV cache, training stability |
| Position | RoPE | Relative temporal awareness |
| Expert routing | MoE, top-2, SwiGLU | Regime-specific specialization |
| Activations | SwiGLU | Better gradient flow vs GELU/ReLU |
| Normalisation | RMSNorm + Layer Scale | Stable training at depth |
| Regularisation | Stochastic Depth | Generalisation |
| Optional SSM | Mamba (vectorised scan) | Long-range dependencies |
| Loss | BalancedDirectionLoss | Joint regression + direction accuracy |

## Model specifications (v6.0 default)

| Spec | Value |
|------|-------|
| Parameters | ~430K |
| Hidden dimension | 96 |
| Layers | 3 |
| Attention heads | 4 (2 KV heads) |
| Experts | 2 (top-2) |
| Prediction heads | 2 |
| Mamba SSM | Disabled (CPU default) |
| Input features | 44 technical indicators |
| Sequence length | 30 timesteps (daily) |

## Available models

### Meridian.AI Stocks

- **File**: `models/Meridian.AI_Stocks.pt`
- **Coverage**: ~50 large-cap US equities per run (AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, SPY, …)
- **Data**: Max daily history, split/dividend adjusted, 44 technical indicators
- **Training**: Automatic hourly CI retrain (GitHub Actions)
- **Tracking**: Comet project `meridianalgo/meridian-ai-stock-v5`

### Meridian.AI Forex

- **File**: `models/Meridian.AI_Forex.pt`
- **Coverage**: 22 currency pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD, …)
- **Data**: Max daily history, 44 technical indicators
- **Training**: Automatic hourly CI retrain (GitHub Actions)
- **Tracking**: Comet project `meridianalgo/meridian-ai-forex-v5`

## Usage

```python
from huggingface_hub import hf_hub_download
from meridianalgo.unified_ml import UnifiedStockML

model_path = hf_hub_download(
    repo_id="meridianal/ARA.AI",
    filename="models/Meridian.AI_Stocks.pt"
)

ml = UnifiedStockML(model_path=model_path)
prediction = ml.predict_ultimate("AAPL", days=5)
print(prediction)
```

```python
from huggingface_hub import hf_hub_download
from meridianalgo.forex_ml import ForexML

model_path = hf_hub_download(
    repo_id="meridianal/ARA.AI",
    filename="models/Meridian.AI_Forex.pt"
)

ml = ForexML(model_path=model_path)
prediction = ml.predict_forex("EUR/USD", days=5)
print(prediction)
```

> Note: the model needs ~200 days of price history to compute its indicators
> (it uses a 200-day moving average), and it outputs a **single next-day**
> return. A multi-day horizon is produced by rolling that one-step prediction
> forward, so confidence drops sharply after the first day.

## Training configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (`weight_decay=0.02`, `betas=(0.9, 0.95)`) |
| LR warmup | Linear ramp, then CosineAnnealingWarmRestarts |
| Loss | BalancedDirectionLoss (60% Huber + 40% weighted BCE) |
| Effective batch size | 256 via gradient accumulation |
| Gradient clipping | Max norm 1.0 |
| EMA | Decay 0.999 — used for validation and the saved checkpoint |
| Data augmentation | Gaussian noise + timestep masking |
| Train/val split | Chronological — last 20% held out (no shuffle) |
| Scaler fit | **Train split only** (no validation leakage) |
| Target clip | ±0.25 (stocks) / ±0.10 (forex) |
| Feature clamping | `[-10, 10]` after z-score normalisation |
| Sample cap | 60K most-recent rows per run |
| CI step budget | Up to 2000 optimizer steps per run |
| Checkpoint write | Atomic (`.tmp` → `os.replace`) |
| Push safety | Sanity gate blocks degenerate models |
| Comet logging | Every step loss/LR/grad-norm + per-epoch metrics + per-symbol dataset audit |

## Checkpoint format

```python
{
    "model_state_dict": ...,       # PyTorch weights
    "model_type": "stock",         # or "forex"
    "architecture": "MeridianModel-2026",
    "version": "6.0.1",
    "input_size": 44,
    "seq_len": 30,
    "dim": 96,
    "num_layers": 3,
    "num_heads": 4,
    "num_kv_heads": 2,
    "num_experts": 2,
    "num_prediction_heads": 2,
    "dropout": 0.15,
    "use_mamba": False,
    "scaler_mean": Tensor,         # shape (30, 44)
    "scaler_std": Tensor,          # shape (30, 44)
    "metadata": {
        "best_val_loss": float,
        "training_history": [...],
        "trained_symbols": [...],
        "training_date": str,
    }
}
```

## Technical indicators (44 features)

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

## Limitations

1. Next-day model only. Multi-day output is recursive and degrades fast past day one.
2. Daily direction is near-efficient; the live edge over a naive baseline is small.
3. Performance degrades during black-swan events and regime shifts.
4. Patterns are statistical and may not persist.
5. Pre-v6 checkpoints in `legacy/` have a known downward-bias bug — do not use them.
6. For research and educational use only — not financial advice.

## Citation

```bibtex
@software{meridianalgo_2026,
  title  = {Meridian.AI: Financial Prediction Engine},
  author = {MeridianAlgo},
  year   = {2026},
  version = {6.0.1},
  url    = {https://github.com/MeridianAlgo/AraAI}
}
```

## Disclaimer

These models are for research and educational purposes only. They do not
constitute financial advice. Trading carries significant risk and past
performance does not guarantee future results. The developers and contributors
are not liable for any financial losses. All trading decisions are yours alone.

## License

MIT License. See the [GitHub repository](https://github.com/MeridianAlgo/AraAI) for details.
