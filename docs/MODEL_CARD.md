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

Meridian.AI is a deep learning system for predicting price movements across stocks and forex pairs. Version 5.0 introduces **MeridianModel** — a clean, verifiably-trained architecture combining grouped query attention, mixture-of-experts routing, and optional Mamba SSM — trained continuously via GitHub Actions every hour.

Every checkpoint produced by v5.0+ is verifiably correct: validation runs on every epoch, direction accuracy is measured against held-out data, and all loss values are guaranteed finite.

## Model versions

| Version | Architecture string | Params | Notes |
|---------|-------------------|--------|-------|
| v5.0 | `MeridianModel-2026` | ~11M (CPU) | Current; 7 training bugs fixed |
| v4.1 | `RevolutionaryFinancialModel-2026` | ~45M | Old checkpoints; still loadable |

Both architecture strings are accepted by the loader — no migration required.

## Architecture

### MeridianBlock

Each layer of MeridianModel is a `MeridianBlock` containing:

1. **RMSNorm** — pre-norm before attention
2. **GroupedQueryAttention (GQA)** — multi-head attention with fewer KV heads; QK-Norm for stability; RoPE position encoding
3. **Optional MambaBlock** — vectorised selective-scan SSM (disabled in CPU default)
4. **Layer Scale** — per-block learnable scalar (init 0.1) for training stability at depth
5. **Stochastic Depth** — drop-path regularisation
6. **RMSNorm** — pre-norm before MoE
7. **MixtureOfExperts** — `num_experts` SwiGLU expert networks, top-2 routing

### Component table

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Attention | GQA + QK-Norm | Reduced KV cache, training stability |
| Position | RoPE | Relative temporal awareness |
| Expert routing | MoE, top-2, SwiGLU | Regime-specific specialization |
| Activations | SwiGLU | Better gradient flow vs GELU/ReLU |
| Normalisation | RMSNorm + Layer Scale | Stable training at depth |
| Regularisation | Stochastic Depth | Generalisation, prevents overfitting |
| Optional SSM | Mamba (vectorised scan) | Long-range sequential dependencies |
| Loss | BalancedDirectionLoss | Joint regression + direction accuracy |

## Model specifications

| Spec | CPU Default (v5.0) | GPU / Large |
|------|--------------------|-------------|
| Parameters | ~11M | ~45M |
| Hidden dimension | 256 | 384 |
| Layers | 6 | 6 |
| Attention heads | 4 | 6 |
| KV heads | 2 | 2 |
| Experts | 4 (top-2) | 4 (top-2) |
| Prediction heads | 4 | 4 |
| Mamba SSM | Disabled | Optional |
| Input features | 44 | 44 |
| Sequence length | 30 timesteps | 30 timesteps |

## Available models

### Meridian.AI Stocks

- **File**: `models/Meridian.AI_Stocks.pt`
- **Coverage**: 49+ equities — AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, SPY, and more
- **Data**: Daily + 2yr hourly + 5yr weekly OHLCV with 44 technical indicators
- **Training**: Automatically retrained every hour (GitHub Actions, `:00`)

### Meridian.AI Forex

- **File**: `models/Meridian.AI_Forex.pt`
- **Coverage**: 22 currency pairs — EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD, etc.
- **Data**: Multi-timeframe OHLCV with 44 technical indicators
- **Training**: Automatically retrained every hour (GitHub Actions, `:30`)

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
prediction = ml.predict("EURUSD=X", days=5)
print(prediction)
```

## Training configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (`weight_decay=0.02`, `betas=(0.9, 0.95)`) |
| LR warmup | 2-epoch linear ramp, 10% → 100% of base LR |
| Scheduler | CosineAnnealingWarmRestarts after warmup |
| Loss | BalancedDirectionLoss (60% Huber + 40% weighted BCE) |
| Effective batch size | 256 via gradient accumulation |
| Gradient clipping | Max norm 1.0 |
| EMA | Decay 0.999 — used for validation and saved checkpoint |
| Data augmentation | Gaussian noise (0.5%) + timestep masking (5%) |
| Early stopping | Patience 15 on EMA validation loss |
| Mixed precision | bfloat16 on CPU, float16 on CUDA |
| Feature clamping | `[-10, 10]` after z-score normalisation |
| Sample cap | 60K most-recent rows per run |
| CI budget | 45 minutes, up to 999 epochs |

## Checkpoint format

Every v5.0 `.pt` file contains:

```python
{
    "model_state_dict": ...,       # PyTorch weights
    "model_type": "stock",         # or "forex"
    "architecture": "MeridianModel-2026",
    "version": "5.0",
    "input_size": 44,
    "seq_len": 30,
    "dim": 256,
    "num_layers": 6,
    "num_heads": 4,
    "num_kv_heads": 2,
    "num_experts": 4,
    "num_prediction_heads": 4,
    "dropout": 0.1,
    "use_mamba": False,
    "mamba_state_dim": 4,
    "scaler_mean": Tensor,         # shape (44,)
    "scaler_std": Tensor,          # shape (44,)
    "metadata": {
        "best_val_loss": float,
        "direction_accuracy": float,   # percent (0–100)
        "target_min": float,
        "target_max": float,
        "training_history": [...],
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

1. Performance may degrade during black swan events or extreme market dislocation.
2. Predictive accuracy decreases as the forecast horizon extends beyond a few days.
3. The model reflects statistical patterns in historical data — these patterns may not persist.
4. Pre-v5.0 checkpoints (before 2026-05-15) have a known bug where validation never ran; direction accuracy is unreliable on those models.
5. For research and educational use only — not financial advice.

## Citation

```bibtex
@software{meridianalgo_2026,
  title  = {Meridian.AI: Financial Prediction Engine},
  author = {MeridianAlgo},
  year   = {2026},
  version = {5.0.0},
  url    = {https://github.com/MeridianAlgo/AraAI}
}
```

## Disclaimer

These models are for research and educational purposes only. They do not constitute financial advice. Trading financial instruments carries significant risk. Past performance does not guarantee future results. The developers and contributors are not liable for any financial losses. All trading decisions are yours alone.

## License

MIT License. See the [GitHub repository](https://github.com/MeridianAlgo/AraAI) for details.
