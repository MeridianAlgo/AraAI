---
library_name: pytorch
license: mit
tags:
- finance
- trading
- time-series
- transformer
- mamba
- state-space-models
- stock-prediction
- forex-prediction
---

# Meridian.AI - Financial Prediction Models

## Overview

Meridian.AI is a 45M-parameter deep learning system for predicting price movements across stocks and forex pairs. It combines Mamba-2 state space models, sparse mixture-of-experts routing, and grouped query attention into a single unified architecture, trained continuously via GitHub Actions.

## Architecture

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Sequence Modeling | Mamba-2 SSM with selective scan | Linear-time long-range dependencies |
| Attention | Grouped Query Attention (GQA) + QK-Norm | Efficient multi-head attention with reduced KV cache |
| Position Encoding | Rotary Position Embeddings (RoPE) | Relative temporal awareness |
| Expert Routing | Mixture of Experts (4 experts, top-2) | Regime-specific specialization |
| Activations | SwiGLU (gated linear units) | Improved gradient flow |
| Normalization | RMSNorm + Layer Scale | Training stability at scale |
| Regularization | Stochastic Depth (drop path) | Better generalization |
| Training | Mixed precision (FP16/BF16) via Accelerate | 2x throughput on compatible hardware |
| Loss | BalancedDirectionLoss (Huber + BCE) | Joint price regression and direction accuracy |

## Model Specifications

```
Architecture: Revolutionary v4.1
Parameters: ~45 Million (33,071,045)
Input Features: 44 technical indicators
Sequence Length: 30 timesteps
Hidden Dimension: 384
Transformer Layers: 6
Attention Heads: 6 (2 KV heads)
Experts: 4 (top-2 routing)
Prediction Heads: 4
```

## Available Models

### Meridian.AI Stocks

- **File:** `models/Meridian.AI_Stocks.pt`
- **Coverage:** 49 equities across sectors (AAPL, MSFT, GOOGL, AMZN, TSLA, etc.)
- **Data:** Daily + hourly + weekly OHLCV with 44 technical indicators

### Meridian.AI Forex

- **File:** `models/Meridian.AI_Forex.pt`
- **Coverage:** 22 currency pairs (EUR/USD, GBP/USD, USD/JPY, etc.)
- **Data:** Multi-timeframe OHLCV with 44 technical indicators

## Usage

```python
from huggingface_hub import hf_hub_download
from meridianalgo.unified_ml import UnifiedStockML

# Download the stocks model
model_path = hf_hub_download(
    repo_id="meridianal/ARA.AI",
    filename="models/Meridian.AI_Stocks.pt"
)

ml = UnifiedStockML(model_path=model_path)
prediction = ml.predict_ultimate("AAPL", days=5)
print(prediction)
```

```python
from meridianalgo.forex_ml import ForexML

# Download the forex model
model_path = hf_hub_download(
    repo_id="meridianal/ARA.AI",
    filename="models/Meridian.AI_Forex.pt"
)

ml = ForexML(model_path=model_path)
prediction = ml.predict("EURUSD=X", days=5)
print(prediction)
```

## Training

- **Optimizer:** AdamW (weight decay 0.02, betas 0.9/0.95)
- **Scheduler:** Cosine annealing with warm restarts
- **Loss:** BalancedDirectionLoss (60% Huber + 40% balanced BCE)
- **Batch Size:** Effective 256 via gradient accumulation
- **EMA:** Exponential moving average of weights (decay 0.999)
- **Data Augmentation:** Gaussian noise (0.5%) + random timestep masking (5%)
- **Early Stopping:** Patience of 20 epochs on EMA validation loss
- **Pipeline:** Automated via GitHub Actions every 6 hours

Metrics are tracked with Comet ML. Version gating ensures stale pre-v4.1 checkpoints are never loaded.

## Technical Indicators (44 features)

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

1. Performance may degrade during black swan events or extreme volatility.
2. Predictive accuracy decreases as forecast horizon extends.
3. Models reflect patterns in historical data which may not repeat.
4. For research and educational use only -- not financial advice.

## Citation

```bibtex
@software{meridianalgo_2026,
  title = {Meridian.AI: Financial Prediction Engine},
  author = {MeridianAlgo},
  year = {2026},
  version = {4.1.0}
}
```

## Disclaimer

These models are for research and educational purposes only. They do not constitute financial advice. Trading financial instruments carries significant risk. The developers and contributors are not liable for any financial losses. All trading decisions are yours alone.

## License

MIT License. See the [GitHub repository](https://github.com/MeridianAlgo/AraAI) for details.
