# Meridian.AI Documentation Index

**Version 5.0.0** | [GitHub](https://github.com/MeridianAlgo/AraAI) | [Hugging Face](https://huggingface.co/meridianal/ARA.AI)

---

## Start here

| Document | Description |
|----------|-------------|
| [Quick Start](QUICK_START.md) | Install, download models, run a prediction — 5 minutes |
| [README](../README.md) | Project overview, architecture summary, training pipeline |

## Model reference

| Document | Description |
|----------|-------------|
| [Model Card](MODEL_CARD.md) | Specs, checkpoint format, Hugging Face usage, limitations |
| [Architecture](ARCHITECTURE.md) | MeridianBlock internals — GQA, RoPE, MoE, Mamba, output heads |
| [Technical Indicators](INDICATORS.md) | All 44 features: formulas, index positions, category descriptions |
| [Loss Functions](LOSS_FUNCTIONS.md) | BalancedDirectionLoss, direction metrics, class weighting |

## Training reference

| Document | Description |
|----------|-------------|
| [Training Guide](TRAINING.md) | Data pipeline, optimiser, LR schedule, EMA, CI workflow structure |
| [Changelog](CHANGELOG.md) | Full version history from v1.0.0-Beta to v5.0.0 |

---

## Key facts

- **Training**: Every hour — stocks at `:00`, forex at `:30` — via GitHub Actions
- **Models**: `models/Meridian.AI_Stocks.pt` and `models/Meridian.AI_Forex.pt` on Hugging Face
- **Architecture**: MeridianModel v5.0 — GQA + MoE + optional Mamba SSM
- **Parameters**: ~11M (CPU default, `dim=256`) / ~45M (GPU large, `dim=384`)
- **Input**: 30 timesteps × 44 technical indicators
- **Output**: Predicted next-day return + confidence (std across prediction heads)
- **v5.0 critical fix**: Validation never ran in pre-v5 models — direction accuracy in old checkpoints is unreliable

## Source layout

```
meridianalgo/
  meridian_model.py        # MeridianModel v5.0 — canonical architecture
  revolutionary_model.py   # Backward-compat shim (re-exports from meridian_model.py)
  large_torch_model.py     # AdvancedMLSystem — training loop, data loading, inference
  direction_loss.py        # BalancedDirectionLoss, DirectionAwareLoss, metrics
  unified_ml.py            # UnifiedStockML — stock prediction API
  forex_ml.py              # ForexML — forex prediction API
  indicators.py            # 44 technical indicator computation
scripts/
  train_stock_model.py     # Stock training entrypoint (called by CI)
  train_forex_model.py     # Forex training entrypoint (called by CI)
  fetch_and_store_data.py  # Market data ingestion → SQLite
  push_elite_models.py     # Hugging Face Hub deployment
.github/workflows/
  meridian-stocks.yml      # Stock CI: hourly at :00
  meridian-forex.yml       # Forex CI: hourly at :30
  lint.yml                 # isort + black + ruff on push/PR
tests/
  conftest.py              # Shared fixtures (checkpoint loading, model building)
  test_checkpoint_health.py      # Metadata health: finite losses, direction > 50%
  test_model_inference.py        # Forward pass shape, state dict, NaN checks
  test_directional_signal.py     # Live accuracy on yfinance data (requires network)
docs/
  INDEX.md                 # This file
  QUICK_START.md           # 5-minute setup guide
  MODEL_CARD.md            # Hugging Face model card
  ARCHITECTURE.md          # Deep-dive: MeridianBlock components
  TRAINING.md              # Data pipeline, loss, LR schedule, CI structure
  INDICATORS.md            # All 44 indicators: index, formula, category
  LOSS_FUNCTIONS.md        # Loss function math and direction metrics
  CHANGELOG.md             # Version history from v1.0.0-Beta
```
