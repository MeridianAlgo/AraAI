# Meridian.AI Documentation Index

**Version 1.0.0 (Production)** | [GitHub](https://github.com/MeridianAlgo/AraAI) | [Hugging Face](https://huggingface.co/meridianal/ARA.AI)

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
| [Changelog](CHANGELOG.md) | The v1.0.0 production release plus the full pre-1.0 development history |

---

## Key facts

- **Training**: Every hour — stocks at `:00`, forex at `:30` — via GitHub Actions
- **Models**: `models/Meridian.AI_Stocks.pt` and `models/Meridian.AI_Forex.pt` on Hugging Face
- **Architecture**: MeridianModel-2026 — GQA + MoE + optional Mamba SSM (checkpoint architecture rev `6.0`)
- **Parameters**: ~430K (`dim=96`, 3 layers, 2 experts) — deliberately compact so the model must extract signal instead of memorising noise
- **Input**: 30 timesteps × 44 technical indicators
- **Output**: Predicted next-day return + confidence (std across prediction heads)
- **Validated**: Walk-forward backtest — forex 63.5% next-day directional (z=8.4, flagship); stocks 51.6% vs a 51.5% drift baseline (experimental, no significant edge)

## Source layout

```
meridianalgo/
  meridian_model.py        # MeridianModel-2026 — canonical architecture
  large_torch_model.py     # AdvancedMLSystem — training loop, data loading, inference
  direction_loss.py        # BalancedDirectionLoss, DirectionAwareLoss, metrics
  unified_ml.py            # UnifiedStockML — stock prediction API + feature engineering
  forex_ml.py              # ForexML — forex prediction API
  utils.py                 # GPU detection and accuracy tracking helpers
scripts/
  train_stocks.py          # Stock training entrypoint (called by CI)
  train_forex.py           # Forex training entrypoint (called by CI)
  fetch_and_store_data.py  # Market data ingestion → SQLite
  push_to_hf.py            # Hugging Face Hub deployment
  sanity_check_model.py    # Post-training gate that blocks degenerate models
  migrate_hf_legacy.py     # Archives older checkpoints into legacy/ on Hugging Face
.github/workflows/
  stocks.yml               # Stock CI: hourly at :00
  forex.yml                # Forex CI: hourly at :30
  daily-model-tests.yml    # Daily checkpoint health + live signal tests
  lint.yml                 # isort + black + ruff on push/PR
tests/
  conftest.py              # Shared fixtures (checkpoint loading, model building)
  test_checkpoint_health.py      # Metadata health: finite losses, noise-aware direction floor
  test_model_inference.py        # Forward pass shape, state dict, NaN checks
  test_directional_signal.py     # Live accuracy on yfinance data (requires network)
  test_predict_denormalization.py # Predictions are scaled back correctly
docs/
  INDEX.md                 # This file
  QUICK_START.md           # 5-minute setup guide
  MODEL_CARD.md            # Hugging Face model card
  ARCHITECTURE.md          # Deep-dive: MeridianBlock components
  TRAINING.md              # Data pipeline, loss, LR schedule, CI structure
  INDICATORS.md            # All 44 indicators: index, formula, category
  LOSS_FUNCTIONS.md        # Loss function math and direction metrics
  CHANGELOG.md             # v1.0.0 release + full pre-1.0 development history
```
