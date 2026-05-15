# Training Guide

How Meridian.AI fetches data, trains models, and deploys checkpoints — end to end.

---

## Pipeline overview

```
GitHub Actions (every hour)
        |
        v
fetch_and_store_data.py          ← yfinance → SQLite (training.db)
        |
        v
train_stock_model.py             ← SQLite → MeridianModel → .pt file
train_forex_model.py
        |
        v
push_elite_models.py             ← .pt file → Hugging Face Hub
```

Each step runs as a separate GitHub Actions job so failures are isolated and artifacts are passed between jobs explicitly.

---

## Data ingestion

**Script**: `scripts/fetch_and_store_data.py`

Data is fetched from Yahoo Finance via `yfinance` and stored in a local SQLite database (`training.db`). Three timeframes are fetched per symbol:

| Timeframe | Period | Resolution |
|-----------|--------|------------|
| Daily | Max available (up to 30+ years) | 1 day |
| Hourly | Last 2 years | 1 hour |
| Weekly | Last 5 years | 1 week |

**Stock symbols** (up to 50 per run): AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, JPM, BAC, V, MA, WMT, JNJ, PG, UNH, HD, and more.

**Forex pairs** (up to 30 per run): EURUSD=X, GBPUSD=X, USDJPY=X, AUDUSD=X, USDCHF=X, USDCAD=X, NZDUSD=X, EURGBP=X, and more.

Each row in `market_data` stores: `symbol`, `timeframe`, `timestamp`, `open`, `high`, `low`, `close`, `volume`.

---

## Feature engineering

**44 technical indicators** are computed from raw OHLCV data by `UnifiedStockML._add_indicators()` and `_extract_features()`:

| Category | Features |
|----------|---------|
| Price | 1-day return, log return, 5-day volatility, ATR (14) |
| Trend | SMA 5/10/20/50/200, EMA 5/10/20/50/200 |
| Momentum | RSI (14), Fast RSI (7), Stochastic RSI, Momentum (10), ROC (10), Williams %R (14) |
| Oscillators | MACD (12,26,9), MACD Signal, MACD Histogram, Stochastic K/D (14), CCI (20) |
| Volatility | Bollinger Upper/Lower/Width/%B (20,2), Keltner Upper/Lower/%K (20,1.5) |
| Volume | Volume SMA ratio, OBV (normalised by 1M) |
| Trend Strength | ADX (14), +DI, -DI, Price/SMA50, Price/SMA200, ATR% |
| Mean Reversion | Z-score (20d), Distance from 52-week high |

Features are computed row-by-row on price history up to the current timestep — no lookahead. `NaN` rows (before indicators warm up) are dropped before training.

### Normalisation

After extraction, each feature is z-score normalised using the mean and standard deviation computed over the training set:

```python
x = (x - mean) / std
x = torch.clamp(x, -10, 10)    # prevents saturation from extreme outliers
```

The `mean` and `std` tensors are saved in the checkpoint (`scaler_mean`, `scaler_std`) so inference normalises identically to training.

### Target variable

The target is the **next-day return**:

```python
target = (close[t+1] - close[t]) / close[t]
```

Outliers are clipped at ±100% for stocks and ±20% for forex before training. Returns beyond these bounds are almost certainly bad data, not real moves, and would dominate the MSE surface.

---

## Model creation and checkpoint loading

**Script**: `meridianalgo/large_torch_model.py` — `AdvancedMLSystem` class

On each CI run, the script attempts to download the current checkpoint from Hugging Face:

```python
hf_hub_download(repo_id="meridianal/ARA.AI", filename="models/Meridian.AI_Stocks.pt")
```

If a valid checkpoint is found (architecture string `"MeridianModel-2026"` or `"RevolutionaryFinancialModel-2026"`, version ≥ 4.1), it is loaded and training continues. If no checkpoint exists or the version is too old, a fresh model is initialised.

**CPU default config** (used in CI):
```python
MeridianModel(
    input_size=44, seq_len=30,
    dim=256, num_layers=6,
    num_heads=4, num_kv_heads=2,
    num_experts=4, num_prediction_heads=4,
    dropout=0.1,
    use_mamba=False, mamba_state_dim=4,
)
```

---

## Training loop

### Data preparation

- Up to **60K most-recent samples** are selected from the database (oldest rows dropped when over limit)
- Samples are split 80/20 train/validation (chronological — validation is always the most recent 20%)
- `DataLoader` with shuffle for train, no shuffle for validation; batch size from gradient accumulation config

### Optimiser

```python
AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.02,
    betas=(0.9, 0.95),
    eps=1e-8,
)
```

### Learning rate schedule

Two-phase `SequentialLR`:

1. **Warmup** (epochs 0–1): `LinearLR` from `0.1 × lr` → `lr` over 2 epochs. Prevents large early gradients from destabilising freshly-initialised weights.
2. **Cosine annealing** (epoch 2+): `CosineAnnealingWarmRestarts(T_0=max(3, (epochs-2)//3), T_mult=1, eta_min=lr×0.01)`. Periodic LR restarts help escape local minima.

### Loss function: BalancedDirectionLoss

```python
loss = 0.6 * HuberLoss(pred, target) + 0.4 * WeightedBCE(pred, direction)
```

**Huber loss** (SmoothL1): Quadratic near zero, linear for large errors. More robust than MSE to outlier returns that survive the clip threshold.

**Weighted BCE**: Predicts direction (up/down) from the raw predicted return (scaled by 10× to get logit-range values). Class weights balance up/down imbalance common in bull markets:

```python
weight_up   = N / (2 * N_up)
weight_down = N / (2 * N_down)
```

This prevents the model from collapsing to "always predict up."

### Gradient accumulation and clipping

Gradients accumulate over micro-batches to simulate an effective batch size of 256. After accumulation:

```python
accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

Gradient clipping prevents exploding gradients from rare extreme-return samples.

### EMA weight averaging

An exponential moving average of model weights is maintained with decay 0.999:

```python
ema_weights = 0.999 * ema_weights + 0.001 * model_weights
```

Validation always evaluates the EMA model, not the live model. The checkpoint saves the EMA weights — this is what gets deployed to Hugging Face.

### Validation (runs every epoch)

After each training epoch:

1. EMA weights are loaded into a copy of the model
2. The validation set (20% of data, held out) is evaluated with no gradient computation
3. `BalancedDirectionLoss` is computed on the validation set
4. `calculate_direction_metrics()` measures direction accuracy (%)
5. If `val_loss < best_val_loss`, the checkpoint is updated

This guarantees `best_val_loss` and `direction_accuracy` are always set — the bug fixed in v5.0.

### Early stopping

Training stops if EMA validation loss does not improve for 15 consecutive epochs. NaN validation epochs (from numerical instability) do not count toward the patience counter.

### Time-based termination

Training runs for up to 45 minutes (`--max-time 45`). When the clock hits the limit, the current best checkpoint is saved and the loop exits cleanly — regardless of whether the epoch finished. This is checked at the end of each epoch.

### Data augmentation

Applied to each training batch:

- **Gaussian noise**: `x += Normal(0, 0.005)` — prevents overfitting to exact feature values
- **Timestep masking**: 5% of timesteps are randomly zeroed — forces the model to be robust to missing data

---

## Checkpoint saving

At the end of training (or on time limit), the best checkpoint is saved:

```python
torch.save({
    "model_state_dict": ema_state_dict,
    "model_type": "stock",                      # or "forex"
    "architecture": "MeridianModel-2026",
    "version": "5.0",
    "input_size": 44,
    "seq_len": 30,
    "dim": model.dim,
    "num_layers": model.num_layers,
    "num_heads": model.num_heads,
    "num_kv_heads": model.num_kv_heads,
    "num_experts": model.num_experts,
    "num_prediction_heads": model.num_prediction_heads,
    "dropout": model.dropout,
    "use_mamba": model.use_mamba,
    "mamba_state_dim": model.mamba_state_dim,
    "scaler_mean": mean_tensor,
    "scaler_std": std_tensor,
    "metadata": {
        "best_val_loss": float,
        "direction_accuracy": float,           # percent, 0–100
        "target_min": float,
        "target_max": float,
        "training_history": [{"epoch": i, "val_loss": v, ...}],
    },
}, path)
```

The `use_mamba` flag is read from the live model object — not hardcoded — so round-trip save/load is always faithful.

---

## Deployment

**Script**: `scripts/push_elite_models.py`

The trained `.pt` file is uploaded to the Hugging Face Hub repository `meridianal/ARA.AI`:

```
models/Meridian.AI_Stocks.pt
models/Meridian.AI_Forex.pt
```

The upload uses the `HF_TOKEN` secret. If the upload fails, the checkpoint artifact is still preserved in GitHub Actions for 7 days.

---

## CI workflow structure

```
setup job
  ├─ Checkout code
  ├─ Install dependencies
  ├─ Fetch market data → training.db
  ├─ Check row count → set has_data output
  └─ Upload training.db artifact

train job (needs: setup, if has_data)
  ├─ Download training.db artifact
  ├─ Download existing model from HF Hub
  ├─ python scripts/train_*.py --max-time 45 --epochs 999
  └─ Upload model artifact (retention: 7 days)

deploy job (needs: train)
  ├─ Download model artifact
  └─ python scripts/push_elite_models.py

cleanup job (always runs)
  ├─ Delete training.db artifact
  ├─ If train succeeded → close any open failure issues
  └─ If train failed → create/update failure issue on GitHub
```

Concurrency groups (`forex-training`, `stock-training`) ensure only one run per asset type is active at a time. When CI triggers two overlapping runs, the second waits in the queue rather than cancelling the first.

---

## Running training locally

```bash
# Fetch data first
python scripts/fetch_and_store_data.py \
  --db-file training.db \
  --asset-type stock \
  --limit 50

# Train (adjust --max-time for your machine)
python scripts/train_stock_model.py \
  --db-file training.db \
  --output models/Meridian.AI_Stocks.pt \
  --use-all-data \
  --epochs 999 \
  --max-time 60 \
  --comet-api-key $COMET_API_KEY   # optional

# Inspect the checkpoint
python -c "
import torch
ckpt = torch.load('models/Meridian.AI_Stocks.pt', map_location='cpu', weights_only=False)
print('val_loss:', ckpt['metadata']['best_val_loss'])
print('direction_acc:', ckpt['metadata']['direction_accuracy'])
print('version:', ckpt['version'])
"
```

---

## Experiment tracking (Comet ML)

When `COMET_API_KEY` is set, each training run logs:

- Train and validation loss per epoch
- Direction accuracy per epoch
- EMA validation loss
- Hyperparameters (dim, num_layers, lr, etc.)
- Time per epoch

View runs at [comet.ml](https://www.comet.ml) under the project configured in the training script.

---

## See also

- [Architecture](ARCHITECTURE.md) — internals of MeridianModel
- [Model card](MODEL_CARD.md) — specs, checkpoint format, Hugging Face usage
- [Quick start](QUICK_START.md) — get up and running in 5 minutes
