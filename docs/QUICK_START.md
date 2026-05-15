# Meridian.AI — Quick Start Guide

**MeridianModel v5.0 | Hourly CI | 11M–45M Parameters**

Get predictions running in under 5 minutes.

---

## Installation

```bash
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI

python -m venv venv
source venv/bin/activate        # Linux/macOS
# or
venv\Scripts\activate           # Windows

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Download the latest models

Models are trained hourly and hosted on Hugging Face. You don't need to train locally.

```python
from huggingface_hub import hf_hub_download

stocks_path = hf_hub_download(
    repo_id="meridianal/ARA.AI",
    filename="models/Meridian.AI_Stocks.pt"
)

forex_path = hf_hub_download(
    repo_id="meridianal/ARA.AI",
    filename="models/Meridian.AI_Forex.pt"
)
```

---

## Run a prediction

### Stocks

```python
from meridianalgo.unified_ml import UnifiedStockML

ml = UnifiedStockML(model_path=stocks_path)
result = ml.predict_ultimate("AAPL", days=5)

print(f"Current price:  ${result['current_price']:.2f}")
print(f"Direction:      {result['direction']}")
for p in result['predictions']:
    print(f"  Day {p['day']}: ${p['predicted_price']:.2f}")
```

### Forex

```python
from meridianalgo.forex_ml import ForexML

ml = ForexML(model_path=forex_path)
result = ml.predict("EURUSD=X", days=5)

print(f"Current rate:   {result['current_price']:.5f}")
print(f"Direction:      {result['direction']}")
for p in result['predictions']:
    print(f"  Day {p['day']}: {p['predicted_price']:.5f}")
```

---

## Supported symbols

**Stocks** — any ticker supported by `yfinance`: `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`, `NVDA`, `JPM`, `SPY`, etc.

**Forex** — append `=X` for yfinance compatibility:
`EURUSD=X`, `GBPUSD=X`, `USDJPY=X`, `AUDUSD=X`, `USDCHF=X`, `USDCAD=X`, etc.

---

## Model specs at a glance

| Property | Value |
|----------|-------|
| Architecture | MeridianModel v5.0 |
| Parameters | ~11M (CPU default) |
| Input | 44 technical indicators |
| Lookback | 30 timesteps |
| Hidden dim | 256 |
| Attention | GQA — 4 heads, 2 KV heads |
| Experts | 4 (top-2 MoE) |
| Training | Hourly via GitHub Actions |

---

## Train locally (optional)

If you want to train your own checkpoint rather than using the hosted model:

### 1. Add secrets to `.env`

```bash
HF_TOKEN=your_huggingface_token
COMET_API_KEY=your_comet_api_key   # optional — disables experiment tracking if absent
```

### 2. Fetch market data

```bash
# Fetch stock data (50 symbols)
python scripts/fetch_and_store_data.py \
  --db-file training.db \
  --asset-type stock \
  --limit 50

# Fetch forex data (22 pairs)
python scripts/fetch_and_store_data.py \
  --db-file training.db \
  --asset-type forex \
  --limit 30
```

### 3. Train

```bash
# Train stock model (45-minute budget, up to 999 epochs)
python scripts/train_stock_model.py \
  --db-file training.db \
  --output models/Meridian.AI_Stocks.pt \
  --use-all-data \
  --epochs 999 \
  --max-time 45

# Train forex model
python scripts/train_forex_model.py \
  --db-file training.db \
  --output models/Meridian.AI_Forex.pt \
  --use-all-data \
  --epochs 999 \
  --max-time 45
```

### 4. Push to Hugging Face (optional)

```bash
python scripts/push_elite_models.py \
  --model-path models/Meridian.AI_Stocks.pt \
  --model-type stock
```

---

## GitHub Actions setup

CI trains new models every hour automatically. To enable in your fork:

1. **Settings → Actions → General** → Enable "Allow all actions and reusable workflows"
2. **Settings → Secrets and variables → Actions** → Add:
   - `HF_TOKEN` — Hugging Face write token
   - `COMET_API_KEY` — Comet ML key (optional)
3. Workflows run automatically. Trigger manually: **Actions → Select workflow → Run workflow**

Available workflows:
| Workflow | Schedule | File |
|----------|----------|------|
| Meridian.AI for Stocks | Every hour at :00 | `meridian-stocks.yml` |
| Meridian.AI for Forex | Every hour at :30 | `meridian-forex.yml` |
| Lint | On push / PR | `lint.yml` |

---

## Run the test suite

```bash
pip install pytest

# All tests (requires model files at models/*.pt)
pytest tests/ -v

# Skip live-network tests
NO_NET=1 pytest tests/ -v

# Checkpoint health only (fast)
pytest tests/test_checkpoint_health.py -v
```

Test files:
| File | What it checks |
|------|---------------|
| `test_checkpoint_health.py` | Required keys, finite losses, direction accuracy > 50%, target range sanity |
| `test_model_inference.py` | Forward pass shape, state dict loading, no NaNs in output |
| `test_directional_signal.py` | Live directional accuracy on real yfinance data |

---

## Code quality

```bash
pip install black isort ruff

black meridianalgo/ scripts/ tests/
isort meridianalgo/ scripts/ tests/
ruff check --fix meridianalgo/ scripts/ tests/
```

---

## Troubleshooting

**`ImportError: meridianalgo not found`**
Run from the project root: `cd AraAI` and ensure your venv is activated.

**CUDA warnings on CPU**
Normal — the model auto-selects CPU when no GPU is present.

**`No existing model found, will train fresh`**
Expected on first run or if `HF_TOKEN` is not set. Training starts from random weights.

**Database errors**
```bash
rm training.db
python scripts/fetch_and_store_data.py --db-file training.db --asset-type stock --limit 10
```

**`best_val_loss` is None in an old checkpoint**
This is the bug fixed in v5.0. The old model never ran validation. Download the latest checkpoint from Hugging Face — any model trained after 2026-05-15 has this fixed.

---

## Further reading

- [Architecture deep-dive](ARCHITECTURE.md) — how MeridianModel works internally
- [Training guide](TRAINING.md) — data pipeline, loss function, LR schedule
- [Model card](MODEL_CARD.md) — Hugging Face model card
- [Changelog](CHANGELOG.md) — full version history
- [GitHub repository](https://github.com/MeridianAlgo/AraAI)
- [Hugging Face Hub](https://huggingface.co/meridianal/ARA.AI)

---

**Version**: 5.0.0 | **Maintained by**: [MeridianAlgo](https://github.com/MeridianAlgo)
