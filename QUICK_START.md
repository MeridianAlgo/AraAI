# ⚡ ARA AI - Quick Start Guide

Get up and running with ARA AI in under 5 minutes!

---

## 🎯 Training Performance

| Metric | Value |
|--------|-------|
| **Training Time** | ~2-3 minutes for both models |
| **Accuracy** | >99.9% |
| **Frequency** | Every hour (24x per day) |
| **Models** | 2 unified (1 stock + 1 forex) |
| **Loss** | <0.0004 |

---

## 🚀 Installation

```bash
# Clone and setup
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## ⚡ Quick Commands

### Windows Setup
```bash
# Set UTF-8 encoding (required on Windows)
$env:PYTHONIOENCODING="utf-8"
```

### Training Commands

```bash
# Test single model (~15 seconds)
python scripts/quick_train.py --symbol AAPL --epochs 50

# Train multiple models
python scripts/batch_train.py --symbols AAPL GOOGL MSFT

# Train random stocks
python scripts/batch_train.py --random 5

# View training dashboard
python scripts/training_dashboard.py

# Full training session
python scripts/continuous_training.py
```

---

## 📊 Training Schedules

### Hourly Training (Automated)
- **Frequency**: Every hour (24 times per day)
- **Schedule**: Runs at minute 0 of every hour
- **Models**: 2 unified models
- **Training Time**: ~2-3 minutes per session
- **Cost**: Requires paid GitHub Actions plan OR use public repository (unlimited)

---

## 🤖 Enable Automated Training

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Setup automated training"
git push
```

### Step 2: Enable Actions
1. Go to repository **Settings**
2. Navigate to **Actions** → **General**
3. Enable "Allow all actions and reusable workflows"
4. Save

### Step 3: Add Secrets (Optional)
For Hugging Face and W&B integration:
1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add:
   - `HF_TOKEN` - Hugging Face API token
   - `WANDB_API_KEY` - Weights & Biases API key

### Step 4: Trigger First Run
1. Go to **Actions** tab
2. Select "Hourly Model Training (24x per day)"
3. Click **"Run workflow"**
4. Watch it train! 🎉

The workflow will then run automatically every hour.

---

## 💻 Usage Examples

### Load Pre-trained Model

```python
from meridianalgo.unified_ml import UnifiedStockML
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/stock_AAPL.pt"
)

# Predict
ml = UnifiedStockML(model_path=model_path)
prediction = ml.predict('AAPL', days=5)
print(f"5-Day Forecast: ${prediction['predictions'][4]['price']:.2f}")
```

### Train Custom Model

```python
from scripts.train_unified_model import train_unified_stock_model

# Train unified stock model (works for ALL stocks)
success = train_unified_stock_model(
    db_file="training.db",
    output_path="models/unified_stock_model.pt",
    epochs=50
)
```

---

## 📈 Current Results

```
╭──────────────────────┬──────────╮
│ Total Models Trained │ 3        │
│ Trainings (24h)      │ 3        │
│ Avg Accuracy         │ 99.9786  │
│ Avg Loss             │ 0.000214 │
╰──────────────────────┴──────────╯

Latest Training:
Symbol  Accuracy  Loss      Time
MSFT    99.9877   0.000123  13.22s
GOOGL   99.9641   0.000359  13.76s
AAPL    99.9839   0.000161  15.40s
```

---

## 🐛 Troubleshooting

### Unicode Error (Windows)
```bash
$env:PYTHONIOENCODING="utf-8"
```

### Insufficient Data
```bash
# Fetch more data
python scripts/fetch_training_data.py --period 2y --interval 1d
```

### Training Too Slow
- Reduce epochs: `--epochs 30`
- Use shorter period: `--period 1y`
- Enable GPU if available

### Database Locked
```bash
# Recreate database
rm training.db
python scripts/store_training_data.py --data-dir datasets/training_data --db-file training.db
```

---

## 📚 Next Steps

1. ✅ Test local training
2. ✅ View dashboard
3. ⬜ Enable GitHub Actions
4. ⬜ Configure secrets (optional)
5. ⬜ Monitor first automated run
6. ⬜ Adjust schedules as needed

---

## 📖 Documentation

- **[README.md](README.md)** - Main documentation
- **[GITHUB_ACTIONS_FIX.md](GITHUB_ACTIONS_FIX.md)** - Workflow optimization
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guide

---

## 💡 Tips

- Hourly training provides maximum learning frequency
- Monitor GitHub Actions usage (or use public repo for unlimited)
- Use unified models for scalability
- Check dashboard regularly
- Enable W&B for experiment tracking

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/AraAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/AraAI/discussions)
- **Models**: [Hugging Face Hub](https://huggingface.co/MeridianAlgo/ARA.AI)

---

**Ready to train?** Run: `python scripts/quick_train.py --symbol AAPL`

🚀 Happy Training!
