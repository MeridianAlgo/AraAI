# ARA AI Training System - Quick Start

## 🚀 Training Performance

**Training Time**: ~15 seconds per model (50 epochs, 2 years data)
**Accuracy**: >99.9%
**Models per Day**: 48 (multi-daily) or 192 (hourly)

## ⚡ Quick Commands

```bash
# Windows: Set UTF-8 encoding first
$env:PYTHONIOENCODING="utf-8"

# Test single model training
python scripts/quick_train.py --symbol AAPL --epochs 50

# Train multiple models
python scripts/batch_train.py --symbols AAPL GOOGL MSFT --epochs 50

# Train random stocks
python scripts/batch_train.py --random 5 --epochs 50

# View training dashboard
python scripts/training_dashboard.py

# Run full training session
python scripts/continuous_training.py
```

## 📊 Training Schedules

### Option 1: Multi-Daily (Recommended for Free Tier)
- **File**: `.github/workflows/multi-daily-training.yml`
- **Frequency**: 6 times per day
- **Schedule**: Every ~4 hours at market-strategic times
- **Models/Day**: 48 (5 stocks + 3 forex per session)
- **GitHub Actions Usage**: ~900 min/month (within free 2,000 limit)

### Option 2: Hourly (Maximum Learning)
- **File**: `.github/workflows/hourly-training.yml`
- **Frequency**: 24 times per day
- **Schedule**: Every hour
- **Models/Day**: 192 (5 stocks + 3 forex per session)
- **GitHub Actions Usage**: ~3,600 min/month (requires paid plan)

## 🎯 Current Training Results

```
╭──────────────────────┬──────────╮
│ Total Models Trained │ 3        │
│ Trainings (24h)      │ 3        │
│ Unique Symbols       │ 3        │
│ Avg Accuracy         │ 99.9786  │
│ Avg Loss             │ 0.000214 │
╰──────────────────────┴──────────╯

Latest Training Sessions:
Symbol  Type        Date                 Accuracy  Loss      Epochs
MSFT    unified_ml  2026-01-11 09:45:25  99.9877   0.000123  50
GOOGL   unified_ml  2026-01-11 09:45:12  99.9641   0.000359  50
AAPL    unified_ml  2026-01-11 09:43:07  99.9839   0.000161  50
```

## 📁 Project Structure

```
AraAI/
├── .github/workflows/
│   ├── multi-daily-training.yml  # 6x per day schedule
│   └── hourly-training.yml       # 24x per day schedule
├── scripts/
│   ├── quick_train.py           # Test single model
│   ├── batch_train.py           # Train multiple models
│   ├── training_dashboard.py    # View training stats
│   ├── continuous_training.py   # Full training session
│   ├── train_model.py          # Stock model training
│   └── train_forex_model.py    # Forex model training
├── models/                      # Trained models
├── datasets/                    # Training data
├── training.db                  # Training database
├── TRAINING_GUIDE.md           # Detailed guide
├── TRAINING_RESULTS.md         # Performance results
└── QUICK_START.md              # This file
```

## 🔧 Setup (First Time)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run setup script (optional)
python setup_training.py

# 3. Test training
python scripts/quick_train.py --symbol AAPL
```

## 🤖 Enable Automated Training

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Add automated training system"
git push
```

### Step 2: Enable GitHub Actions
1. Go to repository Settings
2. Navigate to Actions → General
3. Enable "Allow all actions and reusable workflows"
4. Save

### Step 3: Add Secrets (Optional)
For Hugging Face and W&B integration:
1. Go to Settings → Secrets and variables → Actions
2. Add secrets:
   - `HF_TOKEN` - Hugging Face API token
   - `WANDB_API_KEY` - Weights & Biases API key

### Step 4: Trigger First Run
1. Go to Actions tab
2. Select "Multi-Daily Model Training"
3. Click "Run workflow"
4. Watch the training progress!

## 📈 Monitoring

### Local Dashboard
```bash
python scripts/training_dashboard.py
```

### GitHub Actions
- View logs in Actions tab
- Download training artifacts
- Monitor success/failure rates

### Database Queries
```bash
sqlite3 training.db "SELECT * FROM model_metadata ORDER BY training_date DESC LIMIT 10"
```

## 🎛️ Configuration

Edit `scripts/continuous_training.py`:

```python
# Number of stocks per session
STOCK_COUNT = 5

# Forex pairs to train
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]

# Training epochs
EPOCHS = 50
```

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
- Enable GPU acceleration

### Database Locked
```bash
# Close connections and recreate
rm training.db
python scripts/store_training_data.py --data-dir datasets/training_data --db-file training.db
```

## 📚 Documentation

- **TRAINING_GUIDE.md** - Complete training guide
- **TRAINING_RESULTS.md** - Performance benchmarks
- **CONTRIBUTING.md** - Contribution guidelines
- **README.md** - Main project README

## 🎉 Success Indicators

✓ Training completes in ~15 seconds per model
✓ Accuracy >99.9%
✓ Loss <0.0004
✓ Models saved to `models/` directory
✓ Dashboard shows training history
✓ GitHub Actions runs successfully

## 🚀 Next Steps

1. ✅ Test local training
2. ✅ View dashboard
3. ⬜ Enable GitHub Actions
4. ⬜ Configure secrets (optional)
5. ⬜ Monitor first automated run
6. ⬜ Adjust schedules as needed

## 💡 Tips

- Start with multi-daily schedule (free tier friendly)
- Monitor GitHub Actions usage
- Use batch training for testing
- Check dashboard regularly
- Adjust epochs based on performance
- Enable W&B for experiment tracking

## 🆘 Support

If you encounter issues:
1. Check TRAINING_GUIDE.md
2. Review GitHub Actions logs
3. Run training dashboard
4. Check database: `sqlite3 training.db`
5. Review model files in `models/`

---

**Ready to train?** Run: `python scripts/quick_train.py --symbol AAPL`
