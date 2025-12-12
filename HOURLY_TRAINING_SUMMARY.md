# ⏰ Hourly Training System - Summary

## What You Got

A complete **automated hourly training system** that runs on GitHub Actions:

### 📅 Training Schedule
- **9 AM - 7 PM UTC**: Hourly training (11 runs/day)
- **8 PM UTC**: Full historical training (1 run/day)
- **Total**: 12 automated training runs every day

### 🎯 What Gets Trained
- **10 Stock Models**: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, AMD, INTC
- **5 Forex Models**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD
- **Total**: 15 models × 12 runs = 180 training sessions per day

### 🔄 How It Works

#### Hourly Training (9 AM - 7 PM)
```
9 AM  → Pull 1h data → Train 5 epochs → Done by 10 AM
10 AM → Pull 1h data → Train 5 epochs → Done by 11 AM
11 AM → Pull 1h data → Train 5 epochs → Done by 12 PM
... continues until 7 PM
```

#### Full Training (8 PM)
```
8 PM → Pull 2 years data → Train 100 epochs → Commit models
```

### 🗄️ Database
- **SQLite database** (`training_data.db`)
- **Tracks**: All data, training runs, model performance
- **Timeframe aware**: Models know if they're trained on hourly vs daily data
- **Persistent**: Stored as GitHub artifact

### 📊 Timeframe Awareness
Models know their training context:
```python
{
  "symbol": "AAPL",
  "timeframe": "1h",        # or "2y" for full
  "training_mode": "hourly", # or "full"
  "hour": 14,               # UTC hour
  "interval": "1h",         # data interval
  "accuracy": 0.95
}
```

## Files Created

### Workflow
- `.github/workflows/daily-training.yml` - Main hourly training workflow
- `.github/workflows/README.md` - Workflow documentation

### Scripts
- `scripts/fetch_market_data.py` - Fetch hourly/daily data
- `scripts/store_training_data.py` - Store in database with timeframe
- `scripts/train_model.py` - Train stock models (timeframe aware)
- `scripts/train_forex_model.py` - Train forex models (timeframe aware)
- `scripts/evaluate_models.py` - Evaluate model performance
- `scripts/create_training_summary.py` - Generate reports
- `scripts/query_training_data.py` - Query and analyze database

### Documentation
- `HOURLY_TRAINING.md` - Complete system documentation
- `QUICKSTART_HOURLY.md` - Quick start guide
- `HOURLY_TRAINING_SUMMARY.md` - This file

## Quick Start

### 1. Push to GitHub
```bash
git add .
git commit -m "Add hourly training system"
git push origin main
```

### 2. Enable Actions
- Go to **Actions** tab
- Enable workflows

### 3. Test Run
- Click **"Run workflow"**
- Select `hourly` mode
- Watch it train!

### 4. Query Results
```bash
# Show recent runs
python scripts/query_training_data.py runs

# Show model performance
python scripts/query_training_data.py performance

# Show hourly data
python scripts/query_training_data.py hourly --symbol AAPL

# Show stats
python scripts/query_training_data.py stats
```

## Key Features

### ✅ Automatic
- Runs every hour without intervention
- No manual triggering needed
- Completely hands-off

### ✅ Fast
- Hourly training: 5-10 minutes
- Completes within the hour
- Doesn't block other work

### ✅ Smart
- Models know their timeframe
- Tracks training context
- Learns intraday patterns

### ✅ Persistent
- Database stored as artifact
- Models committed to repo
- Full training history

### ✅ Scalable
- Easy to add more symbols
- Adjust training schedule
- Modify parameters

## Benefits

1. **Always Fresh**: Models updated every hour with latest data
2. **Intraday Patterns**: Captures hourly market movements
3. **Fast Training**: 5 epochs keeps models current
4. **Deep Learning**: Nightly full training maintains accuracy
5. **Timeframe Context**: Models know their training scope
6. **Continuous Improvement**: 12 training runs daily

## Monitoring

### GitHub Actions
- View workflow runs in Actions tab
- Check logs for each job
- Download artifacts

### Query Database
```bash
# Recent training runs
python scripts/query_training_data.py runs

# Model performance by timeframe
python scripts/query_training_data.py performance

# Training stats by hour
python scripts/query_training_data.py by-hour

# Database statistics
python scripts/query_training_data.py stats
```

### Artifacts
- `market-data-*`: Fetched CSV data
- `model-*`: Trained model files
- `training-database`: SQLite database
- `evaluation-results-*`: Performance metrics

## Customization

### Add Symbols
Edit `.github/workflows/daily-training.yml`:
```yaml
--symbols AAPL MSFT YOUR_SYMBOL
```

### Change Schedule
```yaml
schedule:
  - cron: '0 8 * * *'   # Add 8 AM
  - cron: '0 21 * * *'  # Add 9 PM
```

### Adjust Training
```yaml
--epochs 10           # More training
--incremental         # Incremental learning
```

## What's Next?

### 1. Monitor First Day
- Watch first few hourly runs
- Check database growth
- Verify model performance

### 2. Integrate Models
```python
from meridianalgo.unified_ml import UnifiedStockML

ml = UnifiedStockML(model_path='models/AAPL_model.pt')
result = ml.predict('AAPL', days=5)
```

### 3. Build Dashboard
- Query database for metrics
- Visualize training progress
- Track model performance

### 4. Optimize
- Adjust epochs based on performance
- Add more symbols gradually
- Fine-tune schedule

## Troubleshooting

### Workflow Not Running
- Check Actions are enabled
- Verify cron syntax
- Wait for scheduled time

### Training Fails
- Check logs for errors
- Verify symbol validity
- Test with fewer symbols

### Database Issues
- Download from artifacts
- Check disk space
- Verify schema

## Success Metrics

After 24 hours, you should see:
- ✅ 11 hourly training runs completed
- ✅ 1 full training run completed
- ✅ 15 models trained 12 times each
- ✅ Database with all training history
- ✅ Models committed to repository
- ✅ Performance metrics available

## Summary

You now have a **production-ready hourly training system** that:

- 🤖 Trains automatically every hour
- 📊 Pulls fresh market data
- 🗄️ Stores everything in a database
- 🎯 Models know their timeframe
- 📈 Tracks performance over time
- 🚀 Runs completely hands-off

**Your models train themselves with fresh data every hour, all day long! 🎉**

---

**Need Help?**
- Read: `HOURLY_TRAINING.md` (detailed docs)
- Start: `QUICKSTART_HOURLY.md` (quick guide)
- Check: `.github/workflows/README.md` (workflow docs)
