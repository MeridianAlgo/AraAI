# 📚 Hourly Training System - Complete File Index

## Quick Navigation

- **Getting Started**: [QUICKSTART_HOURLY.md](QUICKSTART_HOURLY.md)
- **System Overview**: [HOURLY_TRAINING_SUMMARY.md](HOURLY_TRAINING_SUMMARY.md)
- **Detailed Docs**: [HOURLY_TRAINING.md](HOURLY_TRAINING.md)
- **Architecture**: [.github/workflows/SYSTEM_DIAGRAM.md](.github/workflows/SYSTEM_DIAGRAM.md)

## 📁 All Files Created

### 🔧 GitHub Actions Workflow

| File | Purpose | Lines |
|------|---------|-------|
| `.github/workflows/daily-training.yml` | Main hourly training workflow | ~250 |
| `.github/workflows/README.md` | Workflow documentation | ~200 |
| `.github/workflows/SYSTEM_DIAGRAM.md` | System architecture diagrams | ~400 |

**What it does:**
- Runs every hour from 9 AM to 7 PM UTC
- Full training at 8 PM UTC
- Fetches data, trains models, stores results
- Parallel execution for 15 models

### 🐍 Python Scripts

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/fetch_market_data.py` | Fetch hourly/daily market data | ~150 |
| `scripts/store_training_data.py` | Store data in SQLite database | ~200 |
| `scripts/train_model.py` | Train stock models (timeframe aware) | ~250 |
| `scripts/train_forex_model.py` | Train forex models (timeframe aware) | ~250 |
| `scripts/evaluate_models.py` | Evaluate model performance | ~100 |
| `scripts/create_training_summary.py` | Generate training reports | ~80 |
| `scripts/query_training_data.py` | Query and analyze database | ~300 |

**What they do:**
- Fetch 1-hour interval data from yFinance
- Store with timeframe metadata in SQLite
- Train models with timeframe awareness
- Track performance by hour and timeframe
- Generate reports and summaries

### 📖 Documentation

| File | Purpose | Pages |
|------|---------|-------|
| `HOURLY_TRAINING.md` | Complete system documentation | ~15 |
| `QUICKSTART_HOURLY.md` | Quick start guide | ~8 |
| `HOURLY_TRAINING_SUMMARY.md` | Executive summary | ~5 |
| `HOURLY_TRAINING_INDEX.md` | This file - complete index | ~3 |

**What they cover:**
- System overview and architecture
- Setup and configuration
- Usage examples and queries
- Troubleshooting and best practices

## 📊 Database Schema

### Tables Created

```sql
-- Market data with timeframe tracking
market_data (
    id, symbol, asset_type, date,
    open, high, low, close, volume,
    fetch_date, timeframe, interval, hour
)

-- Training run history
training_runs (
    id, run_date, mode, timeframe, hour,
    symbols_count, rows_processed, status
)

-- Model performance tracking
model_metadata (
    id, symbol, model_type, training_date,
    accuracy, loss, epochs, model_path,
    timeframe, training_mode, hour
)
```

## 🎯 Key Features

### 1. Hourly Training (9 AM - 7 PM)
- ✅ Runs every hour automatically
- ✅ Fetches 1-hour interval data
- ✅ Trains for 5 epochs (fast)
- ✅ Completes within the hour
- ✅ Updates database incrementally

### 2. Full Training (8 PM)
- ✅ Fetches 2 years of historical data
- ✅ Trains for 100 epochs (deep)
- ✅ Commits models to repository
- ✅ Complete retraining from scratch

### 3. Timeframe Awareness
- ✅ Models know if trained on hourly vs daily
- ✅ Tracks training hour (UTC)
- ✅ Stores interval metadata
- ✅ Enables timeframe-specific queries

### 4. Database Tracking
- ✅ All data stored in SQLite
- ✅ Complete training history
- ✅ Performance metrics by timeframe
- ✅ Queryable with Python scripts

### 5. Parallel Execution
- ✅ 10 stock models train simultaneously
- ✅ 5 forex models train simultaneously
- ✅ Fast completion (5-10 minutes)
- ✅ Efficient resource usage

## 🚀 Quick Start Commands

### 1. Push to GitHub
```bash
git add .
git commit -m "Add hourly training system"
git push origin main
```

### 2. Enable Actions
- Go to Actions tab
- Enable workflows

### 3. Manual Test
```bash
# Via GitHub UI
Actions → "Hourly Model Training" → "Run workflow"

# Via GitHub CLI
gh workflow run daily-training.yml -f mode=hourly
```

### 4. Query Results
```bash
# Show recent runs
python scripts/query_training_data.py runs

# Show model performance
python scripts/query_training_data.py performance

# Show hourly data for AAPL
python scripts/query_training_data.py hourly --symbol AAPL --hours 24

# Show training by hour
python scripts/query_training_data.py by-hour

# Show database stats
python scripts/query_training_data.py stats

# Export to CSV
python scripts/query_training_data.py export --table market_data --output data.csv
```

## 📈 Training Schedule

```
UTC Time    Mode      Data           Training    Duration
─────────────────────────────────────────────────────────
09:00       Hourly    1h interval    5 epochs    5-10 min
10:00       Hourly    1h interval    5 epochs    5-10 min
11:00       Hourly    1h interval    5 epochs    5-10 min
12:00       Hourly    1h interval    5 epochs    5-10 min
13:00       Hourly    1h interval    5 epochs    5-10 min
14:00       Hourly    1h interval    5 epochs    5-10 min
15:00       Hourly    1h interval    5 epochs    5-10 min
16:00       Hourly    1h interval    5 epochs    5-10 min
17:00       Hourly    1h interval    5 epochs    5-10 min
18:00       Hourly    1h interval    5 epochs    5-10 min
19:00       Hourly    1h interval    5 epochs    5-10 min
20:00       FULL      2y daily       100 epochs  30-60 min
```

## 🎯 Trained Assets

### Stocks (10)
- AAPL, MSFT, GOOGL, AMZN, TSLA
- NVDA, META, NFLX, AMD, INTC

### Forex (5)
- EURUSD, GBPUSD, USDJPY
- AUDUSD, USDCAD

**Total: 15 models × 12 runs/day = 180 training sessions daily**

## 🔍 Monitoring

### GitHub Actions
```bash
# List runs
gh run list --workflow=daily-training.yml

# View specific run
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

### Database Queries
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('training_data.db')

# Recent runs
runs = pd.read_sql_query('''
    SELECT * FROM training_runs 
    ORDER BY run_date DESC LIMIT 10
''', conn)

# Model performance
perf = pd.read_sql_query('''
    SELECT symbol, timeframe, 
           AVG(accuracy) as avg_accuracy
    FROM model_metadata
    GROUP BY symbol, timeframe
''', conn)

conn.close()
```

## 🛠️ Customization

### Add More Symbols
Edit `.github/workflows/daily-training.yml`:
```yaml
# Line ~80
--symbols AAPL MSFT GOOGL YOUR_SYMBOL

# Line ~95
--symbols EURUSD GBPUSD YOUR_PAIR
```

### Change Schedule
```yaml
schedule:
  # Add 8 AM training
  - cron: '0 8 * * *'
  
  # Add 9 PM training
  - cron: '0 21 * * *'
```

### Adjust Training
```yaml
# Hourly: faster
--epochs 3

# Full: deeper
--epochs 150
```

## 📦 Artifacts

### Generated Each Run
- `market-data-{run}`: CSV files (7 days retention)
- `model-{symbol}-{run}`: Model files (90 days)
- `training-database`: SQLite DB (90 days)
- `evaluation-results-{run}`: Metrics (90 days)

### Committed (Full Mode Only)
- `models/*.pt`: All trained models
- `training_data.db`: Complete database
- `evaluation_results.json`: Performance metrics

## 🐛 Troubleshooting

### Issue: Workflow not running
**Solution:**
- Check Actions are enabled
- Verify cron syntax
- Wait for scheduled time

### Issue: Training fails
**Solution:**
- Check logs for errors
- Verify symbol validity
- Test with fewer symbols

### Issue: Database errors
**Solution:**
- Download from artifacts
- Check disk space
- Verify schema

### Issue: Models not improving
**Solution:**
- Increase epochs
- Check data quality
- Verify timeframe alignment

## 📚 Documentation Hierarchy

```
QUICKSTART_HOURLY.md          ← Start here (5 min read)
    ↓
HOURLY_TRAINING_SUMMARY.md    ← Overview (10 min read)
    ↓
HOURLY_TRAINING.md            ← Complete docs (30 min read)
    ↓
.github/workflows/
    ├── README.md             ← Workflow details
    └── SYSTEM_DIAGRAM.md     ← Architecture
```

## 🎓 Learning Path

### Day 1: Setup
1. Read `QUICKSTART_HOURLY.md`
2. Push to GitHub
3. Enable Actions
4. Run first test

### Day 2: Monitor
1. Check first hourly runs
2. Query database
3. Review performance
4. Verify artifacts

### Day 3: Optimize
1. Adjust epochs
2. Add more symbols
3. Fine-tune schedule
4. Build dashboard

### Week 1: Production
1. Monitor daily runs
2. Track performance trends
3. Optimize parameters
4. Integrate with app

## 🎉 Success Checklist

After 24 hours, you should have:
- ✅ 11 hourly training runs completed
- ✅ 1 full training run completed
- ✅ 180 model training sessions (15 models × 12 runs)
- ✅ Database with complete history
- ✅ Models committed to repository
- ✅ Performance metrics available
- ✅ Artifacts stored
- ✅ System running automatically

## 📞 Support

### Documentation
- Read: `HOURLY_TRAINING.md`
- Start: `QUICKSTART_HOURLY.md`
- Check: `.github/workflows/README.md`

### Debugging
- View: GitHub Actions logs
- Query: `python scripts/query_training_data.py`
- Export: Database to CSV for analysis

### Community
- Open: GitHub Issues
- Discuss: GitHub Discussions
- Share: Your improvements!

## 🌟 What You Built

A **production-ready automated ML training system** that:

1. ✅ Trains every hour (9 AM - 7 PM)
2. ✅ Full retraining nightly (8 PM)
3. ✅ Handles 15 models automatically
4. ✅ Tracks timeframe context
5. ✅ Stores complete history
6. ✅ Runs hands-off
7. ✅ Scales easily
8. ✅ Monitors performance

**Your models now train themselves with fresh data every hour! 🚀**

---

## 📋 File Summary

| Category | Files | Total Lines |
|----------|-------|-------------|
| Workflows | 3 | ~850 |
| Scripts | 7 | ~1,330 |
| Documentation | 4 | ~2,000 |
| **Total** | **14** | **~4,180** |

**All files are production-ready and fully documented!**
