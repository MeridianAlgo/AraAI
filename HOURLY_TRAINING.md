# ⏰ Hourly Training System

## Overview

Your ARA AI system now trains **automatically every hour** from 9 AM to 8 PM UTC, with a full historical retraining at 8 PM. This ensures your models are always fresh with the latest market data.

## 📅 Daily Training Schedule

```
UTC Time    Mode      Data Pulled           Training        Duration
─────────────────────────────────────────────────────────────────────
09:00 AM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
10:00 AM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
11:00 AM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
12:00 PM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
01:00 PM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
02:00 PM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
03:00 PM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
04:00 PM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
05:00 PM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
06:00 PM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
07:00 PM    Hourly    Last 2h (1h interval) 5 epochs        ~5-10 min
08:00 PM    FULL      2 years (daily)       100 epochs      ~30-60 min
```

## 🔄 How It Works

### Hourly Training (9 AM - 7 PM)

1. **Data Collection**
   - Fetches 1-hour interval data from yFinance
   - Gets last 2 hours to ensure fresh data
   - Stores in SQLite database with timeframe metadata

2. **Model Training**
   - Loads last 30 days of hourly data
   - Trains for 5 epochs (fast incremental)
   - Models are aware they're training on hourly data
   - Completes within the hour

3. **Database Update**
   - Adds new hourly data points
   - Records training run metadata
   - Tracks hour, timeframe, and performance

### Full Training (8 PM)

1. **Historical Data Collection**
   - Fetches 2 years of daily data
   - Comprehensive historical dataset
   - Stores with full timeframe metadata

2. **Deep Model Training**
   - Trains from scratch with 100 epochs
   - Uses all historical data
   - Full model retraining
   - Takes 30-60 minutes

3. **Deployment**
   - Commits trained models to repository
   - Updates database
   - Creates performance reports

## 🗄️ Database Structure

### market_data Table
```sql
CREATE TABLE market_data (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    date TIMESTAMP NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    fetch_date TIMESTAMP,
    timeframe TEXT,        -- '1h' or '2y'
    interval TEXT,         -- '1h' or '1d'
    hour INTEGER,          -- UTC hour (0-23)
    UNIQUE(symbol, date, interval)
);
```

### training_runs Table
```sql
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY,
    run_date TIMESTAMP NOT NULL,
    mode TEXT NOT NULL,           -- 'hourly' or 'full'
    timeframe TEXT,               -- '1h' or '2y'
    hour INTEGER,                 -- UTC hour
    symbols_count INTEGER,
    rows_processed INTEGER,
    status TEXT
);
```

### model_metadata Table
```sql
CREATE TABLE model_metadata (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    model_type TEXT NOT NULL,
    training_date TIMESTAMP NOT NULL,
    accuracy REAL,
    loss REAL,
    epochs INTEGER,
    model_path TEXT,
    timeframe TEXT,               -- Training timeframe
    training_mode TEXT,           -- 'hourly' or 'full'
    hour INTEGER                  -- Training hour
);
```

## 🎯 Trained Assets

### Stocks (10 symbols)
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- AMZN (Amazon)
- TSLA (Tesla)
- NVDA (NVIDIA)
- META (Meta)
- NFLX (Netflix)
- AMD (AMD)
- INTC (Intel)

### Forex Pairs (5 pairs)
- EURUSD (Euro/US Dollar)
- GBPUSD (British Pound/US Dollar)
- USDJPY (US Dollar/Japanese Yen)
- AUDUSD (Australian Dollar/US Dollar)
- USDCAD (US Dollar/Canadian Dollar)

**Total: 15 models trained every hour + full retraining at 8 PM**

## 📊 Timeframe Awareness

Models are fully aware of their training context:

```python
# Example model metadata
{
    "symbol": "AAPL",
    "training_date": "2025-12-11T14:00:00",
    "timeframe": "1h",              # Hourly data
    "training_mode": "hourly",      # Hourly training
    "hour": 14,                     # Trained at 2 PM UTC
    "interval": "1h",               # 1-hour data points
    "epochs": 5,
    "accuracy": 0.94,
    "loss": 0.028
}
```

This allows you to:
- Query models by timeframe
- Compare hourly vs daily performance
- Track intraday model evolution
- Understand prediction context
- Analyze time-of-day patterns

## 🚀 Benefits

### 1. Always Fresh
- Models updated every hour
- Latest market data incorporated
- Intraday pattern recognition

### 2. Fast Training
- 5 epochs per hour (5-10 minutes)
- Doesn't block other operations
- Completes within the hour

### 3. Deep Learning
- Nightly full retraining (100 epochs)
- Long-term pattern recognition
- Comprehensive historical analysis

### 4. Timeframe Context
- Models know their training scope
- Hourly vs daily awareness
- Time-of-day tracking

### 5. Continuous Improvement
- 11 hourly updates per day
- 1 full update per day
- 12 total training runs daily

## 📈 Performance Tracking

The system tracks:
- **Hourly accuracy**: How well models perform on 1h data
- **Full accuracy**: Performance on historical data
- **Time-of-day patterns**: Best/worst training hours
- **Symbol performance**: Which assets train best
- **Training duration**: Optimization opportunities

## 🔧 Configuration

### Adding More Symbols

Edit `.github/workflows/daily-training.yml`:

```yaml
# For stocks
--symbols AAPL MSFT GOOGL YOUR_SYMBOL

# For forex
--symbols EURUSD GBPUSD YOUR_PAIR
```

### Changing Schedule

Edit the cron expressions:

```yaml
schedule:
  # Add more hours
  - cron: '0 8 * * *'   # 8 AM
  - cron: '0 21 * * *'  # 9 PM (additional full training)
```

### Adjusting Training

Modify training parameters:

```yaml
# Hourly training
--epochs 5          # Increase for more training
--incremental       # Use incremental learning

# Full training
--epochs 100        # Increase for deeper learning
--use-all-data      # Use all historical data
```

## 📱 Manual Trigger

Run training manually:

1. Go to **Actions** tab in GitHub
2. Select **"Hourly Model Training"**
3. Click **"Run workflow"**
4. Choose mode:
   - `hourly`: Quick 1-hour training
   - `full`: Complete historical retraining

## 🔍 Monitoring

### Check Training Status

```bash
# View workflow runs
gh run list --workflow=daily-training.yml

# View specific run
gh run view <run-id>

# Download artifacts
gh run download <run-id>
```

### Query Database

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('training_data.db')

# View recent training runs
runs = pd.read_sql_query('''
    SELECT * FROM training_runs 
    ORDER BY run_date DESC 
    LIMIT 10
''', conn)

# View model performance by timeframe
performance = pd.read_sql_query('''
    SELECT 
        symbol,
        timeframe,
        AVG(accuracy) as avg_accuracy,
        AVG(loss) as avg_loss,
        COUNT(*) as training_count
    FROM model_metadata
    GROUP BY symbol, timeframe
''', conn)

# View hourly data
hourly_data = pd.read_sql_query('''
    SELECT * FROM market_data
    WHERE interval = '1h'
    AND date >= datetime('now', '-7 days')
    ORDER BY date DESC
''', conn)

conn.close()
```

## 🎓 Best Practices

1. **Monitor first few runs**: Ensure everything works correctly
2. **Check database size**: Hourly data accumulates quickly
3. **Review accuracy trends**: Track model performance over time
4. **Adjust epochs**: Balance speed vs accuracy
5. **Add symbols gradually**: Start with a few, expand later
6. **Use artifacts**: Download models and data for analysis
7. **Set up alerts**: Get notified of training failures

## 🐛 Troubleshooting

### Training Fails
- Check GitHub Actions logs
- Verify yFinance is accessible
- Ensure database isn't corrupted
- Check symbol validity

### Database Too Large
- Implement data retention policy
- Archive old hourly data
- Keep only recent hourly + all daily

### Models Not Improving
- Increase epochs for hourly training
- Check data quality
- Verify timeframe alignment
- Review feature engineering

## 📚 Files Created

```
.github/workflows/
  └── daily-training.yml          # Main workflow (hourly + full)
  └── README.md                   # Workflow documentation

scripts/
  ├── fetch_market_data.py        # Data fetching (hourly aware)
  ├── store_training_data.py      # Database storage (timeframe tracking)
  ├── train_model.py              # Stock training (timeframe aware)
  ├── train_forex_model.py        # Forex training (timeframe aware)
  ├── evaluate_models.py          # Model evaluation
  └── create_training_summary.py  # Performance reports

HOURLY_TRAINING.md                # This file
```

## 🎉 Summary

Your system now:
- ✅ Trains every hour from 9 AM to 7 PM UTC
- ✅ Pulls fresh hourly data each run
- ✅ Completes training within the hour
- ✅ Does full retraining at 8 PM
- ✅ Stores everything in a database
- ✅ Models know their timeframe
- ✅ Tracks performance by hour
- ✅ Runs completely automatically

**Result**: Always-fresh models that understand both intraday patterns (hourly) and long-term trends (daily), with full awareness of their training context!
