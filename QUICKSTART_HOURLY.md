# 🚀 Quick Start: Hourly Training System

Get your hourly training system up and running in 5 minutes!

## Step 1: Push to GitHub

```bash
# Add all files
git add .

# Commit
git commit -m "Add hourly training system"

# Push to GitHub
git push origin main
```

## Step 2: Enable GitHub Actions

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. If prompted, click **"I understand my workflows, go ahead and enable them"**

## Step 3: Verify Workflow

1. In the Actions tab, you should see **"Hourly Model Training"**
2. The workflow will automatically run at:
   - Every hour from 9 AM to 7 PM UTC (hourly training)
   - 8 PM UTC (full historical training)

## Step 4: Manual Test Run (Optional)

Test the system immediately:

1. Go to **Actions** → **"Hourly Model Training"**
2. Click **"Run workflow"** (top right)
3. Select mode: `hourly` or `full`
4. Click **"Run workflow"** button
5. Watch it run! (takes 5-10 minutes for hourly, 30-60 for full)

## Step 5: Monitor Progress

### View Running Workflow

1. Click on the running workflow
2. See real-time logs
3. Check each job's progress

### Check Artifacts

After completion:
1. Scroll down to **Artifacts** section
2. Download:
   - `market-data-*`: CSV files with fetched data
   - `model-*`: Trained model files
   - `training-database`: SQLite database
   - `evaluation-results-*`: Performance metrics

## Step 6: Query Training Data

Once training completes, query the database:

```bash
# Show recent training runs
python scripts/query_training_data.py runs

# Show model performance
python scripts/query_training_data.py performance

# Show hourly data for AAPL
python scripts/query_training_data.py hourly --symbol AAPL

# Show training stats by hour
python scripts/query_training_data.py by-hour

# Show overall database stats
python scripts/query_training_data.py stats

# Export data to CSV
python scripts/query_training_data.py export --table market_data --output data.csv
```

## What Happens Next?

### Hourly (9 AM - 7 PM UTC)

Every hour, the system will:
1. ✅ Fetch latest 1-hour interval data
2. ✅ Store in database with timeframe metadata
3. ✅ Train all models (5 epochs each)
4. ✅ Update model metadata
5. ✅ Complete within the hour

### Full Training (8 PM UTC)

At 8 PM, the system will:
1. ✅ Fetch 2 years of historical data
2. ✅ Store in database
3. ✅ Train all models from scratch (100 epochs)
4. ✅ Commit models to repository
5. ✅ Update database

## Customization

### Add More Symbols

Edit `.github/workflows/daily-training.yml`:

```yaml
# Line ~80 - Stock symbols
--symbols AAPL MSFT GOOGL AMZN TSLA YOUR_SYMBOL

# Line ~95 - Forex pairs
--symbols EURUSD GBPUSD USDJPY YOUR_PAIR
```

### Change Training Hours

Edit the cron schedule:

```yaml
schedule:
  # Add 8 AM training
  - cron: '0 8 * * *'
  
  # Add 9 PM training
  - cron: '0 21 * * *'
```

### Adjust Training Parameters

```yaml
# Hourly training (faster)
--epochs 3          # Reduce from 5

# Full training (deeper)
--epochs 150        # Increase from 100
```

## Monitoring Tips

### 1. Check Workflow Status

```bash
# Install GitHub CLI
gh auth login

# List recent runs
gh run list --workflow=daily-training.yml

# View specific run
gh run view <run-id> --log

# Download artifacts
gh run download <run-id>
```

### 2. Set Up Notifications

1. Go to repository **Settings**
2. Click **Notifications**
3. Enable **Actions** notifications
4. Get email/Slack alerts on failures

### 3. Review Logs

- Click on any workflow run
- Expand job steps to see detailed logs
- Check for errors or warnings

### 4. Monitor Database Size

```bash
# Check database size
ls -lh training_data.db

# Query row counts
python scripts/query_training_data.py stats
```

## Troubleshooting

### Workflow Not Running

**Problem**: Workflow doesn't trigger automatically

**Solution**:
1. Check Actions are enabled (Settings → Actions)
2. Verify workflow file is in `.github/workflows/`
3. Check cron syntax is correct
4. Wait for next scheduled time

### Training Fails

**Problem**: Training job fails with errors

**Solution**:
1. Check logs for specific error
2. Verify symbols are valid
3. Check yFinance is accessible
4. Try manual run with fewer symbols

### Database Issues

**Problem**: Database errors or corruption

**Solution**:
1. Download fresh database from artifacts
2. Delete local database and let it recreate
3. Check disk space in GitHub Actions

### Models Not Improving

**Problem**: Accuracy not increasing

**Solution**:
1. Increase epochs for hourly training
2. Check data quality in database
3. Verify timeframe alignment
4. Review feature engineering

## Next Steps

### 1. Integrate with Your App

```python
from meridianalgo.unified_ml import UnifiedStockML

# Load trained model
ml = UnifiedStockML(model_path='models/AAPL_model.pt')

# Make prediction
result = ml.predict('AAPL', days=5)
print(result)
```

### 2. Build API Endpoint

```python
from fastapi import FastAPI
import sqlite3

app = FastAPI()

@app.get("/model-status/{symbol}")
def get_model_status(symbol: str):
    conn = sqlite3.connect('training_data.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT training_date, accuracy, loss, timeframe, training_mode
        FROM model_metadata
        WHERE symbol = ?
        ORDER BY training_date DESC
        LIMIT 1
    ''', (symbol,))
    
    result = cursor.fetchone()
    conn.close()
    
    return {
        "symbol": symbol,
        "last_trained": result[0],
        "accuracy": result[1],
        "loss": result[2],
        "timeframe": result[3],
        "mode": result[4]
    }
```

### 3. Create Dashboard

Use the query script to build a monitoring dashboard:

```python
import streamlit as st
import pandas as pd
import sqlite3

st.title("Model Training Dashboard")

# Connect to database
conn = sqlite3.connect('training_data.db')

# Show recent runs
st.header("Recent Training Runs")
runs = pd.read_sql_query('''
    SELECT * FROM training_runs 
    ORDER BY run_date DESC 
    LIMIT 20
''', conn)
st.dataframe(runs)

# Show model performance
st.header("Model Performance")
perf = pd.read_sql_query('''
    SELECT symbol, timeframe, AVG(accuracy) as avg_accuracy
    FROM model_metadata
    GROUP BY symbol, timeframe
''', conn)
st.bar_chart(perf.set_index('symbol'))

conn.close()
```

## Success Checklist

- ✅ Workflow file pushed to GitHub
- ✅ GitHub Actions enabled
- ✅ First manual run completed successfully
- ✅ Database created and populated
- ✅ Models trained and saved
- ✅ Can query training data
- ✅ Hourly schedule running automatically
- ✅ Full training at 8 PM working
- ✅ Artifacts being saved
- ✅ Models aware of timeframe

## Support

If you encounter issues:

1. Check the [workflow README](.github/workflows/README.md)
2. Review [HOURLY_TRAINING.md](HOURLY_TRAINING.md)
3. Check GitHub Actions logs
4. Open an issue with error details

## Summary

You now have:
- ✅ **11 hourly training runs** per day (9 AM - 7 PM)
- ✅ **1 full training run** per day (8 PM)
- ✅ **15 models** (10 stocks + 5 forex) trained automatically
- ✅ **Timeframe-aware models** that know their training context
- ✅ **Complete database** with all training history
- ✅ **Automatic deployment** of trained models

**Your models are now training themselves every hour with fresh market data! 🎉**
