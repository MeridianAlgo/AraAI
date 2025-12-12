# GitHub Actions Workflows

## Hourly Training Workflow

The `daily-training.yml` workflow automatically trains your models **every hour** with fresh market data throughout the trading day.

### Schedule

#### 🕐 Hourly Training (9 AM - 7 PM UTC)
Runs every hour from 9 AM to 7 PM UTC:
- **9 AM**: Pull hourly data → Train until 10 AM
- **10 AM**: Pull new hour data → Train until 11 AM
- **11 AM**: Pull new hour data → Train until 12 PM
- **12 PM - 7 PM**: Continue hourly cycle

**Each hourly run:**
- Pulls last 2 hours of 1-hour interval data
- Incremental training with 5 epochs (fast)
- Updates database with fresh hourly data
- Models learn intraday patterns
- Training completes within the hour

#### 🌙 Full Historical Training (8 PM UTC)
Complete retraining with all historical data:
- Pulls 2 years of historical data
- Trains models from scratch with 100 epochs
- Stores all data in database
- Commits trained models to repository
- Deep learning on full dataset

### How It Works

1. **Determine Mode**: Automatically selects hourly or full mode based on UTC time
   - 9 AM - 7 PM: Hourly mode (1h interval data)
   - 8 PM: Full mode (2y historical data)
   - Other times: Skip (no training)

2. **Fetch Data**: Downloads stock and forex data from yFinance
   - Hourly: 1-hour interval data (last 2 hours)
   - Full: Daily data (2 years)
   - Includes timeframe metadata

3. **Store Data**: Saves data to SQLite database
   - Tracks timeframe and interval
   - Records hour of training
   - Maintains training history

4. **Train Models**: Trains models in parallel for multiple symbols
   - Models are aware of their training timeframe
   - Hourly: 5 epochs (fast incremental)
   - Full: 100 epochs (complete retraining)

5. **Evaluate**: Generates performance metrics
   - Tracks accuracy by timeframe
   - Monitors hourly vs full performance

6. **Deploy**: Commits trained models and database
   - Full mode: Commits everything
   - Hourly mode: Updates artifacts only

### Supported Assets

**Stocks:**
- AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, AMD, INTC

**Forex Pairs:**
- EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD

### Manual Trigger

You can manually trigger the workflow:

1. Go to Actions tab in GitHub
2. Select "Hourly Model Training"
3. Click "Run workflow"
4. Choose mode:
   - `hourly`: Quick 1-hour data training
   - `full`: Complete historical retraining

### Database Structure

The workflow maintains a SQLite database (`training_data.db`) with:

- **market_data**: Historical price data
  - Columns: symbol, asset_type, date, OHLCV, timeframe, interval, hour
  - Indexes: symbol+date, timeframe, hour
  - Supports both daily and hourly data

- **training_runs**: Training execution logs
  - Tracks: mode, timeframe, hour, symbols_count, rows_processed
  - Records every training run

- **model_metadata**: Model performance metrics
  - Tracks: accuracy, loss, epochs, timeframe, training_mode, hour
  - Links models to their training context

### Artifacts

The workflow stores:

- Market data CSV files (30 days retention)
- Trained model files (90 days retention)
- Evaluation results (90 days retention)

### Configuration

Edit the workflow file to customize:

- Training schedule (cron expressions)
- Symbols to train
- Training epochs
- Data retention periods

### Requirements

No additional setup required! The workflow uses GitHub-hosted runners with all dependencies installed automatically.

### Monitoring

Check the Actions tab to monitor:
- Training progress
- Model performance
- Error logs
- Execution time

### Tips

- **Hourly training**: Completes in ~5-10 minutes per run
- **Full training**: Takes ~30-60 minutes (runs at 8 PM)
- **Timeframe awareness**: Models know if they're trained on hourly vs daily data
- **Database growth**: Hourly data accumulates faster - database includes cleanup
- **Parallel training**: Multiple symbols train simultaneously
- **Artifact retention**: 
  - Hourly data: 7 days
  - Models: 90 days
  - Database: 90 days (persistent)

### Timeframe Awareness

Models are trained with full awareness of their timeframe:

```python
# Model metadata includes:
{
  "symbol": "AAPL",
  "timeframe": "1h",           # or "2y" for full
  "training_mode": "hourly",   # or "full"
  "hour": 14,                  # UTC hour of training
  "interval": "1h",            # data interval
  "accuracy": 0.95,
  "loss": 0.023
}
```

This allows you to:
- Query models by timeframe
- Compare hourly vs daily performance
- Track model evolution throughout the day
- Understand prediction context

### Trading Day Coverage

The system covers the full trading day:
- **Pre-market**: 9-10 AM UTC
- **Market hours**: 10 AM - 4 PM UTC (approx)
- **After-hours**: 4-7 PM UTC
- **Deep learning**: 8 PM UTC (full historical)

### Benefits

1. **Fresh predictions**: Models updated every hour with latest data
2. **Intraday patterns**: Captures hourly market movements
3. **Fast training**: 5 epochs per hour keeps models current
4. **Deep learning**: Nightly full training maintains long-term accuracy
5. **Timeframe context**: Models know their training scope
6. **Continuous improvement**: 11 hourly updates + 1 full update daily
