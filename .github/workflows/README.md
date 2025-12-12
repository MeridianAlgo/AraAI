# GitHub Actions Workflows

## Daily Training Workflow

The `daily-training.yml` workflow automatically trains your models every day with fresh market data.

### Schedule

- **Night Training (2 AM UTC)**: Full training with all historical data
  - Pulls complete historical data (2 years)
  - Trains models from scratch with 100 epochs
  - Stores all data in database
  - Commits trained models to repository

- **Morning Refresh (8 AM UTC)**: Quick update with fresh data
  - Pulls only last 7 days of data
  - Incremental training with 10 epochs
  - Updates database with new data
  - Fast refresh for morning predictions

### How It Works

1. **Determine Mode**: Automatically selects full or refresh mode based on time
2. **Fetch Data**: Downloads stock and forex data from yFinance
3. **Store Data**: Saves data to SQLite database
4. **Train Models**: Trains models in parallel for multiple symbols
5. **Evaluate**: Generates performance metrics
6. **Deploy**: Commits trained models (full mode only)

### Supported Assets

**Stocks:**
- AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX, AMD, INTC

**Forex Pairs:**
- EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD

### Manual Trigger

You can manually trigger the workflow:

1. Go to Actions tab in GitHub
2. Select "Daily Model Training"
3. Click "Run workflow"
4. Choose mode: `full` or `refresh`

### Database Structure

The workflow maintains a SQLite database with:

- **market_data**: Historical price data
- **training_runs**: Training execution logs
- **model_metadata**: Model performance metrics

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

- Full training takes ~30-60 minutes depending on data size
- Refresh mode completes in ~10-15 minutes
- Models are automatically versioned by date
- Database grows over time - consider periodic cleanup
