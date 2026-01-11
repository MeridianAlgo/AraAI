# ARA AI Training System Guide

## Overview

The ARA AI training system is designed for continuous, automated model training with multiple schedules running throughout the day.

## Training Performance

Based on testing with AAPL stock (502 data points, 50 epochs):
- **Data Fetch**: ~2-3 seconds
- **Data Storage**: ~1 second
- **Model Training**: ~15 seconds
- **Total Time**: ~18-20 seconds per model

## Training Schedules

### 1. Multi-Daily Training (6x per day)
**File**: `.github/workflows/multi-daily-training.yml`

Runs 6 times per day at strategic market hours:
- 02:00 UTC - Pre-market Asia
- 07:00 UTC - European market open
- 13:00 UTC - US pre-market
- 17:00 UTC - US market mid-day
- 21:00 UTC - US market close
- 23:00 UTC - After-hours trading

**Trains**: 5 random stocks + 3 forex pairs per session = **48 models per day**

### 2. Hourly Training (24x per day)
**File**: `.github/workflows/hourly-training.yml`

Runs every hour for maximum learning frequency.

**Trains**: 5 random stocks + 3 forex pairs per session = **192 models per day**

### 3. Manual Training
Run training manually anytime:

```bash
# Quick single model training
python scripts/quick_train.py --symbol AAPL --epochs 50

# Full continuous training session
python scripts/continuous_training.py
```

## Quick Start

### 1. Test Training Locally

```bash
# Set UTF-8 encoding (Windows)
$env:PYTHONIOENCODING="utf-8"

# Run quick training test
python scripts/quick_train.py --symbol AAPL --epochs 50
```

### 2. View Training Dashboard

```bash
python scripts/training_dashboard.py
```

### 3. Enable GitHub Actions

1. Go to your repository settings
2. Navigate to Actions → General
3. Enable workflows
4. Add secrets (if using Hugging Face or W&B):
   - `HF_TOKEN` - Hugging Face API token
   - `WANDB_API_KEY` - Weights & Biases API key

### 4. Trigger Manual Training

Go to Actions tab → Select workflow → Click "Run workflow"

## Training Configuration

Edit `scripts/continuous_training.py` to customize:

```python
# Number of stocks to train per session
STOCK_COUNT = 5

# Forex pairs to train
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]

# Training epochs
EPOCHS = 50
```

## Training Modes

### Full Training
- Uses all historical data
- Best for initial model creation
- Takes longer but more comprehensive

### Incremental Training
- Updates existing models with new data
- Faster training
- Preserves learned patterns

## Model Storage

### Local Storage
Models are saved to `models/` directory:
- `models/stock_AAPL.pt`
- `models/forex_EURUSD.pt`

### Hugging Face Storage (Optional)
Models can be automatically uploaded to Hugging Face Hub for:
- Version control
- Easy deployment
- Team collaboration

## Monitoring

### Training Dashboard
```bash
python scripts/training_dashboard.py
```

Shows:
- Total models trained
- Recent training sessions (24h)
- Average accuracy and loss
- Latest training results

### GitHub Actions Logs
- View training logs in Actions tab
- Download artifacts for detailed analysis
- Monitor training success/failure

## Troubleshooting

### Unicode Encoding Error (Windows)
```bash
$env:PYTHONIOENCODING="utf-8"
```

### Insufficient Data Error
Increase data period:
```bash
python scripts/fetch_training_data.py --period 2y --interval 1d
```

### Training Too Slow
- Reduce epochs: `--epochs 30`
- Use GPU if available
- Reduce number of stocks per session

### Database Locked
```bash
# Close any open connections
# Delete and recreate database
rm training.db
python scripts/store_training_data.py --data-dir datasets/training_data --db-file training.db
```

## Advanced Configuration

### Custom Training Schedule

Edit `.github/workflows/multi-daily-training.yml`:

```yaml
on:
  schedule:
    - cron: '0 */3 * * *'  # Every 3 hours
    - cron: '30 */6 * * *' # Every 6 hours at :30
```

### Training with W&B Tracking

```bash
export WANDB_API_KEY="your_key_here"
python scripts/train_model.py --wandb-project ara-ai --wandb-run-name AAPL-training
```

### Batch Training Multiple Symbols

```python
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
for symbol in symbols:
    train_model(symbol, epochs=50)
```

## Performance Optimization

### Speed Up Training
1. Use GPU: Install CUDA-enabled PyTorch
2. Reduce data: Use shorter periods for testing
3. Parallel training: Train multiple models simultaneously
4. Cache dependencies: GitHub Actions caches pip packages

### Improve Model Quality
1. More data: Use longer historical periods
2. More epochs: Increase training iterations
3. Hyperparameter tuning: Adjust learning rate, batch size
4. Feature engineering: Add technical indicators

## Cost Considerations

### GitHub Actions Free Tier
- 2,000 minutes/month for private repos
- Unlimited for public repos
- Each training session: ~5 minutes
- Multi-daily (6x): ~30 min/day = ~900 min/month ✓
- Hourly (24x): ~120 min/day = ~3,600 min/month (exceeds free tier)

**Recommendation**: Use multi-daily schedule for free tier, or upgrade for hourly training.

## Next Steps

1. ✓ Test local training
2. ✓ View dashboard
3. Enable GitHub Actions
4. Configure secrets (optional)
5. Monitor first automated training
6. Adjust schedules based on needs

## Support

For issues or questions:
- Check GitHub Actions logs
- Review training dashboard
- Check database: `sqlite3 training.db`
- Review model files in `models/` directory
