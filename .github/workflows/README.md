# GitHub Workflows

## Training Workflows

### Stock Training (`train-stock.yml`)
- Runs every 3 hours
- Randomizes timeframe: 1h, 4h, 1d, 1w
- Supports all stocks or sampling
- Separate jobs: Setup → Train → Deploy → Cleanup
- Environment: `stock-training`
- Comet ML: https://www.comet.com/meridianalgo/ara-ai-stock

### Forex Training (`train-forex.yml`)
- Runs every 3 hours at :30
- Randomizes timeframe: 15m, 1h, 4h, 1d
- Supports all pairs or sampling
- Separate jobs: Setup → Train → Deploy → Cleanup
- Environment: `forex-training`
- Comet ML: https://www.comet.com/meridianalgo/ara-ai-forex

### Linting (`lint.yml`)
- Runs on push/PR to main
- Tools: Ruff, Black, isort
- Auto-fixes on push to main
- Creates issues on failure

## Features

- Randomized timeframes for diversity
- Dependency caching between jobs
- Automatic issue creation on failure
- Progress tracking via environments
- Easy cancellation (separate workflows)
- Artifact cleanup after completion

## Manual Trigger

Go to Actions → Select workflow → Run workflow

**Stock Options:**
- `train_all_stocks`: Train with all stocks
- `epochs`: Training epochs (default: 60)
- `sample_size`: Number of stocks (0 = all)

**Forex Options:**
- `epochs`: Training epochs (default: 60)
- `sample_size`: Number of pairs (0 = all)
