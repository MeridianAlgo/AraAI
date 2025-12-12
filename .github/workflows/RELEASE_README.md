# 📦 Weekly Release System

## Overview

The ARA AI system automatically creates a **weekly release** every Sunday at 00:00 UTC with comprehensive statistics, trained models, and performance metrics.

## What Gets Released

### 1. Release Archive (`ara-ai-vX.X.X.tar.gz`)
Contains:
- **Trained Models**: All `.pt` model files
- **Training Database**: Complete `training_data.db`
- **Statistics**: JSON and Markdown formats
- **Performance Metrics**: Weekly training results

### 2. Individual Files
- `release_stats.json` - Statistics in JSON format
- `release_stats.md` - Statistics in Markdown format

### 3. Release Notes
Comprehensive release notes including:
- Weekly statistics summary
- Top performing models
- Training activity metrics
- Data collection stats
- System information

## Release Schedule

**Automatic**: Every Sunday at 00:00 UTC

**Manual**: Can be triggered anytime via GitHub Actions

## Release Process

### 1. Test System (5-10 minutes)
- Tests data fetching
- Tests database operations
- Tests query scripts
- Validates all components

### 2. Collect Statistics (2-3 minutes)
- Queries training database
- Generates weekly metrics
- Creates performance reports
- Formats statistics

### 3. Determine Version (1 minute)
- Auto-increments patch version
- Or uses manual version input
- Creates version tag

### 4. Create Release (3-5 minutes)
- Generates release notes
- Creates release archive
- Pushes Git tag
- Creates GitHub release
- Updates version in files

**Total Time**: ~15-20 minutes

## Version Numbering

Format: `vMAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes (manual)
- **MINOR**: New features (manual)
- **PATCH**: Weekly releases (automatic)

Current: **v5.0.0**

## Manual Release

Trigger a release manually:

1. Go to **Actions** tab
2. Select **"Weekly Release"**
3. Click **"Run workflow"**
4. Enter version (or leave as "auto")
5. Click **"Run workflow"**

## Release Contents

### Statistics Included

```json
{
  "training_runs": {
    "total_runs": 84,
    "hourly_runs": 77,
    "full_runs": 7,
    "total_rows_processed": 125000
  },
  "model_performance": {
    "total_models": 15,
    "total_training_sessions": 1260,
    "overall_avg_accuracy": 0.94,
    "best_accuracy": 0.97,
    "overall_avg_loss": 0.025,
    "best_loss": 0.018
  },
  "top_performers": [
    {
      "symbol": "AAPL",
      "timeframe": "1h",
      "best_accuracy": 0.97,
      "best_loss": 0.018
    }
  ]
}
```

### Release Notes Format

```markdown
# 🚀 ARA AI vX.X.X - Weekly Release

## 📊 Weekly Statistics
- Training runs, model performance, top performers

## 🎯 Trained Assets
- List of all stocks and forex pairs

## 🔄 Training Schedule
- Hourly and full training information

## 📥 Download
- Links to release archive and files

## 🚀 Quick Start
- Usage examples and documentation links
```

## Accessing Releases

### Via GitHub UI
1. Go to repository
2. Click **Releases** (right sidebar)
3. Select desired version
4. Download files

### Via GitHub CLI
```bash
# List releases
gh release list

# View specific release
gh release view v5.0.0

# Download release
gh release download v5.0.0
```

### Via API
```bash
# Get latest release
curl https://api.github.com/repos/MeridianAlgo/AraAI/releases/latest

# Download asset
curl -L -o ara-ai.tar.gz \
  https://github.com/MeridianAlgo/AraAI/releases/download/v5.0.0/ara-ai-v5.0.0.tar.gz
```

## Using Released Models

### Extract Archive
```bash
tar -xzf ara-ai-v5.0.0.tar.gz
cd release_artifacts
```

### Load Models
```python
from meridianalgo.unified_ml import UnifiedStockML

# Load trained model
ml = UnifiedStockML(model_path='models/AAPL_model.pt')

# Make prediction
result = ml.predict('AAPL', days=5)
```

### Query Database
```bash
# Show statistics
python scripts/query_training_data.py stats --db-file training_data.db

# Show model performance
python scripts/query_training_data.py performance --db-file training_data.db
```

## Release Artifacts

### Models Directory
```
models/
├── AAPL_model.pt
├── MSFT_model.pt
├── GOOGL_model.pt
├── AMZN_model.pt
├── TSLA_model.pt
├── NVDA_model.pt
├── META_model.pt
├── NFLX_model.pt
├── AMD_model.pt
├── INTC_model.pt
├── forex_EURUSD_model.pt
├── forex_GBPUSD_model.pt
├── forex_USDJPY_model.pt
├── forex_AUDUSD_model.pt
└── forex_USDCAD_model.pt
```

### Database Schema
```sql
-- Market data with timeframe tracking
market_data (symbol, date, OHLCV, timeframe, interval, hour)

-- Training run history
training_runs (run_date, mode, timeframe, hour, symbols_count, rows_processed)

-- Model performance tracking
model_metadata (symbol, training_date, accuracy, loss, timeframe, training_mode, hour)
```

## Monitoring Releases

### GitHub Actions
- View workflow runs in Actions tab
- Check release creation logs
- Monitor test results

### Release Notifications
Enable notifications:
1. Go to repository
2. Click **Watch** → **Custom**
3. Enable **Releases**
4. Get notified of new releases

### RSS Feed
Subscribe to releases:
```
https://github.com/MeridianAlgo/AraAI/releases.atom
```

## Troubleshooting

### Release Failed
1. Check Actions logs
2. Verify tests passed
3. Check database availability
4. Retry workflow

### Missing Models
- Models only included after training runs
- First release may have limited models
- Subsequent releases will have all models

### Statistics Empty
- Requires at least one week of training
- First release may have minimal stats
- Stats accumulate over time

## Customization

### Change Release Schedule
Edit `.github/workflows/weekly-release.yml`:
```yaml
schedule:
  # Every Sunday at 00:00 UTC
  - cron: '0 0 * * 0'
  
  # Change to Friday at 18:00 UTC
  - cron: '0 18 * * 5'
```

### Modify Release Contents
Edit `scripts/generate_release_notes.py`:
- Add custom sections
- Include additional metrics
- Customize formatting

### Add Release Assets
Edit workflow to include more files:
```yaml
files: |
  ara-ai-${{ needs.determine-version.outputs.version }}.tar.gz
  release_stats.json
  release_stats.md
  your_custom_file.txt
```

## Best Practices

1. **Review Before Release**: Check statistics before manual releases
2. **Test Locally**: Run scripts locally to verify
3. **Monitor First Release**: Watch first automated release closely
4. **Archive Old Releases**: Keep last 10 releases, archive older ones
5. **Document Changes**: Add notes for significant updates

## Release Checklist

Before manual release:
- [ ] All tests passing
- [ ] Database available
- [ ] Models trained
- [ ] Statistics generated
- [ ] Version number correct
- [ ] Release notes reviewed

## Future Enhancements

Planned improvements:
- [ ] Release comparison (week-over-week)
- [ ] Performance trends visualization
- [ ] Automated changelog generation
- [ ] Release quality metrics
- [ ] Notification integrations (Slack, Discord)
- [ ] Docker image releases

## Support

### Documentation
- [Weekly Release System](.github/workflows/RELEASE_README.md) (this file)
- [Hourly Training](../../HOURLY_TRAINING.md)
- [Quick Start](../../QUICKSTART_HOURLY.md)

### Issues
- Report release issues on GitHub
- Include workflow run ID
- Attach relevant logs

---

**Next Release**: Every Sunday at 00:00 UTC  
**Current Version**: v5.0.0  
**Release Frequency**: Weekly (automatic)
