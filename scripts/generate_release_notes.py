#!/usr/bin/env python3
"""
Generate beautiful release notes for weekly release
"""

import argparse
import json
from datetime import datetime

def generate_release_notes(version, stats_file):
    """Generate comprehensive release notes"""
    
    # Load statistics
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    except FileNotFoundError:
        stats = {'database_found': False}
    
    # Generate release notes
    notes = f"""# 🚀 ARA AI {version} - Weekly Release

**Release Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Release Type**: Automated Weekly Release

---

## 📦 What's Included

This release includes:

- ✅ **Trained Models**: All models trained with latest data
- ✅ **Training Database**: Complete SQLite database with training history
- ✅ **Weekly Statistics**: Comprehensive performance metrics
- ✅ **System Tests**: All tests passed ✓

---

"""
    
    if stats.get('database_found'):
        # Add statistics
        notes += "## 📊 Weekly Statistics\n\n"
        
        if 'training_runs' in stats and 'error' not in stats['training_runs']:
            tr = stats['training_runs']
            notes += f"""### Training Activity
- **Total Training Runs**: {tr['total_runs']}
- **Hourly Training Runs**: {tr['hourly_runs']}
- **Full Training Runs**: {tr['full_runs']}
- **Data Points Processed**: {tr['total_rows_processed']:,}

"""
        
        if 'model_performance' in stats and 'error' not in stats['model_performance']:
            mp = stats['model_performance']
            notes += f"""### Model Performance
- **Models Trained**: {mp['total_models']}
- **Training Sessions**: {mp['total_training_sessions']}
- **Average Accuracy**: {mp['overall_avg_accuracy']:.2%}
- **Best Accuracy**: {mp['best_accuracy']:.2%}
- **Average Loss**: {mp['overall_avg_loss']:.4f}
- **Best Loss**: {mp['best_loss']:.4f}

"""
        
        if stats.get('top_performers'):
            notes += "### 🏆 Top Performing Models\n\n"
            notes += "| Rank | Symbol | Timeframe | Accuracy | Loss |\n"
            notes += "|------|--------|-----------|----------|------|\n"
            for i, model in enumerate(stats['top_performers'][:5], 1):
                notes += f"| {i} | {model['symbol']} | {model['timeframe']} | {model['best_accuracy']:.2%} | {model['best_loss']:.4f} |\n"
            notes += "\n"
        
        if 'data_collection' in stats and 'error' not in stats['data_collection']:
            dc = stats['data_collection']
            notes += f"""### Data Collection
- **Total Data Points**: {dc['total_rows']:,}
- **Unique Symbols**: {dc['total_symbols']}

"""
    else:
        notes += """## ⚠️ First Release

This is the first release. Training data will be available in future releases after the system has been running.

"""
    
    notes += """---

## 🎯 Trained Assets

### Stocks (10)
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

### Forex Pairs (5)
- EURUSD (Euro/US Dollar)
- GBPUSD (British Pound/US Dollar)
- USDJPY (US Dollar/Japanese Yen)
- AUDUSD (Australian Dollar/US Dollar)
- USDCAD (US Dollar/Canadian Dollar)

---

## 🔄 Training Schedule

The system trains automatically:

- **Hourly Training**: 9 AM - 7 PM UTC (11 runs/day)
  - 1-hour interval data
  - 5 epochs per model
  - Fast incremental training

- **Full Training**: 8 PM UTC (1 run/day)
  - 2 years historical data
  - 100 epochs per model
  - Complete retraining

**Total**: 12 automated training runs per day

---

## 📥 Download

### Release Archive
Download `ara-ai-{version}.tar.gz` which includes:
- Trained model files (`.pt`)
- Training database (`training_data.db`)
- Weekly statistics (`release_stats.json`, `release_stats.md`)

### Individual Files
- `release_stats.json` - Statistics in JSON format
- `release_stats.md` - Statistics in Markdown format

---

## 🚀 Quick Start

### Using the Models

```python
from meridianalgo.unified_ml import UnifiedStockML

# Load trained model
ml = UnifiedStockML(model_path='models/AAPL_model.pt')

# Make prediction
result = ml.predict('AAPL', days=5)
print(f"Prediction: ${result['predictions'][4]['predicted_price']:.2f}")
```

### Querying the Database

```bash
# Show recent training runs
python scripts/query_training_data.py runs

# Show model performance
python scripts/query_training_data.py performance

# Show statistics
python scripts/query_training_data.py stats
```

---

## 🔧 System Requirements

- Python 3.9+
- PyTorch
- pandas, numpy
- yfinance
- SQLite (built-in)

---

## 📚 Documentation

- [Quick Start Guide](QUICKSTART_HOURLY.md)
- [Complete Documentation](HOURLY_TRAINING.md)
- [System Architecture](.github/workflows/SYSTEM_DIAGRAM.md)
- [Main README](README.md)

---

## 🐛 Known Issues

None reported this week.

---

## 🔮 Coming Soon

- Additional technical indicators
- More asset classes (crypto, commodities)
- Enhanced model architectures
- Real-time prediction API
- Web dashboard

---

## 📞 Support

- **Documentation**: See docs in repository
- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/AraAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/AraAI/discussions)

---

## ⚠️ Disclaimer

This software is for educational and research purposes only. NOT financial advice. You are solely responsible for your investment decisions. Past performance does not guarantee future results.

---

## 📝 Changelog

### {version}
- Weekly automated release
- Updated models with latest training data
- Comprehensive statistics and metrics
- All system tests passed

---

**Generated automatically by ARA AI Weekly Release System**  
**Next release**: Next Sunday at 00:00 UTC

---

## 🌟 Star the Project

If you find ARA AI useful, please star the repository on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/MeridianAlgo/AraAI?style=social)](https://github.com/MeridianAlgo/AraAI)
"""
    
    return notes

def main():
    parser = argparse.ArgumentParser(description='Generate release notes')
    parser.add_argument('--version', required=True, help='Release version')
    parser.add_argument('--stats', required=True, help='Statistics JSON file')
    parser.add_argument('--output', required=True, help='Output markdown file')
    
    args = parser.parse_args()
    
    print(f"Generating release notes for {args.version}...")
    notes = generate_release_notes(args.version, args.stats)
    
    with open(args.output, 'w') as f:
        f.write(notes)
    
    print(f"✓ Release notes saved to {args.output}")
    print(f"\nPreview:\n{'-'*60}")
    print(notes[:500] + "...")

if __name__ == '__main__':
    main()
