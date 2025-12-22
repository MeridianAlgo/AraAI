# Trained Models

This directory contains trained machine learning models for stock and forex prediction.

## Model Files

### Stock Models
- `stock_*.pt` - PyTorch models trained on individual stock symbols
- Format: `stock_{SYMBOL}.pt` (e.g., `stock_AAPL.pt`)
- Updated daily via GitHub Actions workflow

### Forex Models
- `forex_*.pt` - PyTorch models trained on currency pairs
- Format: `forex_{PAIR}.pt` (e.g., `forex_EURUSD.pt`)
- Pairs: EURUSD, GBPUSD, USDJPY
- Updated daily via GitHub Actions workflow

## Model Information

- **Framework**: PyTorch
- **Type**: Unified ML ensemble (XGBoost, LightGBM, Random Forest, Transformers, CNN-LSTM)
- **Training Data**: 2 years of historical market data
- **Update Frequency**: Daily at 6:00 AM UTC
- **Experiment Tracking**: Weights & Biases (wandb)

## Using the Models

### Load a Stock Model
```python
from meridianalgo.unified_ml import UnifiedStockML

ml = UnifiedStockML(model_path='models/stock_AAPL.pt')
prediction = ml.predict('AAPL', days=5)
```

### Load a Forex Model
```python
from meridianalgo.forex_ml import ForexML

forex_ml = ForexML(model_path='models/forex_EURUSD.pt')
prediction = forex_ml.predict('EURUSD', days=5)
```

## Training History

Models are automatically retrained daily. Check the GitHub Actions workflow for:
- Training logs
- Performance metrics
- Experiment tracking on Weights & Biases

## Model Versioning

Each model file is timestamped in the commit history. To access previous versions:
1. Go to GitHub repository
2. Navigate to `models/` directory
3. Click on the file
4. Use "History" to see previous versions

## Storage

- **Local**: Models are stored in this directory
- **Artifacts**: Also available as GitHub Actions artifacts for 90 days
- **Backup**: Committed to repository for permanent storage

## File Size

Model files are typically 50-500 MB depending on the symbol and training data.

## Notes

- Models are trained on historical data and past performance doesn't guarantee future results
- Use predictions as one input among many for investment decisions
- Always validate predictions with current market conditions
- See main README for disclaimer and usage guidelines
