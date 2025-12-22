# Trained Models

This directory contains trained machine learning models for stock and forex prediction.

## Model Storage

Models are automatically stored in two locations:
1. **Hugging Face Hub**: [MeridianAlgo/ARA.AI](https://huggingface.co/MeridianAlgo/ARA.AI) (primary storage)
2. **GitHub Artifacts**: Temporary storage for 7 days

## Model Files

### Stock Models
- `stock_*.pt` - PyTorch models trained on individual stock symbols
- Format: `stock_{SYMBOL}.pt` (e.g., `stock_AAPL.pt`)
- Updated every 2 hours (5 random stocks per cycle)

### Forex Models
- `forex_*.pt` - PyTorch models trained on currency pairs
- Format: `forex_{PAIR}.pt` (e.g., `forex_EURUSD.pt`)
- Pairs: EURUSD, GBPUSD, USDJPY
- Updated every 2 hours

## Model Information

- **Framework**: PyTorch
- **Type**: Unified ML ensemble (XGBoost, LightGBM, Random Forest, Transformers, CNN-LSTM)
- **Training Data**: 2 years of historical market data
- **Update Frequency**: Every 2 hours (12 times daily)
- **Training Mode**: Incremental (continues from previous checkpoint)
- **Epochs per Cycle**: 50
- **Experiment Tracking**: Weights & Biases (wandb)

## Using the Models

### Download from Hugging Face

```python
from huggingface_hub import hf_hub_download

# Download a specific stock model
model_path = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/stock_AAPL.pt"
)

# Download a forex model
forex_model_path = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/forex_EURUSD.pt"
)
```

### Load a Stock Model
```python
from meridianalgo.unified_ml import UnifiedStockML

ml = UnifiedStockML(model_path=model_path)
prediction = ml.predict('AAPL', days=5)
```

### Load a Forex Model
```python
from meridianalgo.forex_ml import ForexML

forex_ml = ForexML(model_path=forex_model_path)
prediction = forex_ml.predict('EURUSD', days=5)
```

## Training Schedule

Models train continuously on a 2-hour cycle:
- **00:00 UTC**: Cycle 1
- **02:00 UTC**: Cycle 2
- **04:00 UTC**: Cycle 3
- **06:00 UTC**: Cycle 4
- **08:00 UTC**: Cycle 5
- **10:00 UTC**: Cycle 6
- **12:00 UTC**: Cycle 7
- **14:00 UTC**: Cycle 8
- **16:00 UTC**: Cycle 9
- **18:00 UTC**: Cycle 10
- **20:00 UTC**: Cycle 11
- **22:00 UTC**: Cycle 12

Each cycle trains 5 random stocks + 3 forex pairs = 8 models per cycle = 96 model updates daily

## Training History

Models are automatically retrained every 2 hours. Check:
- **Hugging Face Hub**: https://huggingface.co/MeridianAlgo/ARA.AI
- **GitHub Actions**: Workflow runs and logs
- **Weights & Biases**: Experiment tracking and metrics

## Model Versioning

Each model update is tracked:
1. **Hugging Face**: Automatic versioning with commit history
2. **W&B**: Experiment runs with timestamps
3. **GitHub Actions**: Workflow run artifacts

## File Size

Model files are typically 50-500 MB depending on the symbol and training data.

## Notes

- Models use incremental training to continuously improve
- Each 2-hour cycle adds 50 more epochs to existing models
- Random stock selection ensures broad market coverage over time
- Forex models focus on the 3 most liquid pairs
- Models are trained on historical data - past performance doesn't guarantee future results
- Use predictions as one input among many for investment decisions
- Always validate predictions with current market conditions
- See main README for disclaimer and usage guidelines
