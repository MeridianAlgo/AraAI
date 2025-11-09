# ARA AI - Complete Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Dataset Training](#dataset-training)
3. [Stock Predictions](#stock-predictions)
4. [Forex Predictions](#forex-predictions)
5. [Model Management](#model-management)
6. [Advanced Usage](#advanced-usage)

---

## Quick Start

### Installation
```bash
git clone https://github.com/MeridianAlgo/AraAI.git
cd AraAI
pip install -r requirements.txt
```

### First Prediction
```bash
# Stock prediction (will train on first run)
python ara.py AAPL

# Forex prediction (will train on first run)
python ara_forex.py EURUSD
```

---

## Dataset Training

### Why Use Dataset Training?

- Train on 5+ years of historical data
- Better accuracy with more training data
- Models saved for instant future predictions
- No need to retrain every time

### Step 1: Download Dataset

**Stock Data:**
```bash
python download_dataset.py AAPL --period 5y --type stock
python download_dataset.py MSFT --period 5y --type stock
python download_dataset.py GOOGL --period 5y --type stock
```

**Forex Data:**
```bash
python download_dataset.py EURUSD --period 5y --type forex
python download_dataset.py GBPUSD --period 5y --type forex
python download_dataset.py USDJPY --period 5y --type forex
```

**Custom Output:**
```bash
python download_dataset.py AAPL --period 10y --output my_data/apple.csv
```

### Step 2: Train Models

**Train Stock Models:**
```bash
python train_from_dataset.py datasets/AAPL.csv --type stock --name AAPL
```

This will:
- Load the dataset
- Train all 9 models (XGBoost, LightGBM, RF, ET, GB, AdaBoost, Ridge, Elastic, Lasso)
- Save models to `models/` directory
- Save scalers and metadata
- Test predictions

**Train Forex Models:**
```bash
python train_from_dataset.py datasets/EURUSD.csv --type forex --name EURUSD
```

This will:
- Load the dataset
- Train all 9 models
- Save models to `models/forex/` directory
- Test predictions with pip calculations

### Step 3: Make Predictions

Once trained, models are automatically loaded:

```bash
# Stock predictions (uses saved models)
python ara.py AAPL --days 7

# Forex predictions (uses saved models)
python ara_forex.py EURUSD --days 7
```

---

## Stock Predictions

### Basic Usage

```bash
# Quick prediction (5 days)
python ara.py AAPL

# 7-day forecast
python ara.py GOOGL --days 7

# 14-day forecast
python ara.py TSLA --days 14
```

### Training Options

```bash
# Force retraining with 2 years of data
python ara.py AAPL --train --period 2y

# Train with 5 years of data
python ara.py MSFT --train --period 5y

# Train with 6 months of data (faster)
python ara.py NVDA --train --period 6mo
```

### Example Output

```
ARA AI v3.0.0 - AAPL Analysis
=====================================

Initializing Ultimate ML System...
Loaded 9 pre-trained models from models
Training date: 2025-11-08T17:30:00
Trained on: AAPL

Predicting AAPL for next 5 days...

AAPL ULTIMATE ML Predictions
Model: ultimate_ensemble_9_models
Accuracy: 98.5% | Features: 44 | Models: 9
Current Price: $245.50

Day 1: $246.85 (+0.55%) - Confidence: 95.0%
Day 2: $248.20 (+1.10%) - Confidence: 87.4%
Day 3: $249.10 (+1.47%) - Confidence: 80.4%
Day 4: $250.05 (+1.85%) - Confidence: 74.0%
Day 5: $251.15 (+2.30%) - Confidence: 68.1%

Prediction completed successfully!
```

---

## Forex Predictions

### Basic Usage

```bash
# Major pairs
python ara_forex.py EURUSD
python ara_forex.py GBPUSD
python ara_forex.py USDJPY

# Cross pairs
python ara_forex.py EURJPY
python ara_forex.py GBPJPY

# Exotic pairs
python ara_forex.py USDMXN
python ara_forex.py USDZAR
```

### Advanced Forex

```bash
# 7-day forecast
python ara_forex.py EURUSD --days 7

# Force retraining
python ara_forex.py GBPUSD --train

# Train with more data
python ara_forex.py USDJPY --train --period 5y
```

### Supported Pairs

**Major Pairs:**
- EURUSD, GBPUSD, USDJPY, USDCHF
- AUDUSD, USDCAD, NZDUSD

**Cross Pairs:**
- EURJPY, GBPJPY, EURGBP, EURAUD
- EURCHF, AUDJPY, GBPAUD, GBPCAD

**Exotic Pairs:**
- USDMXN, USDZAR, USDTRY, USDBRL

### Example Output

```
EURUSD - Forex Prediction Results
=====================================

Pair Information:
   Base: Euro (EUR)
   Quote: US Dollar (USD)
   Type: Major Pair
   Regions: Europe / North America

Current Rate: 1.08450
Trend: Bullish
Volatility: 0.85%

5-Day Forecast:
Date         Rate         Pips         Change       Confidence
2025-11-09   1.08523      +7.3         +0.07%       95.0%
2025-11-10   1.08601      +7.8         +0.07%       87.4%
2025-11-11   1.08685      +8.4         +0.08%       80.4%
2025-11-12   1.08774      +8.9         +0.08%       74.0%
2025-11-13   1.08868      +9.4         +0.09%       68.1%

Summary:
   Average Daily Change: +0.08%
   Final Predicted Rate: 1.08868
   Total Change: +0.39%
   Total Pips: +41.8

Outlook: Bullish
   EUR expected to strengthen vs USD

Market Status: Open (24/5 Market)
```

---

## Model Management

### Check Model Status

Models are automatically saved and loaded. Check status:

```python
from meridianalgo.ultimate_ml import UltimateStockML

ml = UltimateStockML()
status = ml.get_model_status()

print(f"Trained: {status['is_trained']}")
print(f"Models: {status['model_count']}")
print(f"Features: {status['feature_count']}")
print(f"Metadata: {status['training_metadata']}")
```

### Model Locations

- **Stock Models**: `models/`
- **Forex Models**: `models/forex/`
- **CSV Models**: `models/csv/`

### Model Files

Each trained system saves:
- `xgb_model.pkl` - XGBoost model
- `lgb_model.pkl` - LightGBM model
- `gb_model.pkl` - Gradient Boosting model
- `rf_model.pkl` - Random Forest model
- `et_model.pkl` - Extra Trees model
- `adaboost_model.pkl` - AdaBoost model
- `ridge_model.pkl` - Ridge regression model
- `elastic_model.pkl` - Elastic Net model
- `lasso_model.pkl` - Lasso regression model
- `scalers.pkl` - Feature scalers
- `metadata.json` - Training metadata

### Retrain Models

To update models with new data:

```bash
# Download fresh dataset
python download_dataset.py AAPL --period 5y

# Retrain
python train_from_dataset.py datasets/AAPL.csv --type stock --name AAPL
```

Or force retrain during prediction:

```bash
python ara.py AAPL --train --period 5y
```

---

## Advanced Usage

### Custom Datasets

Use your own CSV data:

```bash
# Prepare your CSV with Date,Open,High,Low,Close,Volume columns
python train_from_dataset.py my_data.csv --type stock --name CUSTOM
```

### Batch Training

Train multiple symbols:

```bash
# Download datasets
for symbol in AAPL MSFT GOOGL TSLA NVDA; do
    python download_dataset.py $symbol --period 5y --type stock
done

# Train all
for file in datasets/*.csv; do
    python train_from_dataset.py $file --type stock
done
```

### Python API

```python
from meridianalgo.ultimate_ml import UltimateStockML
from meridianalgo.forex_ml import ForexML

# Stock predictions
ml = UltimateStockML()
result = ml.predict_ultimate('AAPL', days=7)
print(result)

# Forex predictions
forex = ForexML()
result = forex.predict_forex('EURUSD', days=7)
print(result)

# Train from dataset
ml.train_from_dataset('datasets/AAPL.csv', 'AAPL')
```

### Performance Tips

1. **Use Dataset Training**: Train once on 5+ years of data
2. **Save Models**: Models are automatically saved and loaded
3. **Batch Predictions**: Train multiple symbols at once
4. **Longer Periods**: More training data = better accuracy

---

## Troubleshooting

### Models Not Loading

```bash
# Check if models exist
ls models/

# Retrain if needed
python ara.py AAPL --train
```

### Insufficient Data Error

```bash
# Download more data
python download_dataset.py AAPL --period 5y

# Train with more data
python train_from_dataset.py datasets/AAPL.csv --type stock
```

### Prediction Errors

```bash
# Force retrain
python ara.py AAPL --train --period 2y

# Or train from fresh dataset
python download_dataset.py AAPL --period 5y
python train_from_dataset.py datasets/AAPL.csv --type stock
```

---

## Best Practices

1. **Train on Historical Data**: Use 5+ years for best results
2. **Save Models**: Let the system save models automatically
3. **Update Periodically**: Retrain monthly with fresh data
4. **Test Predictions**: Always verify predictions make sense
5. **Use Confidence Scores**: Higher confidence = more reliable

---

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/MeridianAlgo/AraAI/issues)
- Email: support@meridianalgo.com

---

**Version**: 3.0.2  
**Last Updated**: November 8, 2025
