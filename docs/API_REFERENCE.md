# ARA AI - API Reference

## Python API Documentation

### UltimateStockML Class

```python
from meridianalgo.ultimate_ml import UltimateStockML

# Initialize
ml = UltimateStockML(model_dir="models")
```

#### Methods

**train_ultimate_models(target_symbol, period='2y', max_symbols=None, use_parallel=False)**
- Train models on online data (Quick Mode)
- Args:
  - `target_symbol`: Stock symbol (e.g., 'AAPL')
  - `period`: Training period ('6mo', '1y', '2y', '5y')
  - `max_symbols`: Ignored
  - `use_parallel`: Ignored
- Returns: bool (success status)

**train_from_dataset(dataset_path, symbol_name)**
- Train models from CSV dataset (Advanced Mode)
- Args:
  - `dataset_path`: Path to CSV file
  - `symbol_name`: Symbol name for metadata
- Returns: bool (success status)

**predict_ultimate(symbol, days=5)**
- Make predictions
- Args:
  - `symbol`: Stock symbol
  - `days`: Number of days to predict
- Returns: dict with predictions

**get_model_status()**
- Get model status
- Returns: dict with training info

### ForexML Class

```python
from meridianalgo.forex_ml import ForexML

# Initialize
forex = ForexML(model_dir="models/forex")
```

#### Methods

**predict_forex(pair, days=5, period='2y')**
- Predict forex pair
- Args:
  - `pair`: Currency pair (e.g., 'EURUSD')
  - `days`: Number of days to predict
  - `period`: Training period if not trained
- Returns: dict with predictions including pips

**get_forex_symbol(pair)**
- Convert pair to Yahoo Finance symbol
- Args:
  - `pair`: Currency pair
- Returns: str (Yahoo Finance symbol)

**get_pair_info(pair)**
- Get currency pair information
- Args:
  - `pair`: Currency pair
- Returns: dict with pair details

### Example Usage

```python
from meridianalgo.ultimate_ml import UltimateStockML
from meridianalgo.forex_ml import ForexML

# Stock predictions
ml = UltimateStockML()

# Quick Mode - train on-the-fly
ml.train_ultimate_models('AAPL', period='2y')
result = ml.predict_ultimate('AAPL', days=7)

# Advanced Mode - train from dataset
ml.train_from_dataset('datasets/AAPL.csv', 'AAPL')
result = ml.predict_ultimate('AAPL', days=7)

# Access predictions
for pred in result['predictions']:
    print(f"Day {pred['day']}: ${pred['predicted_price']:.2f}")
    print(f"Confidence: {pred['confidence']:.1%}")

# Forex predictions
forex = ForexML()

# Quick Mode
forex.train_ultimate_models('EURUSD=X', period='2y')
result = forex.predict_forex('EURUSD', days=7)

# Advanced Mode
forex.train_from_dataset('datasets/EURUSD.csv', 'EURUSD')
result = forex.predict_forex('EURUSD', days=7)

# Access forex predictions
for pred in result['predictions']:
    print(f"Day {pred['day']}: {pred['predicted_price']:.5f}")
    print(f"Pips: {pred['pips']:+.1f}")
    print(f"Confidence: {pred['confidence']:.1%}")
```

### Response Format

**Stock Prediction Response:**
```python
{
    'symbol': 'AAPL',
    'current_price': 245.50,
    'predictions': [
        {
            'day': 1,
            'date': '2025-11-09',
            'predicted_price': 246.85,
            'predicted_return': 0.0055,
            'confidence': 0.95
        },
        # ... more days
    ],
    'model_accuracy': 98.5,
    'timestamp': '2025-11-08T17:30:00',
    'trained_on': 'AAPL'
}
```

**Forex Prediction Response:**
```python
{
    'pair': 'EUR/USD',
    'pair_info': {
        'base_currency': 'EUR',
        'quote_currency': 'USD',
        'base_name': 'Euro',
        'quote_name': 'US Dollar',
        'type': 'Major'
    },
    'current_price': 1.08450,
    'predictions': [
        {
            'day': 1,
            'date': '2025-11-09',
            'predicted_price': 1.08523,
            'predicted_return': 0.0007,
            'pips': 7.3,
            'confidence': 0.95
        },
        # ... more days
    ],
    'model_accuracy': 95.0,
    'volatility': 0.85,
    'trend': 'Bullish',
    'timestamp': '2025-11-08T17:30:00',
    'model_type': 'forex_ultimate_ensemble'
}
```

### Model Status Response

```python
{
    'is_trained': True,
    'models': ['xgb', 'lgb', 'gb', 'rf', 'et', 'adaboost', 'ridge', 'elastic', 'lasso'],
    'feature_count': 44,
    'model_count': 9,
    'training_metadata': {
        'training_date': '2025-11-08T17:30:00',
        'symbol': 'AAPL',
        'data_points': 502,
        'date_range': '2023-11-08 to 2025-11-07'
    }
}
```

### Error Handling

```python
try:
    result = ml.predict_ultimate('AAPL', days=7)
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        # Process predictions
        pass
except Exception as e:
    print(f"Prediction failed: {e}")
```

### Console Manager

```python
from meridianalgo.console import ConsoleManager

console = ConsoleManager()

console.print_header("My Header")
console.print_info("Information message")
console.print_success("Success message")
console.print_warning("Warning message")
console.print_error("Error message")
```

---

**Version**: 3.0.2  
**Last Updated**: November 8, 2025
