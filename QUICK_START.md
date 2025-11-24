# 🚀 ARA AI Quick Start Guide

## System Status: ✅ FULLY OPERATIONAL

All 180 modules tested and passing. The system is ready for production use!

## Quick Predictions

### Basic Usage
```bash
# Predict AAPL for 5 days (default)
python scripts/ara.py AAPL

# Predict TSLA for 7 days
python scripts/ara.py TSLA --days 7

# Predict MSFT for 3 days
python scripts/ara.py MSFT --days 3
```

### Recent Test Results
- **AAPL**: +4.79% over 5 days (Bullish)
- **TSLA**: +70.77% over 7 days (Strong Bullish)
- **MSFT**: +8.12% over 3 days (Strong Bullish)

## Available Scripts

### Stock Predictions
```bash
# Main prediction script
python scripts/ara.py <SYMBOL> --days <N>

# CSV-based predictions
python scripts/ara_csv.py <CSV_FILE>

# Forex predictions
python scripts/ara_forex.py <PAIR>
```

### Model Training
```bash
# Train on AAPL data
python scripts/train_aapl_model.py

# Train advanced models
python scripts/train_advanced_models.py

# Train from custom dataset
python scripts/train_from_dataset.py
```

### API Server
```bash
# Start the FastAPI server
python scripts/run_api.py

# Access at: http://localhost:8000
# API docs: http://localhost:8000/docs
```

## System Testing

### Run Comprehensive Tests
```bash
# Test all 180 modules
python test_all_modules.py

# Run unit tests
pytest tests/

# Run specific test
pytest tests/test_integration.py
```

## Key Features

### 1. ML Models
- ✅ Transformer models
- ✅ CNN-LSTM networks
- ✅ Ensemble systems
- ✅ Regime detection
- ✅ Auto-retraining

### 2. Data Sources
- ✅ Yahoo Finance (stocks)
- ✅ Crypto exchanges
- ✅ DeFi protocols
- ✅ On-chain data
- ✅ Custom CSV files

### 3. Analysis Tools
- ✅ Technical indicators (44+)
- ✅ Sentiment analysis
- ✅ Pattern recognition
- ✅ Correlation analysis
- ✅ Risk metrics

### 4. Risk Management
- ✅ Portfolio optimization
- ✅ Risk calculators
- ✅ Constraint management
- ✅ Backtesting engine

### 5. Production Features
- ✅ REST API
- ✅ WebSocket streaming
- ✅ Authentication
- ✅ Rate limiting
- ✅ Monitoring & alerts

## Python API Usage

### Basic Prediction
```python
from meridianalgo.unified_ml import UnifiedStockML

# Initialize
ml = UnifiedStockML()

# Make prediction
result = ml.predict('AAPL', days=5)
print(result)
```

### Using ARA Package
```python
from ara.data.base_provider import BaseDataProvider
from ara.features.calculator import IndicatorCalculator
from ara.models.ensemble import EnhancedEnsemble

# Fetch data
provider = BaseDataProvider()
data = provider.fetch_historical('AAPL', period='1y')

# Calculate indicators
calc = IndicatorCalculator()
features = calc.calculate(data, ['rsi', 'macd', 'bb'])

# Make predictions
model = EnhancedEnsemble()
predictions = model.predict(features)
```

## Configuration

### Config File
Edit `ara/config/config.example.yaml` and save as `config.yaml`

### Environment Variables
```bash
# API Keys (optional)
export ALPHA_VANTAGE_API_KEY=your_key
export NEWS_API_KEY=your_key

# Database (optional)
export DATABASE_URL=postgresql://...

# Redis (optional)
export REDIS_URL=redis://localhost:6379
```

## Documentation

- 📖 [README.md](README.md) - Main documentation
- 🧪 [TEST_REPORT.md](TEST_REPORT.md) - Test results
- 📊 [SYSTEM_STATUS.md](SYSTEM_STATUS.md) - System health
- 📚 [docs/](docs/) - Full documentation
- 💡 [examples/](examples/) - Code examples

## Troubleshooting

### Import Errors
All import errors have been fixed. If you encounter any:
```bash
python test_all_modules.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Model Not Found
```bash
# Download or train models
python scripts/train_aapl_model.py
```

## Performance Tips

1. **Use GPU** - Set `use_gpu: true` in config for faster training
2. **Enable Caching** - Reduces API calls and speeds up predictions
3. **Batch Predictions** - Process multiple symbols together
4. **Parallel Processing** - Use multi-worker mode for large datasets

## Support

- 📧 Issues: Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- 🔒 Security: See [docs/SECURITY.md](docs/SECURITY.md)
- 📖 API Docs: See [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)
- ❓ FAQ: See [docs/FAQ.md](docs/FAQ.md)

---

**System Status: 🟢 ALL SYSTEMS OPERATIONAL**

Last Updated: November 16, 2025
