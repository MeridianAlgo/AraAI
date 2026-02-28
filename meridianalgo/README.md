# meridianalgo

Core Python package for Meridian.AI stock prediction system.

## Active Modules

- `unified_ml.py` - Unified stock prediction model
- `forex_ml.py` - Forex prediction model
- `revolutionary_model.py` - Revolutionary 2026 architecture (71M parameters)
- `large_torch_model.py` - Advanced ML training system
- `direction_loss.py` - Direction-aware loss functions
- `utils.py` - Utility functions and helpers

## Usage

```python
from meridianalgo.unified_ml import UnifiedStockML
from meridianalgo.forex_ml import ForexML

# Stock predictions
ml = UnifiedStockML()
result = ml.predict_ultimate('AAPL', days=5)

# Forex predictions
forex = ForexML()
result = forex.predict_forex('EURUSD', days=5)
```

**Maintained by**: MeridianAlgo  
**Last Updated**: February 2026
