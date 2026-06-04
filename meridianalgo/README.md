# meridianalgo

Core Python package for Meridian.AI stock prediction system.

## Active Modules

- `meridian_model.py` - MeridianModel-2026 architecture (~430K parameters: GQA + MoE + optional Mamba SSM)
- `unified_ml.py` - Unified stock prediction model and feature engineering
- `forex_ml.py` - Forex prediction model
- `large_torch_model.py` - Advanced ML training system (training loop, checkpoint load/save, inference)
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
**Version**: 1.0.0 (Production)
