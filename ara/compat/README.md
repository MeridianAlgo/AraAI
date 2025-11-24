# Backward Compatibility Layer

This module provides backward compatibility with the old MeridianAlgo API (v3.x), allowing existing code to work with the new ARA AI architecture (v4.0) with minimal changes.

## ⚠️ Deprecation Notice

**This compatibility layer is deprecated and will be removed in version 5.0.0 (January 2025).**

Please migrate to the new API as soon as possible. See [Migration Guide](../../docs/MIGRATION_GUIDE.md) for details.

## Features

- **API Wrappers**: Drop-in replacements for old API functions and classes
- **Model Migration**: Automatic migration of old model formats
- **Data Migration**: Migration tools for cache and configuration files
- **Deprecation Warnings**: Clear warnings with migration guidance
- **Format Conversion**: Automatic conversion between old and new response formats

## Quick Start

### Using Compatibility Wrappers

```python
# Old code (v3.x) - still works with warnings
from ara.compat import predict_stock

result = predict_stock('AAPL', days=5)
print(result)

# Warning: predict_stock is deprecated. Use ara.api.PredictionEngine instead.
```

### Migrating Models

```python
from ara.compat.migration import ModelMigrator
from pathlib import Path

# Migrate old models to new format
migrator = ModelMigrator(verbose=True)
summary = migrator.migrate_model_directory(
    old_dir=Path('models'),
    new_dir=Path('.ara_cache/models'),
    backup=True
)

print(f"Migrated {summary['migrated']} models")
```

### Migrating Data

```python
from ara.compat.migration import DataMigrator
from pathlib import Path

# Migrate cache files
migrator = DataMigrator(verbose=True)
summary = migrator.migrate_cache_directory(
    old_cache_dir=Path('.ara_cache'),
    new_cache_dir=Path('.ara_cache/market_data'),
    backup=True
)

print(f"Migrated {summary['migrated']} cache files")
```

## Module Structure

```
ara/compat/
├── __init__.py           # Main exports
├── wrappers.py           # API compatibility wrappers
├── migration.py          # Model and data migration tools
├── deprecation.py        # Deprecation warning system
└── README.md            # This file
```

## API Reference

### Wrappers

#### `AraAI` (Deprecated)

Backward compatible wrapper for the old AraAI class.

```python
from ara.compat import AraAI

ara = AraAI(verbose=True)
result = ara.predict('AAPL', days=5)
```

**Deprecated**: Use `ara.api.PredictionEngine` instead.

#### `StockPredictor` (Deprecated)

Simplified interface for stock prediction.

```python
from ara.compat import StockPredictor

predictor = StockPredictor(verbose=True)
result = predictor.predict('AAPL', days=5)
```

**Deprecated**: Use `ara.api.PredictionEngine` instead.

#### `predict_stock()` (Deprecated)

Convenience function for quick predictions.

```python
from ara.compat import predict_stock

result = predict_stock('AAPL', days=5, verbose=True)
```

**Deprecated**: Use `ara.api.PredictionEngine.predict()` instead.

#### `analyze_stock()` (Deprecated)

Analyze stock with technical indicators.

```python
from ara.compat import analyze_stock

analysis = analyze_stock('AAPL', verbose=True)
```

**Deprecated**: Use `ara.features.FeatureCalculator` instead.

### Migration Tools

#### `ModelMigrator`

Migrates old model formats to new format.

```python
from ara.compat.migration import ModelMigrator

migrator = ModelMigrator(verbose=True)

# Migrate entire directory
summary = migrator.migrate_model_directory(
    old_dir='models',
    new_dir='.ara_cache/models',
    backup=True
)

# Migrate single file
result = migrator.migrate_model_file(
    old_file='models/demo_model.pkl',
    new_dir='.ara_cache/models'
)
```

**Methods**:
- `migrate_model_directory(old_dir, new_dir, backup=True)`: Migrate entire directory
- `migrate_model_file(old_file, new_dir)`: Migrate single file
- `get_migration_log()`: Get migration log

#### `DataMigrator`

Migrates cache files and data structures.

```python
from ara.compat.migration import DataMigrator

migrator = DataMigrator(verbose=True)

# Migrate cache directory
summary = migrator.migrate_cache_directory(
    old_cache_dir='.ara_cache',
    new_cache_dir='.ara_cache/market_data',
    backup=True
)

# Validate migration
validation = migrator.validate_migration('.ara_cache/market_data')
```

**Methods**:
- `migrate_cache_directory(old_cache_dir, new_cache_dir, backup=True)`: Migrate cache
- `validate_migration(new_dir)`: Validate migrated data

### Deprecation System

#### `@deprecated` Decorator

Mark functions as deprecated with clear migration guidance.

```python
from ara.compat.deprecation import deprecated, DeprecationLevel

@deprecated(
    reason="Old API structure",
    version="4.0.0",
    removal_version="5.0.0",
    alternative="ara.api.PredictionEngine.predict()",
    level=DeprecationLevel.WARNING
)
def old_function():
    pass
```

#### `DeprecationLevel`

Severity levels for deprecation warnings:

- `INFO`: Feature will be deprecated in future
- `WARNING`: Feature is deprecated, still works
- `ERROR`: Feature is deprecated, may not work correctly
- `REMOVED`: Feature has been removed

#### `DeprecationTimeline`

Timeline for deprecation:

- **v4.0.0** (Current): Compatibility layer introduced
- **v4.5.0** (June 2024): Warnings become more prominent
- **v5.0.0** (January 2025): Compatibility layer removed

## Migration Strategy

### Phase 1: Quick Fix (Immediate)

Use compatibility wrappers to keep existing code working:

```python
# Change imports only
from ara.compat import predict_stock  # Instead of meridianalgo.core

result = predict_stock('AAPL', days=5)
```

### Phase 2: Migrate Data (Short-term)

Migrate models and cache files:

```python
from ara.compat.migration import ModelMigrator, DataMigrator

# Migrate models
model_migrator = ModelMigrator(verbose=True)
model_migrator.migrate_model_directory('models', '.ara_cache/models')

# Migrate cache
data_migrator = DataMigrator(verbose=True)
data_migrator.migrate_cache_directory('.ara_cache', '.ara_cache/market_data')
```

### Phase 3: Refactor Code (Long-term)

Migrate to new async API:

```python
from ara.api.prediction_engine import PredictionEngine
import asyncio

async def get_prediction(symbol):
    engine = PredictionEngine()
    result = await engine.predict(symbol, days=5)
    return result

result = asyncio.run(get_prediction('AAPL'))
```

## Response Format Conversion

The compatibility layer automatically converts between old and new formats:

### Old Format (v3.x)

```json
{
  "symbol": "AAPL",
  "current_price": 150.0,
  "predictions": [
    {
      "day": 1,
      "date": "2024-01-02",
      "predicted_price": 152.0,
      "change": 2.0,
      "change_pct": 1.33,
      "confidence": 0.85
    }
  ],
  "timestamp": "2024-01-01T00:00:00"
}
```

### New Format (v4.0)

```json
{
  "symbol": "AAPL",
  "asset_type": "stock",
  "current_price": 150.0,
  "predictions": [
    {
      "day": 1,
      "date": "2024-01-02",
      "predicted_price": 152.0,
      "predicted_return": 1.33,
      "confidence": 0.85,
      "lower_bound": 148.0,
      "upper_bound": 156.0
    }
  ],
  "confidence": {
    "overall": 0.85,
    "model_agreement": 0.90,
    "data_quality": 0.95
  },
  "explanations": {...},
  "regime": {...},
  "timestamp": "2024-01-01T00:00:00",
  "model_version": "4.0.0"
}
```

The compatibility layer handles this conversion automatically.

## Troubleshooting

### Issue: Deprecation Warnings

**Problem**: Getting deprecation warnings in console.

**Solution**: This is expected. The warnings guide you to migrate to the new API. To suppress warnings temporarily:

```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
```

However, it's better to migrate to the new API.

### Issue: Model Migration Fails

**Problem**: Model migration fails with errors.

**Solution**: Check the migration log for details:

```python
migrator = ModelMigrator(verbose=True)
summary = migrator.migrate_model_directory('models', '.ara_cache/models')

# Check log
log = migrator.get_migration_log()
for entry in log:
    if not entry['success']:
        print(f"Failed: {entry['file']}")
        print(f"Error: {entry['error']}")
```

### Issue: Response Format Different

**Problem**: Code expects old response format but gets new format.

**Solution**: Use compatibility wrappers which automatically convert formats:

```python
# Use compatibility wrapper
from ara.compat import predict_stock

result = predict_stock('AAPL', days=5)
# Returns old format automatically
```

## Best Practices

1. **Use Compatibility Layer Temporarily**: Don't rely on it long-term
2. **Migrate Incrementally**: Start with data migration, then code
3. **Test Thoroughly**: Test with compatibility layer before full migration
4. **Monitor Warnings**: Pay attention to deprecation warnings
5. **Plan Migration**: Complete migration before v5.0.0 (January 2025)

## Support

For migration help:

- **Migration Guide**: [docs/MIGRATION_GUIDE.md](../../docs/MIGRATION_GUIDE.md)
- **GitHub Issues**: Report migration issues
- **Documentation**: See [docs/](../../docs/) for full documentation

## Timeline

| Version | Date | Status |
|---------|------|--------|
| v4.0.0 | January 2024 | ✅ Compatibility layer available |
| v4.5.0 | June 2024 | ⚠️ Enhanced warnings |
| v5.0.0 | January 2025 | ❌ Compatibility layer removed |

**Recommendation**: Complete migration by December 2024 to avoid disruption.
