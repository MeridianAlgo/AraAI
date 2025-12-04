# Backward Compatibility - Quick Reference

## ⚠️ Deprecation Notice

**This compatibility layer will be removed in version 5.0.0 (January 2025)**

## Quick Migration

### Old Code (v3.x)
```python
from meridianalgo.core import predict_stock
result = predict_stock('AAPL', days=5)
```

### Quick Fix (Temporary)
```python
from ara.compat import predict_stock  # Just change import
result = predict_stock('AAPL', days=5)
```

### New Code (v4.0) - Recommended
```python
from ara.api.prediction_engine import PredictionEngine
import asyncio

async def main():
    engine = PredictionEngine()
    return await engine.predict('AAPL', days=5)

result = asyncio.run(main())
```

## Migration Tools

### Migrate Models
```python
from ara.compat.migration import ModelMigrator

migrator = ModelMigrator(verbose=True)
migrator.migrate_model_directory(
    old_dir='models',
    new_dir='.ara_cache/models',
    backup=True
)
```

### Migrate Cache
```python
from ara.compat.migration import DataMigrator

migrator = DataMigrator(verbose=True)
migrator.migrate_cache_directory(
    old_cache_dir='.ara_cache',
    new_cache_dir='.ara_cache/market_data',
    backup=True
)
```

## Timeline

| Version | Date | Status |
|---------|------|--------|
| v4.0.0 | Jan 2024 | ✅ Compatibility available |
| v4.5.0 | Jun 2024 | ⚠️ Enhanced warnings |
| v5.0.0 | Jan 2025 | ❌ Compatibility removed |

## Common Issues

### Issue: Import Error
```python
# Old
from meridianalgo.core import AraAI

# Fix
from ara.compat import AraAI
```

### Issue: Async Error
```python
# Old (sync)
result = predict_stock('AAPL')

# New (async)
result = asyncio.run(engine.predict('AAPL'))
```

### Issue: Response Format
```python
# Old format
change_pct = result['predictions'][0]['change_pct']

# New format
predicted_return = result['predictions'][0]['predicted_return']
```

## Resources

- **Full Guide**: [docs/MIGRATION_GUIDE.md](../../docs/MIGRATION_GUIDE.md)
- **Examples**: [examples/compatibility_demo.py](../../examples/compatibility_demo.py)
- **Tests**: [tests/test_compatibility.py](../../tests/test_compatibility.py)

## Support

- GitHub Issues: Report problems
- Documentation: See docs/ folder
- Examples: See examples/ folder
