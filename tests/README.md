# Tests

Test suite for ARA AI.

## Test Files

- `test_ultimate_ml.py` - Ultimate ML system tests
- `test_console.py` - Console manager tests

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=meridianalgo --cov-report=html

# Specific test
pytest tests/test_ultimate_ml.py -v
```

## Test Requirements

- pytest
- pytest-cov
- pytest-xdist
- pytest-timeout

**Maintained by**: MeridianAlgo  
**Last Updated**: October 22, 2025
