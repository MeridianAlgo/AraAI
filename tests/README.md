# ARA AI Test Suite

Comprehensive test suite for ARA AI prediction system with 80%+ code coverage.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── mocks/                      # Mock implementations
│   ├── data_providers.py       # Mock data providers
│   ├── ml_models.py           # Mock ML models
│   └── databases.py           # Mock databases and caches
├── fixtures/                   # Test data generators
│   └── sample_data.py         # Sample market data
├── helpers/                    # Test utilities
│   └── api_helpers.py         # API testing helpers
├── performance/                # Performance benchmarks
│   └── test_benchmarks.py     # Speed and memory tests
├── load/                       # Load testing
│   └── api_load_test.py       # API load tests with Locust
├── property/                   # Property-based tests
│   └── test_properties.py     # Hypothesis tests
└── test_*.py                  # Unit and integration tests
```

## Test Categories

### Unit Tests
Test individual components in isolation.

```bash
# Run unit tests only
pytest tests/ -v -m "unit"

# Run specific component tests
pytest tests/test_config.py -v
pytest tests/test_indicators.py -v
pytest tests/test_models.py -v
```

### Integration Tests
Test multiple components working together.

```bash
# Run integration tests
pytest tests/ -v -m "integration"

# Run end-to-end tests
pytest tests/test_e2e.py -v
pytest tests/test_integration.py -v
```

### API Tests
Test REST API endpoints.

```bash
# Run API tests
pytest tests/ -v -m "api"

# Run with API server
pytest tests/test_auth.py -v
pytest tests/test_api_*.py -v
```

### Performance Tests
Benchmark critical paths and measure performance.

```bash
# Run performance benchmarks
pytest tests/performance/ -v --benchmark-only

# Run with detailed stats
pytest tests/performance/ -v --benchmark-verbose
```

### Property-Based Tests
Test with automatically generated inputs using Hypothesis.

```bash
# Run property-based tests
pytest tests/property/ -v

# Run with more examples
pytest tests/property/ -v --hypothesis-show-statistics
```

### Load Tests
Test API under load with simulated users.

```bash
# Run load tests with Locust
locust -f tests/load/api_load_test.py --host=http://localhost:8000

# Run headless (automated)
locust -f tests/load/api_load_test.py --headless --users 50 --spawn-rate 10 --run-time 60s
```

## Running Tests

### Quick Test Run
```bash
# Fast tests only (skip slow tests)
pytest tests/ -v -m "not slow"

# With parallel execution
pytest tests/ -v -n auto
```

### Comprehensive Test Run
```bash
# All tests with coverage
pytest tests/ -v --cov=ara --cov=meridianalgo --cov-report=html --cov-report=term

# Generate coverage report
pytest tests/ --cov=ara --cov-report=html
open htmlcov/index.html  # View coverage report
```

### Specific Test Scenarios
```bash
# Test specific asset type
pytest tests/ -v -k "crypto"
pytest tests/ -v -k "forex"

# Test specific feature
pytest tests/ -v -k "sentiment"
pytest tests/ -v -k "backtest"
pytest tests/ -v -k "portfolio"

# Test with specific marker
pytest tests/ -v -m "slow"
pytest tests/ -v -m "requires_network"
```

### CI/CD Testing
```bash
# Run tests as in CI
pytest tests/ -v --maxfail=10 --tb=short

# Quick validation before commit
pytest tests/ -v -m "not slow" --maxfail=5
```

## Test Configuration

### pytest.ini
Configuration for test discovery, markers, and options.

### conftest.py
Shared fixtures available to all tests:
- `mock_data_provider` - Mock data provider
- `mock_ml_model` - Mock ML model
- `sample_stock_data` - Sample stock data
- `sample_crypto_data` - Sample crypto data
- `api_client` - Test API client
- `temp_dir` - Temporary directory

## Writing Tests

### Unit Test Example
```python
import pytest
from ara.features.calculator import FeatureCalculator

@pytest.mark.unit
def test_feature_calculation(sample_stock_data):
    """Test feature calculation."""
    calculator = FeatureCalculator()
    features = calculator.calculate(sample_stock_data)
    
    assert 'sma_20' in features
    assert len(features) == len(sample_stock_data)
```

### Integration Test Example
```python
import pytest

@pytest.mark.integration
@pytest.mark.asyncio
async def test_prediction_workflow(mock_data_provider, mock_ml_model):
    """Test complete prediction workflow."""
    from ara.prediction import PredictionEngine
    
    engine = PredictionEngine(
        data_provider=mock_data_provider,
        model=mock_ml_model
    )
    
    result = await engine.predict("AAPL", days=5)
    
    assert result.symbol == "AAPL"
    assert len(result.predictions) == 5
    assert result.confidence > 0
```

### API Test Example
```python
import pytest

@pytest.mark.api
def test_prediction_endpoint(api_client, auth_headers):
    """Test prediction API endpoint."""
    response = api_client.post(
        "/api/v1/predict",
        json={"symbol": "AAPL", "days": 5},
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
```

### Property-Based Test Example
```python
from hypothesis import given, strategies as st

@pytest.mark.property
@given(prices=st.lists(st.floats(min_value=1, max_value=1000), min_size=10))
def test_price_validation(prices):
    """Test price validation with random inputs."""
    from ara.data.validation import validate_prices
    
    result = validate_prices(prices)
    assert all(p > 0 for p in result)
```

## Test Requirements

Install test dependencies:

```bash
# Core testing
pip install pytest pytest-cov pytest-asyncio pytest-xdist pytest-timeout

# Performance testing
pip install pytest-benchmark

# Property-based testing
pip install hypothesis

# Load testing
pip install locust

# Code quality
pip install black flake8 mypy pylint

# Security scanning
pip install bandit safety
```

Or install all at once:
```bash
pip install -r requirements-dev.txt
```

## Coverage Goals

- **Overall Coverage**: 80%+
- **Core Components**: 90%+
- **API Endpoints**: 95%+
- **Critical Paths**: 100%

Check current coverage:
```bash
pytest tests/ --cov=ara --cov-report=term-missing
```

## Continuous Integration

Tests run automatically on:
- Every push to main/develop
- Every pull request
- Nightly (comprehensive suite)
- Weekly (full backtest validation)

See `.github/workflows/` for CI/CD configuration.

## Troubleshooting

### Tests Failing
```bash
# Run with verbose output
pytest tests/ -vv

# Run with full traceback
pytest tests/ --tb=long

# Run single test with debugging
pytest tests/test_file.py::test_name -vv -s
```

### Slow Tests
```bash
# Profile test execution
pytest tests/ --durations=10

# Skip slow tests
pytest tests/ -m "not slow"
```

### Import Errors
```bash
# Ensure ARA is installed
pip install -e .

# Check Python path
python -c "import ara; print(ara.__file__)"
```

## Best Practices

1. **Use fixtures** for common setup
2. **Mock external dependencies** (APIs, databases)
3. **Test edge cases** and error conditions
4. **Keep tests fast** (< 1 second per test)
5. **Use descriptive names** for tests
6. **Add docstrings** explaining what is tested
7. **Test one thing** per test function
8. **Use markers** to categorize tests
9. **Avoid test interdependencies**
10. **Clean up** after tests (use fixtures)

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure tests pass locally
3. Check coverage doesn't decrease
4. Add appropriate markers
5. Update this README if needed

**Maintained by**: ARA AI Team  
**Last Updated**: November 15, 2025
