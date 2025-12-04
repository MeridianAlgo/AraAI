# Workflow and Testing Guide

This document provides comprehensive testing workflows and CI/CD guidelines for ARA AI.

## Table of Contents

- [Testing Overview](#testing-overview)
- [Local Testing](#local-testing)
- [CI/CD Workflows](#cicd-workflows)
- [Test Categories](#test-categories)
- [Performance Testing](#performance-testing)
- [Integration Testing](#integration-testing)

## Testing Overview

ARA AI uses a comprehensive testing strategy to ensure reliability and correctness:

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical operations
- **Security Tests**: Validate security features
- **End-to-End Tests**: Test complete workflows

### Test Statistics

- Total test modules: 180+
- Test coverage target: 80%+
- Average test execution time: < 5 minutes

## Local Testing

### Running All Tests

```bash
# Run complete test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=ara --cov=meridianalgo --cov-report=html

# Run with detailed output
pytest tests/ -v --tb=long

# Run specific test categories
pytest tests/test_unit/ -v          # Unit tests only
pytest tests/test_integration/ -v  # Integration tests only
pytest tests/test_security.py -v   # Security tests only
```

### Running Specific Tests

```bash
# Run tests in a specific file
pytest tests/test_features.py -v

# Run a specific test function
pytest tests/test_features.py::test_rsi_calculation -v

# Run tests matching a pattern
pytest tests/ -k "prediction" -v

# Run tests with specific markers
pytest -m "slow" -v           # Run slow tests
pytest -m "not slow" -v       # Skip slow tests
```

### Test Output Options

```bash
# Quiet mode (minimal output)
pytest tests/ -q

# Verbose mode (detailed output)
pytest tests/ -v

# Show local variables on failures
pytest tests/ -l

# Stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf

# Show print statements
pytest tests/ -s
```

## CI/CD Workflows

### GitHub Actions Setup

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8 mypy bandit safety
    
    - name: Code formatting check
      run: |
        black --check ara/ meridianalgo/
    
    - name: Lint with flake8
      run: |
        flake8 ara/ meridianalgo/ --max-line-length=100 --count --show-source --statistics
    
    - name: Type checking
      run: |
        mypy ara/ meridianalgo/ --ignore-missing-imports
    
    - name: Security scan
      run: |
        bandit -r ara/ meridianalgo/
        safety check
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=ara --cov=meridianalgo --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
```

### Model Training Workflow

Create `.github/workflows/train.yml`:

```yaml
name: Train Models

on:
  workflow_dispatch:
    inputs:
      index:
        description: 'Stock index to train'
        required: true
        default: 'quick'
        type: choice
        options:
          - quick
          - sp500
          - nasdaq
          - all
      epochs:
        description: 'Number of epochs'
        required: false
        default: '1000'
      strict:
        description: 'Enable strict mode'
        required: false
        default: false
        type: boolean

jobs:
  train:
    runs-on: ubuntu-latest
    timeout-minutes: 360
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Train models
      run: |
        python scripts/train_all.py \
          --index ${{ github.event.inputs.index }} \
          --epochs ${{ github.event.inputs.epochs }} \
          ${{ github.event.inputs.strict && '--strict' || '' }}
    
    - name: Upload trained models
      uses: actions/upload-artifact@v3
      with:
        name: trained-models-${{ github.event.inputs.index }}
        path: models/*.pt
        retention-days: 30
    
    - name: Create training report
      run: |
        echo "Training completed for index: ${{ github.event.inputs.index }}" > training-report.txt
        echo "Epochs: ${{ github.event.inputs.epochs }}" >> training-report.txt
        echo "Strict mode: ${{ github.event.inputs.strict }}" >> training-report.txt
    
    - name: Upload training report
      uses: actions/upload-artifact@v3
      with:
        name: training-report
        path: training-report.txt
```

### Security Scanning Workflow

Create `.github/workflows/security.yml`:

```yaml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install security tools
      run: |
        pip install bandit safety
    
    - name: Run Bandit
      run: |
        bandit -r ara/ meridianalgo/ -f json -o bandit-report.json
    
    - name: Run Safety
      run: |
        safety check --json > safety-report.json
      continue-on-error: true
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json
```

### Release Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install build twine
    
    - name: Run tests
      run: |
        pytest tests/ -v
    
    - name: Build package
      run: |
        python -m build
    
    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: |
        twine upload dist/*
```

## Test Categories

### Unit Tests

Test individual components in isolation:

```python
# tests/test_unit/test_indicators.py
import pytest
from ara.features.calculator import IndicatorCalculator

def test_rsi_calculation():
    """Test RSI indicator calculation."""
    calc = IndicatorCalculator()
    data = [100, 102, 101, 103, 105, 107, 106]
    
    result = calc.calculate_rsi(data, period=3)
    
    assert result is not None
    assert 0 <= result <= 100

def test_macd_calculation():
    """Test MACD indicator calculation."""
    calc = IndicatorCalculator()
    data = list(range(100, 200))
    
    macd, signal, histogram = calc.calculate_macd(data)
    
    assert macd is not None
    assert signal is not None
    assert histogram is not None
```

### Integration Tests

Test component interactions:

```python
# tests/test_integration/test_prediction_pipeline.py
import pytest
from meridianalgo.unified_ml import UnifiedStockML

def test_full_prediction_pipeline():
    """Test complete prediction pipeline."""
    ml = UnifiedStockML()
    
    # This tests: data fetch -> feature calc -> model predict
    result = ml.predict('AAPL', days=5)
    
    assert result is not None
    assert 'predictions' in result
    assert len(result['predictions']) == 5
    assert all('price' in p for p in result['predictions'])
```

### Performance Tests

Benchmark critical operations:

```python
# tests/test_performance/test_benchmarks.py
import pytest
import time
from meridianalgo.unified_ml import UnifiedStockML

@pytest.mark.benchmark
def test_prediction_performance():
    """Ensure predictions complete within time limit."""
    ml = UnifiedStockML()
    
    start = time.time()
    result = ml.predict('AAPL', days=5)
    duration = time.time() - start
    
    assert duration < 2.0, f"Prediction took {duration}s, should be < 2s"

@pytest.mark.benchmark
def test_batch_prediction_performance():
    """Test batch prediction performance."""
    ml = UnifiedStockML()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    start = time.time()
    for symbol in symbols:
        ml.predict(symbol, days=5)
    duration = time.time() - start
    
    assert duration < 10.0, f"Batch predictions took {duration}s"
```

### Security Tests

Validate security features:

```python
# tests/test_security.py
import pytest
from ara.security import InputSanitizer, XSSProtection, SQLProtection

def test_sql_injection_prevention():
    """Test SQL injection prevention."""
    malicious_input = "'; DROP TABLE users; --"
    
    # Should detect and reject
    with pytest.raises(ValueError):
        InputSanitizer.sanitize_string(malicious_input, allow_sql=False)

def test_xss_prevention():
    """Test XSS prevention."""
    malicious_html = "<script>alert('xss')</script><p>Content</p>"
    
    clean = XSSProtection.sanitize_html(malicious_html)
    
    assert "<script>" not in clean
    assert "alert" not in clean
    assert "<p>Content</p>" in clean
```

## Performance Testing

### Load Testing

```bash
# Install locust
pip install locust

# Create locustfile.py
cat > locustfile.py << 'EOF'
from locust import HttpUser, task, between

class AraAIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def predict(self):
        self.client.post("/api/v1/predict", json={
            "symbol": "AAPL",
            "days": 5
        })
    
    @task(1)
    def health_check(self):
        self.client.get("/health")
EOF

# Run load test
locust -f locustfile.py --host=http://localhost:8000
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile a script
python -m memory_profiler scripts/ara.py AAPL --days 5

# Profile a function
@profile
def my_function():
    # Function code
    pass
```

### CPU Profiling

```bash
# Install profiling tools
pip install py-spy

# Profile running process
py-spy record -o profile.svg -- python scripts/ara.py AAPL --days 5

# Interactive profiling
py-spy top -- python scripts/ara.py AAPL --days 5
```

## Integration Testing

### API Integration Tests

```python
# tests/test_integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from ara.api.app import app

client = TestClient(app)

def test_predict_endpoint():
    """Test prediction API endpoint."""
    response = client.post("/api/v1/predict", json={
        "symbol": "AAPL",
        "days": 5
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 5

def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Database Integration Tests

```python
# tests/test_integration/test_database.py
import pytest
from ara.data.database import DatabaseManager

@pytest.fixture
def db():
    """Create test database."""
    db = DatabaseManager(test_mode=True)
    yield db
    db.cleanup()

def test_save_and_retrieve_prediction(db):
    """Test saving and retrieving predictions."""
    prediction = {
        "symbol": "AAPL",
        "predictions": [{"day": 1, "price": 175.0}]
    }
    
    # Save
    pred_id = db.save_prediction(prediction)
    
    # Retrieve
    retrieved = db.get_prediction(pred_id)
    
    assert retrieved["symbol"] == "AAPL"
    assert len(retrieved["predictions"]) == 1
```

## Test Configuration

### pytest.ini

Create `pytest.ini` in project root:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    security: marks tests as security tests
    performance: marks tests as performance tests
    benchmark: marks tests as benchmarks
```

### Coverage Configuration

Create `.coveragerc`:

```ini
[run]
source = ara,meridianalgo
omit = 
    */tests/*
    */test_*.py
    */__pycache__/*
    */venv/*

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
```

## Continuous Monitoring

### Test Metrics to Track

- Test pass rate
- Code coverage percentage
- Test execution time
- Failed test trends
- Security scan results
- Performance benchmarks

### Monitoring Tools

- **Codecov**: Code coverage tracking
- **SonarQube**: Code quality analysis
- **Snyk**: Dependency vulnerability scanning
- **GitHub Actions**: CI/CD workflow execution

## Best Practices

### Writing Tests

1. **Use descriptive names**: `test_rsi_calculation_with_valid_data`
2. **Follow AAA pattern**: Arrange, Act, Assert
3. **One assertion per test** (when possible)
4. **Use fixtures** for setup/teardown
5. **Mock external dependencies**
6. **Test edge cases** and error conditions

### Test Organization

```
tests/
├── conftest.py           # Shared fixtures
├── test_unit/            # Unit tests
│   ├── test_features.py
│   ├── test_models.py
│   └── test_utils.py
├── test_integration/     # Integration tests
│   ├── test_api.py
│   ├── test_database.py
│   └── test_pipeline.py
├── test_security.py      # Security tests
└── test_performance/     # Performance tests
    └── test_benchmarks.py
```

### CI/CD Best Practices

1. **Run tests on every push**
2. **Test multiple Python versions**
3. **Cache dependencies** for faster builds
4. **Fail fast** on critical errors
5. **Generate and track** coverage reports
6. **Run security scans** regularly
7. **Automate releases** with tags

## Troubleshooting

### Common Test Issues

**Tests failing locally but passing in CI:**
- Check Python version differences
- Verify environment variables
- Check for local file dependencies

**Slow test execution:**
- Use markers to skip slow tests during development
- Parallelize tests with pytest-xdist
- Mock external API calls

**Flaky tests:**
- Add retry logic for network-dependent tests
- Use fixtures for consistent test data
- Avoid time-dependent assertions

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Locust Documentation](https://docs.locust.io/)

---

**Last Updated**: 2025-11-25
