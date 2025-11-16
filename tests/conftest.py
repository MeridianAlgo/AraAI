"""
Pytest configuration and shared fixtures for ARA AI test suite.

This module provides:
- Mock data providers for testing
- Sample datasets for different scenarios
- Mock ML models for fast testing
- Test database fixtures
- API test client helpers
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import tempfile
import shutil

# Mock implementations
from tests.mocks.data_providers import MockDataProvider, MockCryptoProvider
from tests.mocks.ml_models import MockMLModel, MockTransformer, MockEnsemble
from tests.mocks.databases import MockDatabase, MockCache
from tests.fixtures.sample_data import (
    generate_stock_data,
    generate_crypto_data,
    generate_forex_data,
    generate_features_data
)


# ============================================================================
# Session-scoped fixtures (created once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp(prefix="ara_test_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_stock_data():
    """Generate sample stock market data."""
    return generate_stock_data(
        symbol="AAPL",
        days=252,  # 1 year of trading days
        start_price=150.0
    )


@pytest.fixture(scope="session")
def sample_crypto_data():
    """Generate sample cryptocurrency data."""
    return generate_crypto_data(
        symbol="BTC-USD",
        days=365,  # 1 year
        start_price=40000.0
    )


@pytest.fixture(scope="session")
def sample_forex_data():
    """Generate sample forex data."""
    return generate_forex_data(
        symbol="EURUSD",
        days=252,
        start_price=1.10
    )


# ============================================================================
# Function-scoped fixtures (created for each test)
# ============================================================================

@pytest.fixture
def mock_data_provider(sample_stock_data):
    """Mock data provider for testing."""
    provider = MockDataProvider()
    provider.set_data("AAPL", sample_stock_data)
    return provider


@pytest.fixture
def mock_crypto_provider(sample_crypto_data):
    """Mock cryptocurrency data provider."""
    provider = MockCryptoProvider()
    provider.set_data("BTC-USD", sample_crypto_data)
    return provider


@pytest.fixture
def mock_ml_model():
    """Mock ML model for fast testing."""
    return MockMLModel(accuracy=0.85)


@pytest.fixture
def mock_transformer():
    """Mock Transformer model."""
    return MockTransformer(accuracy=0.88)


@pytest.fixture
def mock_ensemble():
    """Mock ensemble model."""
    return MockEnsemble(num_models=5, accuracy=0.90)


@pytest.fixture
def mock_database(temp_dir):
    """Mock database for testing."""
    db_path = temp_dir / "test.db"
    return MockDatabase(db_path)


@pytest.fixture
def mock_cache():
    """Mock cache for testing."""
    return MockCache()


@pytest.fixture
def sample_features():
    """Generate sample feature data."""
    return generate_features_data(days=100)


# ============================================================================
# API Testing fixtures
# ============================================================================

@pytest.fixture
def api_client():
    """Create a test client for API testing."""
    from fastapi.testclient import TestClient
    from ara.api.app import app
    
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Generate authentication headers for API testing."""
    return {
        "Authorization": "Bearer test_token_12345",
        "Content-Type": "application/json"
    }


@pytest.fixture
def test_api_key():
    """Generate a test API key."""
    return "test_api_key_abcdef123456"


# ============================================================================
# Configuration fixtures
# ============================================================================

@pytest.fixture
def test_config(temp_dir):
    """Create test configuration."""
    return {
        "data": {
            "cache_dir": str(temp_dir / "cache"),
            "models_dir": str(temp_dir / "models"),
        },
        "api": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": True
        },
        "models": {
            "ensemble_size": 5,
            "training_epochs": 10
        },
        "testing": True
    }


# ============================================================================
# Parametrized fixtures for multiple scenarios
# ============================================================================

@pytest.fixture(params=["stock", "crypto", "forex"])
def asset_type(request):
    """Parametrized fixture for different asset types."""
    return request.param


@pytest.fixture(params=[5, 10, 30])
def prediction_days(request):
    """Parametrized fixture for different prediction horizons."""
    return request.param


@pytest.fixture(params=["bull", "bear", "sideways", "high_volatility"])
def market_regime(request):
    """Parametrized fixture for different market regimes."""
    return request.param


# ============================================================================
# Cleanup fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Cleanup code here if needed
    pass


# ============================================================================
# Markers and test configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for multiple components"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmark tests"
    )
    config.addinivalue_line(
        "markers", "api: API endpoint tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that take more than 5 seconds"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file names
        if "test_api" in item.nodeid:
            item.add_marker(pytest.mark.api)
        if "test_e2e" in item.nodeid or "test_end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.performance)
