"""
Test the testing infrastructure itself.

Validates that fixtures, mocks, and test utilities work correctly.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.mark.unit
def test_mock_data_provider(mock_data_provider, sample_stock_data):
    """Test mock data provider fixture."""
    # Data should be pre-loaded
    assert "AAPL" in mock_data_provider.get_supported_symbols()
    
    # Should be able to fetch data
    import asyncio
    data = asyncio.run(mock_data_provider.fetch_historical("AAPL"))
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert "close" in data.columns


@pytest.mark.unit
def test_mock_ml_model(mock_ml_model):
    """Test mock ML model fixture."""
    X = np.random.randn(100, 50)
    y = np.random.randn(100)
    
    # Train model
    result = mock_ml_model.train(X, y)
    assert "accuracy" in result
    assert result["accuracy"] > 0
    
    # Make predictions
    predictions, confidences = mock_ml_model.predict(X)
    assert len(predictions) == len(X)
    assert len(confidences) == len(X)
    assert all(0 <= c <= 1 for c in confidences)


@pytest.mark.unit
def test_mock_cache(mock_cache):
    """Test mock cache fixture."""
    # Set and get
    mock_cache.set("test_key", "test_value")
    assert mock_cache.get("test_key") == "test_value"
    
    # Non-existent key
    assert mock_cache.get("nonexistent") is None
    
    # Delete
    mock_cache.delete("test_key")
    assert mock_cache.get("test_key") is None


@pytest.mark.unit
def test_sample_data_fixtures(sample_stock_data, sample_crypto_data, sample_forex_data):
    """Test sample data fixtures."""
    # Stock data
    assert isinstance(sample_stock_data, pd.DataFrame)
    assert len(sample_stock_data) == 252
    assert "close" in sample_stock_data.columns
    
    # Crypto data
    assert isinstance(sample_crypto_data, pd.DataFrame)
    assert len(sample_crypto_data) == 365
    
    # Forex data
    assert isinstance(sample_forex_data, pd.DataFrame)
    assert len(sample_forex_data) == 252


@pytest.mark.unit
def test_temp_dir_fixture(temp_dir):
    """Test temporary directory fixture."""
    from pathlib import Path
    
    assert isinstance(temp_dir, Path)
    assert temp_dir.exists()
    
    # Should be able to create files
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()


@pytest.mark.unit
def test_test_config_fixture(test_config):
    """Test configuration fixture."""
    assert isinstance(test_config, dict)
    assert "data" in test_config
    assert "api" in test_config
    assert test_config["testing"] is True


@pytest.mark.unit
@pytest.mark.parametrize("asset_type", ["stock", "crypto", "forex"])
def test_asset_type_parametrization(asset_type):
    """Test parametrized asset type fixture."""
    assert asset_type in ["stock", "crypto", "forex"]


@pytest.mark.unit
@pytest.mark.parametrize("days", [5, 10, 30])
def test_prediction_days_parametrization(days):
    """Test parametrized prediction days fixture."""
    assert days in [5, 10, 30]
    assert days > 0


@pytest.mark.unit
def test_mock_ensemble(mock_ensemble):
    """Test mock ensemble model."""
    X = np.random.randn(100, 50)
    y = np.random.randn(100)
    
    # Train ensemble
    result = mock_ensemble.train(X, y)
    assert "models_trained" in result
    assert result["models_trained"] > 0
    
    # Make predictions
    predictions, confidences = mock_ensemble.predict(X)
    assert len(predictions) == len(X)


@pytest.mark.unit
def test_mock_database(mock_database):
    """Test mock database fixture."""
    # Insert data
    record_id = mock_database.insert("test_table", {"name": "test", "value": 123})
    assert record_id > 0
    
    # Select data
    results = mock_database.select("test_table")
    assert len(results) == 1
    assert results[0]["name"] == "test"
    
    # Update data
    success = mock_database.update("test_table", record_id, {"value": 456})
    assert success
    
    # Verify update
    results = mock_database.select("test_table")
    assert results[0]["value"] == 456


@pytest.mark.unit
def test_sample_features_fixture(sample_features):
    """Test sample features fixture."""
    assert isinstance(sample_features, pd.DataFrame)
    assert len(sample_features) == 100
    assert sample_features.shape[1] == 50  # 50 features


@pytest.mark.integration
def test_fixtures_work_together(mock_data_provider, mock_ml_model, mock_cache):
    """Test that fixtures work together."""
    import asyncio
    
    # Fetch data
    data = asyncio.run(mock_data_provider.fetch_historical("AAPL"))
    
    # Cache it
    mock_cache.set("AAPL_data", data)
    
    # Retrieve from cache
    cached_data = mock_cache.get("AAPL_data")
    assert cached_data is not None
    
    # Use for model training
    X = np.random.randn(100, 50)
    y = np.random.randn(100)
    result = mock_ml_model.train(X, y)
    
    assert result["accuracy"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
