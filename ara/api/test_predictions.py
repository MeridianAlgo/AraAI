"""
Tests for prediction endpoints
"""

import pytest
from fastapi.testclient import TestClient
from ara.api.app import create_app
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_predict_stock_mocked(client):
    """Test stock prediction with mocked provider and model"""
    # Mock data fetch
    mock_df = pd.DataFrame(
        {
            "open": np.random.rand(100) + 100,
            "high": np.random.rand(100) + 101,
            "low": np.random.rand(100) + 99,
            "close": np.random.rand(100) + 100,
            "volume": np.random.rand(100) * 1000000,
        },
        index=pd.date_range("2023-01-01", periods=100),
    )

    with patch("ara.data.stock_provider.YahooFinanceProvider.fetch_historical") as mock_fetch:
        mock_fetch.return_value = mock_df

        # Test endpoint
        response = client.post(
            "/api/v1/predict", json={"symbol": "AAPL", "days": 5, "asset_type": "stock"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert len(data["predictions"]) == 5
        assert "confidence" in data
        assert "regime" in data


def test_predict_forex_mocked(client):
    """Test forex prediction with mocked provider"""
    mock_df = pd.DataFrame(
        {
            "open": np.random.rand(100) + 1,
            "high": np.random.rand(100) + 1.1,
            "low": np.random.rand(100) + 0.9,
            "close": np.random.rand(100) + 1,
            "volume": np.random.rand(100) * 1000,
        },
        index=pd.date_range("2023-01-01", periods=100),
    )

    with patch("ara.data.stock_provider.YahooFinanceProvider.fetch_historical") as mock_fetch:
        mock_fetch.return_value = mock_df

        response = client.post(
            "/api/v1/predict", json={"symbol": "EURUSD", "days": 3, "asset_type": "forex"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "EURUSD"
        assert data["asset_type"] == "forex"
