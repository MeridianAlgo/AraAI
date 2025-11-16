"""
Tests for core interfaces and base classes
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from ara.core.interfaces import IDataProvider, IMLModel, IFeatureEngine, AssetType
from ara.models.base_model import BaseModel
from ara.core.exceptions import ValidationError, ModelError


class MockDataProvider(IDataProvider):
    """Mock data provider for testing"""
    
    async def fetch_historical(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        # Return mock data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        return pd.DataFrame({
            'Date': dates,
            'Open': np.random.randn(100) + 100,
            'High': np.random.randn(100) + 101,
            'Low': np.random.randn(100) + 99,
            'Close': np.random.randn(100) + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
    
    async def fetch_realtime(self, symbol: str) -> dict:
        return {
            'symbol': symbol,
            'price': 100.0,
            'volume': 1000000,
            'timestamp': datetime.now()
        }
    
    async def stream_data(self, symbol: str, callback) -> None:
        pass
    
    def get_supported_symbols(self) -> list:
        return ['AAPL', 'MSFT', 'GOOGL']
    
    def get_provider_name(self) -> str:
        return "MockProvider"
    
    def get_asset_type(self) -> AssetType:
        return AssetType.STOCK


class MockModel(BaseModel):
    """Mock model for testing"""
    
    def __init__(self):
        super().__init__("mock_model")
        self.model = "mock"
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, **kwargs) -> dict:
        self.validate_input(X, y)
        self.is_trained = True
        self.training_date = datetime.now().isoformat()
        return {'loss': 0.1, 'accuracy': 0.9}
    
    def predict(self, X: np.ndarray) -> tuple:
        self.validate_input(X)
        predictions = np.random.randn(len(X))
        confidence = np.random.rand(len(X))
        return predictions, confidence
    
    def explain(self, X: np.ndarray) -> dict:
        return {'feature_importance': {'feature_1': 0.5, 'feature_2': 0.3}}


class MockFeatureEngine(IFeatureEngine):
    """Mock feature engine for testing"""
    
    def calculate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        features = data.copy()
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['rsi'] = 50.0
        return features
    
    def get_feature_names(self) -> list:
        return ['sma_20', 'rsi']
    
    def get_feature_importance(self) -> dict:
        return {'sma_20': 0.6, 'rsi': 0.4}
    
    def validate_data(self, data: pd.DataFrame) -> tuple:
        if 'Close' not in data.columns:
            return False, ['Missing Close column']
        return True, []


def test_data_provider_interface():
    """Test IDataProvider interface"""
    provider = MockDataProvider()
    
    # Test get_supported_symbols
    symbols = provider.get_supported_symbols()
    assert 'AAPL' in symbols
    
    # Test get_provider_name
    assert provider.get_provider_name() == "MockProvider"
    
    # Test get_asset_type
    assert provider.get_asset_type() == AssetType.STOCK
    
    # Note: async methods (fetch_historical, fetch_realtime, stream_data) 
    # will be tested in integration tests with actual implementations


def test_base_model_validation():
    """Test BaseModel input validation"""
    model = MockModel()
    
    # Valid input
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model.validate_input(X, y)
    
    # Invalid X (not numpy array)
    with pytest.raises(ValidationError):
        model.validate_input([1, 2, 3], y)
    
    # Invalid X (wrong dimensions)
    with pytest.raises(ValidationError):
        model.validate_input(np.array([1, 2, 3]), y)
    
    # Invalid X (contains NaN)
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValidationError):
        model.validate_input(X_nan, y)
    
    # Invalid y (wrong length)
    with pytest.raises(ValidationError):
        model.validate_input(X, np.random.randn(50))


def test_base_model_train_predict():
    """Test BaseModel train and predict"""
    model = MockModel()
    
    # Train model
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    metrics = model.train(X, y)
    
    assert model.is_trained
    assert 'accuracy' in metrics
    assert model.training_date is not None
    
    # Predict
    predictions, confidence = model.predict(X)
    assert len(predictions) == len(X)
    assert len(confidence) == len(X)
    
    # Explain
    explanations = model.explain(X)
    assert 'feature_importance' in explanations


def test_base_model_save_load(tmp_path):
    """Test BaseModel save and load"""
    model = MockModel()
    
    # Cannot save untrained model
    with pytest.raises(ModelError):
        model.save(tmp_path / "model")
    
    # Train model
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    model.train(X, y)
    
    # Save model
    model_path = tmp_path / "model"
    model.save(model_path)
    
    assert (tmp_path / "model.pkl").exists()
    assert (tmp_path / "model.json").exists()
    
    # Load model
    new_model = MockModel()
    new_model.load(model_path)
    
    assert new_model.is_trained
    assert new_model.model == "mock"


def test_base_model_info():
    """Test BaseModel get_model_info"""
    model = MockModel()
    
    info = model.get_model_info()
    assert info['model_name'] == 'mock_model'
    assert info['is_trained'] is False
    
    # Update metadata
    model.update_metadata('custom_key', 'custom_value')
    info = model.get_model_info()
    assert info['metadata']['custom_key'] == 'custom_value'


def test_feature_engine_interface():
    """Test IFeatureEngine interface"""
    engine = MockFeatureEngine()
    
    # Create sample data
    data = pd.DataFrame({
        'Date': pd.date_range(end=datetime.now(), periods=100, freq='D'),
        'Close': np.random.randn(100) + 100
    })
    
    # Calculate features
    features = engine.calculate_features(data)
    assert 'sma_20' in features.columns
    assert 'rsi' in features.columns
    
    # Get feature names
    names = engine.get_feature_names()
    assert 'sma_20' in names
    assert 'rsi' in names
    
    # Get feature importance
    importance = engine.get_feature_importance()
    assert importance['sma_20'] == 0.6
    
    # Validate data
    is_valid, issues = engine.validate_data(data)
    assert is_valid
    assert len(issues) == 0
    
    # Invalid data
    invalid_data = pd.DataFrame({'Open': [1, 2, 3]})
    is_valid, issues = engine.validate_data(invalid_data)
    assert not is_valid
    assert len(issues) > 0


def test_asset_type_enum():
    """Test AssetType enum"""
    assert AssetType.STOCK.value == "stock"
    assert AssetType.CRYPTO.value == "crypto"
    assert AssetType.FOREX.value == "forex"
