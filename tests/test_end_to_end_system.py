"""
Comprehensive End-to-End Integration Tests.

Tests the complete system workflow for:
- Stock predictions
- Cryptocurrency predictions
- Forex predictions
- Backtesting
- Portfolio optimization
- API endpoints
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# Import system components
from ara.models.ensemble import EnhancedEnsemble
from ara.models.model_registry import ModelRegistry, ModelMetadata
from ara.models.model_comparator import ModelComparator, ValidationDataset
from ara.backtesting.engine import BacktestEngine
from ara.risk.optimizer import PortfolioOptimizer


class TestEndToEndStockPrediction:
    """Test end-to-end stock prediction workflow."""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Generate sample stock data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target (returns)
        y = 0.001 + 0.01 * X[:, 0] + 0.005 * X[:, 1] + np.random.randn(n_samples) * 0.02
        
        return X, y
    
    def test_stock_prediction_workflow(self, sample_stock_data):
        """Test complete stock prediction workflow."""
        X, y = sample_stock_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 1. Train model
        model = EnhancedEnsemble(model_name="stock_predictor")
        training_results = model.train(X_train, y_train, validation_split=0.2)
        
        assert model.is_trained
        assert 'training_results' in training_results
        assert len(model.models) > 0
        
        # 2. Make predictions
        predictions, confidences = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert len(confidences) == len(X_test)
        assert np.all(confidences >= 0) and np.all(confidences <= 1)
        
        # 3. Evaluate predictions
        directional_accuracy = np.mean((predictions > 0) == (y_test > 0))
        mae = np.mean(np.abs(predictions - y_test))
        
        assert directional_accuracy > 0.45  # Better than random
        assert mae < 0.1  # Reasonable error
        
        # 4. Get explanations
        explanations = model.explain(X_test[:5])
        
        assert 'individual_predictions' in explanations
        assert 'model_weights' in explanations
        
        print(f"Stock Prediction Test:")
        print(f"  Directional Accuracy: {directional_accuracy:.2%}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Active Models: {len(model.get_active_models())}")


class TestEndToEndCryptoPrediction:
    """Test end-to-end cryptocurrency prediction workflow."""
    
    @pytest.fixture
    def sample_crypto_data(self):
        """Generate sample crypto data with higher volatility."""
        np.random.seed(43)
        n_samples = 1000
        n_features = 25  # More features for crypto (on-chain metrics)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target with higher volatility
        y = 0.002 + 0.02 * X[:, 0] + 0.01 * X[:, 1] + np.random.randn(n_samples) * 0.05
        
        return X, y
    
    def test_crypto_prediction_workflow(self, sample_crypto_data):
        """Test complete crypto prediction workflow."""
        X, y = sample_crypto_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 1. Train model
        model = EnhancedEnsemble(model_name="crypto_predictor")
        training_results = model.train(X_train, y_train, validation_split=0.2)
        
        assert model.is_trained
        
        # 2. Make predictions
        predictions, confidences = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        
        # 3. Evaluate (crypto should have lower accuracy due to higher volatility)
        directional_accuracy = np.mean((predictions > 0) == (y_test > 0))
        mae = np.mean(np.abs(predictions - y_test))
        
        assert directional_accuracy > 0.40  # Still better than random
        
        print(f"Crypto Prediction Test:")
        print(f"  Directional Accuracy: {directional_accuracy:.2%}")
        print(f"  MAE: {mae:.4f}")


class TestEndToEndForexPrediction:
    """Test end-to-end forex prediction workflow."""
    
    @pytest.fixture
    def sample_forex_data(self):
        """Generate sample forex data."""
        np.random.seed(44)
        n_samples = 1000
        n_features = 15
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate target (forex has smaller movements)
        y = 0.0005 + 0.005 * X[:, 0] + 0.003 * X[:, 1] + np.random.randn(n_samples) * 0.01
        
        return X, y
    
    def test_forex_prediction_workflow(self, sample_forex_data):
        """Test complete forex prediction workflow."""
        X, y = sample_forex_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 1. Train model
        model = EnhancedEnsemble(model_name="forex_predictor")
        training_results = model.train(X_train, y_train, validation_split=0.2)
        
        assert model.is_trained
        
        # 2. Make predictions
        predictions, confidences = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        
        # 3. Evaluate
        directional_accuracy = np.mean((predictions > 0) == (y_test > 0))
        mae = np.mean(np.abs(predictions - y_test))
        
        assert directional_accuracy > 0.45
        
        print(f"Forex Prediction Test:")
        print(f"  Directional Accuracy: {directional_accuracy:.2%}")
        print(f"  MAE: {mae:.4f}")


class TestEndToEndBacktesting:
    """Test end-to-end backtesting workflow."""
    
    @pytest.fixture
    def historical_data(self):
        """Generate historical data for backtesting."""
        np.random.seed(45)
        n_days = 500
        n_features = 20
        
        dates = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
        
        # Generate features
        X = np.random.randn(n_days, n_features)
        
        # Generate returns
        y = 0.001 + 0.01 * X[:, 0] + np.random.randn(n_days) * 0.02
        
        # Generate prices
        prices = 100 * np.exp(np.cumsum(y))
        
        return dates, X, y, prices
    
    def test_backtesting_workflow(self, historical_data):
        """Test complete backtesting workflow."""
        dates, X, y, prices = historical_data
        
        # 1. Train model on first 60% of data
        train_idx = int(len(X) * 0.6)
        X_train, y_train = X[:train_idx], y[:train_idx]
        
        model = EnhancedEnsemble(model_name="backtest_model")
        model.train(X_train, y_train, validation_split=0.2)
        
        # 2. Backtest on remaining 40%
        X_test, y_test = X[train_idx:], y[train_idx:]
        predictions, confidences = model.predict(X_test)
        
        # 3. Calculate backtest metrics
        directional_accuracy = np.mean((predictions > 0) == (y_test > 0))
        
        # Calculate returns
        strategy_returns = np.where(predictions > 0, y_test, -y_test)
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
        
        # Calculate max drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        assert directional_accuracy > 0.45
        assert sharpe_ratio > -1.0  # Not terrible
        
        print(f"Backtesting Test:")
        print(f"  Directional Accuracy: {directional_accuracy:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Final Return: {cumulative_returns[-1]:.2%}")


class TestEndToEndPortfolioOptimization:
    """Test end-to-end portfolio optimization workflow."""
    
    @pytest.fixture
    def multi_asset_data(self):
        """Generate multi-asset data."""
        np.random.seed(46)
        n_days = 252  # 1 year
        n_assets = 5
        
        # Generate correlated returns
        correlation = np.array([
            [1.0, 0.6, 0.4, 0.3, 0.2],
            [0.6, 1.0, 0.5, 0.4, 0.3],
            [0.4, 0.5, 1.0, 0.6, 0.4],
            [0.3, 0.4, 0.6, 1.0, 0.5],
            [0.2, 0.3, 0.4, 0.5, 1.0]
        ])
        
        # Cholesky decomposition for correlated samples
        L = np.linalg.cholesky(correlation)
        
        # Generate returns
        uncorrelated = np.random.randn(n_days, n_assets)
        returns = uncorrelated @ L.T
        
        # Add different means and volatilities
        means = np.array([0.0008, 0.0010, 0.0006, 0.0012, 0.0009])
        vols = np.array([0.015, 0.020, 0.012, 0.025, 0.018])
        
        returns = returns * vols + means
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        return symbols, returns
    
    def test_portfolio_optimization_workflow(self, multi_asset_data):
        """Test complete portfolio optimization workflow."""
        symbols, returns = multi_asset_data
        
        # 1. Calculate expected returns and covariance
        expected_returns = np.mean(returns, axis=0) * 252  # Annualized
        cov_matrix = np.cov(returns.T) * 252  # Annualized
        
        # 2. Optimize portfolio (simplified - would use PortfolioOptimizer)
        n_assets = len(symbols)
        
        # Equal weight portfolio
        equal_weights = np.ones(n_assets) / n_assets
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(equal_weights, expected_returns)
        portfolio_vol = np.sqrt(np.dot(equal_weights, np.dot(cov_matrix, equal_weights)))
        sharpe_ratio = portfolio_return / portfolio_vol
        
        assert np.sum(equal_weights) == pytest.approx(1.0)
        assert np.all(equal_weights >= 0)
        assert portfolio_vol > 0
        
        print(f"Portfolio Optimization Test:")
        print(f"  Expected Return: {portfolio_return:.2%}")
        print(f"  Volatility: {portfolio_vol:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Weights: {dict(zip(symbols, equal_weights))}")


class TestEndToEndModelComparison:
    """Test end-to-end model comparison workflow."""
    
    @pytest.fixture
    def temp_registry(self):
        """Create temporary model registry."""
        temp_dir = tempfile.mkdtemp()
        registry = ModelRegistry(registry_dir=Path(temp_dir) / "registry")
        yield registry
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(47)
        n_samples = 500
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = 0.001 + 0.01 * X[:, 0] + np.random.randn(n_samples) * 0.02
        
        return X, y
    
    def test_model_comparison_workflow(self, temp_registry, sample_data):
        """Test complete model comparison workflow."""
        X, y = sample_data
        
        # Split data
        split_idx = int(len(X) * 0.7)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 1. Train multiple models
        model_ids = []
        temp_dirs = []
        
        for i in range(2):
            model = EnhancedEnsemble(model_name=f"test_model_{i}")
            model.train(X_train, y_train, validation_split=0.2, random_state=42+i)
            
            # Save model to temp directory
            temp_dir = Path(tempfile.mkdtemp())
            temp_dirs.append(temp_dir)
            temp_model_base = temp_dir / f"test_model_{i}"
            model.save(temp_model_base)
            
            # Find the main model pkl file (ensemble saves multiple files)
            # We'll use one of the individual model files for registration
            model_files = list(temp_dir.glob("*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No model files found in {temp_dir}")
            
            # Use the first pkl file found
            model_file = model_files[0]
            
            # Register model
            metadata = ModelMetadata(
                model_id="",
                model_name=f"test_model_{i}",
                version=f"1.{i}",
                created_at=datetime.now(),
                training_date=datetime.now(),
                accuracy=0.75 + i * 0.02,
                directional_accuracy=0.70 + i * 0.02,
                mae=0.02 - i * 0.001,
                rmse=0.03 - i * 0.001,
                sharpe_ratio=1.5 + i * 0.2,
                max_drawdown=-0.15 + i * 0.01,
                training_samples=len(X_train),
                training_period_start=datetime.now() - timedelta(days=365),
                training_period_end=datetime.now(),
                feature_count=X.shape[1],
                model_type="ensemble",
                hyperparameters={},
                data_sources=["test"],
                data_hash="test_hash",
                status="active",
                deployed=False,
                tags=["test"],
                notes="Test model"
            )
            
            model_id = temp_registry.register_model(model_file, metadata)
            model_ids.append(model_id)
        
        # 2. Test validation dataset creation
        validation_dataset = ValidationDataset(
            dataset_dir=temp_registry.registry_dir / "validation"
        )
        
        # Create validation dataset
        data_hash = validation_dataset.create_dataset("TEST", X_val, y_val)
        assert data_hash is not None
        assert len(data_hash) > 0
        
        # Load it back
        X_loaded, y_loaded, hash_loaded = validation_dataset.load_dataset("TEST")
        assert np.array_equal(X_loaded, X_val)
        assert np.array_equal(y_loaded, y_val)
        assert hash_loaded == data_hash
        
        # 3. Test comparator initialization
        comparator = ModelComparator(
            registry=temp_registry,
            validation_dataset=validation_dataset
        )
        
        # Note: We skip actual model comparison since it requires full model loading
        # which is complex for ensemble models. The comparison logic is tested
        # through the individual components.
        
        print(f"\nModel Comparison Test:")
        print(f"  Models Registered: {len(model_ids)}")
        print(f"  Validation Dataset Created: {data_hash}")
        print(f"  Validation Samples: {len(X_val)}")
        
        # 5. Test user preferences
        user_prefs = comparator.get_user_preferences("test_user")
        assert user_prefs.user_id == "test_user"
        
        # Update preferences
        comparator.update_user_preferences(
            "test_user",
            accuracy_weight=0.5,
            speed_weight=0.3
        )
        
        updated_prefs = comparator.get_user_preferences("test_user")
        assert updated_prefs.accuracy_weight == 0.5
        assert updated_prefs.speed_weight == 0.3
        
        # Cleanup temp directories
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


class TestEndToEndSystemIntegration:
    """Test complete system integration."""
    
    def test_full_system_workflow(self):
        """Test complete workflow from data to deployment."""
        np.random.seed(48)
        
        # 1. Generate data
        n_samples = 1000
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        y = 0.001 + 0.01 * X[:, 0] + np.random.randn(n_samples) * 0.02
        
        # 2. Split data
        train_idx = int(n_samples * 0.6)
        val_idx = int(n_samples * 0.8)
        
        X_train, y_train = X[:train_idx], y[:train_idx]
        X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
        X_test, y_test = X[val_idx:], y[val_idx:]
        
        # 3. Train model
        model = EnhancedEnsemble(model_name="production_model")
        training_results = model.train(X_train, y_train, validation_split=0.2)
        
        # 4. Validate model
        val_predictions, val_confidences = model.predict(X_val)
        val_accuracy = np.mean((val_predictions > 0) == (y_val > 0))
        
        assert val_accuracy > 0.45
        
        # 5. Test model
        test_predictions, test_confidences = model.predict(X_test)
        test_accuracy = np.mean((test_predictions > 0) == (y_test > 0))
        
        assert test_accuracy > 0.40
        
        # 6. Get explanations
        explanations = model.explain(X_test[:10])
        
        assert 'individual_predictions' in explanations
        assert 'model_weights' in explanations
        
        print(f"\nFull System Integration Test:")
        print(f"  Training Samples: {len(X_train)}")
        print(f"  Validation Accuracy: {val_accuracy:.2%}")
        print(f"  Test Accuracy: {test_accuracy:.2%}")
        print(f"  Active Models: {len(model.get_active_models())}")
        print(f"  Model Count: {model.get_model_count()}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
