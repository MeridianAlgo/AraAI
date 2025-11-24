"""
Tests for Enhanced Ensemble System
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from ara.models.ensemble import EnhancedEnsemble
from ara.models.regime_adaptive import (
    RegimeAdaptiveWeighting, 
    MarketRegime, 
    detect_market_regime
)
from ara.models.adaptive_ensemble import AdaptiveEnsembleSystem


@pytest.fixture
def sample_data():
    """Generate sample training data"""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.randn(200) * 0.01  # Small returns
    prices = 100 + np.cumsum(y * 100)  # Price series
    return X, y, prices


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


class TestEnhancedEnsemble:
    """Test EnhancedEnsemble class"""
    
    def test_initialization(self):
        """Test ensemble initialization"""
        ensemble = EnhancedEnsemble("test_ensemble")
        
        assert ensemble.model_name == "test_ensemble"
        assert not ensemble.is_trained
        assert len(ensemble.model_weights) > 0
        assert sum(ensemble.model_weights.values()) == pytest.approx(1.0, abs=1e-6)
    
    def test_training(self, sample_data):
        """Test ensemble training"""
        X, y, _ = sample_data
        ensemble = EnhancedEnsemble("test_ensemble")
        
        results = ensemble.train(X, y, validation_split=0.2, n_estimators=10)
        
        assert ensemble.is_trained
        assert 'training_results' in results
        assert 'model_weights' in results
        assert ensemble.scaler_mean is not None
        assert ensemble.scaler_std is not None
    
    def test_prediction(self, sample_data):
        """Test ensemble prediction"""
        X, y, _ = sample_data
        ensemble = EnhancedEnsemble("test_ensemble")
        
        # Train
        ensemble.train(X[:150], y[:150], n_estimators=10)
        
        # Predict
        X_test = X[150:]
        predictions, confidence = ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert len(confidence) == len(X_test)
        assert np.all(confidence >= 0) and np.all(confidence <= 1)
    
    def test_explain(self, sample_data):
        """Test ensemble explanations"""
        X, y, _ = sample_data
        ensemble = EnhancedEnsemble("test_ensemble")
        
        # Train
        ensemble.train(X[:150], y[:150], n_estimators=10)
        
        # Explain
        explanations = ensemble.explain(X[150:151])
        
        assert 'individual_predictions' in explanations
        assert 'model_weights' in explanations
        assert 'feature_importance' in explanations
    
    def test_save_load(self, sample_data, temp_dir):
        """Test saving and loading ensemble"""
        X, y, _ = sample_data
        ensemble = EnhancedEnsemble("test_ensemble")
        
        # Train
        ensemble.train(X[:150], y[:150], n_estimators=10)
        
        # Make prediction
        X_test = X[150:160]
        pred_before, conf_before = ensemble.predict(X_test)
        
        # Save
        save_path = temp_dir / "test_ensemble"
        ensemble.save(save_path)
        
        # Load
        new_ensemble = EnhancedEnsemble("loaded_ensemble")
        new_ensemble.load(save_path)
        
        # Predict with loaded model
        pred_after, conf_after = new_ensemble.predict(X_test)
        
        # Verify
        np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=6)
        np.testing.assert_array_almost_equal(conf_before, conf_after, decimal=6)
    
    def test_model_count(self, sample_data):
        """Test model counting"""
        X, y, _ = sample_data
        ensemble = EnhancedEnsemble("test_ensemble")
        
        ensemble.train(X, y, n_estimators=10)
        
        model_count = ensemble.get_model_count()
        active_models = ensemble.get_active_models()
        
        assert model_count >= 10  # At least 10 models
        assert len(active_models) > 0
        assert all(isinstance(name, str) for name in active_models)


class TestRegimeAdaptiveWeighting:
    """Test RegimeAdaptiveWeighting class"""
    
    def test_initialization(self):
        """Test initialization"""
        weighting = RegimeAdaptiveWeighting(window_size=30)
        
        assert weighting.window_size == 30
        assert weighting.current_regime == MarketRegime.SIDEWAYS
        assert len(weighting.regime_base_weights) == 4  # 4 regimes
    
    def test_regime_weights(self):
        """Test getting regime weights"""
        weighting = RegimeAdaptiveWeighting()
        
        for regime in MarketRegime:
            weights = weighting.get_regime_weights(regime)
            
            assert isinstance(weights, dict)
            assert len(weights) > 0
            assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    
    def test_track_prediction(self):
        """Test prediction tracking"""
        weighting = RegimeAdaptiveWeighting()
        
        # Track some predictions
        for i in range(10):
            weighting.track_prediction(
                'xgboost',
                prediction=0.01,
                actual=0.012,
                regime=MarketRegime.BULL
            )
        
        # Get performance
        perf = weighting.get_model_performance('xgboost', MarketRegime.BULL)
        
        assert perf['count'] == 10
        assert perf['mae'] > 0
    
    def test_adaptive_weights(self):
        """Test adaptive weight calculation"""
        weighting = RegimeAdaptiveWeighting()
        
        # Track performance for multiple models
        for model in ['xgboost', 'lightgbm', 'random_forest']:
            for i in range(20):
                # xgboost performs better
                error = 0.001 if model == 'xgboost' else 0.005
                weighting.track_prediction(
                    model,
                    prediction=0.01,
                    actual=0.01 + error,
                    regime=MarketRegime.BULL
                )
        
        # Get adaptive weights
        weights = weighting.get_adaptive_weights(MarketRegime.BULL, adaptation_strength=0.8)
        
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
        # xgboost should have higher weight due to better performance
        assert weights.get('xgboost', 0) > weights.get('lightgbm', 0)
    
    def test_select_best_models(self):
        """Test model selection"""
        weighting = RegimeAdaptiveWeighting()
        
        # Track performance
        models = ['xgboost', 'lightgbm', 'random_forest', 'ridge', 'lasso']
        for model in models:
            for i in range(15):
                weighting.track_prediction(
                    model,
                    prediction=0.01,
                    actual=0.011,
                    regime=MarketRegime.BULL
                )
        
        # Select best models
        best_models = weighting.select_best_models(3, MarketRegime.BULL)
        
        assert len(best_models) == 3
        assert all(isinstance(name, str) for name in best_models)
    
    def test_save_load(self, temp_dir):
        """Test saving and loading"""
        weighting = RegimeAdaptiveWeighting()
        
        # Track some data
        for i in range(10):
            weighting.track_prediction(
                'xgboost',
                prediction=0.01,
                actual=0.012,
                regime=MarketRegime.BULL
            )
        
        # Save
        save_path = temp_dir / "regime_weights.json"
        weighting.save(save_path)
        
        # Load
        new_weighting = RegimeAdaptiveWeighting()
        new_weighting.load(save_path)
        
        # Verify
        assert new_weighting.window_size == weighting.window_size
        assert new_weighting.current_regime == weighting.current_regime


class TestMarketRegimeDetection:
    """Test market regime detection"""
    
    def test_bull_market_detection(self):
        """Test bull market detection"""
        # Generate upward trending prices
        prices = np.linspace(100, 120, 100)
        regime = detect_market_regime(prices)
        
        assert regime == MarketRegime.BULL
    
    def test_bear_market_detection(self):
        """Test bear market detection"""
        # Generate downward trending prices
        prices = np.linspace(100, 80, 100)
        regime = detect_market_regime(prices)
        
        assert regime == MarketRegime.BEAR
    
    def test_sideways_market_detection(self):
        """Test sideways market detection"""
        # Generate range-bound prices
        prices = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 2
        regime = detect_market_regime(prices)
        
        assert regime == MarketRegime.SIDEWAYS
    
    def test_volatile_market_detection(self):
        """Test high volatility detection"""
        # Generate highly volatile prices
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 5)
        regime = detect_market_regime(prices)
        
        assert regime == MarketRegime.HIGH_VOLATILITY


class TestAdaptiveEnsembleSystem:
    """Test AdaptiveEnsembleSystem class"""
    
    def test_initialization(self):
        """Test system initialization"""
        system = AdaptiveEnsembleSystem("test_system")
        
        assert system.model_name == "test_system"
        assert system.ensemble is not None
        assert system.regime_weighting is not None
    
    def test_training(self, sample_data):
        """Test system training"""
        X, y, prices = sample_data
        system = AdaptiveEnsembleSystem("test_system")
        
        results = system.train(X, y, prices=prices, n_estimators=10)
        
        assert 'regime' in results
        assert 'regime_weights' in results
        assert system.ensemble.is_trained
    
    def test_prediction(self, sample_data):
        """Test system prediction"""
        X, y, prices = sample_data
        system = AdaptiveEnsembleSystem("test_system")
        
        # Train
        system.train(X[:150], y[:150], prices=prices[:150], n_estimators=10)
        
        # Predict
        X_test = X[150:]
        predictions, confidence = system.predict(X_test, prices=prices[150:])
        
        assert len(predictions) == len(X_test)
        assert len(confidence) == len(X_test)
    
    def test_prediction_with_tracking(self, sample_data):
        """Test prediction with performance tracking"""
        X, y, prices = sample_data
        system = AdaptiveEnsembleSystem("test_system")
        
        # Train
        system.train(X[:150], y[:150], prices=prices[:150], n_estimators=10)
        
        # Predict with tracking
        X_test = X[150:160]
        y_test = y[150:160]
        result = system.predict_with_tracking(X_test, prices=prices[150:160], actual=y_test)
        
        assert 'predictions' in result
        assert 'confidence' in result
        assert 'regime' in result
        assert 'model_weights' in result
    
    def test_performance_report(self, sample_data):
        """Test performance report generation"""
        X, y, prices = sample_data
        system = AdaptiveEnsembleSystem("test_system")
        
        # Train
        system.train(X, y, prices=prices, n_estimators=10)
        
        # Get report
        report = system.get_performance_report()
        
        assert 'current_regime' in report
        assert 'model_count' in report
        assert 'active_models' in report
        assert 'model_weights' in report
    
    def test_save_load(self, sample_data, temp_dir):
        """Test saving and loading complete system"""
        X, y, prices = sample_data
        system = AdaptiveEnsembleSystem("test_system")
        
        # Train
        system.train(X[:150], y[:150], prices=prices[:150], n_estimators=10)
        
        # Predict
        X_test = X[150:160]
        pred_before, conf_before = system.predict(X_test)
        
        # Save
        save_path = temp_dir / "test_system"
        system.save(save_path)
        
        # Load
        new_system = AdaptiveEnsembleSystem("loaded_system")
        new_system.load(save_path)
        
        # Predict with loaded system
        pred_after, conf_after = new_system.predict(X_test)
        
        # Verify
        np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
