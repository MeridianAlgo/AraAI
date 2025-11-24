"""
Tests for Market Regime Detection System
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

from ara.models.regime_detector import (
    RegimeType, RegimeFeatures, HiddenMarkovModel, RegimeDetector
)
from ara.models.regime_adjustments import RegimeAdaptivePredictions


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    n_days = 500
    
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Generate synthetic price data with different regimes
    prices = []
    base_price = 100
    
    for i in range(n_days):
        if i < 150:  # Bull market
            trend = 0.001
            volatility = 0.01
        elif i < 300:  # Bear market
            trend = -0.002
            volatility = 0.015
        elif i < 400:  # Sideways
            trend = 0.0
            volatility = 0.008
        else:  # High volatility
            trend = 0.0
            volatility = 0.03
        
        change = np.random.randn() * volatility + trend
        base_price *= (1 + change)
        prices.append(base_price)
    
    prices = np.array(prices)
    
    # Generate OHLCV
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_days) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n_days)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    }, index=dates)
    
    return data


class TestRegimeFeatures:
    """Test RegimeFeatures class"""
    
    def test_initialization(self):
        """Test RegimeFeatures initialization"""
        features = RegimeFeatures()
        assert features is not None
        assert hasattr(features, 'scaler')
        assert hasattr(features, 'feature_names')
    
    def test_calculate_features(self, sample_data):
        """Test feature calculation"""
        features = RegimeFeatures()
        feature_matrix = features.calculate_features(sample_data)
        
        assert feature_matrix is not None
        assert feature_matrix.shape[0] == len(sample_data)
        assert feature_matrix.shape[1] > 0
        assert len(features.feature_names) == feature_matrix.shape[1]
        
        # Check for NaN or inf values
        assert not np.any(np.isnan(feature_matrix))
        assert not np.any(np.isinf(feature_matrix))
    
    def test_feature_names(self, sample_data):
        """Test that feature names are generated"""
        features = RegimeFeatures()
        feature_matrix = features.calculate_features(sample_data)
        
        assert len(features.feature_names) > 0
        assert 'momentum_5d' in features.feature_names
        assert 'volatility_10d' in features.feature_names
    
    def test_scaler_transform(self, sample_data):
        """Test feature scaling"""
        features = RegimeFeatures()
        feature_matrix = features.calculate_features(sample_data)
        
        # Fit and transform
        scaled = features.fit_transform(feature_matrix)
        
        assert scaled.shape == feature_matrix.shape
        assert not np.any(np.isnan(scaled))
        
        # Check that scaling worked (mean ~0, std ~1)
        assert np.abs(np.mean(scaled)) < 0.5
        assert np.abs(np.std(scaled) - 1.0) < 0.5


class TestHiddenMarkovModel:
    """Test HiddenMarkovModel class"""
    
    def test_initialization(self):
        """Test HMM initialization"""
        hmm = HiddenMarkovModel(n_states=4, n_features=10)
        
        assert hmm.n_states == 4
        assert hmm.n_features == 10
        assert hmm.transition_matrix.shape == (4, 4)
        assert hmm.means.shape == (4, 10)
        assert len(hmm.covariances) == 4
    
    def test_transition_matrix_properties(self):
        """Test transition matrix properties"""
        hmm = HiddenMarkovModel(n_states=4, n_features=10)
        
        # Each row should sum to 1
        row_sums = hmm.transition_matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(4))
        
        # All probabilities should be between 0 and 1
        assert np.all(hmm.transition_matrix >= 0)
        assert np.all(hmm.transition_matrix <= 1)
    
    def test_emission_probability(self):
        """Test emission probability calculation"""
        hmm = HiddenMarkovModel(n_states=4, n_features=10)
        
        observation = np.random.randn(10)
        prob = hmm._emission_probability(observation, 0)
        
        assert prob > 0
        assert prob <= 1
    
    def test_forward_algorithm(self, sample_data):
        """Test forward algorithm"""
        features = RegimeFeatures()
        feature_matrix = features.calculate_features(sample_data)
        feature_matrix = features.fit_transform(feature_matrix)
        
        hmm = HiddenMarkovModel(n_states=4, n_features=feature_matrix.shape[1])
        
        alpha, log_likelihood = hmm.forward_algorithm(feature_matrix[:100])
        
        assert alpha.shape == (100, 4)
        assert not np.isnan(log_likelihood)
        
        # Each row should sum to approximately 1 (normalized)
        row_sums = alpha.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(100), decimal=5)
    
    def test_viterbi_algorithm(self, sample_data):
        """Test Viterbi algorithm"""
        features = RegimeFeatures()
        feature_matrix = features.calculate_features(sample_data)
        feature_matrix = features.fit_transform(feature_matrix)
        
        hmm = HiddenMarkovModel(n_states=4, n_features=feature_matrix.shape[1])
        
        states = hmm.viterbi_algorithm(feature_matrix[:100])
        
        assert len(states) == 100
        assert np.all(states >= 0)
        assert np.all(states < 4)
    
    def test_fit(self, sample_data):
        """Test HMM fitting"""
        features = RegimeFeatures()
        feature_matrix = features.calculate_features(sample_data)
        feature_matrix = features.fit_transform(feature_matrix)
        
        hmm = HiddenMarkovModel(n_states=4, n_features=feature_matrix.shape[1])
        
        # Fit should not raise errors
        hmm.fit(feature_matrix[:200], n_iterations=10)
        
        # Check that parameters were updated
        assert hmm.transition_matrix is not None
        assert hmm.means is not None
    
    def test_predict(self, sample_data):
        """Test prediction"""
        features = RegimeFeatures()
        feature_matrix = features.calculate_features(sample_data)
        feature_matrix = features.fit_transform(feature_matrix)
        
        hmm = HiddenMarkovModel(n_states=4, n_features=feature_matrix.shape[1])
        hmm.fit(feature_matrix[:200], n_iterations=10)
        
        states = hmm.predict(feature_matrix[200:250])
        
        assert len(states) == 50
        assert np.all(states >= 0)
        assert np.all(states < 4)


class TestRegimeDetector:
    """Test RegimeDetector class"""
    
    def test_initialization(self):
        """Test RegimeDetector initialization"""
        detector = RegimeDetector(lookback_period=252)
        
        assert detector.lookback_period == 252
        assert detector.feature_extractor is not None
        assert not detector.is_fitted
        assert len(detector.regime_mapping) == 4
    
    def test_fit(self, sample_data):
        """Test fitting regime detector"""
        detector = RegimeDetector(lookback_period=252)
        detector.fit(sample_data)
        
        assert detector.is_fitted
        assert detector.hmm is not None
    
    def test_detect(self, sample_data):
        """Test regime detection"""
        detector = RegimeDetector(lookback_period=252)
        detector.fit(sample_data)
        
        regime_info = detector.detect(sample_data.tail(100))
        
        assert 'current_regime' in regime_info
        assert 'confidence' in regime_info
        assert 'regime_probabilities' in regime_info
        assert 'transition_probabilities' in regime_info
        assert 'stability_score' in regime_info
        assert 'duration_in_regime' in regime_info
        
        # Check regime is valid
        assert regime_info['current_regime'] in [r.value for r in RegimeType]
        
        # Check confidence is between 0 and 1
        assert 0 <= regime_info['confidence'] <= 1
        
        # Check probabilities sum to 1
        prob_sum = sum(regime_info['regime_probabilities'].values())
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_detect_without_fit(self, sample_data):
        """Test that detect raises error if not fitted"""
        detector = RegimeDetector()
        
        with pytest.raises(ValueError, match="must be fitted"):
            detector.detect(sample_data)
    
    def test_regime_stability(self, sample_data):
        """Test regime stability calculation"""
        detector = RegimeDetector(lookback_period=252)
        detector.fit(sample_data)
        
        regime_info = detector.detect(sample_data.tail(100))
        
        assert 'stability_score' in regime_info
        assert 0 <= regime_info['stability_score'] <= 1
    
    def test_regime_duration(self, sample_data):
        """Test regime duration tracking"""
        detector = RegimeDetector(lookback_period=252)
        detector.fit(sample_data)
        
        # Detect multiple times
        for i in range(5):
            start_idx = 100 + i * 20
            regime_info = detector.detect(sample_data.iloc[start_idx:start_idx+100])
        
        assert len(detector.regime_history) > 0
        assert regime_info['duration_in_regime'] >= 0
    
    def test_get_regime_statistics(self, sample_data):
        """Test regime statistics"""
        detector = RegimeDetector(lookback_period=252)
        detector.fit(sample_data)
        
        # Detect multiple times to build history
        for i in range(10):
            start_idx = 100 + i * 20
            detector.detect(sample_data.iloc[start_idx:start_idx+100])
        
        stats = detector.get_regime_statistics()
        
        assert 'total_observations' in stats
        assert 'current_regime' in stats
        assert stats['total_observations'] > 0
    
    def test_save_load(self, sample_data):
        """Test saving and loading detector state"""
        detector = RegimeDetector(lookback_period=252)
        detector.fit(sample_data)
        
        regime_info = detector.detect(sample_data.tail(100))
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "detector.json"
            detector.save(save_path)
            
            assert save_path.exists()
            
            # Load
            new_detector = RegimeDetector()
            new_detector.load(save_path)
            
            assert new_detector.is_fitted
            assert new_detector.lookback_period == detector.lookback_period
            assert new_detector.current_regime == detector.current_regime
            
            # Verify detection works
            new_regime_info = new_detector.detect(sample_data.tail(100))
            assert new_regime_info['current_regime'] == regime_info['current_regime']


class TestRegimeAdaptivePredictions:
    """Test RegimeAdaptivePredictions class"""
    
    def test_initialization(self):
        """Test initialization"""
        adaptive = RegimeAdaptivePredictions()
        
        assert adaptive.regime_detector is not None
        assert len(adaptive.regime_params) == 4
        assert len(adaptive.alert_thresholds) > 0
    
    def test_adjust_prediction_horizon(self):
        """Test prediction horizon adjustment"""
        adaptive = RegimeAdaptivePredictions()
        
        # Test each regime
        for regime in RegimeType:
            adjusted = adaptive.adjust_prediction_horizon(30, regime)
            
            assert adjusted > 0
            assert adjusted <= 30
            
            # High volatility should have shortest horizon
            if regime == RegimeType.HIGH_VOLATILITY:
                assert adjusted <= 15
    
    def test_adjust_confidence_intervals(self):
        """Test confidence interval adjustment"""
        adaptive = RegimeAdaptivePredictions()
        
        predictions = np.array([100, 101, 102, 103, 104])
        base_std = 1.0
        
        for regime in RegimeType:
            lower, upper = adaptive.adjust_confidence_intervals(
                predictions, base_std, regime
            )
            
            assert len(lower) == len(predictions)
            assert len(upper) == len(predictions)
            assert np.all(lower < predictions)
            assert np.all(upper > predictions)
            assert np.all(lower < upper)
    
    def test_calculate_regime_confidence(self):
        """Test regime confidence calculation"""
        adaptive = RegimeAdaptivePredictions()
        
        regime_info = {
            'current_regime': RegimeType.BULL.value,
            'confidence': 0.8,
            'stability_score': 0.7
        }
        
        adjusted = adaptive.calculate_regime_confidence(0.75, regime_info)
        
        assert 0 <= adjusted <= 1
    
    def test_get_regime_feature_importance(self):
        """Test feature importance retrieval"""
        adaptive = RegimeAdaptivePredictions()
        
        for regime in RegimeType:
            importance = adaptive.get_regime_feature_importance(regime)
            
            assert isinstance(importance, dict)
            assert len(importance) > 0
            assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_get_preferred_models(self):
        """Test preferred models retrieval"""
        adaptive = RegimeAdaptivePredictions()
        
        for regime in RegimeType:
            models = adaptive.get_preferred_models(regime)
            
            assert isinstance(models, list)
            assert len(models) > 0
            assert all(isinstance(m, str) for m in models)
    
    def test_check_regime_change(self):
        """Test regime change detection"""
        adaptive = RegimeAdaptivePredictions()
        
        regime_info = {
            'current_regime': RegimeType.BULL.value,
            'confidence': 0.85,
            'stability_score': 0.7
        }
        
        # Should detect change
        alert = adaptive.check_regime_change(RegimeType.BEAR, regime_info)
        
        assert alert is not None
        assert alert['type'] == 'regime_change'
        assert alert['previous_regime'] == RegimeType.BEAR.value
        assert alert['new_regime'] == RegimeType.BULL.value
        assert 'recommendations' in alert
    
    def test_check_uncertainty_alert(self):
        """Test uncertainty alert"""
        adaptive = RegimeAdaptivePredictions()
        
        # Low confidence should trigger alert
        regime_info = {
            'current_regime': RegimeType.SIDEWAYS.value,
            'confidence': 0.4,
            'stability_score': 0.3
        }
        
        alert = adaptive.check_uncertainty_alert(regime_info)
        
        assert alert is not None
        assert alert['type'] == 'high_uncertainty'
        assert 'recommendations' in alert
    
    def test_check_volatility_spike(self):
        """Test volatility spike detection"""
        adaptive = RegimeAdaptivePredictions()
        
        # High volatility should trigger alert
        alert = adaptive.check_volatility_spike(0.05, 0.02)
        
        assert alert is not None
        assert alert['type'] == 'volatility_spike'
        assert alert['multiplier'] > 2.0
    
    def test_apply_regime_adjustments(self, sample_data):
        """Test full regime adjustment workflow"""
        detector = RegimeDetector(lookback_period=252)
        detector.fit(sample_data)
        
        adaptive = RegimeAdaptivePredictions(regime_detector=detector)
        
        predictions = np.linspace(100, 110, 30)
        
        result = adaptive.apply_regime_adjustments(
            predictions=predictions,
            requested_days=30,
            base_confidence=0.75,
            base_std=2.0,
            data=sample_data.tail(100)
        )
        
        assert 'adjusted_predictions' in result
        assert 'adjusted_days' in result
        assert 'lower_bounds' in result
        assert 'upper_bounds' in result
        assert 'adjusted_confidence' in result
        assert 'regime_info' in result
        assert 'feature_importance' in result
        assert 'preferred_models' in result
        assert 'alerts' in result
        assert 'adjustment_summary' in result
        
        # Check that adjustments were applied
        assert len(result['adjusted_predictions']) <= 30
        assert 0 <= result['adjusted_confidence'] <= 1
    
    def test_get_recent_alerts(self):
        """Test getting recent alerts"""
        adaptive = RegimeAdaptivePredictions()
        
        # Generate some alerts
        regime_info = {
            'current_regime': RegimeType.HIGH_VOLATILITY.value,
            'confidence': 0.4,
            'stability_score': 0.3
        }
        
        adaptive.check_uncertainty_alert(regime_info)
        adaptive.check_volatility_spike(0.05, 0.02)
        
        alerts = adaptive.get_recent_alerts(n=10)
        
        assert len(alerts) >= 2
        assert all('type' in alert for alert in alerts)
    
    def test_save_load(self, sample_data):
        """Test saving and loading state"""
        detector = RegimeDetector(lookback_period=252)
        detector.fit(sample_data)
        
        adaptive = RegimeAdaptivePredictions(regime_detector=detector)
        
        # Generate some state
        regime_info = {
            'current_regime': RegimeType.BULL.value,
            'confidence': 0.8,
            'stability_score': 0.7
        }
        adaptive.check_uncertainty_alert(regime_info)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "adaptive.json"
            adaptive.save(save_path)
            
            assert save_path.exists()
            
            # Load
            new_adaptive = RegimeAdaptivePredictions()
            new_adaptive.load(save_path)
            
            assert len(new_adaptive.alerts) == len(adaptive.alerts)
            assert new_adaptive.alert_thresholds == adaptive.alert_thresholds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
