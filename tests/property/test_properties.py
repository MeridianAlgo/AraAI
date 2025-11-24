"""
Property-based tests using Hypothesis.

These tests verify that functions behave correctly for a wide range of inputs.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    from hypothesis import given, strategies as st, settings, assume
    from hypothesis.extra.numpy import arrays
    from hypothesis.extra.pandas import data_frames, column
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    pytest.skip("Hypothesis not installed", allow_module_level=True)

from tests.mocks.ml_models import MockMLModel
from tests.mocks.data_providers import MockDataProvider


@pytest.mark.property
class TestPredictionProperties:
    """Property-based tests for predictions."""
    
    @given(
        X=arrays(
            dtype=np.float64,
            shape=st.tuples(
                st.integers(min_value=10, max_value=1000),
                st.integers(min_value=5, max_value=100)
            ),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False)
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_prediction_shape_matches_input(self, X):
        """Predictions should have same length as input."""
        model = MockMLModel()
        model.train(X, np.random.randn(len(X)))
        
        predictions, confidences = model.predict(X)
        
        assert len(predictions) == len(X)
        assert len(confidences) == len(X)
        
    @given(
        X=arrays(
            dtype=np.float64,
            shape=(100, 50),
            elements=st.floats(min_value=-10, max_value=10, allow_nan=False)
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_confidence_in_valid_range(self, X):
        """Confidence scores should be between 0 and 1."""
        model = MockMLModel(accuracy=0.85)
        
        predictions, confidences = model.predict(X)
        
        assert np.all(confidences >= 0.0)
        assert np.all(confidences <= 1.0)
        
    @given(
        accuracy=st.floats(min_value=0.5, max_value=1.0)
    )
    @settings(max_examples=20)
    def test_model_accuracy_preserved(self, accuracy):
        """Model should maintain configured accuracy."""
        model = MockMLModel(accuracy=accuracy)
        X = np.random.randn(100, 50)
        
        _, confidences = model.predict(X)
        
        # Confidence should be close to configured accuracy
        assert np.allclose(confidences, accuracy, atol=0.01)


@pytest.mark.property
class TestDataValidationProperties:
    """Property-based tests for data validation."""
    
    @given(
        prices=st.lists(
            st.floats(min_value=0.01, max_value=10000),
            min_size=10,
            max_size=1000
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_price_data_always_positive(self, prices):
        """Price data should always be positive."""
        df = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': [p * 1.1 for p in prices],
            'low': [p * 0.9 for p in prices]
        })
        
        assert (df['close'] > 0).all()
        assert (df['open'] > 0).all()
        assert (df['high'] > 0).all()
        assert (df['low'] > 0).all()
        
    @given(
        data=data_frames([
            column('close', dtype=float, elements=st.floats(min_value=1, max_value=1000)),
            column('volume', dtype=int, elements=st.integers(min_value=0, max_value=1000000000))
        ], index=st.integers(min_value=0, max_value=1000))
    )
    @settings(max_examples=20, deadline=None)
    def test_ohlc_relationships(self, data):
        """High should be >= Low for all rows."""
        # Add high and low columns
        data['high'] = data['close'] * 1.05
        data['low'] = data['close'] * 0.95
        
        assert (data['high'] >= data['low']).all()


@pytest.mark.property
class TestFeatureCalculationProperties:
    """Property-based tests for feature calculations."""
    
    @given(
        window=st.integers(min_value=2, max_value=50),
        data_size=st.integers(min_value=100, max_value=500)
    )
    @settings(max_examples=30, deadline=None)
    def test_moving_average_properties(self, window, data_size):
        """Moving average should have expected properties."""
        assume(window < data_size)
        
        prices = np.random.randn(data_size).cumsum() + 100
        df = pd.DataFrame({'close': prices})
        
        ma = df['close'].rolling(window).mean()
        
        # MA should have NaN for first (window-1) values
        assert ma.iloc[:window-1].isna().all()
        
        # MA should be defined after window
        assert ma.iloc[window:].notna().all()
        
        # MA should smooth the data (lower std dev)
        if len(ma.dropna()) > 0:
            assert ma.dropna().std() <= df['close'].std()
            
    @given(
        period=st.integers(min_value=5, max_value=50)
    )
    @settings(max_examples=20)
    def test_rsi_bounds(self, period):
        """RSI should always be between 0 and 100."""
        # Generate price data
        prices = np.random.randn(200).cumsum() + 100
        
        # Calculate simple RSI
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
        assert 0 <= rsi <= 100


@pytest.mark.property
class TestCacheProperties:
    """Property-based tests for caching."""
    
    @given(
        key=st.text(min_size=1, max_size=100),
        value=st.integers()
    )
    @settings(max_examples=50)
    def test_cache_get_returns_set_value(self, key, value):
        """Cache should return the value that was set."""
        from tests.mocks.databases import MockCache
        
        cache = MockCache()
        cache.set(key, value)
        
        retrieved = cache.get(key)
        assert retrieved == value
        
    @given(
        keys=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=100, unique=True)
    )
    @settings(max_examples=30, deadline=None)
    def test_cache_isolation(self, keys):
        """Different keys should not interfere with each other."""
        from tests.mocks.databases import MockCache
        
        cache = MockCache()
        
        # Set different values for each key
        for i, key in enumerate(keys):
            cache.set(key, i)
            
        # Verify each key has correct value
        for i, key in enumerate(keys):
            assert cache.get(key) == i


@pytest.mark.property
class TestPortfolioProperties:
    """Property-based tests for portfolio calculations."""
    
    @given(
        weights=st.lists(
            st.floats(min_value=0, max_value=1),
            min_size=2,
            max_size=10
        )
    )
    @settings(max_examples=30)
    def test_portfolio_weights_sum_to_one(self, weights):
        """Portfolio weights should sum to 1 after normalization."""
        # Normalize weights
        total = sum(weights)
        assume(total > 0)
        
        normalized = [w / total for w in weights]
        
        assert abs(sum(normalized) - 1.0) < 1e-10
        
    @given(
        returns=st.lists(
            st.floats(min_value=-0.5, max_value=0.5),
            min_size=10,
            max_size=100
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_sharpe_ratio_properties(self, returns):
        """Sharpe ratio should have expected properties."""
        returns_array = np.array(returns)
        
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return > 0:
            sharpe = mean_return / std_return
            
            # Sharpe ratio should be finite
            assert np.isfinite(sharpe)
            
            # If mean return is positive, Sharpe should be positive
            if mean_return > 0:
                assert sharpe > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "property"])
