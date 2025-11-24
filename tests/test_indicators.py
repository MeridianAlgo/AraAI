"""
Tests for technical indicators.
"""

import pytest
import pandas as pd
import numpy as np
from ara.features import IndicatorCalculator, get_registry


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close + np.random.rand(100) * 2
    low = close - np.random.rand(100) * 2
    open_price = close + np.random.randn(100)
    volume = np.random.randint(1000000, 10000000, 100)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


def test_indicator_registry():
    """Test that indicators are registered."""
    registry = get_registry()
    
    # Check that indicators are registered
    indicators = registry.list_indicators()
    assert len(indicators) > 0
    
    # Check categories
    categories = registry.list_categories()
    assert 'trend' in categories
    assert 'momentum' in categories
    assert 'volatility' in categories
    assert 'volume' in categories


def test_trend_indicators(sample_data):
    """Test trend indicators."""
    calc = IndicatorCalculator()
    
    # Test SMA
    result = calc.calculate(sample_data, 'sma', {'period': 20})
    assert 'sma_20' in result.columns
    assert not result['sma_20'].isna().all()
    
    # Test EMA
    result = calc.calculate(sample_data, 'ema', {'period': 20})
    assert 'ema_20' in result.columns
    
    # Test MACD
    result = calc.calculate(sample_data, 'macd')
    assert 'macd' in result.columns
    assert 'macd_signal' in result.columns
    assert 'macd_histogram' in result.columns


def test_momentum_indicators(sample_data):
    """Test momentum indicators."""
    calc = IndicatorCalculator()
    
    # Test RSI
    result = calc.calculate(sample_data, 'rsi', {'period': 14})
    assert 'rsi_14' in result.columns
    assert result['rsi_14'].max() <= 100
    assert result['rsi_14'].min() >= 0
    
    # Test Stochastic
    result = calc.calculate(sample_data, 'stochastic')
    assert 'stoch_k' in result.columns
    assert 'stoch_d' in result.columns


def test_volatility_indicators(sample_data):
    """Test volatility indicators."""
    calc = IndicatorCalculator()
    
    # Test Bollinger Bands
    result = calc.calculate(sample_data, 'bollinger_bands')
    assert 'bb_upper' in result.columns
    assert 'bb_middle' in result.columns
    assert 'bb_lower' in result.columns
    
    # Test ATR
    result = calc.calculate(sample_data, 'atr')
    assert 'atr' in result.columns
    assert (result['atr'] >= 0).all()


def test_volume_indicators(sample_data):
    """Test volume indicators."""
    calc = IndicatorCalculator()
    
    # Test OBV
    result = calc.calculate(sample_data, 'obv')
    assert 'obv' in result.columns
    
    # Test MFI
    result = calc.calculate(sample_data, 'mfi')
    assert 'mfi' in result.columns
    assert result['mfi'].max() <= 100
    assert result['mfi'].min() >= 0


def test_multiple_indicators(sample_data):
    """Test calculating multiple indicators at once."""
    calc = IndicatorCalculator()
    
    indicators = ['sma', 'ema', 'rsi', 'bollinger_bands']
    result = calc.calculate(sample_data, indicators)
    
    assert 'sma_20' in result.columns
    assert 'ema_20' in result.columns
    assert 'rsi_14' in result.columns
    assert 'bb_upper' in result.columns


def test_multi_timeframe(sample_data):
    """Test multi-timeframe analysis."""
    calc = IndicatorCalculator()
    
    # Note: This test uses daily data, so we can't test intraday timeframes
    # Just verify the function works
    result = calc.calculate(sample_data, 'sma')
    assert 'sma_20' in result.columns


def test_caching(sample_data):
    """Test indicator caching."""
    calc = IndicatorCalculator(enable_cache=True)
    
    # Calculate twice - second should be from cache
    result1 = calc.calculate(sample_data, 'sma')
    result2 = calc.calculate(sample_data, 'sma')
    
    pd.testing.assert_frame_equal(result1, result2)
    
    # Clear cache
    calc.clear_cache()


def test_get_available_indicators():
    """Test getting available indicators."""
    calc = IndicatorCalculator()
    
    # Get all indicators
    all_indicators = calc.get_available_indicators()
    assert len(all_indicators) > 0
    
    # Get by category
    trend_indicators = calc.get_available_indicators('trend')
    assert len(trend_indicators) > 0
    assert 'sma' in trend_indicators


def test_get_indicator_info():
    """Test getting indicator information."""
    calc = IndicatorCalculator()
    
    info = calc.get_indicator_info('sma')
    assert info is not None
    assert info['name'] == 'sma'
    assert info['category'] == 'trend'
    assert 'period' in info['parameters']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
