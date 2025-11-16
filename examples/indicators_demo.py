"""
Technical Indicators Demo

This script demonstrates how to use the advanced technical indicators system.
"""

import pandas as pd
import numpy as np
from ara.features import IndicatorCalculator, get_registry


def create_sample_data(days=100):
    """Create sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=days, freq='D')
    
    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(days) * 2)
    high = close + np.random.rand(days) * 2
    low = close - np.random.rand(days) * 2
    open_price = close + np.random.randn(days)
    volume = np.random.randint(1000000, 10000000, days)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    return df


def demo_basic_usage():
    """Demonstrate basic indicator usage."""
    print("=" * 80)
    print("BASIC INDICATOR USAGE")
    print("=" * 80)
    
    # Create sample data
    data = create_sample_data()
    print(f"\nSample data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Initialize calculator
    calc = IndicatorCalculator()
    
    # Calculate a single indicator
    print("\n1. Calculating SMA (20-period)...")
    result = calc.calculate(data, 'sma', {'period': 20})
    print(f"   Added column: sma_20")
    print(f"   Latest value: {result['sma_20'].iloc[-1]:.2f}")
    
    # Calculate multiple indicators
    print("\n2. Calculating multiple indicators...")
    indicators = ['ema', 'rsi', 'macd', 'bollinger_bands']
    result = calc.calculate(data, indicators)
    print(f"   Added columns: {[col for col in result.columns if col not in data.columns]}")
    
    return result


def demo_indicator_categories():
    """Demonstrate indicators by category."""
    print("\n" + "=" * 80)
    print("INDICATORS BY CATEGORY")
    print("=" * 80)
    
    calc = IndicatorCalculator()
    registry = get_registry()
    
    categories = registry.list_categories()
    print(f"\nAvailable categories: {categories}")
    
    for category in categories:
        indicators = calc.get_available_indicators(category)
        print(f"\n{category.upper()} ({len(indicators)} indicators):")
        print(f"  {', '.join(indicators[:10])}")
        if len(indicators) > 10:
            print(f"  ... and {len(indicators) - 10} more")


def demo_trend_indicators():
    """Demonstrate trend indicators."""
    print("\n" + "=" * 80)
    print("TREND INDICATORS")
    print("=" * 80)
    
    data = create_sample_data()
    calc = IndicatorCalculator()
    
    # Moving averages
    print("\n1. Moving Averages:")
    result = calc.calculate(data, ['sma', 'ema', 'wma'], {
        'sma': {'period': 20},
        'ema': {'period': 20},
        'wma': {'period': 20}
    })
    print(f"   SMA(20): {result['sma_20'].iloc[-1]:.2f}")
    print(f"   EMA(20): {result['ema_20'].iloc[-1]:.2f}")
    print(f"   WMA(20): {result['wma_20'].iloc[-1]:.2f}")
    
    # MACD
    print("\n2. MACD:")
    result = calc.calculate(data, 'macd')
    print(f"   MACD Line: {result['macd'].iloc[-1]:.2f}")
    print(f"   Signal Line: {result['macd_signal'].iloc[-1]:.2f}")
    print(f"   Histogram: {result['macd_histogram'].iloc[-1]:.2f}")
    
    # ADX
    print("\n3. ADX (Trend Strength):")
    result = calc.calculate(data, 'adx')
    print(f"   ADX: {result['adx'].iloc[-1]:.2f}")
    print(f"   +DI: {result['plus_di'].iloc[-1]:.2f}")
    print(f"   -DI: {result['minus_di'].iloc[-1]:.2f}")


def demo_momentum_indicators():
    """Demonstrate momentum indicators."""
    print("\n" + "=" * 80)
    print("MOMENTUM INDICATORS")
    print("=" * 80)
    
    data = create_sample_data()
    calc = IndicatorCalculator()
    
    # RSI
    print("\n1. RSI (Relative Strength Index):")
    result = calc.calculate(data, 'rsi', {'period': 14})
    rsi_value = result['rsi_14'].iloc[-1]
    print(f"   RSI(14): {rsi_value:.2f}")
    if rsi_value > 70:
        print("   Status: Overbought")
    elif rsi_value < 30:
        print("   Status: Oversold")
    else:
        print("   Status: Neutral")
    
    # Stochastic
    print("\n2. Stochastic Oscillator:")
    result = calc.calculate(data, 'stochastic')
    print(f"   %K: {result['stoch_k'].iloc[-1]:.2f}")
    print(f"   %D: {result['stoch_d'].iloc[-1]:.2f}")
    
    # Williams %R
    print("\n3. Williams %R:")
    result = calc.calculate(data, 'williams_r')
    print(f"   Williams %R: {result['williams_r'].iloc[-1]:.2f}")


def demo_volatility_indicators():
    """Demonstrate volatility indicators."""
    print("\n" + "=" * 80)
    print("VOLATILITY INDICATORS")
    print("=" * 80)
    
    data = create_sample_data()
    calc = IndicatorCalculator()
    
    # Bollinger Bands
    print("\n1. Bollinger Bands:")
    result = calc.calculate(data, 'bollinger_bands')
    current_price = result['close'].iloc[-1]
    print(f"   Upper Band: {result['bb_upper'].iloc[-1]:.2f}")
    print(f"   Middle Band: {result['bb_middle'].iloc[-1]:.2f}")
    print(f"   Lower Band: {result['bb_lower'].iloc[-1]:.2f}")
    print(f"   Current Price: {current_price:.2f}")
    print(f"   BB %: {result['bb_percent'].iloc[-1]:.2f}")
    
    # ATR
    print("\n2. ATR (Average True Range):")
    result = calc.calculate(data, 'atr')
    print(f"   ATR(14): {result['atr'].iloc[-1]:.2f}")
    
    # Historical Volatility
    print("\n3. Historical Volatility:")
    result = calc.calculate(data, 'historical_volatility')
    print(f"   HV(20): {result['historical_volatility'].iloc[-1]:.2f}%")


def demo_volume_indicators():
    """Demonstrate volume indicators."""
    print("\n" + "=" * 80)
    print("VOLUME INDICATORS")
    print("=" * 80)
    
    data = create_sample_data()
    calc = IndicatorCalculator()
    
    # OBV
    print("\n1. On-Balance Volume:")
    result = calc.calculate(data, 'obv')
    print(f"   OBV: {result['obv'].iloc[-1]:,.0f}")
    
    # MFI
    print("\n2. Money Flow Index:")
    result = calc.calculate(data, 'mfi')
    mfi_value = result['mfi'].iloc[-1]
    print(f"   MFI(14): {mfi_value:.2f}")
    if mfi_value > 80:
        print("   Status: Overbought")
    elif mfi_value < 20:
        print("   Status: Oversold")
    else:
        print("   Status: Neutral")
    
    # VWAP
    print("\n3. Volume Weighted Average Price:")
    result = calc.calculate(data, 'vwap')
    print(f"   VWAP: {result['vwap'].iloc[-1]:.2f}")
    print(f"   Current Price: {result['close'].iloc[-1]:.2f}")


def demo_support_resistance():
    """Demonstrate support/resistance indicators."""
    print("\n" + "=" * 80)
    print("SUPPORT & RESISTANCE")
    print("=" * 80)
    
    data = create_sample_data()
    calc = IndicatorCalculator()
    
    # Pivot Points
    print("\n1. Standard Pivot Points:")
    result = calc.calculate(data, 'pivot_standard')
    print(f"   R3: {result['r3'].iloc[-1]:.2f}")
    print(f"   R2: {result['r2'].iloc[-1]:.2f}")
    print(f"   R1: {result['r1'].iloc[-1]:.2f}")
    print(f"   Pivot: {result['pivot'].iloc[-1]:.2f}")
    print(f"   S1: {result['s1'].iloc[-1]:.2f}")
    print(f"   S2: {result['s2'].iloc[-1]:.2f}")
    print(f"   S3: {result['s3'].iloc[-1]:.2f}")
    
    # Fibonacci Retracement
    print("\n2. Fibonacci Retracement:")
    result = calc.calculate(data, 'fibonacci_retracement')
    print(f"   100%: {result['fib_100'].iloc[-1]:.2f}")
    print(f"   78.6%: {result['fib_786'].iloc[-1]:.2f}")
    print(f"   61.8%: {result['fib_618'].iloc[-1]:.2f}")
    print(f"   50.0%: {result['fib_500'].iloc[-1]:.2f}")
    print(f"   38.2%: {result['fib_382'].iloc[-1]:.2f}")
    print(f"   23.6%: {result['fib_236'].iloc[-1]:.2f}")
    print(f"   0%: {result['fib_0'].iloc[-1]:.2f}")


def demo_performance():
    """Demonstrate performance with caching."""
    print("\n" + "=" * 80)
    print("PERFORMANCE & CACHING")
    print("=" * 80)
    
    import time
    
    data = create_sample_data(500)  # Larger dataset
    
    # Without caching
    calc_no_cache = IndicatorCalculator(enable_cache=False)
    start = time.time()
    for _ in range(3):
        calc_no_cache.calculate(data, ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'])
    time_no_cache = time.time() - start
    
    # With caching
    calc_with_cache = IndicatorCalculator(enable_cache=True)
    start = time.time()
    for _ in range(3):
        calc_with_cache.calculate(data, ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands'])
    time_with_cache = time.time() - start
    
    print(f"\nCalculating 5 indicators 3 times on 500 data points:")
    print(f"  Without caching: {time_no_cache:.3f}s")
    print(f"  With caching: {time_with_cache:.3f}s")
    print(f"  Speedup: {time_no_cache/time_with_cache:.1f}x")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("TECHNICAL INDICATORS DEMO")
    print("Advanced Technical Analysis System")
    print("=" * 80)
    
    # Run demos
    demo_basic_usage()
    demo_indicator_categories()
    demo_trend_indicators()
    demo_momentum_indicators()
    demo_volatility_indicators()
    demo_volume_indicators()
    demo_support_resistance()
    demo_performance()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - ara/features/trend.py - Trend indicators")
    print("  - ara/features/momentum.py - Momentum indicators")
    print("  - ara/features/volatility.py - Volatility indicators")
    print("  - ara/features/volume.py - Volume indicators")
    print("  - ara/features/patterns.py - Pattern recognition")
    print("  - ara/features/support_resistance.py - Support/Resistance")
    print()


if __name__ == '__main__':
    main()
