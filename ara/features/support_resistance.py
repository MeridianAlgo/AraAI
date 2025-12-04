"""
Support and Resistance Detection

This module implements various methods for detecting support and resistance levels.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from ara.features.indicator_registry import get_registry


class SupportResistance:
    """Collection of support and resistance detection methods."""
    
    @staticmethod
    def pivot_points_standard(data: pd.DataFrame) -> pd.DataFrame:
        """
        Standard Pivot Points.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            DataFrame with pivot point levels
        """
        result = data.copy()
        
        # Calculate pivot point
        pivot = (result['high'].shift(1) + result['low'].shift(1) + result['close'].shift(1)) / 3
        
        # Calculate support and resistance levels
        result['pivot'] = pivot
        result['r1'] = 2 * pivot - result['low'].shift(1)
        result['r2'] = pivot + (result['high'].shift(1) - result['low'].shift(1))
        result['r3'] = result['high'].shift(1) + 2 * (pivot - result['low'].shift(1))
        result['s1'] = 2 * pivot - result['high'].shift(1)
        result['s2'] = pivot - (result['high'].shift(1) - result['low'].shift(1))
        result['s3'] = result['low'].shift(1) - 2 * (result['high'].shift(1) - pivot)
        
        return result
    
    @staticmethod
    def pivot_points_fibonacci(data: pd.DataFrame) -> pd.DataFrame:
        """
        Fibonacci Pivot Points.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            DataFrame with Fibonacci pivot levels
        """
        result = data.copy()
        
        # Calculate pivot point
        pivot = (result['high'].shift(1) + result['low'].shift(1) + result['close'].shift(1)) / 3
        range_hl = result['high'].shift(1) - result['low'].shift(1)
        
        # Calculate Fibonacci levels
        result['pivot_fib'] = pivot
        result['r1_fib'] = pivot + 0.382 * range_hl
        result['r2_fib'] = pivot + 0.618 * range_hl
        result['r3_fib'] = pivot + 1.000 * range_hl
        result['s1_fib'] = pivot - 0.382 * range_hl
        result['s2_fib'] = pivot - 0.618 * range_hl
        result['s3_fib'] = pivot - 1.000 * range_hl
        
        return result
    
    @staticmethod
    def pivot_points_camarilla(data: pd.DataFrame) -> pd.DataFrame:
        """
        Camarilla Pivot Points.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            DataFrame with Camarilla pivot levels
        """
        result = data.copy()
        
        close = result['close'].shift(1)
        range_hl = result['high'].shift(1) - result['low'].shift(1)
        
        # Calculate Camarilla levels
        result['pivot_cam'] = close
        result['r1_cam'] = close + range_hl * 1.1 / 12
        result['r2_cam'] = close + range_hl * 1.1 / 6
        result['r3_cam'] = close + range_hl * 1.1 / 4
        result['r4_cam'] = close + range_hl * 1.1 / 2
        result['s1_cam'] = close - range_hl * 1.1 / 12
        result['s2_cam'] = close - range_hl * 1.1 / 6
        result['s3_cam'] = close - range_hl * 1.1 / 4
        result['s4_cam'] = close - range_hl * 1.1 / 2
        
        return result
    
    @staticmethod
    def fibonacci_retracement(
        data: pd.DataFrame,
        window: int = 50,
        trend: str = 'auto'
    ) -> pd.DataFrame:
        """
        Fibonacci Retracement Levels.
        
        Args:
            data: DataFrame with OHLC data
            window: Window for high/low detection
            trend: 'up', 'down', or 'auto'
            
        Returns:
            DataFrame with Fibonacci retracement levels
        """
        result = data.copy()
        
        # Find swing high and low
        swing_high = result['high'].rolling(window=window).max()
        swing_low = result['low'].rolling(window=window).min()
        
        # Determine trend
        if trend == 'auto':
            is_uptrend = result['close'] > result['close'].shift(window)
        else:
            is_uptrend = trend == 'up'
        
        # Calculate Fibonacci levels
        if isinstance(is_uptrend, pd.Series):
            # Handle series
            range_val = swing_high - swing_low
            result['fib_0'] = np.where(is_uptrend, swing_low, swing_high)
            result['fib_236'] = np.where(is_uptrend, swing_low + 0.236 * range_val, swing_high - 0.236 * range_val)
            result['fib_382'] = np.where(is_uptrend, swing_low + 0.382 * range_val, swing_high - 0.382 * range_val)
            result['fib_500'] = np.where(is_uptrend, swing_low + 0.500 * range_val, swing_high - 0.500 * range_val)
            result['fib_618'] = np.where(is_uptrend, swing_low + 0.618 * range_val, swing_high - 0.618 * range_val)
            result['fib_786'] = np.where(is_uptrend, swing_low + 0.786 * range_val, swing_high - 0.786 * range_val)
            result['fib_100'] = np.where(is_uptrend, swing_high, swing_low)
        else:
            # Handle boolean
            range_val = swing_high - swing_low
            if is_uptrend:
                result['fib_0'] = swing_low
                result['fib_236'] = swing_low + 0.236 * range_val
                result['fib_382'] = swing_low + 0.382 * range_val
                result['fib_500'] = swing_low + 0.500 * range_val
                result['fib_618'] = swing_low + 0.618 * range_val
                result['fib_786'] = swing_low + 0.786 * range_val
                result['fib_100'] = swing_high
            else:
                result['fib_0'] = swing_high
                result['fib_236'] = swing_high - 0.236 * range_val
                result['fib_382'] = swing_high - 0.382 * range_val
                result['fib_500'] = swing_high - 0.500 * range_val
                result['fib_618'] = swing_high - 0.618 * range_val
                result['fib_786'] = swing_high - 0.786 * range_val
                result['fib_100'] = swing_low
        
        return result
    
    @staticmethod
    def price_clusters(
        data: pd.DataFrame,
        window: int = 100,
        num_clusters: int = 5,
        tolerance: float = 0.02
    ) -> pd.DataFrame:
        """
        Dynamic Support/Resistance using Price Clusters.
        
        Args:
            data: DataFrame with OHLC data
            window: Lookback window
            num_clusters: Number of S/R levels to identify
            tolerance: Price clustering tolerance
            
        Returns:
            DataFrame with cluster-based S/R levels
        """
        result = data.copy()
        
        # Collect all price points (highs and lows)
        for i in range(len(result)):
            if i < window:
                continue
            
            # Get recent highs and lows
            recent_highs = result['high'].iloc[max(0, i-window):i].values
            recent_lows = result['low'].iloc[max(0, i-window):i].values
            all_prices = np.concatenate([recent_highs, recent_lows])
            
            # Simple clustering: find price levels with multiple touches
            clusters = []
            for price in np.unique(all_prices):
                # Count prices within tolerance
                nearby = np.abs(all_prices - price) / price < tolerance
                if np.sum(nearby) >= 3:  # At least 3 touches
                    clusters.append(price)
            
            # Sort and take top clusters
            clusters = sorted(set(clusters))
            
            # Assign to result (pad with NaN if fewer clusters)
            for j in range(num_clusters):
                col_name = f'cluster_level_{j+1}'
                if col_name not in result.columns:
                    result[col_name] = np.nan
                
                if j < len(clusters):
                    result.loc[result.index[i], col_name] = clusters[j]
        
        return result
    
    @staticmethod
    def volume_profile(
        data: pd.DataFrame,
        window: int = 100,
        num_levels: int = 20
    ) -> pd.DataFrame:
        """
        Volume Profile for Key Levels.
        
        Args:
            data: DataFrame with OHLCV data
            window: Lookback window
            num_levels: Number of price levels to analyze
            
        Returns:
            DataFrame with volume profile levels
        """
        result = data.copy()
        
        result['poc'] = np.nan  # Point of Control (highest volume)
        result['vah'] = np.nan  # Value Area High
        result['val'] = np.nan  # Value Area Low
        
        for i in range(window, len(result)):
            # Get price range
            price_min = result['low'].iloc[i-window:i].min()
            price_max = result['high'].iloc[i-window:i].max()
            
            # Create price bins
            price_bins = np.linspace(price_min, price_max, num_levels)
            volume_at_price = np.zeros(num_levels - 1)
            
            # Accumulate volume at each price level
            for j in range(i-window, i):
                price = result['close'].iloc[j]
                volume = result['volume'].iloc[j]
                
                # Find which bin this price falls into
                bin_idx = np.digitize(price, price_bins) - 1
                if 0 <= bin_idx < len(volume_at_price):
                    volume_at_price[bin_idx] += volume
            
            # Find Point of Control (highest volume)
            poc_idx = np.argmax(volume_at_price)
            result.loc[result.index[i], 'poc'] = price_bins[poc_idx]
            
            # Find Value Area (70% of volume around POC)
            total_volume = np.sum(volume_at_price)
            target_volume = total_volume * 0.70
            
            # Expand from POC until we reach 70% volume
            lower_idx = poc_idx
            upper_idx = poc_idx
            accumulated_volume = volume_at_price[poc_idx]
            
            while accumulated_volume < target_volume and (lower_idx > 0 or upper_idx < len(volume_at_price) - 1):
                # Expand to side with more volume
                lower_vol = volume_at_price[lower_idx - 1] if lower_idx > 0 else 0
                upper_vol = volume_at_price[upper_idx + 1] if upper_idx < len(volume_at_price) - 1 else 0
                
                if lower_vol > upper_vol and lower_idx > 0:
                    lower_idx -= 1
                    accumulated_volume += lower_vol
                elif upper_idx < len(volume_at_price) - 1:
                    upper_idx += 1
                    accumulated_volume += upper_vol
                else:
                    break
            
            result.loc[result.index[i], 'val'] = price_bins[lower_idx]
            result.loc[result.index[i], 'vah'] = price_bins[upper_idx]
        
        return result
    
    @staticmethod
    def swing_points(
        data: pd.DataFrame,
        window: int = 5
    ) -> pd.DataFrame:
        """
        Identify Swing Highs and Swing Lows.
        
        Args:
            data: DataFrame with OHLC data
            window: Window for swing detection
            
        Returns:
            DataFrame with swing high/low markers
        """
        result = data.copy()
        
        # Swing highs: local maxima
        swing_high = pd.Series(False, index=result.index)
        for i in range(window, len(result) - window):
            is_high = True
            for j in range(1, window + 1):
                if result['high'].iloc[i] <= result['high'].iloc[i-j] or \
                   result['high'].iloc[i] <= result['high'].iloc[i+j]:
                    is_high = False
                    break
            swing_high.iloc[i] = is_high
        
        # Swing lows: local minima
        swing_low = pd.Series(False, index=result.index)
        for i in range(window, len(result) - window):
            is_low = True
            for j in range(1, window + 1):
                if result['low'].iloc[i] >= result['low'].iloc[i-j] or \
                   result['low'].iloc[i] >= result['low'].iloc[i+j]:
                    is_low = False
                    break
            swing_low.iloc[i] = is_low
        
        result['swing_high'] = swing_high.astype(int)
        result['swing_low'] = swing_low.astype(int)
        
        # Mark the actual price levels
        result['swing_high_price'] = np.where(swing_high, result['high'], np.nan)
        result['swing_low_price'] = np.where(swing_low, result['low'], np.nan)
        
        return result


# Register support/resistance indicators
def register_support_resistance_indicators():
    """Register support/resistance indicators with the global registry."""
    registry = get_registry()
    
    # Standard Pivot Points
    registry.register(
        name='pivot_standard',
        func=SupportResistance.pivot_points_standard,
        category='support_resistance',
        description='Standard Pivot Points',
        parameters={},
        required_columns=['high', 'low', 'close'],
        output_columns=['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3']
    )
    
    # Fibonacci Pivot Points
    registry.register(
        name='pivot_fibonacci',
        func=SupportResistance.pivot_points_fibonacci,
        category='support_resistance',
        description='Fibonacci Pivot Points',
        parameters={},
        required_columns=['high', 'low', 'close'],
        output_columns=['pivot_fib', 'r1_fib', 'r2_fib', 'r3_fib', 's1_fib', 's2_fib', 's3_fib']
    )
    
    # Camarilla Pivot Points
    registry.register(
        name='pivot_camarilla',
        func=SupportResistance.pivot_points_camarilla,
        category='support_resistance',
        description='Camarilla Pivot Points',
        parameters={},
        required_columns=['high', 'low', 'close'],
        output_columns=['pivot_cam', 'r1_cam', 'r2_cam', 'r3_cam', 'r4_cam', 's1_cam', 's2_cam', 's3_cam', 's4_cam']
    )
    
    # Fibonacci Retracement
    registry.register(
        name='fibonacci_retracement',
        func=SupportResistance.fibonacci_retracement,
        category='support_resistance',
        description='Fibonacci Retracement Levels',
        parameters={'window': 50, 'trend': 'auto'},
        required_columns=['high', 'low', 'close'],
        output_columns=['fib_0', 'fib_236', 'fib_382', 'fib_500', 'fib_618', 'fib_786', 'fib_100']
    )
    
    # Volume Profile
    registry.register(
        name='volume_profile',
        func=SupportResistance.volume_profile,
        category='support_resistance',
        description='Volume Profile Levels',
        parameters={'window': 100, 'num_levels': 20},
        required_columns=['high', 'low', 'close', 'volume'],
        output_columns=['poc', 'vah', 'val']
    )
    
    # Swing Points
    registry.register(
        name='swing_points',
        func=SupportResistance.swing_points,
        category='support_resistance',
        description='Swing Highs and Lows',
        parameters={'window': 5},
        required_columns=['high', 'low'],
        output_columns=['swing_high', 'swing_low', 'swing_high_price', 'swing_low_price']
    )


# Auto-register on import
register_support_resistance_indicators()
