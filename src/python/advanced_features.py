"""
Advanced Feature Engineering and Market Analysis
Includes Ichimoku, Fibonacci, Volume Profile, and automated feature selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.feature_selection import (
    mutual_info_regression, f_regression, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for market microstructure and patterns"""
    
    def __init__(self):
        # Initialize extractors lazily to avoid forward reference issues
        self.feature_extractors = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def _initialize_extractors(self):
        """Initialize feature extractors lazily"""
        if not self.feature_extractors:
            self.feature_extractors = {
                'ichimoku': IchimokuExtractor(),
                'fibonacci': FibonacciExtractor(),
                'volume_profile': VolumeProfileExtractor(),
                'support_resistance': SupportResistanceExtractor(),
                'chart_patterns': ChartPatternExtractor(),
                'market_structure': MarketStructureExtractor()
            }
        
    def extract_all_features(self, price_data: List, volume_data: List = None) -> Dict[str, float]:
        """Extract comprehensive feature set from price and volume data"""
        
        if len(price_data) < 50:  # Need sufficient data for advanced features
            logger.warning("Insufficient data for advanced feature extraction")
            return self._get_default_features()
            
        try:
            # Initialize extractors lazily
            self._initialize_extractors()
            
            # Convert to pandas for easier manipulation
            df = self._prepare_dataframe(price_data, volume_data)
            
            features = {}
            
            # Extract features from each extractor
            for name, extractor in self.feature_extractors.items():
                try:
                    extracted = extractor.extract(df)
                    features.update(extracted)
                except Exception as e:
                    logger.warning(f"Failed to extract {name} features: {e}")
                    
            # Add basic technical indicators if not present
            features.update(self._extract_basic_indicators(df))
            
            # Clean and validate features
            features = self._clean_features(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._get_default_features()
    
    def _prepare_dataframe(self, price_data, volume_data):
        """Prepare pandas DataFrame from price and volume data"""
        
        # Extract OHLCV data
        data = []
        for i, price_point in enumerate(price_data):
            row = {
                'open': price_point.open_price,
                'high': price_point.high_price,
                'low': price_point.low_price,
                'close': price_point.close_price,
                'volume': volume_data[i] if volume_data and i < len(volume_data) else price_point.volume,
                'date': price_point.date
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _extract_basic_indicators(self, df):
        """Extract basic technical indicators as fallback"""
        
        features = {}
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            if len(close) >= period:
                ma = np.mean(close[-period:])
                features[f'sma_{period}'] = ma / close[-1] - 1  # Normalized
                
        # RSI
        if len(close) >= 14:
            features['rsi'] = self._calculate_rsi(close, 14)
            
        # MACD
        if len(close) >= 26:
            macd, signal = self._calculate_macd(close)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_histogram'] = macd - signal
            
        # Bollinger Bands
        if len(close) >= 20:
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(close, 20)
            features['bb_position'] = (close[-1] - bb_lower) / (bb_upper - bb_lower)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
        # Volume indicators
        if len(volume) >= 20:
            features['volume_ratio'] = volume[-1] / np.mean(volume[-20:])
            features['volume_trend'] = np.corrcoef(range(10), volume[-10:])[0, 1] if len(volume) >= 10 else 0
            
        return features
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi / 100  # Normalize to 0-1
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._ema([macd_line] * signal, signal)  # Simplified
        
        return macd_line / prices[-1], signal_line / prices[-1]  # Normalized
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, lower, sma
    
    def _ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
            
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
            
        return ema
    
    def _clean_features(self, features):
        """Clean and validate features"""
        cleaned = {}
        
        for name, value in features.items():
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                cleaned[name] = float(value)
            elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                # Take last value for arrays
                val = value[-1] if hasattr(value, '__getitem__') else value
                if not (np.isnan(val) or np.isinf(val)):
                    cleaned[name] = float(val)
                    
        return cleaned
    
    def _get_default_features(self):
        """Return default features when extraction fails"""
        return {
            'sma_5': 0.0, 'sma_10': 0.0, 'sma_20': 0.0,
            'rsi': 0.5, 'macd': 0.0, 'macd_signal': 0.0,
            'bb_position': 0.5, 'volume_ratio': 1.0
        }

class IchimokuExtractor:
    """Extract Ichimoku Cloud indicators"""
    
    def extract(self, df):
        """Extract Ichimoku features"""
        features = {}
        
        if len(df) < 52:  # Need at least 52 periods for Ichimoku
            return features
            
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        tenkan_sen = (np.max(high[-9:]) + np.min(low[-9:])) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        kijun_sen = (np.max(high[-26:]) + np.min(low[-26:])) / 2
        
        # Senkou Span A: (Tenkan-sen + Kijun-sen) / 2
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        
        # Senkou Span B: (52-period high + 52-period low) / 2
        senkou_span_b = (np.max(high[-52:]) + np.min(low[-52:])) / 2
        
        # Chikou Span: Current close price
        chikou_span = close[-1]
        
        # Normalize features
        current_price = close[-1]
        features.update({
            'ichimoku_tenkan_ratio': tenkan_sen / current_price - 1,
            'ichimoku_kijun_ratio': kijun_sen / current_price - 1,
            'ichimoku_senkou_a_ratio': senkou_span_a / current_price - 1,
            'ichimoku_senkou_b_ratio': senkou_span_b / current_price - 1,
            'ichimoku_cloud_thickness': abs(senkou_span_a - senkou_span_b) / current_price,
            'ichimoku_price_vs_cloud': self._price_vs_cloud_position(current_price, senkou_span_a, senkou_span_b),
            'ichimoku_tk_cross': 1 if tenkan_sen > kijun_sen else -1,
            'ichimoku_cloud_color': 1 if senkou_span_a > senkou_span_b else -1
        })
        
        return features
    
    def _price_vs_cloud_position(self, price, span_a, span_b):
        """Determine price position relative to cloud"""
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if price > cloud_top:
            return 1  # Above cloud
        elif price < cloud_bottom:
            return -1  # Below cloud
        else:
            return 0  # Inside cloud

class FibonacciExtractor:
    """Extract Fibonacci retracement levels"""
    
    def extract(self, df):
        """Extract Fibonacci features"""
        features = {}
        
        if len(df) < 20:
            return features
            
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Find recent swing high and low
        swing_high, swing_low = self._find_swing_points(high, low, lookback=20)
        
        if swing_high is None or swing_low is None:
            return features
            
        # Calculate Fibonacci levels
        fib_levels = self._calculate_fibonacci_levels(swing_high, swing_low)
        
        current_price = close[-1]
        
        # Find closest Fibonacci level
        closest_level, distance = self._find_closest_level(current_price, fib_levels)
        
        features.update({
            'fib_closest_level': closest_level,
            'fib_distance_to_closest': distance / current_price,
            'fib_retracement_ratio': (current_price - swing_low) / (swing_high - swing_low) if swing_high != swing_low else 0.5,
            'fib_swing_range': (swing_high - swing_low) / current_price
        })
        
        # Add individual level distances
        for level_name, level_price in fib_levels.items():
            features[f'fib_distance_{level_name}'] = abs(current_price - level_price) / current_price
            
        return features
    
    def _find_swing_points(self, high, low, lookback=20):
        """Find recent swing high and low points"""
        if len(high) < lookback:
            return None, None
            
        recent_high = np.max(high[-lookback:])
        recent_low = np.min(low[-lookback:])
        
        return recent_high, recent_low
    
    def _calculate_fibonacci_levels(self, swing_high, swing_low):
        """Calculate Fibonacci retracement levels"""
        diff = swing_high - swing_low
        
        levels = {
            '0': swing_low,
            '236': swing_low + 0.236 * diff,
            '382': swing_low + 0.382 * diff,
            '500': swing_low + 0.500 * diff,
            '618': swing_low + 0.618 * diff,
            '786': swing_low + 0.786 * diff,
            '1000': swing_high
        }
        
        return levels
    
    def _find_closest_level(self, price, levels):
        """Find closest Fibonacci level to current price"""
        min_distance = float('inf')
        closest_level = 0.5
        
        for level_name, level_price in levels.items():
            distance = abs(price - level_price)
            if distance < min_distance:
                min_distance = distance
                closest_level = float(level_name) / 1000  # Normalize
                
        return closest_level, min_distance

class VolumeProfileExtractor:
    """Extract volume profile analysis"""
    
    def extract(self, df):
        """Extract volume profile features"""
        features = {}
        
        if len(df) < 20 or 'volume' not in df.columns:
            return features
            
        # Create price-volume profile
        price_levels, volume_at_levels = self._create_volume_profile(df)
        
        if len(price_levels) == 0:
            return features
            
        current_price = df['close'].iloc[-1]
        
        # Point of Control (POC) - price level with highest volume
        poc_idx = np.argmax(volume_at_levels)
        poc_price = price_levels[poc_idx]
        
        # Value Area (70% of volume)
        value_area_high, value_area_low = self._calculate_value_area(price_levels, volume_at_levels)
        
        features.update({
            'vp_poc_distance': (current_price - poc_price) / current_price,
            'vp_value_area_position': self._get_value_area_position(current_price, value_area_high, value_area_low),
            'vp_value_area_width': (value_area_high - value_area_low) / current_price,
            'vp_volume_imbalance': self._calculate_volume_imbalance(df),
            'vp_high_volume_node': 1 if abs(current_price - poc_price) / current_price < 0.01 else 0
        })
        
        return features
    
    def _create_volume_profile(self, df, num_levels=20):
        """Create volume profile with specified number of price levels"""
        
        price_min = df['low'].min()
        price_max = df['high'].max()
        
        if price_max <= price_min:
            return [], []
            
        price_levels = np.linspace(price_min, price_max, num_levels)
        volume_at_levels = np.zeros(num_levels)
        
        for _, row in df.iterrows():
            # Find which price level this bar belongs to
            level_idx = np.searchsorted(price_levels, row['close']) - 1
            level_idx = max(0, min(level_idx, num_levels - 1))
            volume_at_levels[level_idx] += row['volume']
            
        return price_levels, volume_at_levels
    
    def _calculate_value_area(self, price_levels, volume_at_levels, percentage=0.7):
        """Calculate value area containing specified percentage of volume"""
        
        total_volume = np.sum(volume_at_levels)
        target_volume = total_volume * percentage
        
        # Start from POC and expand
        poc_idx = np.argmax(volume_at_levels)
        
        included_volume = volume_at_levels[poc_idx]
        low_idx = high_idx = poc_idx
        
        while included_volume < target_volume and (low_idx > 0 or high_idx < len(price_levels) - 1):
            # Expand to the side with more volume
            low_volume = volume_at_levels[low_idx - 1] if low_idx > 0 else 0
            high_volume = volume_at_levels[high_idx + 1] if high_idx < len(price_levels) - 1 else 0
            
            if low_volume >= high_volume and low_idx > 0:
                low_idx -= 1
                included_volume += volume_at_levels[low_idx]
            elif high_idx < len(price_levels) - 1:
                high_idx += 1
                included_volume += volume_at_levels[high_idx]
            else:
                break
                
        return price_levels[high_idx], price_levels[low_idx]
    
    def _get_value_area_position(self, price, va_high, va_low):
        """Get position relative to value area"""
        if price > va_high:
            return 1  # Above value area
        elif price < va_low:
            return -1  # Below value area
        else:
            return 0  # Inside value area
    
    def _calculate_volume_imbalance(self, df):
        """Calculate volume imbalance between up and down moves"""
        
        up_volume = 0
        down_volume = 0
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                up_volume += df['volume'].iloc[i]
            else:
                down_volume += df['volume'].iloc[i]
                
        total_volume = up_volume + down_volume
        if total_volume == 0:
            return 0
            
        return (up_volume - down_volume) / total_volume

class SupportResistanceExtractor:
    """Extract support and resistance levels"""
    
    def extract(self, df):
        """Extract support/resistance features"""
        features = {}
        
        if len(df) < 20:
            return features
            
        # Find support and resistance levels
        support_levels = self._find_support_levels(df)
        resistance_levels = self._find_resistance_levels(df)
        
        current_price = df['close'].iloc[-1]
        
        # Distance to nearest support/resistance
        nearest_support = self._find_nearest_level(current_price, support_levels, below=True)
        nearest_resistance = self._find_nearest_level(current_price, resistance_levels, below=False)
        
        features.update({
            'sr_nearest_support_distance': (current_price - nearest_support) / current_price if nearest_support else 0.1,
            'sr_nearest_resistance_distance': (nearest_resistance - current_price) / current_price if nearest_resistance else 0.1,
            'sr_support_strength': len(support_levels) / 10,  # Normalized
            'sr_resistance_strength': len(resistance_levels) / 10,  # Normalized
            'sr_price_position': self._get_price_position(current_price, support_levels, resistance_levels)
        })
        
        return features
    
    def _find_support_levels(self, df, window=5):
        """Find support levels using local minima"""
        lows = df['low'].values
        support_levels = []
        
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                support_levels.append(lows[i])
                
        return support_levels
    
    def _find_resistance_levels(self, df, window=5):
        """Find resistance levels using local maxima"""
        highs = df['high'].values
        resistance_levels = []
        
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                resistance_levels.append(highs[i])
                
        return resistance_levels
    
    def _find_nearest_level(self, price, levels, below=True):
        """Find nearest support (below) or resistance (above) level"""
        if not levels:
            return None
            
        if below:
            # Find highest level below current price
            below_levels = [level for level in levels if level < price]
            return max(below_levels) if below_levels else None
        else:
            # Find lowest level above current price
            above_levels = [level for level in levels if level > price]
            return min(above_levels) if above_levels else None
    
    def _get_price_position(self, price, support_levels, resistance_levels):
        """Get price position between support and resistance"""
        nearest_support = self._find_nearest_level(price, support_levels, below=True)
        nearest_resistance = self._find_nearest_level(price, resistance_levels, below=False)
        
        if nearest_support and nearest_resistance:
            return (price - nearest_support) / (nearest_resistance - nearest_support)
        else:
            return 0.5  # Neutral position

class ChartPatternExtractor:
    """Extract chart pattern features"""
    
    def extract(self, df):
        """Extract chart pattern features"""
        features = {}
        
        if len(df) < 30:
            return features
            
        close = df['close'].values
        
        # Trend analysis
        features.update(self._analyze_trend(close))
        
        # Volatility patterns
        features.update(self._analyze_volatility(df))
        
        # Price action patterns
        features.update(self._analyze_price_action(df))
        
        return features
    
    def _analyze_trend(self, prices):
        """Analyze trend characteristics"""
        features = {}
        
        # Linear trend
        x = np.arange(len(prices))
        trend_slope = np.polyfit(x, prices, 1)[0]
        features['trend_slope'] = trend_slope / prices[-1]  # Normalized
        
        # Trend strength (R-squared)
        trend_line = np.polyval([trend_slope, prices[0]], x)
        ss_res = np.sum((prices - trend_line) ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        features['trend_strength'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Recent trend vs overall trend
        recent_slope = np.polyfit(x[-10:], prices[-10:], 1)[0] if len(prices) >= 10 else trend_slope
        features['trend_acceleration'] = (recent_slope - trend_slope) / abs(trend_slope) if trend_slope != 0 else 0
        
        return features
    
    def _analyze_volatility(self, df):
        """Analyze volatility patterns"""
        features = {}
        
        # Calculate returns
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return features
            
        # Volatility metrics
        features['volatility'] = returns.std()
        features['volatility_skew'] = returns.skew()
        features['volatility_kurtosis'] = returns.kurtosis()
        
        # Volatility clustering
        abs_returns = abs(returns)
        features['volatility_clustering'] = abs_returns.autocorr(lag=1)
        
        # GARCH-like effect
        features['volatility_persistence'] = abs_returns.rolling(5).std().std() if len(returns) >= 5 else 0
        
        return features
    
    def _analyze_price_action(self, df):
        """Analyze price action patterns"""
        features = {}
        
        # Candlestick patterns
        features.update(self._detect_candlestick_patterns(df))
        
        # Gap analysis
        features.update(self._analyze_gaps(df))
        
        return features
    
    def _detect_candlestick_patterns(self, df):
        """Detect basic candlestick patterns"""
        features = {}
        
        if len(df) < 3:
            return features
            
        # Get last few candles
        last_candles = df.tail(3)
        
        # Doji pattern
        for i, (_, candle) in enumerate(last_candles.iterrows()):
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range > 0:
                doji_ratio = body_size / total_range
                features[f'doji_pattern_{i}'] = 1 if doji_ratio < 0.1 else 0
            else:
                features[f'doji_pattern_{i}'] = 0
                
        # Hammer/Hanging man
        last_candle = df.iloc[-1]
        body_size = abs(last_candle['close'] - last_candle['open'])
        lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
        upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        
        if body_size > 0:
            features['hammer_pattern'] = 1 if lower_shadow > 2 * body_size and upper_shadow < body_size else 0
        else:
            features['hammer_pattern'] = 0
            
        return features
    
    def _analyze_gaps(self, df):
        """Analyze price gaps"""
        features = {}
        
        if len(df) < 2:
            return features
            
        # Calculate gaps
        gaps = []
        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            curr_open = df['open'].iloc[i]
            gap = (curr_open - prev_close) / prev_close
            gaps.append(gap)
            
        if gaps:
            features['avg_gap_size'] = np.mean(np.abs(gaps))
            features['gap_frequency'] = sum(1 for gap in gaps if abs(gap) > 0.01) / len(gaps)
            features['recent_gap'] = gaps[-1] if gaps else 0
            
        return features

class MarketStructureExtractor:
    """Extract market structure features"""
    
    def extract(self, df):
        """Extract market structure features"""
        features = {}
        
        if len(df) < 20:
            return features
            
        # Market regime analysis
        features.update(self._analyze_market_regime(df))
        
        # Liquidity analysis
        features.update(self._analyze_liquidity(df))
        
        return features
    
    def _analyze_market_regime(self, df):
        """Analyze current market regime"""
        features = {}
        
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return features
            
        # Regime classification using volatility and returns
        avg_return = returns.mean()
        volatility = returns.std()
        
        # Simple regime classification
        if avg_return > 0.001 and volatility < 0.02:
            regime = 1  # Bull market
        elif avg_return < -0.001 and volatility < 0.02:
            regime = -1  # Bear market
        elif volatility > 0.03:
            regime = 0  # High volatility
        else:
            regime = 0.5  # Sideways
            
        features['market_regime'] = regime
        features['regime_volatility'] = volatility
        features['regime_return'] = avg_return
        
        return features
    
    def _analyze_liquidity(self, df):
        """Analyze liquidity characteristics"""
        features = {}
        
        # Volume-based liquidity measures
        volume = df['volume'].values
        
        if len(volume) < 10:
            return features
            
        # Volume consistency
        features['volume_consistency'] = 1 / (1 + np.std(volume) / np.mean(volume))
        
        # Volume trend
        x = np.arange(len(volume))
        volume_trend = np.polyfit(x, volume, 1)[0]
        features['volume_trend'] = volume_trend / np.mean(volume)
        
        # Price-volume relationship
        price_changes = df['close'].pct_change().dropna()
        volume_changes = df['volume'].pct_change().dropna()
        
        if len(price_changes) == len(volume_changes) and len(price_changes) > 5:
            correlation = np.corrcoef(abs(price_changes), volume_changes[1:])[0, 1]
            features['price_volume_correlation'] = correlation if not np.isnan(correlation) else 0
            
        return features

class AutoFeatureSelector:
    """Automated feature selection using multiple methods"""
    
    def __init__(self):
        self.selection_methods = {
            'mutual_info': self._mutual_info_selection,
            'f_regression': self._f_regression_selection,
            'lasso': self._lasso_selection,
            'random_forest': self._random_forest_selection
        }
        self.selected_features = []
        
    def select_features(self, X, y, method='ensemble', max_features=50):
        """Select most predictive features"""
        
        if X.shape[1] <= max_features:
            return list(range(X.shape[1]))  # Return all features if not too many
            
        if method == 'ensemble':
            selected_features = self._ensemble_feature_selection(X, y, max_features)
        else:
            selected_features = self.selection_methods[method](X, y, max_features)
            
        self.selected_features = selected_features
        return selected_features
    
    def _ensemble_feature_selection(self, X, y, max_features):
        """Use ensemble of selection methods"""
        
        feature_scores = defaultdict(list)
        
        for method_name, selector in self.selection_methods.items():
            try:
                scores = selector(X, y, X.shape[1])  # Get all scores first
                
                # Normalize scores to 0-1 range
                if len(scores) > 0:
                    min_score, max_score = min(scores), max(scores)
                    if max_score > min_score:
                        normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
                    else:
                        normalized_scores = [0.5] * len(scores)
                        
                    for i, score in enumerate(normalized_scores):
                        feature_scores[i].append(score)
                        
            except Exception as e:
                logger.warning(f"Feature selection method {method_name} failed: {e}")
                
        # Average scores across methods
        final_scores = {}
        for feature_idx, scores in feature_scores.items():
            final_scores[feature_idx] = np.mean(scores)
            
        # Select top features
        sorted_features = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, score in sorted_features[:max_features]]
        
        return selected_indices
    
    def _mutual_info_selection(self, X, y, max_features):
        """Mutual information feature selection"""
        try:
            scores = mutual_info_regression(X, y, random_state=42)
            top_indices = np.argsort(scores)[-max_features:]
            return top_indices.tolist()
        except:
            return list(range(min(max_features, X.shape[1])))
    
    def _f_regression_selection(self, X, y, max_features):
        """F-regression feature selection"""
        try:
            scores, _ = f_regression(X, y)
            top_indices = np.argsort(scores)[-max_features:]
            return top_indices.tolist()
        except:
            return list(range(min(max_features, X.shape[1])))
    
    def _lasso_selection(self, X, y, max_features):
        """LASSO-based feature selection"""
        try:
            lasso = Lasso(alpha=0.01, random_state=42)
            selector = SelectFromModel(lasso, max_features=max_features)
            selector.fit(X, y)
            return selector.get_support(indices=True).tolist()
        except:
            return list(range(min(max_features, X.shape[1])))
    
    def _random_forest_selection(self, X, y, max_features):
        """Random Forest importance-based selection"""
        try:
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = SelectFromModel(rf, max_features=max_features)
            selector.fit(X, y)
            return selector.get_support(indices=True).tolist()
        except:
            return list(range(min(max_features, X.shape[1])))

if __name__ == "__main__":
    print("Testing Advanced Feature Engineering...")
    
    # Create sample data
    from models import StockData
    from datetime import datetime, timedelta
    
    # Generate sample stock data
    sample_data = []
    base_price = 100.0
    base_date = datetime.now() - timedelta(days=100)
    
    for i in range(100):
        price_change = np.random.normal(0, 0.02)
        new_price = base_price * (1 + price_change)
        
        stock_data = StockData(
            symbol="TEST",
            date=base_date + timedelta(days=i),
            open_price=base_price,
            high_price=new_price * 1.01,
            low_price=new_price * 0.99,
            close_price=new_price,
            volume=int(np.random.normal(1000000, 200000))
        )
        sample_data.append(stock_data)
        base_price = new_price
    
    # Test feature extraction
    feature_engineer = AdvancedFeatureEngineer()
    features = feature_engineer.extract_all_features(sample_data)
    
    print(f"Extracted {len(features)} features:")
    for name, value in list(features.items())[:10]:  # Show first 10
        print(f"  {name}: {value:.6f}")
    
    print("\nAdvanced feature engineering test completed!")