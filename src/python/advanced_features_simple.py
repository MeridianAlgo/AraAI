"""
Simplified Advanced Feature Engineering
Basic implementation to get the system working
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Simplified advanced feature engineering"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_all_features(self, price_data: List, volume_data: List = None) -> Dict[str, float]:
        """Extract comprehensive feature set from price and volume data"""
        
        if len(price_data) < 20:  # Need sufficient data
            logger.warning("Insufficient data for advanced feature extraction")
            return self._get_default_features()
            
        try:
            # Convert to pandas for easier manipulation
            df = self._prepare_dataframe(price_data, volume_data)
            
            features = {}
            
            # Extract basic indicators
            features.update(self._extract_basic_indicators(df))
            
            # Extract advanced indicators
            features.update(self._extract_advanced_indicators(df))
            
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
            # Ensure date is offset-naive
            date_val = price_point.date
            if hasattr(date_val, 'tzinfo') and date_val.tzinfo is not None:
                date_val = date_val.replace(tzinfo=None)
            row = {
                'open': price_point.open_price,
                'high': price_point.high_price,
                'low': price_point.low_price,
                'close': price_point.close_price,
                'volume': volume_data[i] if volume_data and i < len(volume_data) else price_point.volume,
                'date': date_val
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _extract_basic_indicators(self, df):
        """Extract basic technical indicators"""
        
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
            features['bb_position'] = (close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0.1
            
        # Volume indicators
        if len(volume) >= 20:
            avg_volume = np.mean(volume[-20:])
            features['volume_ratio'] = volume[-1] / avg_volume if avg_volume != 0 else 1.0
            if len(volume) >= 10:
                try:
                    features['volume_trend'] = np.corrcoef(range(10), volume[-10:])[0, 1]
                except:
                    features['volume_trend'] = 0.0
            else:
                features['volume_trend'] = 0.0
            
        return features
    
    def _extract_advanced_indicators(self, df):
        """Extract advanced technical indicators"""
        
        features = {}
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Ichimoku-like indicators
        if len(df) >= 26:
            # Simplified Ichimoku
            tenkan = (np.max(high[-9:]) + np.min(low[-9:])) / 2 if len(high) >= 9 else close[-1]
            kijun = (np.max(high[-26:]) + np.min(low[-26:])) / 2
            
            features['ichimoku_tenkan_ratio'] = tenkan / close[-1] - 1
            features['ichimoku_kijun_ratio'] = kijun / close[-1] - 1
            features['ichimoku_tk_cross'] = 1 if tenkan > kijun else -1
            
        # Fibonacci-like levels
        if len(high) >= 20:
            recent_high = np.max(high[-20:])
            recent_low = np.min(low[-20:])
            
            if recent_high != recent_low:
                fib_ratio = (close[-1] - recent_low) / (recent_high - recent_low)
                features['fib_retracement_ratio'] = fib_ratio
                features['fib_swing_range'] = (recent_high - recent_low) / close[-1]
            else:
                features['fib_retracement_ratio'] = 0.5
                features['fib_swing_range'] = 0.01
                
        # Support/Resistance levels
        if len(df) >= 20:
            # Find local minima and maxima
            support_levels = self._find_support_levels(low)
            resistance_levels = self._find_resistance_levels(high)
            
            current_price = close[-1]
            
            # Distance to nearest levels
            nearest_support = max([s for s in support_levels if s < current_price], default=current_price * 0.95)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=current_price * 1.05)
            
            features['sr_support_distance'] = (current_price - nearest_support) / current_price
            features['sr_resistance_distance'] = (nearest_resistance - current_price) / current_price
            features['sr_support_strength'] = len(support_levels) / 10
            features['sr_resistance_strength'] = len(resistance_levels) / 10
            
        # Volatility patterns
        returns = df['close'].pct_change().dropna()
        if len(returns) >= 10:
            features['volatility'] = returns.std()
            features['volatility_skew'] = returns.skew() if len(returns) > 2 else 0.0
            
            # Trend analysis
            x = np.arange(len(close))
            if len(close) > 1:
                trend_slope = np.polyfit(x, close, 1)[0]
                features['trend_slope'] = trend_slope / close[-1]
            else:
                features['trend_slope'] = 0.0
                
        # Market regime indicators
        if len(returns) >= 10:
            avg_return = returns.mean()
            volatility = returns.std()
            
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
    
    def _find_support_levels(self, lows, window=5):
        """Find support levels using local minima"""
        support_levels = []
        
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                support_levels.append(lows[i])
                
        return support_levels
    
    def _find_resistance_levels(self, highs, window=5):
        """Find resistance levels using local maxima"""
        resistance_levels = []
        
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                resistance_levels.append(highs[i])
                
        return resistance_levels
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 1.0
        
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
            'bb_position': 0.5, 'volume_ratio': 1.0,
            'ichimoku_tenkan_ratio': 0.0, 'ichimoku_kijun_ratio': 0.0,
            'fib_retracement_ratio': 0.5, 'volatility': 0.02,
            'trend_slope': 0.0, 'market_regime': 0.5
        }

class AutoFeatureSelector:
    """Simplified automated feature selection"""
    
    def __init__(self):
        self.selected_features = []
        
    def select_features(self, X, y, method='simple', max_features=50):
        """Select most predictive features using simple correlation"""
        
        if X.shape[1] <= max_features:
            return list(range(X.shape[1]))  # Return all features if not too many
            
        try:
            # Simple correlation-based selection
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                correlations.append((i, abs(corr) if not np.isnan(corr) else 0))
                
            # Sort by correlation and take top features
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, corr in correlations[:max_features]]
            
            self.selected_features = selected_indices
            return selected_indices
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}")
            return list(range(min(max_features, X.shape[1])))

if __name__ == "__main__":
    print("Testing Simplified Advanced Feature Engineering...")
    
    # Create sample data
    from models import StockData
    from datetime import datetime, timedelta
    
    # Generate sample stock data
    sample_data = []
    base_price = 100.0
    base_date = datetime.now() - timedelta(days=60)
    
    for i in range(60):
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
    for name, value in list(features.items())[:15]:  # Show first 15
        print(f"  {name}: {value:.6f}")
    
    print("\nSimplified advanced feature engineering test completed!")