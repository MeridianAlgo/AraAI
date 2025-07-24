"""
Technical Indicators Calculation Engine for ML Stock Predictor
Implements RSI, MACD, SMA, EMA, Bollinger Bands, Stochastic, Williams %R, and CCI
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate various technical indicators for stock analysis"""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index (RSI)"""
        if len(prices) < period + 1:
            return [50.0] * len(prices)  # Default neutral RSI
        
        prices = np.array(prices)
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi_values = []
        
        # Fill initial values with neutral RSI (one for each price point)
        for _ in range(period):
            rsi_values.append(50.0)
        
        # Calculate RSI for remaining values
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        # Add one more value to match the length of prices
        if len(rsi_values) < len(prices):
            rsi_values.append(rsi_values[-1])
        
        return rsi_values
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            default_val = 0.0
            return {
                'macd': [default_val] * len(prices),
                'signal': [default_val] * len(prices),
                'histogram': [default_val] * len(prices)
            }
        
        prices = np.array(prices)
        
        # Calculate EMAs
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        # MACD line
        macd_line = np.array(ema_fast) - np.array(ema_slow)
        
        # Signal line (EMA of MACD)
        signal_line = self.calculate_ema(macd_line.tolist(), signal)
        
        # Histogram
        histogram = macd_line - np.array(signal_line)
        
        return {
            'macd': macd_line.tolist(),
            'signal': signal_line,
            'histogram': histogram.tolist()
        }
    
    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average (SMA)"""
        if len(prices) < period:
            return [np.mean(prices)] * len(prices)
        
        sma_values = []
        prices = np.array(prices)
        
        # Fill initial values
        for i in range(period - 1):
            sma_values.append(np.mean(prices[:i+1]))
        
        # Calculate SMA for remaining values
        for i in range(period - 1, len(prices)):
            sma = np.mean(prices[i - period + 1:i + 1])
            sma_values.append(sma)
        
        return sma_values
    
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average (EMA)"""
        if len(prices) == 0:
            return []
        
        prices = np.array(prices)
        alpha = 2.0 / (period + 1)
        
        ema_values = [prices[0]]  # Start with first price
        
        for i in range(1, len(prices)):
            ema = alpha * prices[i] + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict[str, List[float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            avg_price = np.mean(prices)
            return {
                'upper': [avg_price * 1.02] * len(prices),
                'middle': [avg_price] * len(prices),
                'lower': [avg_price * 0.98] * len(prices)
            }
        
        sma = self.calculate_sma(prices, period)
        prices = np.array(prices)
        
        upper_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if i < period - 1:
                # Use available data for initial calculations
                window = prices[:i+1]
            else:
                window = prices[i - period + 1:i + 1]
            
            std = np.std(window)
            upper_band.append(sma[i] + (std_dev * std))
            lower_band.append(sma[i] - (std_dev * std))
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    def calculate_stochastic(self, high: List[float], low: List[float], close: List[float], 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, List[float]]:
        """Calculate Stochastic Oscillator"""
        if len(close) < k_period:
            return {
                'k': [50.0] * len(close),
                'd': [50.0] * len(close)
            }
        
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        
        k_values = []
        
        for i in range(len(close)):
            if i < k_period - 1:
                # Use available data
                period_high = np.max(high[:i+1])
                period_low = np.min(low[:i+1])
            else:
                period_high = np.max(high[i - k_period + 1:i + 1])
                period_low = np.min(low[i - k_period + 1:i + 1])
            
            if period_high == period_low:
                k = 50.0
            else:
                k = ((close[i] - period_low) / (period_high - period_low)) * 100
            
            k_values.append(k)
        
        # Calculate %D (SMA of %K)
        d_values = self.calculate_sma(k_values, d_period)
        
        return {
            'k': k_values,
            'd': d_values
        }
    
    def calculate_williams_r(self, high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
        """Calculate Williams %R"""
        if len(close) < period:
            return [-50.0] * len(close)  # Default neutral value
        
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        
        williams_r = []
        
        for i in range(len(close)):
            if i < period - 1:
                period_high = np.max(high[:i+1])
                period_low = np.min(low[:i+1])
            else:
                period_high = np.max(high[i - period + 1:i + 1])
                period_low = np.min(low[i - period + 1:i + 1])
            
            if period_high == period_low:
                wr = -50.0
            else:
                wr = ((period_high - close[i]) / (period_high - period_low)) * -100
            
            williams_r.append(wr)
        
        return williams_r
    
    def calculate_cci(self, high: List[float], low: List[float], close: List[float], period: int = 20) -> List[float]:
        """Calculate Commodity Channel Index (CCI)"""
        if len(close) < period:
            return [0.0] * len(close)
        
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        
        # Calculate Typical Price
        typical_price = (high + low + close) / 3
        
        cci_values = []
        
        for i in range(len(typical_price)):
            if i < period - 1:
                tp_window = typical_price[:i+1]
            else:
                tp_window = typical_price[i - period + 1:i + 1]
            
            sma_tp = np.mean(tp_window)
            mean_deviation = np.mean(np.abs(tp_window - sma_tp))
            
            if mean_deviation == 0:
                cci = 0.0
            else:
                cci = (typical_price[i] - sma_tp) / (0.015 * mean_deviation)
            
            cci_values.append(cci)
        
        return cci_values
    
    def calculate_all_indicators(self, ohlcv_data: List[Dict]) -> List[Dict]:
        """Calculate all technical indicators for OHLCV data"""
        if not ohlcv_data:
            return []
        
        # Extract price arrays
        high_prices = [d['high_price'] for d in ohlcv_data]
        low_prices = [d['low_price'] for d in ohlcv_data]
        close_prices = [d['close_price'] for d in ohlcv_data]
        volumes = [d['volume'] for d in ohlcv_data]
        
        logger.info(f"Calculating indicators for {len(ohlcv_data)} data points")
        
        # Calculate all indicators
        rsi = self.calculate_rsi(close_prices)
        macd_data = self.calculate_macd(close_prices)
        sma_5 = self.calculate_sma(close_prices, 5)
        sma_10 = self.calculate_sma(close_prices, 10)
        sma_20 = self.calculate_sma(close_prices, 20)
        sma_50 = self.calculate_sma(close_prices, 50)
        ema_12 = self.calculate_ema(close_prices, 12)
        ema_26 = self.calculate_ema(close_prices, 26)
        bollinger = self.calculate_bollinger_bands(close_prices)
        stochastic = self.calculate_stochastic(high_prices, low_prices, close_prices)
        williams_r = self.calculate_williams_r(high_prices, low_prices, close_prices)
        cci = self.calculate_cci(high_prices, low_prices, close_prices)
        
        # Combine with original data
        enhanced_data = []
        for i, data in enumerate(ohlcv_data):
            enhanced = data.copy()
            enhanced['indicators'] = {
                'rsi': rsi[i],
                'macd': macd_data['macd'][i],
                'macd_signal': macd_data['signal'][i],
                'macd_histogram': macd_data['histogram'][i],
                'sma_5': sma_5[i],
                'sma_10': sma_10[i],
                'sma_20': sma_20[i],
                'sma_50': sma_50[i],
                'ema_12': ema_12[i],
                'ema_26': ema_26[i],
                'bollinger_upper': bollinger['upper'][i],
                'bollinger_middle': bollinger['middle'][i],
                'bollinger_lower': bollinger['lower'][i],
                'stochastic_k': stochastic['k'][i],
                'stochastic_d': stochastic['d'][i],
                'williams_r': williams_r[i],
                'cci': cci[i]
            }
            enhanced_data.append(enhanced)
        
        logger.info("✅ All technical indicators calculated successfully")
        return enhanced_data
    
    def normalize_indicators(self, indicators_data: List[Dict]) -> List[Dict]:
        """Normalize indicator values for neural network input"""
        if not indicators_data:
            return []
        
        # Extract all indicator values
        all_indicators = {}
        indicator_names = list(indicators_data[0]['indicators'].keys())
        
        for name in indicator_names:
            all_indicators[name] = [d['indicators'][name] for d in indicators_data]
        
        # Normalize each indicator separately
        normalized_indicators = {}
        for name, values in all_indicators.items():
            values_array = np.array(values).reshape(-1, 1)
            
            # Handle special cases for different indicator ranges
            if name == 'rsi' or name.startswith('stochastic'):
                # RSI and Stochastic are already 0-100, just divide by 100
                normalized_values = values_array / 100.0
            elif name == 'williams_r':
                # Williams %R is -100 to 0, normalize to 0-1
                normalized_values = (values_array + 100) / 100.0
            elif name.startswith('macd'):
                # MACD can vary widely, use standard normalization
                if np.std(values_array) > 0:
                    normalized_values = self.scaler.fit_transform(values_array)
                else:
                    normalized_values = values_array * 0  # All zeros if no variation
            else:
                # For other indicators, use min-max normalization
                if np.max(values_array) != np.min(values_array):
                    normalized_values = self.scaler.fit_transform(values_array)
                else:
                    normalized_values = values_array * 0 + 0.5  # Neutral value
            
            normalized_indicators[name] = normalized_values.flatten().tolist()
        
        # Update the data with normalized indicators
        normalized_data = []
        for i, data in enumerate(indicators_data):
            normalized = data.copy()
            normalized['normalized_indicators'] = {}
            for name in indicator_names:
                normalized['normalized_indicators'][name] = normalized_indicators[name][i]
            normalized_data.append(normalized)
        
        logger.info("✅ Indicators normalized for ML model input")
        return normalized_data
    
    def get_feature_vector(self, indicators: Dict[str, float], normalized: bool = True) -> List[float]:
        """Get feature vector for ML model input"""
        indicator_names = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'stochastic_k', 'stochastic_d',
            'williams_r', 'cci'
        ]
        
        if normalized and 'normalized_indicators' in indicators:
            return [indicators['normalized_indicators'].get(name, 0.0) for name in indicator_names]
        else:
            return [indicators.get(name, 0.0) for name in indicator_names]

# Global instance
technical_indicators = TechnicalIndicators()

if __name__ == "__main__":
    print("Technical Indicators Engine - Setting up...")
    print("Available indicators: RSI, MACD, SMA, EMA, Bollinger Bands, Stochastic, Williams %R, CCI")