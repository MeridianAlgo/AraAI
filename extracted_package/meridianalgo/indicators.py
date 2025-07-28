"""
Technical Indicators Module for MeridianAlgo
Provides common technical analysis indicators
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


class Indicators:
    """
    Collection of technical analysis indicators
    """
    
    @staticmethod
    def sma(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        
        Args:
            data: Price data (close prices)
            period: Period for the moving average
            
        Returns:
            pd.Series: Simple Moving Average
        """
        return pd.Series(data).rolling(window=period).mean()
    
    @staticmethod
    def ema(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        
        Args:
            data: Price data (close prices)
            period: Period for the moving average
            
        Returns:
            pd.Series: Exponential Moving Average
        """
        return pd.Series(data).ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            data: Price data (close prices)
            period: Period for RSI calculation (default: 14)
            
        Returns:
            pd.Series: RSI values
        """
        data = pd.Series(data)
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: Union[pd.Series, np.ndarray], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price data (close prices)
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            
        Returns:
            tuple: (macd_line, signal_line, histogram)
        """
        data = pd.Series(data)
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: Union[pd.Series, np.ndarray], period: int = 20, std_dev: float = 2) -> tuple:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Price data (close prices)
            period: Period for moving average (default: 20)
            std_dev: Number of standard deviations (default: 2)
            
        Returns:
            tuple: (upper_band, middle_band, lower_band)
        """
        data = pd.Series(data)
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def stochastic(high: Union[pd.Series, np.ndarray], 
                  low: Union[pd.Series, np.ndarray], 
                  close: Union[pd.Series, np.ndarray], 
                  k_period: int = 14, d_period: int = 3) -> tuple:
        """
        Calculate Stochastic Oscillator
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            
        Returns:
            tuple: (%K, %D)
        """
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: Union[pd.Series, np.ndarray], 
            low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray], 
            period: int = 14) -> pd.Series:
        """
        Calculate Average True Range
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for ATR calculation (default: 14)
            
        Returns:
            pd.Series: ATR values
        """
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def volume_sma(volume: Union[pd.Series, np.ndarray], period: int = 20) -> pd.Series:
        """
        Calculate Volume Simple Moving Average
        
        Args:
            volume: Volume data
            period: Period for moving average (default: 20)
            
        Returns:
            pd.Series: Volume SMA
        """
        return pd.Series(volume).rolling(window=period).mean()
    
    @staticmethod
    def price_channels(high: Union[pd.Series, np.ndarray], 
                      low: Union[pd.Series, np.ndarray], 
                      period: int = 20) -> tuple:
        """
        Calculate Price Channels (Donchian Channels)
        
        Args:
            high: High prices
            low: Low prices
            period: Period for channels (default: 20)
            
        Returns:
            tuple: (upper_channel, lower_channel, middle_channel)
        """
        high = pd.Series(high)
        low = pd.Series(low)
        
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return upper_channel, lower_channel, middle_channel
    
    @staticmethod
    def williams_r(high: Union[pd.Series, np.ndarray], 
                  low: Union[pd.Series, np.ndarray], 
                  close: Union[pd.Series, np.ndarray], 
                  period: int = 14) -> pd.Series:
        """
        Calculate Williams %R
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for calculation (default: 14)
            
        Returns:
            pd.Series: Williams %R values
        """
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    @staticmethod
    def cci(high: Union[pd.Series, np.ndarray], 
            low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray], 
            period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for calculation (default: 20)
            
        Returns:
            pd.Series: CCI values
        """
        high = pd.Series(high)
        low = pd.Series(low)
        close = pd.Series(close)
        
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci 