"""
MeridianAlgo Smart Trader Analysis Module
Advanced volatility analysis and technical indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

def detect_volatility_spikes(data_df: pd.DataFrame) -> Dict:
    """
    Detect future volatility spikes based on historical patterns
    
    This advanced algorithm analyzes historical volatility patterns to predict
    future volatility spikes, helping traders prepare for market turbulence.
    
    Args:
        data_df: DataFrame with OHLCV data
        
    Returns:
        Dict with spike probability, expected timing, and magnitude
    """
    try:
        prices = data_df['Close'].values
        volumes = data_df['Volume'].values
        
        # Calculate rolling volatility (20-day window)
        returns = np.diff(prices) / prices[:-1]
        rolling_vol = []
        
        for i in range(19, len(returns)):
            window_vol = np.std(returns[i-19:i+1]) * 100
            rolling_vol.append(window_vol)
        
        if len(rolling_vol) < 10:
            return {
                'spike_probability': 0, 
                'expected_spike_days': 0, 
                'spike_magnitude': 0,
                'current_volatility': 0,
                'risk_level': 'Unknown'
            }
        
        # Identify historical volatility spikes (>2 std deviations)
        vol_mean = np.mean(rolling_vol)
        vol_std = np.std(rolling_vol)
        spike_threshold = vol_mean + (2 * vol_std)
        
        # Find spike patterns
        spike_indices = [i for i, vol in enumerate(rolling_vol) if vol > spike_threshold]
        
        if len(spike_indices) < 2:
            return {
                'spike_probability': 15, 
                'expected_spike_days': 0, 
                'spike_magnitude': 0,
                'current_volatility': rolling_vol[-1] if rolling_vol else 0,
                'risk_level': 'Low'
            }
        
        # Calculate spike frequency and patterns
        spike_intervals = np.diff(spike_indices)
        avg_interval = np.mean(spike_intervals) if len(spike_intervals) > 0 else 30
        
        # Days since last spike
        days_since_spike = len(rolling_vol) - max(spike_indices) if spike_indices else 999
        
        # Advanced probability calculation based on historical patterns
        if days_since_spike > avg_interval * 0.8:
            spike_probability = min(85, 30 + (days_since_spike / avg_interval) * 25)
        else:
            spike_probability = max(10, 30 - (days_since_spike / avg_interval) * 20)
        
        # Expected spike timing
        expected_days = max(1, int(avg_interval - days_since_spike)) if days_since_spike < avg_interval else int(avg_interval * 0.3)
        
        # Expected magnitude based on historical spikes
        historical_spikes = [rolling_vol[i] for i in spike_indices]
        expected_magnitude = np.mean(historical_spikes) if historical_spikes else vol_mean * 1.5
        
        # Volume confirmation (high volume often precedes volatility spikes)
        recent_volume_trend = np.mean(volumes[-5:]) / np.mean(volumes[-20:])
        if recent_volume_trend > 1.2:
            spike_probability *= 1.15
        
        # Risk level classification
        if spike_probability > 60:
            risk_level = 'High'
        elif spike_probability > 35:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'spike_probability': min(spike_probability, 90),
            'expected_spike_days': min(expected_days, 15),
            'spike_magnitude': expected_magnitude,
            'current_volatility': rolling_vol[-1] if rolling_vol else vol_mean,
            'risk_level': risk_level,
            'volume_confirmation': recent_volume_trend > 1.2,
            'historical_spikes': len(spike_indices),
            'avg_spike_interval': avg_interval
        }
        
    except Exception as e:
        return {
            'spike_probability': 20, 
            'expected_spike_days': 7, 
            'spike_magnitude': 0,
            'current_volatility': 0,
            'risk_level': 'Unknown',
            'error': str(e)
        }

def calculate_technical_indicators(data_df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive technical indicators
    
    Args:
        data_df: DataFrame with OHLCV data
        
    Returns:
        Dict with calculated technical indicators
    """
    try:
        # Try to use the advanced indicators module
        try:
            import sys
            import os
            sys.path.append('src/python')
            from indicators import technical_indicators
            
            # Convert DataFrame to list format
            data_list = []
            for idx, row in data_df.iterrows():
                data_list.append({
                    'close': row['Close'],
                    'high': row['High'],
                    'low': row['Low'],
                    'volume': row['Volume'],
                    'open': row['Open']
                })
            
            indicators_data = {}
            if len(data_list) >= 14:
                indicators_data['rsi'] = technical_indicators.calculate_rsi([d['close'] for d in data_list])
                indicators_data['macd'] = technical_indicators.calculate_macd([d['close'] for d in data_list])
                indicators_data['sma_20'] = technical_indicators.calculate_sma([d['close'] for d in data_list], 20)
                indicators_data['ema_12'] = technical_indicators.calculate_ema([d['close'] for d in data_list], 12)
            
            return indicators_data
            
        except ImportError:
            # Fallback to basic indicators
            return calculate_basic_indicators(data_df)
            
    except Exception as e:
        return {'error': str(e)}

def calculate_basic_indicators(data_df: pd.DataFrame) -> Dict:
    """
    Calculate basic technical indicators as fallback
    
    Args:
        data_df: DataFrame with OHLCV data
        
    Returns:
        Dict with basic technical indicators
    """
    try:
        closes = data_df['Close'].values
        highs = data_df['High'].values
        lows = data_df['Low'].values
        
        indicators = {}
        
        # Simple Moving Average (20-period)
        if len(closes) >= 20:
            sma_20 = []
            for i in range(19, len(closes)):
                sma_20.append(np.mean(closes[i-19:i+1]))
            indicators['sma_20'] = sma_20
        
        # Exponential Moving Average (12-period)
        if len(closes) >= 12:
            ema_12 = []
            multiplier = 2 / (12 + 1)
            ema = closes[0]
            
            for price in closes:
                ema = (price * multiplier) + (ema * (1 - multiplier))
                ema_12.append(ema)
            
            indicators['ema_12'] = ema_12[11:]  # Skip first 11 values
        
        # RSI (14-period)
        if len(closes) >= 15:
            rsi_values = []
            period = 14
            
            for i in range(period, len(closes)):
                price_changes = np.diff(closes[i-period:i+1])
                gains = np.where(price_changes > 0, price_changes, 0)
                losses = np.where(price_changes < 0, -price_changes, 0)
                
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                rsi_values.append(rsi)
            
            indicators['rsi'] = rsi_values
        
        return indicators
        
    except Exception as e:
        return {'error': str(e)}

def analyze_stock(symbol: str, days: int = 60, epochs: int = 10, verbose: bool = False) -> Dict:
    """
    Convenient function to analyze a stock with Smart Trader
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
        days: Historical data days (default: 60)
        epochs: Training epochs (default: 10)
        verbose: Enable verbose output (default: False)
        
    Returns:
        Dict with analysis results
    """
    from .core import SmartTrader
    
    trader = SmartTrader(verbose=verbose)
    return trader.analyze(symbol, days, epochs)

def get_market_sentiment(data_df: pd.DataFrame) -> Dict:
    """
    Analyze market sentiment based on price and volume patterns
    
    Args:
        data_df: DataFrame with OHLCV data
        
    Returns:
        Dict with sentiment analysis
    """
    try:
        prices = data_df['Close'].values
        volumes = data_df['Volume'].values
        
        # Price trend analysis
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-20:])
        
        if short_ma > long_ma * 1.02:
            price_sentiment = 'Bullish'
            price_strength = min(((short_ma / long_ma) - 1) * 100, 10)
        elif short_ma < long_ma * 0.98:
            price_sentiment = 'Bearish'
            price_strength = min(((long_ma / short_ma) - 1) * 100, 10)
        else:
            price_sentiment = 'Neutral'
            price_strength = 0
        
        # Volume analysis
        recent_volume = np.mean(volumes[-5:])
        avg_volume = np.mean(volumes[-20:])
        volume_ratio = recent_volume / avg_volume
        
        if volume_ratio > 1.2:
            volume_sentiment = 'High Interest'
        elif volume_ratio < 0.8:
            volume_sentiment = 'Low Interest'
        else:
            volume_sentiment = 'Normal'
        
        # Overall sentiment
        if price_sentiment == 'Bullish' and volume_ratio > 1.1:
            overall_sentiment = 'Strong Bullish'
        elif price_sentiment == 'Bearish' and volume_ratio > 1.1:
            overall_sentiment = 'Strong Bearish'
        else:
            overall_sentiment = price_sentiment
        
        return {
            'overall_sentiment': overall_sentiment,
            'price_sentiment': price_sentiment,
            'price_strength': price_strength,
            'volume_sentiment': volume_sentiment,
            'volume_ratio': volume_ratio,
            'confidence': min(85, 60 + (price_strength * 2) + (abs(volume_ratio - 1) * 20))
        }
        
    except Exception as e:
        return {
            'overall_sentiment': 'Unknown',
            'error': str(e)
        }