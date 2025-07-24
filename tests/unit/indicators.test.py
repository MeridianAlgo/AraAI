#!/usr/bin/env python3
"""
Unit tests for Technical Indicators Engine
"""

import sys
import os
sys.path.append('src/python')

import unittest
import numpy as np
from indicators import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    
    def setUp(self):
        self.ti = TechnicalIndicators()
        # Sample price data for testing
        self.sample_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
        self.sample_high = [101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 115, 114]
        self.sample_low = [99, 101, 100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112]
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        rsi = self.ti.calculate_rsi(self.sample_prices)
        
        # RSI should be between 0 and 100
        self.assertTrue(all(0 <= r <= 100 for r in rsi))
        self.assertEqual(len(rsi), len(self.sample_prices))
        
        # Test with insufficient data
        short_prices = [100, 102]
        rsi_short = self.ti.calculate_rsi(short_prices)
        self.assertEqual(rsi_short, [50.0, 50.0])
    
    def test_calculate_sma(self):
        """Test Simple Moving Average calculation"""
        sma_5 = self.ti.calculate_sma(self.sample_prices, 5)
        
        self.assertEqual(len(sma_5), len(self.sample_prices))
        
        # Check that SMA is calculated correctly for the 5th element
        expected_sma_5th = np.mean(self.sample_prices[:5])
        self.assertAlmostEqual(sma_5[4], expected_sma_5th, places=2)
        
        # Test with insufficient data
        short_prices = [100, 102]
        sma_short = self.ti.calculate_sma(short_prices, 5)
        self.assertEqual(len(sma_short), 2)
    
    def test_calculate_ema(self):
        """Test Exponential Moving Average calculation"""
        ema = self.ti.calculate_ema(self.sample_prices, 5)
        
        self.assertEqual(len(ema), len(self.sample_prices))
        self.assertEqual(ema[0], self.sample_prices[0])  # First value should be the first price
        
        # EMA should be different from SMA
        sma = self.ti.calculate_sma(self.sample_prices, 5)
        self.assertNotEqual(ema[-1], sma[-1])
    
    def test_calculate_macd(self):
        """Test MACD calculation"""
        macd_data = self.ti.calculate_macd(self.sample_prices)
        
        self.assertIn('macd', macd_data)
        self.assertIn('signal', macd_data)
        self.assertIn('histogram', macd_data)
        
        self.assertEqual(len(macd_data['macd']), len(self.sample_prices))
        self.assertEqual(len(macd_data['signal']), len(self.sample_prices))
        self.assertEqual(len(macd_data['histogram']), len(self.sample_prices))
        
        # Test with insufficient data
        short_prices = [100, 102]
        macd_short = self.ti.calculate_macd(short_prices)
        self.assertEqual(macd_short['macd'], [0.0, 0.0])
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        bb = self.ti.calculate_bollinger_bands(self.sample_prices)
        
        self.assertIn('upper', bb)
        self.assertIn('middle', bb)
        self.assertIn('lower', bb)
        
        self.assertEqual(len(bb['upper']), len(self.sample_prices))
        self.assertEqual(len(bb['middle']), len(self.sample_prices))
        self.assertEqual(len(bb['lower']), len(self.sample_prices))
        
        # Upper band should be higher than middle, middle higher than lower
        for i in range(len(self.sample_prices)):
            self.assertGreater(bb['upper'][i], bb['middle'][i])
            self.assertGreater(bb['middle'][i], bb['lower'][i])
    
    def test_calculate_stochastic(self):
        """Test Stochastic Oscillator calculation"""
        stoch = self.ti.calculate_stochastic(self.sample_high, self.sample_low, self.sample_prices)
        
        self.assertIn('k', stoch)
        self.assertIn('d', stoch)
        
        self.assertEqual(len(stoch['k']), len(self.sample_prices))
        self.assertEqual(len(stoch['d']), len(self.sample_prices))
        
        # Stochastic values should be between 0 and 100
        self.assertTrue(all(0 <= k <= 100 for k in stoch['k']))
        self.assertTrue(all(0 <= d <= 100 for d in stoch['d']))
    
    def test_calculate_williams_r(self):
        """Test Williams %R calculation"""
        wr = self.ti.calculate_williams_r(self.sample_high, self.sample_low, self.sample_prices)
        
        self.assertEqual(len(wr), len(self.sample_prices))
        
        # Williams %R should be between -100 and 0
        self.assertTrue(all(-100 <= w <= 0 for w in wr))
    
    def test_calculate_cci(self):
        """Test Commodity Channel Index calculation"""
        cci = self.ti.calculate_cci(self.sample_high, self.sample_low, self.sample_prices)
        
        self.assertEqual(len(cci), len(self.sample_prices))
        
        # CCI can vary widely, just check it's numeric
        self.assertTrue(all(isinstance(c, (int, float)) for c in cci))
    
    def test_calculate_all_indicators(self):
        """Test calculation of all indicators together"""
        # Prepare OHLCV data
        ohlcv_data = []
        for i in range(len(self.sample_prices)):
            ohlcv_data.append({
                'symbol': 'TEST',
                'date': f'2023-01-{i+1:02d}',
                'open_price': self.sample_prices[i] - 0.5,
                'high_price': self.sample_high[i],
                'low_price': self.sample_low[i],
                'close_price': self.sample_prices[i],
                'volume': 1000000 + i * 10000
            })
        
        enhanced_data = self.ti.calculate_all_indicators(ohlcv_data)
        
        self.assertEqual(len(enhanced_data), len(ohlcv_data))
        
        # Check that all indicators are present
        indicators = enhanced_data[0]['indicators']
        expected_indicators = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26',
            'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
            'stochastic_k', 'stochastic_d',
            'williams_r', 'cci'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, indicators)
            self.assertIsInstance(indicators[indicator], (int, float, np.integer, np.floating))
    
    def test_normalize_indicators(self):
        """Test indicator normalization"""
        # Prepare test data with indicators
        test_data = [
            {
                'indicators': {
                    'rsi': 70.0,
                    'macd': 1.5,
                    'williams_r': -20.0,
                    'stochastic_k': 80.0
                }
            },
            {
                'indicators': {
                    'rsi': 30.0,
                    'macd': -1.5,
                    'williams_r': -80.0,
                    'stochastic_k': 20.0
                }
            }
        ]
        
        normalized_data = self.ti.normalize_indicators(test_data)
        
        self.assertEqual(len(normalized_data), len(test_data))
        
        # Check that normalized indicators are present
        self.assertIn('normalized_indicators', normalized_data[0])
        
        # RSI normalization (should be divided by 100)
        self.assertAlmostEqual(normalized_data[0]['normalized_indicators']['rsi'], 0.7, places=2)
        self.assertAlmostEqual(normalized_data[1]['normalized_indicators']['rsi'], 0.3, places=2)
        
        # Williams %R normalization (should be converted from -100,0 to 0,1)
        self.assertAlmostEqual(normalized_data[0]['normalized_indicators']['williams_r'], 0.8, places=2)
        self.assertAlmostEqual(normalized_data[1]['normalized_indicators']['williams_r'], 0.2, places=2)
    
    def test_get_feature_vector(self):
        """Test feature vector extraction"""
        indicators = {
            'rsi': 70.0,
            'macd': 1.5,
            'macd_signal': 1.2,
            'macd_histogram': 0.3,
            'sma_5': 105.0,
            'sma_10': 104.0,
            'sma_20': 103.0,
            'sma_50': 102.0,
            'ema_12': 105.5,
            'ema_26': 104.5,
            'bollinger_upper': 107.0,
            'bollinger_middle': 105.0,
            'bollinger_lower': 103.0,
            'stochastic_k': 80.0,
            'stochastic_d': 75.0,
            'williams_r': -20.0,
            'cci': 50.0
        }
        
        feature_vector = self.ti.get_feature_vector(indicators, normalized=False)
        
        # Should have 17 features
        self.assertEqual(len(feature_vector), 17)
        
        # First feature should be RSI
        self.assertEqual(feature_vector[0], 70.0)
        
        # Test with missing indicators (should default to 0.0)
        partial_indicators = {'rsi': 70.0}
        partial_vector = self.ti.get_feature_vector(partial_indicators, normalized=False)
        self.assertEqual(len(partial_vector), 17)
        self.assertEqual(partial_vector[0], 70.0)
        self.assertEqual(partial_vector[1], 0.0)  # Missing MACD should be 0.0

if __name__ == '__main__':
    unittest.main()