"""
AI Analyzer Module for MeridianAlgo
Provides AI-powered market analysis and insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
class AIAnalyzer:
    """
    AI-powered market analyzer for comprehensive stock analysis using Yahoo Finance data
    """
    
    def __init__(self):
        """
        Initialize the AI analyzer (no API keys required - uses Yahoo Finance)
        """
        pass
        
    def analyze_market_sentiment(self, data: pd.DataFrame) -> Dict:
        """
        Analyze market sentiment based on price and volume data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict: Market sentiment analysis
        """
        try:
            # Calculate sentiment indicators
            recent_data = data.tail(20)
            
            # Price momentum analysis
            price_changes = recent_data['Close'].pct_change().dropna()
            positive_days = (price_changes > 0).sum()
            negative_days = (price_changes < 0).sum()
            
            # Volume analysis
            avg_volume = recent_data['Volume'].mean()
            recent_volume = recent_data['Volume'].tail(5).mean()
            volume_trend = "increasing" if recent_volume > avg_volume * 1.1 else "decreasing" if recent_volume < avg_volume * 0.9 else "stable"
            
            # Volatility analysis
            volatility = price_changes.std() * 100
            volatility_level = "high" if volatility > 3 else "medium" if volatility > 1.5 else "low"
            
            # Overall sentiment
            if positive_days > negative_days * 1.5:
                sentiment = "bullish"
                sentiment_score = min(80 + (positive_days - negative_days) * 2, 95)
            elif negative_days > positive_days * 1.5:
                sentiment = "bearish"
                sentiment_score = max(20 - (negative_days - positive_days) * 2, 5)
            else:
                sentiment = "neutral"
                sentiment_score = 50
            
            return {
                'sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'positive_days': int(positive_days),
                'negative_days': int(negative_days),
                'volume_trend': volume_trend,
                'volatility_level': volatility_level,
                'volatility_value': round(volatility, 2),
                'analysis_period': len(recent_data)
            }
            
        except Exception as e:
            return {
                'sentiment': 'unknown',
                'sentiment_score': 50,
                'error': str(e)
            }
    
    def detect_market_regime(self, data: pd.DataFrame) -> Dict:
        """
        Detect current market regime (trending, ranging, volatile)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict: Market regime analysis
        """
        try:
            # Calculate moving averages
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            # Trend analysis
            current_price = data['Close'].iloc[-1]
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            
            # Determine trend direction
            if current_price > sma_20 > sma_50:
                trend = "uptrend"
                trend_strength = min(((current_price - sma_50) / sma_50) * 100, 100)
            elif current_price < sma_20 < sma_50:
                trend = "downtrend"
                trend_strength = min(((sma_50 - current_price) / sma_50) * 100, 100)
            else:
                trend = "sideways"
                trend_strength = 50
            
            # Range analysis
            recent_high = data['High'].tail(20).max()
            recent_low = data['Low'].tail(20).min()
            price_range = (recent_high - recent_low) / recent_low * 100
            
            # Volatility regime
            returns = data['Close'].pct_change().tail(20)
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            if volatility > 30:
                volatility_regime = "high_volatility"
            elif volatility > 15:
                volatility_regime = "medium_volatility"
            else:
                volatility_regime = "low_volatility"
            
            # Overall regime
            if trend != "sideways" and volatility < 25:
                regime = f"trending_{trend.split('trend')[0]}"
                regime_confidence = min(trend_strength + 20, 95)
            elif price_range < 10 and volatility < 20:
                regime = "ranging"
                regime_confidence = 75
            else:
                regime = "volatile"
                regime_confidence = 60
            
            return {
                'regime': regime,
                'regime_confidence': round(regime_confidence, 1),
                'trend': trend,
                'trend_strength': round(trend_strength, 1),
                'volatility_regime': volatility_regime,
                'annualized_volatility': round(volatility, 1),
                'price_range_20d': round(price_range, 1)
            }
            
        except Exception as e:
            return {
                'regime': 'unknown',
                'regime_confidence': 50,
                'error': str(e)
            }
    
    def calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> Dict:
        """
        Calculate support and resistance levels
        
        Args:
            data: DataFrame with OHLCV data
            window: Window for calculating levels
            
        Returns:
            Dict: Support and resistance levels
        """
        try:
            recent_data = data.tail(window * 2)
            
            # Find local minima and maxima
            highs = recent_data['High']
            lows = recent_data['Low']
            
            # Simple support/resistance calculation
            resistance_levels = []
            support_levels = []
            
            # Find peaks and troughs
            for i in range(1, len(highs) - 1):
                if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                    resistance_levels.append(highs.iloc[i])
                if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                    support_levels.append(lows.iloc[i])
            
            # Get strongest levels
            current_price = data['Close'].iloc[-1]
            
            # Resistance: levels above current price
            resistance_above = [r for r in resistance_levels if r > current_price]
            nearest_resistance = min(resistance_above) if resistance_above else None
            
            # Support: levels below current price
            support_below = [s for s in support_levels if s < current_price]
            nearest_support = max(support_below) if support_below else None
            
            # Calculate strength based on how many times levels were tested
            resistance_strength = len([r for r in resistance_levels if abs(r - (nearest_resistance or 0)) < current_price * 0.02])
            support_strength = len([s for s in support_levels if abs(s - (nearest_support or 0)) < current_price * 0.02])
            
            return {
                'nearest_resistance': round(nearest_resistance, 2) if nearest_resistance else None,
                'nearest_support': round(nearest_support, 2) if nearest_support else None,
                'resistance_strength': resistance_strength,
                'support_strength': support_strength,
                'all_resistance_levels': [round(r, 2) for r in sorted(resistance_levels, reverse=True)[:5]],
                'all_support_levels': [round(s, 2) for s in sorted(support_levels, reverse=True)[:5]],
                'current_price': round(current_price, 2)
            }
            
        except Exception as e:
            return {
                'nearest_resistance': None,
                'nearest_support': None,
                'error': str(e)
            }
    
    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict:
        """
        Analyze volume profile and patterns
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict: Volume analysis
        """
        try:
            # Volume trend analysis
            recent_volume = data['Volume'].tail(10)
            avg_volume = data['Volume'].tail(50).mean()
            
            volume_trend = recent_volume.mean() / avg_volume
            
            if volume_trend > 1.5:
                volume_analysis = "significantly_above_average"
            elif volume_trend > 1.2:
                volume_analysis = "above_average"
            elif volume_trend < 0.8:
                volume_analysis = "below_average"
            elif volume_trend < 0.5:
                volume_analysis = "significantly_below_average"
            else:
                volume_analysis = "average"
            
            # Price-volume relationship
            price_changes = data['Close'].pct_change().tail(10)
            volume_changes = data['Volume'].pct_change().tail(10)
            
            # Calculate correlation
            correlation = np.corrcoef(price_changes.dropna(), volume_changes.dropna())[0, 1]
            
            if np.isnan(correlation):
                correlation = 0
            
            # Volume spikes
            volume_threshold = avg_volume * 2
            volume_spikes = (data['Volume'].tail(20) > volume_threshold).sum()
            
            return {
                'volume_trend': volume_analysis,
                'volume_ratio': round(volume_trend, 2),
                'price_volume_correlation': round(correlation, 2),
                'volume_spikes_20d': int(volume_spikes),
                'average_volume': int(avg_volume),
                'recent_volume': int(recent_volume.mean())
            }
            
        except Exception as e:
            return {
                'volume_trend': 'unknown',
                'error': str(e)
            }
    
    def get_ai_insights(self, symbol: str, analysis_data: Dict) -> Optional[str]:
        """
        Get AI-powered insights using external API (if available)
        
        Args:
            symbol: Stock symbol
            analysis_data: Analysis data to send to AI
            
        Returns:
            Optional[str]: AI insights or None if not available
        """
        try:
            # Generate insights based on Yahoo Finance data analysis
            sentiment = analysis_data.get('sentiment', {})
            regime = analysis_data.get('regime', {})
            support_resistance = analysis_data.get('support_resistance', {})
            volume = analysis_data.get('volume', {})
            
            insights = []
            
            # Sentiment analysis
            sentiment_score = sentiment.get('sentiment_score', 50)
            if sentiment_score > 70:
                insights.append("Strong bullish sentiment detected")
            elif sentiment_score < 30:
                insights.append("Bearish sentiment prevailing")
            else:
                insights.append("Neutral market sentiment")
            
            # Volume analysis
            volume_trend = volume.get('volume_trend', 'stable')
            if volume_trend == 'increasing':
                insights.append("Volume surge indicates strong interest")
            elif volume_trend == 'decreasing':
                insights.append("Low volume suggests consolidation")
            
            # Market regime
            regime_type = regime.get('regime', 'neutral')
            if regime_type == 'trending':
                insights.append("Trending market - momentum strategies favored")
            elif regime_type == 'mean_reverting':
                insights.append("Mean-reverting market - contrarian approach suggested")
            
            # Support/Resistance
            sr_strength = support_resistance.get('support_strength', 0.5)
            if sr_strength > 0.7:
                insights.append("Strong support levels provide downside protection")
            elif sr_strength < 0.3:
                insights.append("Weak support - potential for further decline")
            
            analysis_text = f"ANALYSIS: {'. '.join(insights)}. "
            
            # Add market outlook
            if sentiment_score > 60 and volume_trend == 'increasing':
                analysis_text += "Positive outlook with strong momentum."
            elif sentiment_score < 40 and volume_trend == 'decreasing':
                analysis_text += "Cautious outlook - monitor for reversal signals."
            else:
                analysis_text += "Mixed signals - wait for clearer direction."
            
            return analysis_text
            
        except Exception as e:
            return f"Analysis based on technical indicators: Market showing mixed signals. Monitor key levels and volume for direction."
    
    def comprehensive_analysis(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Perform comprehensive AI analysis
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Stock symbol
            
        Returns:
            Dict: Comprehensive analysis results
        """
        try:
            # Perform all analyses
            sentiment = self.analyze_market_sentiment(data)
            regime = self.detect_market_regime(data)
            support_resistance = self.calculate_support_resistance(data)
            volume = self.analyze_volume_profile(data)
            
            analysis_data = {
                'sentiment': sentiment,
                'regime': regime,
                'support_resistance': support_resistance,
                'volume': volume
            }
            
            # Get AI insights if available
            ai_insights = self.get_ai_insights(symbol, analysis_data)
            
            # Calculate overall score
            sentiment_score = sentiment.get('sentiment_score', 50)
            regime_confidence = regime.get('regime_confidence', 50)
            overall_score = (sentiment_score + regime_confidence) / 2
            
            return {
                'symbol': symbol.upper(),
                'timestamp': datetime.now().isoformat(),
                'overall_score': round(overall_score, 1),
                'sentiment_analysis': sentiment,
                'market_regime': regime,
                'support_resistance': support_resistance,
                'volume_analysis': volume,
                'ai_insights': ai_insights,
                'analysis_summary': {
                    'bullish_signals': self._count_bullish_signals(analysis_data),
                    'bearish_signals': self._count_bearish_signals(analysis_data),
                    'neutral_signals': self._count_neutral_signals(analysis_data)
                }
            }
            
        except Exception as e:
            return {
                'symbol': symbol.upper(),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _count_bullish_signals(self, data: Dict) -> int:
        """Count bullish signals in the analysis"""
        count = 0
        
        # Sentiment signals
        if data['sentiment'].get('sentiment') == 'bullish':
            count += 1
        if data['sentiment'].get('positive_days', 0) > data['sentiment'].get('negative_days', 0):
            count += 1
        
        # Regime signals
        if 'uptrend' in data['regime'].get('trend', ''):
            count += 1
        
        # Volume signals
        if data['volume'].get('volume_trend') in ['above_average', 'significantly_above_average']:
            count += 1
        
        return count
    
    def _count_bearish_signals(self, data: Dict) -> int:
        """Count bearish signals in the analysis"""
        count = 0
        
        # Sentiment signals
        if data['sentiment'].get('sentiment') == 'bearish':
            count += 1
        if data['sentiment'].get('negative_days', 0) > data['sentiment'].get('positive_days', 0):
            count += 1
        
        # Regime signals
        if 'downtrend' in data['regime'].get('trend', ''):
            count += 1
        
        # Volume signals
        if data['volume'].get('volume_trend') in ['below_average', 'significantly_below_average']:
            count += 1
        
        return count
    
    def _count_neutral_signals(self, data: Dict) -> int:
        """Count neutral signals in the analysis"""
        count = 0
        
        if data['sentiment'].get('sentiment') == 'neutral':
            count += 1
        if data['regime'].get('trend') == 'sideways':
            count += 1
        if data['volume'].get('volume_trend') == 'average':
            count += 1
        
        return count