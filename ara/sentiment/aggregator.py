"""
Sentiment Aggregator

Aggregates sentiment from multiple sources and calculates overall sentiment metrics.
"""

import asyncio
from typing import List, Dict, Optional, Any
import logging
import numpy as np

from ara.sentiment.base_analyzer import (
    SentimentResult,
    SentimentScore,
    SentimentType,
)
from ara.sentiment.twitter_analyzer import TwitterSentimentAnalyzer
from ara.sentiment.reddit_analyzer import RedditSentimentAnalyzer
from ara.sentiment.news_analyzer import NewsSentimentAnalyzer

logger = logging.getLogger(__name__)


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources.

    Features:
    - Combine Twitter, Reddit, and news sentiment
    - Weighted aggregation based on source reliability
    - Sentiment momentum calculation
    - Sentiment divergence detection
    """

    # Default weights for each source
    DEFAULT_WEIGHTS = {
        SentimentType.NEWS: 0.4,
        SentimentType.TWITTER: 0.3,
        SentimentType.REDDIT: 0.3,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sentiment aggregator.

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}

        # Initialize analyzers
        self.twitter_analyzer = TwitterSentimentAnalyzer(config)
        self.reddit_analyzer = RedditSentimentAnalyzer(config)
        self.news_analyzer = NewsSentimentAnalyzer(config)

        # Custom weights (if provided)
        self.weights = self.config.get("sentiment_weights", self.DEFAULT_WEIGHTS.copy())

        # Historical sentiment for momentum calculation
        self.sentiment_history: Dict[str, List[SentimentResult]] = {}

    async def analyze(
        self,
        symbol: str,
        lookback_hours: int = 24,
        price_returns: Optional[List[float]] = None,
    ) -> SentimentResult:
        """
        Analyze sentiment from all sources.

        Args:
            symbol: Stock/crypto symbol to analyze
            lookback_hours: Hours of data to analyze
            price_returns: Optional price returns for divergence calculation

        Returns:
            SentimentResult with aggregated sentiment
        """
        logger.info(f"Analyzing sentiment for {symbol}")

        # Fetch sentiment from all sources in parallel
        tasks = [
            self.twitter_analyzer.analyze(symbol, lookback_hours),
            self.reddit_analyzer.analyze(symbol, lookback_hours),
            self.news_analyzer.analyze(symbol, lookback_hours),
        ]

        sentiment_scores = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_scores = []
        for score in sentiment_scores:
            if isinstance(score, SentimentScore):
                valid_scores.append(score)
            elif isinstance(score, Exception):
                logger.error(f"Error getting sentiment: {score}")

        if not valid_scores:
            logger.warning(f"No valid sentiment data for {symbol}")
            return SentimentResult(
                symbol=symbol,
                overall_sentiment=0.0,
                confidence=0.0,
                sentiment_scores=[],
                momentum=0.0,
                divergence=None,
                trending=False,
                total_volume=0,
            )

        # Calculate overall sentiment
        overall_sentiment = self._calculate_weighted_sentiment(valid_scores)

        # Calculate confidence
        confidence = self._calculate_overall_confidence(valid_scores)

        # Calculate momentum
        momentum = self._calculate_sentiment_momentum(symbol, valid_scores)

        # Calculate divergence if price data provided
        divergence = None
        if price_returns:
            divergence = self._calculate_divergence(valid_scores, price_returns)

        # Check if trending
        trending = self._is_trending(valid_scores)

        # Total volume
        total_volume = sum(s.volume for s in valid_scores)

        result = SentimentResult(
            symbol=symbol,
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            sentiment_scores=valid_scores,
            momentum=momentum,
            divergence=divergence,
            trending=trending,
            total_volume=total_volume,
        )

        # Store in history
        self._add_to_history(symbol, result)

        return result

    def _calculate_weighted_sentiment(self, scores: List[SentimentScore]) -> float:
        """
        Calculate weighted average sentiment.

        Args:
            scores: List of sentiment scores

        Returns:
            Weighted sentiment (-1 to +1)
        """
        if not scores:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for score in scores:
            # Get weight for this source
            source_weight = self.weights.get(score.source, 0.33)

            # Combine with confidence
            weight = source_weight * score.confidence

            weighted_sum += score.score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _calculate_overall_confidence(self, scores: List[SentimentScore]) -> float:
        """
        Calculate overall confidence score.

        Args:
            scores: List of sentiment scores

        Returns:
            Confidence (0 to 1)
        """
        if not scores:
            return 0.0

        # Average confidence weighted by source weights
        weighted_confidence = 0.0
        total_weight = 0.0

        for score in scores:
            source_weight = self.weights.get(score.source, 0.33)
            weighted_confidence += score.confidence * source_weight
            total_weight += source_weight

        if total_weight == 0:
            return 0.0

        avg_confidence = weighted_confidence / total_weight

        # Boost confidence if multiple sources agree
        sentiments = [s.score for s in scores]
        agreement = 1.0 / (1.0 + np.std(sentiments))

        # Combined confidence
        return float(min(1.0, avg_confidence * 0.7 + agreement * 0.3))

    def _calculate_sentiment_momentum(
        self, symbol: str, current_scores: List[SentimentScore]
    ) -> float:
        """
        Calculate sentiment momentum (rate of change).

        Args:
            symbol: Symbol
            current_scores: Current sentiment scores

        Returns:
            Momentum (-1 to +1)
        """
        # Get historical sentiment
        history = self.sentiment_history.get(symbol, [])

        if len(history) < 2:
            return 0.0

        # Calculate current sentiment
        current_sentiment = self._calculate_weighted_sentiment(current_scores)

        # Get recent historical sentiment (last 6 hours)
        recent_history = [h for h in history[-10:]]  # Last 10 data points

        if not recent_history:
            return 0.0

        # Calculate average historical sentiment
        historical_sentiment = np.mean([h.overall_sentiment for h in recent_history])

        # Momentum is the change
        momentum = current_sentiment - historical_sentiment

        return float(np.clip(momentum, -1.0, 1.0))

    def _calculate_divergence(
        self, scores: List[SentimentScore], price_returns: List[float]
    ) -> Optional[float]:
        """
        Calculate divergence between sentiment and price action.

        Args:
            scores: Sentiment scores
            price_returns: Price returns

        Returns:
            Divergence score (positive = bullish divergence, negative = bearish)
        """
        if not scores or not price_returns:
            return None

        # Get sentiment values
        sentiments = [s.score for s in scores]

        # Need same length
        min_len = min(len(sentiments), len(price_returns))
        if min_len < 2:
            return None

        sentiments = sentiments[:min_len]
        price_returns = price_returns[:min_len]

        # Calculate correlation
        correlation = np.corrcoef(sentiments, price_returns)[0, 1]

        if np.isnan(correlation):
            return None

        # Divergence is when correlation is negative
        # Positive divergence: sentiment up, price down (bullish)
        # Negative divergence: sentiment down, price up (bearish)
        return float(-correlation)

    def _is_trending(self, scores: List[SentimentScore]) -> bool:
        """
        Determine if symbol is trending based on volume.

        Args:
            scores: Sentiment scores

        Returns:
            True if trending
        """
        if not scores:
            return False

        # Check total volume
        total_volume = sum(s.volume for s in scores)

        # Thresholds for trending
        TRENDING_THRESHOLD = 100  # Minimum mentions

        return total_volume >= TRENDING_THRESHOLD

    def _add_to_history(self, symbol: str, result: SentimentResult) -> None:
        """
        Add sentiment result to history.

        Args:
            symbol: Symbol
            result: Sentiment result
        """
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = []

        self.sentiment_history[symbol].append(result)

        # Keep only last 100 results
        if len(self.sentiment_history[symbol]) > 100:
            self.sentiment_history[symbol] = self.sentiment_history[symbol][-100:]

    async def stream_sentiment(self, symbol: str, callback: callable) -> None:
        """
        Stream real-time sentiment updates from all sources.

        Args:
            symbol: Symbol to monitor
            callback: Function to call with new sentiment data
        """
        logger.info(f"Starting sentiment stream for {symbol}")

        # Create tasks for each analyzer
        tasks = [
            self._stream_with_aggregation(symbol, callback),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info(f"Sentiment stream cancelled for {symbol}")

    async def _stream_with_aggregation(self, symbol: str, callback: callable) -> None:
        """
        Stream with periodic aggregation.

        Args:
            symbol: Symbol to monitor
            callback: Callback function
        """
        while True:
            try:
                # Analyze sentiment
                result = await self.analyze(symbol, lookback_hours=1)

                # Call callback
                await callback(result)

                # Wait before next update
                await asyncio.sleep(60)  # Update every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sentiment stream: {e}")
                await asyncio.sleep(60)

    def get_sentiment_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get sentiment summary for a symbol.

        Args:
            symbol: Symbol

        Returns:
            Dictionary with sentiment summary
        """
        history = self.sentiment_history.get(symbol, [])

        if not history:
            return {
                "symbol": symbol,
                "data_points": 0,
                "avg_sentiment": 0.0,
                "current_sentiment": 0.0,
                "trend": "unknown",
            }

        sentiments = [h.overall_sentiment for h in history]

        return {
            "symbol": symbol,
            "data_points": len(history),
            "avg_sentiment": float(np.mean(sentiments)),
            "current_sentiment": sentiments[-1] if sentiments else 0.0,
            "min_sentiment": float(np.min(sentiments)),
            "max_sentiment": float(np.max(sentiments)),
            "std_sentiment": float(np.std(sentiments)),
            "trend": self._get_trend(sentiments),
        }

    def _get_trend(self, sentiments: List[float]) -> str:
        """
        Get sentiment trend.

        Args:
            sentiments: List of sentiment values

        Returns:
            Trend: 'bullish', 'bearish', or 'neutral'
        """
        if len(sentiments) < 2:
            return "neutral"

        # Simple linear regression
        x = np.arange(len(sentiments))
        slope = np.polyfit(x, sentiments, 1)[0]

        if slope > 0.1:
            return "bullish"
        elif slope < -0.1:
            return "bearish"
        else:
            return "neutral"
