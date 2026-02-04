"""
Twitter Sentiment Analyzer

Analyzes sentiment from Twitter using the Twitter API v2 and FinBERT model.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

from ara.sentiment.base_analyzer import SentimentAnalyzer, SentimentScore, SentimentType
from ara.sentiment.finbert_model import FinBERTModel
from ara.core.exceptions import AraAIException

logger = logging.getLogger(__name__)


class TwitterAPIError(AraAIException):
    """Twitter API related errors"""

    pass


class TwitterSentimentAnalyzer(SentimentAnalyzer):
    """
    Analyzes sentiment from Twitter using API v2.

    Features:
    - Fetch tweets for stock symbols and crypto assets
    - Real-time tweet streaming
    - Sentiment scoring using FinBERT
    - Influencer weighting based on follower count
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Twitter sentiment analyzer.

        Args:
            config: Configuration with Twitter API credentials
                - twitter_bearer_token: Twitter API v2 bearer token
                - min_followers: Minimum followers for influencer weighting (default: 1000)
                - max_tweets: Maximum tweets to fetch (default: 100)
        """
        self.bearer_token = None
        self.min_followers = 1000
        self.max_tweets = 100
        self.finbert = None
        self._client = None

        super().__init__(config)

        self.bearer_token = self.config.get("twitter_bearer_token")
        self.min_followers = self.config.get("min_followers", 1000)
        self.max_tweets = self.config.get("max_tweets", 100)
        self.finbert = FinBERTModel()

    def _validate_config(self) -> None:
        """Validate Twitter API configuration"""
        if not self.bearer_token:
            logger.warning(
                "Twitter bearer token not provided. Twitter sentiment analysis will be disabled."
            )

    def _get_client(self):
        """Get or create Twitter API client"""
        if self._client is None and self.bearer_token:
            try:
                import tweepy

                self._client = tweepy.Client(bearer_token=self.bearer_token)
            except ImportError:
                raise TwitterAPIError(
                    "tweepy library not installed. Install with: pip install tweepy"
                )
            except Exception as e:
                raise TwitterAPIError(f"Failed to initialize Twitter client: {e}")
        return self._client

    async def analyze(self, symbol: str, lookback_hours: int = 24) -> SentimentScore:
        """
        Analyze Twitter sentiment for a symbol.

        Args:
            symbol: Stock/crypto symbol to analyze
            lookback_hours: Hours of tweets to analyze

        Returns:
            SentimentScore with aggregated Twitter sentiment
        """
        if not self.bearer_token:
            logger.warning(f"Twitter API not configured, returning neutral sentiment for {symbol}")
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                volume=0,
                timestamp=datetime.now(),
                source=SentimentType.TWITTER,
                metadata={"error": "API not configured"},
            )

        try:
            # Fetch tweets
            tweets = await self._fetch_tweets(symbol, lookback_hours)

            if not tweets:
                logger.info(f"No tweets found for {symbol}")
                return SentimentScore(
                    score=0.0,
                    confidence=0.0,
                    volume=0,
                    timestamp=datetime.now(),
                    source=SentimentType.TWITTER,
                )

            # Analyze sentiment for each tweet
            sentiments = []
            weights = []

            for tweet in tweets:
                # Get sentiment from FinBERT
                sentiment = self.finbert.analyze_text(tweet["text"])

                # Calculate weight based on engagement and follower count
                weight = self._calculate_tweet_weight(tweet)

                sentiments.append(sentiment)
                weights.append(weight)

            # Calculate weighted average sentiment
            import numpy as np

            weights_array = np.array(weights)
            sentiments_array = np.array(sentiments)

            if weights_array.sum() > 0:
                weighted_sentiment = np.average(sentiments_array, weights=weights_array)
            else:
                weighted_sentiment = np.mean(sentiments_array)

            # Calculate confidence based on volume and agreement
            confidence = self._calculate_confidence(sentiments, len(tweets))

            return SentimentScore(
                score=float(weighted_sentiment),
                confidence=float(confidence),
                volume=len(tweets),
                timestamp=datetime.now(),
                source=SentimentType.TWITTER,
                metadata={
                    "influencer_tweets": sum(
                        1 for t in tweets if t.get("followers", 0) >= self.min_followers
                    ),
                    "avg_engagement": np.mean([t.get("engagement", 0) for t in tweets]),
                },
            )

        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment for {symbol}: {e}")
            raise TwitterAPIError(f"Failed to analyze Twitter sentiment: {e}")

    async def _fetch_tweets(self, symbol: str, lookback_hours: int) -> List[Dict[str, Any]]:
        """
        Fetch tweets mentioning the symbol.

        Args:
            symbol: Symbol to search for
            lookback_hours: Hours to look back

        Returns:
            List of tweet dictionaries
        """
        client = self._get_client()
        if not client:
            return []

        # Normalize symbol and create search query
        normalized_symbol = self._normalize_symbol(symbol)
        query = f"${normalized_symbol} OR #{normalized_symbol} -is:retweet lang:en"

        # Calculate start time
        start_time = datetime.now() - timedelta(hours=lookback_hours)

        try:
            # Search tweets
            response = await asyncio.to_thread(
                client.search_recent_tweets,
                query=query,
                max_results=min(self.max_tweets, 100),  # API limit
                start_time=start_time,
                tweet_fields=["created_at", "public_metrics", "author_id"],
                user_fields=["public_metrics"],
                expansions=["author_id"],
            )

            if not response.data:
                return []

            # Build user lookup
            users = {}
            if response.includes and "users" in response.includes:
                for user in response.includes["users"]:
                    users[user.id] = user

            # Process tweets
            tweets = []
            for tweet in response.data:
                user = users.get(tweet.author_id)
                followers = user.public_metrics["followers_count"] if user else 0

                tweets.append(
                    {
                        "text": tweet.text,
                        "created_at": tweet.created_at,
                        "followers": followers,
                        "engagement": (
                            tweet.public_metrics["like_count"]
                            + tweet.public_metrics["retweet_count"]
                            + tweet.public_metrics["reply_count"]
                        ),
                        "likes": tweet.public_metrics["like_count"],
                        "retweets": tweet.public_metrics["retweet_count"],
                    }
                )

            return tweets

        except Exception as e:
            logger.error(f"Error fetching tweets for {symbol}: {e}")
            return []

    def _calculate_tweet_weight(self, tweet: Dict[str, Any]) -> float:
        """
        Calculate weight for a tweet based on influence and engagement.

        Args:
            tweet: Tweet dictionary

        Returns:
            Weight value (higher = more influential)
        """
        # Base weight
        weight = 1.0

        # Influencer multiplier (followers)
        followers = tweet.get("followers", 0)
        if followers >= self.min_followers:
            # Logarithmic scaling for followers
            import math

            weight *= 1 + math.log10(followers / self.min_followers)

        # Engagement multiplier
        engagement = tweet.get("engagement", 0)
        if engagement > 0:
            weight *= 1 + math.log10(1 + engagement)

        return weight

    def _calculate_confidence(self, sentiments: List[float], volume: int) -> float:
        """
        Calculate confidence score based on sentiment agreement and volume.

        Args:
            sentiments: List of sentiment scores
            volume: Number of tweets

        Returns:
            Confidence score (0 to 1)
        """
        if not sentiments:
            return 0.0

        import numpy as np

        # Agreement factor (lower std = higher agreement)
        std = np.std(sentiments)
        agreement = 1.0 / (1.0 + std)

        # Volume factor (more tweets = higher confidence, with diminishing returns)
        volume_factor = min(1.0, np.log10(1 + volume) / 2.0)

        # Combined confidence
        confidence = agreement * 0.6 + volume_factor * 0.4

        return float(min(1.0, confidence))

    async def stream_sentiment(self, symbol: str, callback: callable) -> None:
        """
        Stream real-time Twitter sentiment updates.

        Args:
            symbol: Symbol to monitor
            callback: Function to call with new sentiment data
        """
        client = self._get_client()
        if not client:
            logger.warning("Twitter streaming not available without API credentials")
            return

        self._normalize_symbol(symbol)

        try:
            # Note: Twitter API v2 streaming requires elevated access
            # This is a simplified implementation
            logger.info(f"Starting Twitter stream for {symbol}")

            # For now, implement polling-based "streaming"
            while True:
                try:
                    sentiment = await self.analyze(symbol, lookback_hours=1)
                    await callback(sentiment)
                    await asyncio.sleep(60)  # Poll every minute
                except Exception as e:
                    logger.error(f"Error in Twitter stream: {e}")
                    await asyncio.sleep(60)

        except asyncio.CancelledError:
            logger.info(f"Twitter stream cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Fatal error in Twitter stream: {e}")
            raise TwitterAPIError(f"Twitter streaming failed: {e}")
