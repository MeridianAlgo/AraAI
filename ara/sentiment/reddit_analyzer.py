"""
Reddit Sentiment Analyzer

Analyzes sentiment from Reddit posts and comments using PRAW and FinBERT.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

from ara.sentiment.base_analyzer import SentimentAnalyzer, SentimentScore, SentimentType
from ara.sentiment.finbert_model import FinBERTModel
from ara.core.exceptions import AraAIException

logger = logging.getLogger(__name__)


class RedditAPIError(AraAIException):
    """Reddit API related errors"""

    pass


class RedditSentimentAnalyzer(SentimentAnalyzer):
    """
    Analyzes sentiment from Reddit using PRAW.

    Features:
    - Monitor r/wallstreetbets, r/stocks, r/cryptocurrency
    - Analyze post titles, content, and comments
    - Sentiment scoring with upvote weighting
    - Track trending tickers and mentions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Reddit sentiment analyzer.

        Args:
            config: Configuration with Reddit API credentials
                - reddit_client_id: Reddit API client ID
                - reddit_client_secret: Reddit API client secret
                - reddit_user_agent: Reddit API user agent
                - subreddits: List of subreddits to monitor (default: wallstreetbets, stocks, cryptocurrency)
                - max_posts: Maximum posts to fetch per subreddit (default: 50)
                - include_comments: Whether to analyze comments (default: True)
                - max_comments_per_post: Maximum comments per post (default: 10)
        """
        self.client_id = None
        self.client_secret = None
        self.user_agent = "ARA-AI-Sentiment-Analyzer/1.0"
        self.subreddits = ["wallstreetbets", "stocks", "cryptocurrency"]
        self.max_posts = 50
        self.include_comments = True
        self.max_comments_per_post = 10
        self.finbert = None
        self._reddit = None

        super().__init__(config)

        self.client_id = self.config.get("reddit_client_id")
        self.client_secret = self.config.get("reddit_client_secret")
        self.user_agent = self.config.get(
            "reddit_user_agent", "ARA-AI-Sentiment-Analyzer/1.0"
        )
        self.subreddits = self.config.get(
            "subreddits", ["wallstreetbets", "stocks", "cryptocurrency"]
        )
        self.max_posts = self.config.get("max_posts", 50)
        self.include_comments = self.config.get("include_comments", True)
        self.max_comments_per_post = self.config.get("max_comments_per_post", 10)
        self.finbert = FinBERTModel()

    def _validate_config(self) -> None:
        """Validate Reddit API configuration"""
        if not self.client_id or not self.client_secret:
            logger.warning(
                "Reddit API credentials not provided. Reddit sentiment analysis will be disabled."
            )

    def _get_reddit(self):
        """Get or create Reddit API client"""
        if self._reddit is None and self.client_id and self.client_secret:
            try:
                import praw

                self._reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
            except ImportError:
                raise RedditAPIError(
                    "praw library not installed. Install with: pip install praw"
                )
            except Exception as e:
                raise RedditAPIError(f"Failed to initialize Reddit client: {e}")
        return self._reddit

    async def analyze(self, symbol: str, lookback_hours: int = 24) -> SentimentScore:
        """
        Analyze Reddit sentiment for a symbol.

        Args:
            symbol: Stock/crypto symbol to analyze
            lookback_hours: Hours of posts to analyze

        Returns:
            SentimentScore with aggregated Reddit sentiment
        """
        if not self.client_id or not self.client_secret:
            logger.warning(
                f"Reddit API not configured, returning neutral sentiment for {symbol}"
            )
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                volume=0,
                timestamp=datetime.now(),
                source=SentimentType.REDDIT,
                metadata={"error": "API not configured"},
            )

        try:
            # Fetch posts and comments
            posts = await self._fetch_posts(symbol, lookback_hours)

            if not posts:
                logger.info(f"No Reddit posts found for {symbol}")
                return SentimentScore(
                    score=0.0,
                    confidence=0.0,
                    volume=0,
                    timestamp=datetime.now(),
                    source=SentimentType.REDDIT,
                )

            # Analyze sentiment for each post/comment
            sentiments = []
            weights = []

            for post in posts:
                # Analyze post title and body
                text = f"{post['title']} {post['body']}"
                sentiment = self.finbert.analyze_text(text)
                weight = self._calculate_post_weight(post)

                sentiments.append(sentiment)
                weights.append(weight)

                # Analyze comments if enabled
                if self.include_comments and post.get("comments"):
                    for comment in post["comments"]:
                        comment_sentiment = self.finbert.analyze_text(comment["text"])
                        comment_weight = self._calculate_comment_weight(comment)

                        sentiments.append(comment_sentiment)
                        weights.append(comment_weight)

            # Calculate weighted average sentiment
            import numpy as np

            weights_array = np.array(weights)
            sentiments_array = np.array(sentiments)

            if weights_array.sum() > 0:
                weighted_sentiment = np.average(sentiments_array, weights=weights_array)
            else:
                weighted_sentiment = np.mean(sentiments_array)

            # Calculate confidence
            confidence = self._calculate_confidence(sentiments, len(posts))

            # Count mentions
            total_mentions = len(posts)
            if self.include_comments:
                total_mentions += sum(len(p.get("comments", [])) for p in posts)

            return SentimentScore(
                score=float(weighted_sentiment),
                confidence=float(confidence),
                volume=total_mentions,
                timestamp=datetime.now(),
                source=SentimentType.REDDIT,
                metadata={
                    "posts": len(posts),
                    "comments": total_mentions - len(posts),
                    "avg_upvotes": np.mean([p["upvotes"] for p in posts]),
                    "subreddits": list(set(p["subreddit"] for p in posts)),
                },
            )

        except Exception as e:
            logger.error(f"Error analyzing Reddit sentiment for {symbol}: {e}")
            raise RedditAPIError(f"Failed to analyze Reddit sentiment: {e}")

    async def _fetch_posts(
        self, symbol: str, lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch Reddit posts mentioning the symbol.

        Args:
            symbol: Symbol to search for
            lookback_hours: Hours to look back

        Returns:
            List of post dictionaries
        """
        reddit = self._get_reddit()
        if not reddit:
            return []

        normalized_symbol = self._normalize_symbol(symbol)
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

        posts = []

        try:
            for subreddit_name in self.subreddits:
                subreddit_posts = await self._fetch_subreddit_posts(
                    reddit, subreddit_name, normalized_symbol, cutoff_time
                )
                posts.extend(subreddit_posts)

            return posts

        except Exception as e:
            logger.error(f"Error fetching Reddit posts for {symbol}: {e}")
            return []

    async def _fetch_subreddit_posts(
        self, reddit, subreddit_name: str, symbol: str, cutoff_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Fetch posts from a specific subreddit.

        Args:
            reddit: Reddit client
            subreddit_name: Name of subreddit
            symbol: Symbol to search for
            cutoff_time: Minimum post time

        Returns:
            List of post dictionaries
        """
        posts = []

        try:
            subreddit = await asyncio.to_thread(reddit.subreddit, subreddit_name)

            # Search for symbol in subreddit
            search_results = await asyncio.to_thread(
                lambda: list(
                    subreddit.search(symbol, limit=self.max_posts, time_filter="day")
                )
            )

            for submission in search_results:
                # Check if post is within time window
                post_time = datetime.fromtimestamp(submission.created_utc)
                if post_time < cutoff_time:
                    continue

                # Check if symbol is actually mentioned
                text = f"{submission.title} {submission.selftext}".upper()
                if symbol.upper() not in text and f"${symbol.upper()}" not in text:
                    continue

                post_data = {
                    "title": submission.title,
                    "body": submission.selftext,
                    "upvotes": submission.score,
                    "upvote_ratio": submission.upvote_ratio,
                    "num_comments": submission.num_comments,
                    "created_utc": post_time,
                    "subreddit": subreddit_name,
                    "url": submission.url,
                    "comments": [],
                }

                # Fetch top comments if enabled
                if self.include_comments:
                    post_data["comments"] = await self._fetch_comments(submission)

                posts.append(post_data)

        except Exception as e:
            logger.error(f"Error fetching from r/{subreddit_name}: {e}")

        return posts

    async def _fetch_comments(self, submission) -> List[Dict[str, Any]]:
        """
        Fetch top comments from a submission.

        Args:
            submission: Reddit submission object

        Returns:
            List of comment dictionaries
        """
        comments = []

        try:
            # Load comments
            await asyncio.to_thread(submission.comments.replace_more, limit=0)

            # Get top comments
            top_comments = await asyncio.to_thread(
                lambda: list(submission.comments)[: self.max_comments_per_post]
            )

            for comment in top_comments:
                if hasattr(comment, "body") and comment.body:
                    comments.append(
                        {
                            "text": comment.body,
                            "upvotes": comment.score,
                            "created_utc": datetime.fromtimestamp(comment.created_utc),
                        }
                    )

        except Exception as e:
            logger.debug(f"Error fetching comments: {e}")

        return comments

    def _calculate_post_weight(self, post: Dict[str, Any]) -> float:
        """
        Calculate weight for a post based on upvotes and engagement.

        Args:
            post: Post dictionary

        Returns:
            Weight value
        """
        import math

        # Base weight
        weight = 1.0

        # Upvote multiplier (logarithmic)
        upvotes = max(1, post.get("upvotes", 1))
        weight *= math.log10(1 + upvotes)

        # Upvote ratio multiplier (controversial posts get lower weight)
        upvote_ratio = post.get("upvote_ratio", 0.5)
        weight *= upvote_ratio

        # Comment engagement multiplier
        num_comments = post.get("num_comments", 0)
        if num_comments > 0:
            weight *= 1 + math.log10(1 + num_comments) * 0.5

        return weight

    def _calculate_comment_weight(self, comment: Dict[str, Any]) -> float:
        """
        Calculate weight for a comment based on upvotes.

        Args:
            comment: Comment dictionary

        Returns:
            Weight value
        """
        import math

        upvotes = max(1, comment.get("upvotes", 1))
        return math.log10(1 + upvotes) * 0.5  # Comments weighted less than posts

    def _calculate_confidence(self, sentiments: List[float], volume: int) -> float:
        """
        Calculate confidence score.

        Args:
            sentiments: List of sentiment scores
            volume: Number of posts/comments

        Returns:
            Confidence score (0 to 1)
        """
        if not sentiments:
            return 0.0

        import numpy as np

        # Agreement factor
        std = np.std(sentiments)
        agreement = 1.0 / (1.0 + std)

        # Volume factor
        volume_factor = min(1.0, np.log10(1 + volume) / 2.0)

        # Combined confidence
        confidence = agreement * 0.6 + volume_factor * 0.4

        return float(min(1.0, confidence))

    async def stream_sentiment(self, symbol: str, callback: callable) -> None:
        """
        Stream real-time Reddit sentiment updates.

        Args:
            symbol: Symbol to monitor
            callback: Function to call with new sentiment data
        """
        reddit = self._get_reddit()
        if not reddit:
            logger.warning("Reddit streaming not available without API credentials")
            return

        try:
            logger.info(f"Starting Reddit stream for {symbol}")

            # Implement polling-based streaming
            while True:
                try:
                    sentiment = await self.analyze(symbol, lookback_hours=1)
                    await callback(sentiment)
                    await asyncio.sleep(300)  # Poll every 5 minutes
                except Exception as e:
                    logger.error(f"Error in Reddit stream: {e}")
                    await asyncio.sleep(300)

        except asyncio.CancelledError:
            logger.info(f"Reddit stream cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Fatal error in Reddit stream: {e}")
            raise RedditAPIError(f"Reddit streaming failed: {e}")

    def get_trending_tickers(
        self, subreddit_name: str = "wallstreetbets", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get trending stock tickers from a subreddit.

        Args:
            subreddit_name: Subreddit to analyze
            limit: Number of top tickers to return

        Returns:
            List of trending tickers with mention counts
        """
        reddit = self._get_reddit()
        if not reddit:
            return []

        try:
            subreddit = reddit.subreddit(subreddit_name)

            # Get hot posts
            hot_posts = list(subreddit.hot(limit=100))

            # Extract tickers
            ticker_counts = {}
            for post in hot_posts:
                text = f"{post.title} {post.selftext}"
                tickers = self._extract_cashtags(text)

                for ticker in tickers:
                    if ticker not in ticker_counts:
                        ticker_counts[ticker] = {"count": 0, "upvotes": 0}
                    ticker_counts[ticker]["count"] += 1
                    ticker_counts[ticker]["upvotes"] += post.score

            # Sort by mention count
            trending = [
                {"ticker": ticker, **data}
                for ticker, data in sorted(
                    ticker_counts.items(), key=lambda x: x[1]["count"], reverse=True
                )[:limit]
            ]

            return trending

        except Exception as e:
            logger.error(f"Error getting trending tickers: {e}")
            return []
