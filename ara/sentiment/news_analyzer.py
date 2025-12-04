"""
News Sentiment Analyzer

Analyzes sentiment from financial news articles using multiple news APIs.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

from ara.sentiment.base_analyzer import SentimentAnalyzer, SentimentScore, SentimentType
from ara.sentiment.finbert_model import FinBERTModel
from ara.core.exceptions import AraAIException

logger = logging.getLogger(__name__)


class NewsAPIError(AraAIException):
    """News API related errors"""
    pass


class NewsSentimentAnalyzer(SentimentAnalyzer):
    """
    Analyzes sentiment from financial news articles.
    
    Features:
    - Integrate NewsAPI and Alpha Vantage News
    - Fetch news articles for symbols
    - Analyze headlines and article content
    - Source credibility weighting
    - Track news momentum and volume
    """
    
    # Source credibility scores (0 to 1)
    SOURCE_CREDIBILITY = {
        'reuters': 1.0,
        'bloomberg': 1.0,
        'wsj': 0.95,
        'financial times': 0.95,
        'cnbc': 0.9,
        'marketwatch': 0.9,
        'seeking alpha': 0.85,
        'yahoo finance': 0.8,
        'benzinga': 0.75,
        'default': 0.5,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize news sentiment analyzer.
        
        Args:
            config: Configuration with news API credentials
                - newsapi_key: NewsAPI.org API key
                - alphavantage_key: Alpha Vantage API key
                - max_articles: Maximum articles to fetch (default: 50)
                - analyze_content: Whether to analyze full content (default: False, headlines only)
        """
        self.newsapi_key = None
        self.alphavantage_key = None
        self.max_articles = 50
        self.analyze_content = False
        self.finbert = None
        
        super().__init__(config)
        
        self.newsapi_key = self.config.get('newsapi_key')
        self.alphavantage_key = self.config.get('alphavantage_key')
        self.max_articles = self.config.get('max_articles', 50)
        self.analyze_content = self.config.get('analyze_content', False)
        self.finbert = FinBERTModel()
    
    def _validate_config(self) -> None:
        """Validate news API configuration"""
        if not self.newsapi_key and not self.alphavantage_key:
            logger.warning("No news API keys provided. News sentiment analysis will be disabled.")
    
    async def analyze(self, symbol: str, lookback_hours: int = 24) -> SentimentScore:
        """
        Analyze news sentiment for a symbol.
        
        Args:
            symbol: Stock/crypto symbol to analyze
            lookback_hours: Hours of news to analyze
            
        Returns:
            SentimentScore with aggregated news sentiment
        """
        if not self.newsapi_key and not self.alphavantage_key:
            logger.warning(f"News APIs not configured, returning neutral sentiment for {symbol}")
            return SentimentScore(
                score=0.0,
                confidence=0.0,
                volume=0,
                timestamp=datetime.now(),
                source=SentimentType.NEWS,
                metadata={'error': 'API not configured'}
            )
        
        try:
            # Fetch articles from available sources
            articles = await self._fetch_articles(symbol, lookback_hours)
            
            if not articles:
                logger.info(f"No news articles found for {symbol}")
                return SentimentScore(
                    score=0.0,
                    confidence=0.0,
                    volume=0,
                    timestamp=datetime.now(),
                    source=SentimentType.NEWS
                )
            
            # Analyze sentiment for each article
            sentiments = []
            weights = []
            
            for article in articles:
                # Analyze headline (and content if enabled)
                text = article['title']
                if self.analyze_content and article.get('content'):
                    text += f" {article['content']}"
                
                sentiment = self.finbert.analyze_text(text)
                weight = self._calculate_article_weight(article)
                
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
            
            # Calculate confidence
            confidence = self._calculate_confidence(sentiments, len(articles))
            
            # Calculate news momentum
            momentum = self._calculate_news_momentum(articles)
            
            return SentimentScore(
                score=float(weighted_sentiment),
                confidence=float(confidence),
                volume=len(articles),
                timestamp=datetime.now(),
                source=SentimentType.NEWS,
                metadata={
                    'sources': list(set(a['source'] for a in articles)),
                    'momentum': momentum,
                    'avg_credibility': np.mean([self._get_source_credibility(a['source']) for a in articles]),
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            raise NewsAPIError(f"Failed to analyze news sentiment: {e}")
    
    async def _fetch_articles(self, symbol: str, lookback_hours: int) -> List[Dict[str, Any]]:
        """
        Fetch news articles from all available sources.
        
        Args:
            symbol: Symbol to search for
            lookback_hours: Hours to look back
            
        Returns:
            List of article dictionaries
        """
        articles = []
        
        # Fetch from NewsAPI
        if self.newsapi_key:
            newsapi_articles = await self._fetch_from_newsapi(symbol, lookback_hours)
            articles.extend(newsapi_articles)
        
        # Fetch from Alpha Vantage
        if self.alphavantage_key:
            av_articles = await self._fetch_from_alphavantage(symbol, lookback_hours)
            articles.extend(av_articles)
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in articles:
            title_lower = article['title'].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_articles.append(article)
        
        # Sort by published date (newest first)
        unique_articles.sort(key=lambda x: x['published_at'], reverse=True)
        
        return unique_articles[:self.max_articles]
    
    async def _fetch_from_newsapi(self, symbol: str, lookback_hours: int) -> List[Dict[str, Any]]:
        """
        Fetch articles from NewsAPI.org.
        
        Args:
            symbol: Symbol to search for
            lookback_hours: Hours to look back
            
        Returns:
            List of article dictionaries
        """
        try:
            from newsapi import NewsApiClient
            
            newsapi = NewsApiClient(api_key=self.newsapi_key)
            
            # Calculate date range
            from_date = (datetime.now() - timedelta(hours=lookback_hours)).strftime('%Y-%m-%d')
            
            # Normalize symbol for search
            normalized_symbol = self._normalize_symbol(symbol)
            
            # Search for articles
            response = await asyncio.to_thread(
                newsapi.get_everything,
                q=normalized_symbol,
                from_param=from_date,
                language='en',
                sort_by='publishedAt',
                page_size=min(self.max_articles, 100)
            )
            
            articles = []
            if response.get('articles'):
                for article in response['articles']:
                    articles.append({
                        'title': article.get('title', ''),
                        'content': article.get('description', ''),
                        'source': article.get('source', {}).get('name', 'unknown'),
                        'published_at': datetime.strptime(
                            article['publishedAt'],
                            '%Y-%m-%dT%H:%M:%SZ'
                        ) if article.get('publishedAt') else datetime.now(),
                        'url': article.get('url', ''),
                        'author': article.get('author', ''),
                    })
            
            return articles
            
        except ImportError:
            logger.warning("newsapi-python library not installed. Install with: pip install newsapi-python")
            return []
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []
    
    async def _fetch_from_alphavantage(self, symbol: str, lookback_hours: int) -> List[Dict[str, Any]]:
        """
        Fetch articles from Alpha Vantage News API.
        
        Args:
            symbol: Symbol to search for
            lookback_hours: Hours to look back
            
        Returns:
            List of article dictionaries
        """
        try:
            import aiohttp
            
            normalized_symbol = self._normalize_symbol(symbol)
            
            # Alpha Vantage News & Sentiment API
            url = 'https://www.alphavantage.co/query'
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': normalized_symbol,
                'apikey': self.alphavantage_key,
                'limit': min(self.max_articles, 50),
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
            
            articles = []
            if 'feed' in data:
                cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
                
                for item in data['feed']:
                    # Parse timestamp
                    time_str = item.get('time_published', '')
                    try:
                        published_at = datetime.strptime(time_str, '%Y%m%dT%H%M%S')
                    except:
                        published_at = datetime.now()
                    
                    # Check if within time window
                    if published_at < cutoff_time:
                        continue
                    
                    articles.append({
                        'title': item.get('title', ''),
                        'content': item.get('summary', ''),
                        'source': item.get('source', 'unknown'),
                        'published_at': published_at,
                        'url': item.get('url', ''),
                        'author': ', '.join(item.get('authors', [])),
                    })
            
            return articles
            
        except ImportError:
            logger.warning("aiohttp library not installed. Install with: pip install aiohttp")
            return []
        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {e}")
            return []
    
    def _calculate_article_weight(self, article: Dict[str, Any]) -> float:
        """
        Calculate weight for an article based on source credibility and recency.
        
        Args:
            article: Article dictionary
            
        Returns:
            Weight value
        """
        import math
        
        # Base weight from source credibility
        source = article.get('source', '').lower()
        credibility = self._get_source_credibility(source)
        weight = credibility
        
        # Recency multiplier (more recent = higher weight)
        published_at = article.get('published_at', datetime.now())
        hours_ago = (datetime.now() - published_at).total_seconds() / 3600
        
        # Exponential decay: weight decreases as article gets older
        recency_factor = math.exp(-hours_ago / 24)  # Half-life of 24 hours
        weight *= (0.5 + 0.5 * recency_factor)  # Scale between 0.5 and 1.0
        
        return weight
    
    def _get_source_credibility(self, source: str) -> float:
        """
        Get credibility score for a news source.
        
        Args:
            source: Source name
            
        Returns:
            Credibility score (0 to 1)
        """
        source_lower = source.lower()
        
        for key, score in self.SOURCE_CREDIBILITY.items():
            if key in source_lower:
                return score
        
        return self.SOURCE_CREDIBILITY['default']
    
    def _calculate_confidence(self, sentiments: List[float], volume: int) -> float:
        """
        Calculate confidence score.
        
        Args:
            sentiments: List of sentiment scores
            volume: Number of articles
            
        Returns:
            Confidence score (0 to 1)
        """
        if not sentiments:
            return 0.0
        
        import numpy as np
        
        # Agreement factor
        std = np.std(sentiments)
        agreement = 1.0 / (1.0 + std)
        
        # Volume factor (more articles = higher confidence)
        volume_factor = min(1.0, np.log10(1 + volume) / 2.0)
        
        # Combined confidence
        confidence = (agreement * 0.7 + volume_factor * 0.3)
        
        return float(min(1.0, confidence))
    
    def _calculate_news_momentum(self, articles: List[Dict[str, Any]]) -> float:
        """
        Calculate news momentum (rate of news publication).
        
        Args:
            articles: List of articles
            
        Returns:
            Momentum score (positive = increasing news volume)
        """
        if len(articles) < 2:
            return 0.0
        
        # Split into recent and older articles
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=6)
        
        recent_count = sum(1 for a in articles if a['published_at'] >= recent_cutoff)
        older_count = len(articles) - recent_count
        
        if older_count == 0:
            return 1.0 if recent_count > 0 else 0.0
        
        # Momentum is the ratio of recent to older (normalized)
        momentum = (recent_count / 6) / (older_count / 18)  # Normalize by time period
        
        # Scale to -1 to +1 range
        return float(min(1.0, max(-1.0, (momentum - 1.0))))
    
    async def stream_sentiment(self, symbol: str, callback: callable) -> None:
        """
        Stream real-time news sentiment updates.
        
        Args:
            symbol: Symbol to monitor
            callback: Function to call with new sentiment data
        """
        if not self.newsapi_key and not self.alphavantage_key:
            logger.warning("News streaming not available without API credentials")
            return
        
        try:
            logger.info(f"Starting news stream for {symbol}")
            
            # Implement polling-based streaming
            while True:
                try:
                    sentiment = await self.analyze(symbol, lookback_hours=6)
                    await callback(sentiment)
                    await asyncio.sleep(600)  # Poll every 10 minutes
                except Exception as e:
                    logger.error(f"Error in news stream: {e}")
                    await asyncio.sleep(600)
                    
        except asyncio.CancelledError:
            logger.info(f"News stream cancelled for {symbol}")
        except Exception as e:
            logger.error(f"Fatal error in news stream: {e}")
            raise NewsAPIError(f"News streaming failed: {e}")
