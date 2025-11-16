"""
Base Sentiment Analyzer

Provides abstract base class for all sentiment analyzers with common functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np


class SentimentType(Enum):
    """Types of sentiment sources"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    ALTERNATIVE = "alternative"


@dataclass
class SentimentScore:
    """Individual sentiment score from a single source"""
    score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    volume: int  # Number of mentions/posts
    timestamp: datetime
    source: SentimentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate sentiment score values"""
        if not -1 <= self.score <= 1:
            raise ValueError(f"Sentiment score must be between -1 and 1, got {self.score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.volume < 0:
            raise ValueError(f"Volume must be non-negative, got {self.volume}")


@dataclass
class SentimentResult:
    """Aggregated sentiment analysis result"""
    symbol: str
    overall_sentiment: float  # -1 to +1
    confidence: float  # 0 to 1
    sentiment_scores: List[SentimentScore]
    momentum: float  # Rate of change in sentiment
    divergence: Optional[float] = None  # Divergence from price action
    trending: bool = False
    total_volume: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_sentiment_by_source(self, source: SentimentType) -> List[SentimentScore]:
        """Get sentiment scores from a specific source"""
        return [s for s in self.sentiment_scores if s.source == source]
    
    def get_weighted_sentiment(self) -> float:
        """Calculate volume-weighted sentiment"""
        if not self.sentiment_scores:
            return 0.0
        
        total_weight = sum(s.volume * s.confidence for s in self.sentiment_scores)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(s.score * s.volume * s.confidence for s in self.sentiment_scores)
        return weighted_sum / total_weight


class SentimentAnalyzer(ABC):
    """
    Abstract base class for sentiment analyzers.
    
    All sentiment analyzers must implement the analyze method to fetch
    and analyze sentiment data for a given symbol.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sentiment analyzer.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate required configuration parameters"""
        pass
    
    @abstractmethod
    async def analyze(self, symbol: str, lookback_hours: int = 24) -> SentimentScore:
        """
        Analyze sentiment for a given symbol.
        
        Args:
            symbol: Stock/crypto symbol to analyze
            lookback_hours: Number of hours to look back for data
            
        Returns:
            SentimentScore with analysis results
        """
        pass
    
    @abstractmethod
    async def stream_sentiment(self, symbol: str, callback: callable) -> None:
        """
        Stream real-time sentiment updates.
        
        Args:
            symbol: Stock/crypto symbol to monitor
            callback: Function to call with new sentiment data
        """
        pass
    
    def calculate_momentum(self, scores: List[SentimentScore], window_hours: int = 6) -> float:
        """
        Calculate sentiment momentum (rate of change).
        
        Args:
            scores: List of sentiment scores ordered by time
            window_hours: Time window for momentum calculation
            
        Returns:
            Momentum value (positive = increasing sentiment, negative = decreasing)
        """
        if len(scores) < 2:
            return 0.0
        
        # Split into recent and older scores
        cutoff_time = datetime.now()
        recent_scores = [s for s in scores if (cutoff_time - s.timestamp).total_seconds() / 3600 <= window_hours]
        older_scores = [s for s in scores if (cutoff_time - s.timestamp).total_seconds() / 3600 > window_hours]
        
        if not recent_scores or not older_scores:
            return 0.0
        
        # Calculate average sentiment for each period
        recent_avg = np.mean([s.score for s in recent_scores])
        older_avg = np.mean([s.score for s in older_scores])
        
        # Momentum is the change
        return float(recent_avg - older_avg)
    
    def calculate_divergence(self, sentiment_scores: List[SentimentScore], 
                           price_returns: List[float]) -> Optional[float]:
        """
        Calculate divergence between sentiment and price action.
        
        Args:
            sentiment_scores: List of sentiment scores
            price_returns: List of price returns (same length as sentiment_scores)
            
        Returns:
            Divergence score (positive = bullish divergence, negative = bearish divergence)
        """
        if len(sentiment_scores) != len(price_returns) or len(sentiment_scores) < 2:
            return None
        
        sentiments = np.array([s.score for s in sentiment_scores])
        returns = np.array(price_returns)
        
        # Calculate correlation
        correlation = np.corrcoef(sentiments, returns)[0, 1]
        
        # Divergence is when correlation is negative (sentiment and price moving opposite)
        if np.isnan(correlation):
            return None
        
        return float(-correlation)  # Negative correlation = divergence
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol for searching (remove exchange suffixes, etc.).
        
        Args:
            symbol: Raw symbol string
            
        Returns:
            Normalized symbol
        """
        # Remove common suffixes
        symbol = symbol.upper()
        for suffix in ['-USD', '.US', ':US', '-USDT', '-BTC']:
            if symbol.endswith(suffix):
                symbol = symbol[:-len(suffix)]
        
        return symbol
    
    def _extract_cashtags(self, text: str) -> List[str]:
        """
        Extract cashtags ($SYMBOL) from text.
        
        Args:
            text: Text to search
            
        Returns:
            List of found symbols
        """
        import re
        pattern = r'\$([A-Z]{1,5})\b'
        return re.findall(pattern, text.upper())
