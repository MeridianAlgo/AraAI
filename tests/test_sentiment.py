"""
Tests for Sentiment Analysis Module
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from ara.sentiment.base_analyzer import (
    SentimentAnalyzer,
    SentimentScore,
    SentimentResult,
    SentimentType
)
from ara.sentiment.finbert_model import FinBERTModel
from ara.sentiment.twitter_analyzer import TwitterSentimentAnalyzer
from ara.sentiment.reddit_analyzer import RedditSentimentAnalyzer
from ara.sentiment.news_analyzer import NewsSentimentAnalyzer
from ara.sentiment.aggregator import SentimentAggregator
from ara.sentiment.alternative_data import AlternativeDataProvider


class TestSentimentScore:
    """Test SentimentScore dataclass"""
    
    def test_valid_sentiment_score(self):
        """Test creating valid sentiment score"""
        score = SentimentScore(
            score=0.5,
            confidence=0.8,
            volume=100,
            timestamp=datetime.now(),
            source=SentimentType.TWITTER
        )
        
        assert score.score == 0.5
        assert score.confidence == 0.8
        assert score.volume == 100
        assert score.source == SentimentType.TWITTER
    
    def test_invalid_sentiment_score(self):
        """Test validation of sentiment score"""
        with pytest.raises(ValueError):
            SentimentScore(
                score=2.0,  # Invalid: > 1
                confidence=0.8,
                volume=100,
                timestamp=datetime.now(),
                source=SentimentType.TWITTER
            )
    
    def test_invalid_confidence(self):
        """Test validation of confidence"""
        with pytest.raises(ValueError):
            SentimentScore(
                score=0.5,
                confidence=1.5,  # Invalid: > 1
                volume=100,
                timestamp=datetime.now(),
                source=SentimentType.TWITTER
            )


class TestSentimentResult:
    """Test SentimentResult dataclass"""
    
    def test_sentiment_result_creation(self):
        """Test creating sentiment result"""
        scores = [
            SentimentScore(0.5, 0.8, 100, datetime.now(), SentimentType.TWITTER),
            SentimentScore(0.3, 0.7, 50, datetime.now(), SentimentType.REDDIT),
        ]
        
        result = SentimentResult(
            symbol='AAPL',
            overall_sentiment=0.4,
            confidence=0.75,
            sentiment_scores=scores,
            momentum=0.1,
            total_volume=150
        )
        
        assert result.symbol == 'AAPL'
        assert result.overall_sentiment == 0.4
        assert len(result.sentiment_scores) == 2
    
    def test_get_sentiment_by_source(self):
        """Test filtering sentiment by source"""
        scores = [
            SentimentScore(0.5, 0.8, 100, datetime.now(), SentimentType.TWITTER),
            SentimentScore(0.3, 0.7, 50, datetime.now(), SentimentType.REDDIT),
            SentimentScore(0.4, 0.9, 75, datetime.now(), SentimentType.TWITTER),
        ]
        
        result = SentimentResult(
            symbol='AAPL',
            overall_sentiment=0.4,
            confidence=0.75,
            sentiment_scores=scores,
            momentum=0.1,
            total_volume=225
        )
        
        twitter_scores = result.get_sentiment_by_source(SentimentType.TWITTER)
        assert len(twitter_scores) == 2
    
    def test_weighted_sentiment(self):
        """Test weighted sentiment calculation"""
        scores = [
            SentimentScore(0.5, 0.8, 100, datetime.now(), SentimentType.TWITTER),
            SentimentScore(0.3, 0.6, 50, datetime.now(), SentimentType.REDDIT),
        ]
        
        result = SentimentResult(
            symbol='AAPL',
            overall_sentiment=0.4,
            confidence=0.75,
            sentiment_scores=scores,
            momentum=0.1,
            total_volume=150
        )
        
        weighted = result.get_weighted_sentiment()
        assert -1 <= weighted <= 1


class TestFinBERTModel:
    """Test FinBERT model wrapper"""
    
    def test_finbert_initialization(self):
        """Test FinBERT model initialization"""
        model = FinBERTModel()
        assert model is not None
    
    def test_analyze_text(self):
        """Test text sentiment analysis"""
        model = FinBERTModel()
        
        # Test positive sentiment
        positive_text = "Stock prices are soaring! Great earnings report!"
        sentiment = model.analyze_text(positive_text)
        assert -1 <= sentiment <= 1
        
        # Test negative sentiment
        negative_text = "Terrible losses, stock crashing badly"
        sentiment = model.analyze_text(negative_text)
        assert -1 <= sentiment <= 1
    
    def test_empty_text(self):
        """Test handling of empty text"""
        model = FinBERTModel()
        sentiment = model.analyze_text("")
        assert sentiment == 0.0
    
    def test_batch_analysis(self):
        """Test batch sentiment analysis"""
        model = FinBERTModel()
        
        texts = [
            "Great news for investors!",
            "Stock market crash imminent",
            "Neutral market conditions"
        ]
        
        sentiments = model.analyze_batch(texts)
        assert len(sentiments) == 3
        assert all(-1 <= s <= 1 for s in sentiments)


class TestTwitterSentimentAnalyzer:
    """Test Twitter sentiment analyzer"""
    
    @pytest.mark.asyncio
    async def test_twitter_without_api_key(self):
        """Test Twitter analyzer without API key"""
        config = {}
        analyzer = TwitterSentimentAnalyzer(config)
        
        result = await analyzer.analyze('AAPL')
        
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.volume == 0
        assert result.source == SentimentType.TWITTER
    
    def test_normalize_symbol(self):
        """Test symbol normalization"""
        config = {}
        analyzer = TwitterSentimentAnalyzer(config)
        
        assert analyzer._normalize_symbol('AAPL-USD') == 'AAPL'
        assert analyzer._normalize_symbol('BTC-USDT') == 'BTC'
        assert analyzer._normalize_symbol('MSFT') == 'MSFT'
    
    def test_extract_cashtags(self):
        """Test cashtag extraction"""
        config = {}
        analyzer = TwitterSentimentAnalyzer(config)
        
        text = "Buying $AAPL and $TSLA today! $MSFT looking good too."
        cashtags = analyzer._extract_cashtags(text)
        
        assert 'AAPL' in cashtags
        assert 'TSLA' in cashtags
        assert 'MSFT' in cashtags


class TestRedditSentimentAnalyzer:
    """Test Reddit sentiment analyzer"""
    
    @pytest.mark.asyncio
    async def test_reddit_without_api_key(self):
        """Test Reddit analyzer without API key"""
        config = {}
        analyzer = RedditSentimentAnalyzer(config)
        
        result = await analyzer.analyze('AAPL')
        
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.volume == 0
        assert result.source == SentimentType.REDDIT
    
    def test_calculate_post_weight(self):
        """Test post weight calculation"""
        config = {}
        analyzer = RedditSentimentAnalyzer(config)
        
        post = {
            'upvotes': 1000,
            'upvote_ratio': 0.95,
            'num_comments': 100
        }
        
        weight = analyzer._calculate_post_weight(post)
        assert weight > 0
    
    def test_calculate_comment_weight(self):
        """Test comment weight calculation"""
        config = {}
        analyzer = RedditSentimentAnalyzer(config)
        
        comment = {'upvotes': 50}
        weight = analyzer._calculate_comment_weight(comment)
        assert weight > 0


class TestNewsSentimentAnalyzer:
    """Test news sentiment analyzer"""
    
    @pytest.mark.asyncio
    async def test_news_without_api_key(self):
        """Test news analyzer without API key"""
        config = {}
        analyzer = NewsSentimentAnalyzer(config)
        
        result = await analyzer.analyze('AAPL')
        
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert result.volume == 0
        assert result.source == SentimentType.NEWS
    
    def test_source_credibility(self):
        """Test source credibility scoring"""
        config = {}
        analyzer = NewsSentimentAnalyzer(config)
        
        assert analyzer._get_source_credibility('Reuters') == 1.0
        assert analyzer._get_source_credibility('Bloomberg') == 1.0
        assert analyzer._get_source_credibility('Unknown Source') == 0.5
    
    def test_calculate_article_weight(self):
        """Test article weight calculation"""
        config = {}
        analyzer = NewsSentimentAnalyzer(config)
        
        article = {
            'source': 'Reuters',
            'published_at': datetime.now()
        }
        
        weight = analyzer._calculate_article_weight(article)
        assert weight > 0


class TestSentimentAggregator:
    """Test sentiment aggregator"""
    
    @pytest.mark.asyncio
    async def test_aggregator_initialization(self):
        """Test aggregator initialization"""
        config = {}
        aggregator = SentimentAggregator(config)
        
        assert aggregator.twitter_analyzer is not None
        assert aggregator.reddit_analyzer is not None
        assert aggregator.news_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_analyze_without_api_keys(self):
        """Test analysis without API keys"""
        config = {}
        aggregator = SentimentAggregator(config)
        
        result = await aggregator.analyze('AAPL')
        
        assert result.symbol == 'AAPL'
        assert -1 <= result.overall_sentiment <= 1
        assert 0 <= result.confidence <= 1
    
    def test_calculate_weighted_sentiment(self):
        """Test weighted sentiment calculation"""
        config = {}
        aggregator = SentimentAggregator(config)
        
        scores = [
            SentimentScore(0.5, 0.8, 100, datetime.now(), SentimentType.TWITTER),
            SentimentScore(0.3, 0.7, 50, datetime.now(), SentimentType.REDDIT),
            SentimentScore(0.6, 0.9, 75, datetime.now(), SentimentType.NEWS),
        ]
        
        weighted = aggregator._calculate_weighted_sentiment(scores)
        assert -1 <= weighted <= 1
    
    def test_is_trending(self):
        """Test trending detection"""
        config = {}
        aggregator = SentimentAggregator(config)
        
        # High volume - should be trending
        high_volume_scores = [
            SentimentScore(0.5, 0.8, 200, datetime.now(), SentimentType.TWITTER),
        ]
        assert aggregator._is_trending(high_volume_scores) == True
        
        # Low volume - not trending
        low_volume_scores = [
            SentimentScore(0.5, 0.8, 10, datetime.now(), SentimentType.TWITTER),
        ]
        assert aggregator._is_trending(low_volume_scores) == False


class TestAlternativeDataProvider:
    """Test alternative data provider"""
    
    @pytest.mark.asyncio
    async def test_alternative_data_initialization(self):
        """Test alternative data provider initialization"""
        config = {}
        provider = AlternativeDataProvider(config)
        
        assert provider is not None
    
    @pytest.mark.asyncio
    async def test_get_alternative_data(self):
        """Test getting alternative data"""
        config = {}
        provider = AlternativeDataProvider(config)
        
        data = await provider.get_alternative_data('AAPL')
        
        assert data['symbol'] == 'AAPL'
        assert 'google_trends' in data
        assert 'insider_trading' in data
        assert 'institutional_holdings' in data
    
    def test_calculate_trend(self):
        """Test trend calculation"""
        config = {}
        provider = AlternativeDataProvider(config)
        
        # Upward trend
        up_values = [1, 2, 3, 4, 5]
        assert provider._calculate_trend(up_values) == 'up'
        
        # Downward trend
        down_values = [5, 4, 3, 2, 1]
        assert provider._calculate_trend(down_values) == 'down'
        
        # Flat trend
        flat_values = [3, 3, 3, 3, 3]
        assert provider._calculate_trend(flat_values) == 'flat'
    
    def test_calculate_momentum(self):
        """Test momentum calculation"""
        config = {}
        provider = AlternativeDataProvider(config)
        
        values = [1, 2, 3, 4, 5, 6]
        momentum = provider._calculate_momentum(values)
        
        assert -1 <= momentum <= 1
    
    def test_engineer_features(self):
        """Test feature engineering"""
        config = {}
        provider = AlternativeDataProvider(config)
        
        alt_data = {
            'google_trends': {
                'current_interest': 50,
                'avg_interest': 40,
                'momentum': 0.2,
                'trend': 'up'
            }
        }
        
        features = provider.engineer_features(alt_data)
        
        assert 'trends_current' in features
        assert 'trends_avg' in features
        assert 'trends_momentum' in features
        assert 'trends_up' in features


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
