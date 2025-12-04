# Sentiment Analysis Module

This module provides comprehensive sentiment analysis for financial assets using multiple data sources.

## Features

- **Multi-Source Analysis**: Twitter, Reddit, and financial news
- **FinBERT Integration**: State-of-the-art financial sentiment model
- **Alternative Data**: Google Trends, insider trading, institutional holdings
- **Real-Time Streaming**: Live sentiment updates
- **Sentiment Aggregation**: Weighted combination of multiple sources
- **Momentum Calculation**: Track sentiment changes over time
- **Divergence Detection**: Identify sentiment-price divergences

## Components

### Base Analyzer (`base_analyzer.py`)
- Abstract base class for all sentiment analyzers
- Common functionality for sentiment calculation
- Momentum and divergence calculations

### Twitter Analyzer (`twitter_analyzer.py`)
- Twitter API v2 integration using tweepy
- Real-time tweet streaming
- Influencer weighting based on follower count
- Engagement-based scoring

### Reddit Analyzer (`reddit_analyzer.py`)
- Reddit API integration using PRAW
- Monitor r/wallstreetbets, r/stocks, r/cryptocurrency
- Upvote-weighted sentiment
- Trending ticker detection

### News Analyzer (`news_analyzer.py`)
- NewsAPI and Alpha Vantage News integration
- Source credibility weighting
- News momentum tracking
- Headline and content analysis

### Alternative Data (`alternative_data.py`)
- Google Trends search interest
- SEC EDGAR insider trading data
- Institutional holdings (13F filings)
- Feature engineering for ML models

### Sentiment Aggregator (`aggregator.py`)
- Combines all sentiment sources
- Weighted aggregation
- Overall confidence calculation
- Historical tracking

### FinBERT Model (`finbert_model.py`)
- FinBERT sentiment model wrapper
- VADER fallback for lightweight analysis
- Batch processing support

## Installation

### Required Dependencies

```bash
# Core sentiment analysis
pip install transformers torch vaderSentiment

# Twitter integration
pip install tweepy

# Reddit integration
pip install praw

# News integration
pip install newsapi-python aiohttp

# Alternative data
pip install pytrends
```

### Optional Dependencies

For GPU acceleration:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

Create a configuration dictionary with API credentials:

```python
config = {
    # Twitter API v2
    'twitter_bearer_token': 'your_bearer_token',
    'min_followers': 1000,
    'max_tweets': 100,
    
    # Reddit API
    'reddit_client_id': 'your_client_id',
    'reddit_client_secret': 'your_client_secret',
    'reddit_user_agent': 'ARA-AI/1.0',
    'subreddits': ['wallstreetbets', 'stocks', 'cryptocurrency'],
    
    # News APIs
    'newsapi_key': 'your_newsapi_key',
    'alphavantage_key': 'your_alphavantage_key',
    
    # Sentiment weights
    'sentiment_weights': {
        'news': 0.4,
        'twitter': 0.3,
        'reddit': 0.3,
    }
}
```

## Usage

### Basic Sentiment Analysis

```python
from ara.sentiment import SentimentAggregator

# Initialize aggregator
aggregator = SentimentAggregator(config)

# Analyze sentiment
result = await aggregator.analyze('AAPL', lookback_hours=24)

print(f"Overall Sentiment: {result.overall_sentiment:.2f}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Volume: {result.total_volume}")
print(f"Momentum: {result.momentum:.2f}")
print(f"Trending: {result.trending}")
```

### Individual Source Analysis

```python
from ara.sentiment import TwitterSentimentAnalyzer, RedditSentimentAnalyzer

# Twitter sentiment
twitter = TwitterSentimentAnalyzer(config)
twitter_sentiment = await twitter.analyze('AAPL')

# Reddit sentiment
reddit = RedditSentimentAnalyzer(config)
reddit_sentiment = await reddit.analyze('AAPL')
```

### Real-Time Streaming

```python
async def on_sentiment_update(result):
    print(f"New sentiment: {result.overall_sentiment:.2f}")

# Stream sentiment updates
await aggregator.stream_sentiment('AAPL', on_sentiment_update)
```

### Alternative Data

```python
from ara.sentiment import AlternativeDataProvider

# Get alternative data
alt_data = AlternativeDataProvider(config)
data = await alt_data.get_alternative_data('AAPL')

# Get engineered features
features = await alt_data.get_features('AAPL')
```

## API Reference

### SentimentResult

```python
@dataclass
class SentimentResult:
    symbol: str
    overall_sentiment: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    sentiment_scores: List[SentimentScore]
    momentum: float  # Rate of change
    divergence: Optional[float]  # Sentiment-price divergence
    trending: bool
    total_volume: int
    timestamp: datetime
```

### SentimentScore

```python
@dataclass
class SentimentScore:
    score: float  # -1 to +1
    confidence: float  # 0 to 1
    volume: int  # Number of mentions
    timestamp: datetime
    source: SentimentType  # TWITTER, REDDIT, NEWS
    metadata: Dict[str, Any]
```

## Performance Considerations

- **FinBERT**: Requires ~500MB GPU memory, ~2GB RAM
- **Twitter API**: Rate limited to 450 requests per 15 minutes
- **Reddit API**: Rate limited to 60 requests per minute
- **NewsAPI**: Free tier limited to 100 requests per day
- **Caching**: Implement caching to reduce API calls

## Best Practices

1. **API Rate Limits**: Respect API rate limits and implement backoff
2. **Error Handling**: Handle API failures gracefully with fallbacks
3. **Caching**: Cache sentiment results to reduce API calls
4. **Batch Processing**: Process multiple symbols in parallel
5. **Monitoring**: Track API usage and sentiment quality

## Limitations

- **API Dependencies**: Requires valid API keys for full functionality
- **Rate Limits**: Subject to third-party API rate limits
- **Data Latency**: Some sources have inherent delays
- **Language**: Currently supports English only
- **Coverage**: Not all symbols have sufficient social media coverage

## Future Enhancements

- [ ] Multi-language support
- [ ] Additional data sources (StockTwits, Discord, Telegram)
- [ ] Sentiment-based trading signals
- [ ] Historical sentiment database
- [ ] Advanced NLP models (GPT-based analysis)
- [ ] Sentiment anomaly detection
- [ ] Cross-asset sentiment correlation

## References

- [FinBERT Paper](https://arxiv.org/abs/1908.10063)
- [Twitter API v2 Documentation](https://developer.twitter.com/en/docs/twitter-api)
- [Reddit API Documentation](https://www.reddit.com/dev/api/)
- [NewsAPI Documentation](https://newsapi.org/docs)
- [Alpha Vantage Documentation](https://www.alphavantage.co/documentation/)
