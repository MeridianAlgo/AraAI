"""
Sentiment Analysis Demo

Demonstrates the sentiment analysis capabilities of ARA AI.
"""

import asyncio
import os
from datetime import datetime

from ara.sentiment import (
    SentimentAggregator,
    TwitterSentimentAnalyzer,
    RedditSentimentAnalyzer,
    NewsSentimentAnalyzer,
    AlternativeDataProvider,
)


async def demo_individual_sources():
    """Demonstrate individual sentiment sources"""
    print("=" * 80)
    print("Individual Sentiment Sources Demo")
    print("=" * 80)
    
    # Configuration (use environment variables for API keys)
    config = {
        'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
        'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
        'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'newsapi_key': os.getenv('NEWSAPI_KEY'),
        'alphavantage_key': os.getenv('ALPHAVANTAGE_KEY'),
    }
    
    symbol = 'AAPL'
    
    # Twitter sentiment
    print(f"\n📱 Twitter Sentiment for {symbol}")
    print("-" * 80)
    twitter = TwitterSentimentAnalyzer(config)
    twitter_sentiment = await twitter.analyze(symbol, lookback_hours=24)
    print(f"Score: {twitter_sentiment.score:+.3f}")
    print(f"Confidence: {twitter_sentiment.confidence:.3f}")
    print(f"Volume: {twitter_sentiment.volume} tweets")
    print(f"Metadata: {twitter_sentiment.metadata}")
    
    # Reddit sentiment
    print(f"\n🤖 Reddit Sentiment for {symbol}")
    print("-" * 80)
    reddit = RedditSentimentAnalyzer(config)
    reddit_sentiment = await reddit.analyze(symbol, lookback_hours=24)
    print(f"Score: {reddit_sentiment.score:+.3f}")
    print(f"Confidence: {reddit_sentiment.confidence:.3f}")
    print(f"Volume: {reddit_sentiment.volume} posts/comments")
    print(f"Metadata: {reddit_sentiment.metadata}")
    
    # News sentiment
    print(f"\n📰 News Sentiment for {symbol}")
    print("-" * 80)
    news = NewsSentimentAnalyzer(config)
    news_sentiment = await news.analyze(symbol, lookback_hours=24)
    print(f"Score: {news_sentiment.score:+.3f}")
    print(f"Confidence: {news_sentiment.confidence:.3f}")
    print(f"Volume: {news_sentiment.volume} articles")
    print(f"Metadata: {news_sentiment.metadata}")


async def demo_aggregated_sentiment():
    """Demonstrate aggregated sentiment analysis"""
    print("\n" + "=" * 80)
    print("Aggregated Sentiment Analysis Demo")
    print("=" * 80)
    
    config = {
        'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
        'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
        'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'newsapi_key': os.getenv('NEWSAPI_KEY'),
        'alphavantage_key': os.getenv('ALPHAVANTAGE_KEY'),
    }
    
    symbols = ['AAPL', 'TSLA', 'BTC-USD']
    
    aggregator = SentimentAggregator(config)
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}")
        print("-" * 80)
        
        result = await aggregator.analyze(symbol, lookback_hours=24)
        
        # Display results
        print(f"Overall Sentiment: {result.overall_sentiment:+.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Total Volume: {result.total_volume}")
        print(f"Momentum: {result.momentum:+.3f}")
        print(f"Trending: {'Yes' if result.trending else 'No'}")
        
        # Show breakdown by source
        print("\nBreakdown by Source:")
        for score in result.sentiment_scores:
            print(f"  {score.source.value:10s}: {score.score:+.3f} "
                  f"(confidence: {score.confidence:.3f}, volume: {score.volume})")
        
        # Interpretation
        if result.overall_sentiment > 0.3:
            sentiment_label = "🟢 Bullish"
        elif result.overall_sentiment < -0.3:
            sentiment_label = "🔴 Bearish"
        else:
            sentiment_label = "🟡 Neutral"
        
        print(f"\nInterpretation: {sentiment_label}")


async def demo_alternative_data():
    """Demonstrate alternative data integration"""
    print("\n" + "=" * 80)
    print("Alternative Data Demo")
    print("=" * 80)
    
    config = {}
    symbol = 'AAPL'
    
    alt_data = AlternativeDataProvider(config)
    
    print(f"\n🔍 Alternative Data for {symbol}")
    print("-" * 80)
    
    data = await alt_data.get_alternative_data(symbol)
    
    # Google Trends
    if data['google_trends']:
        trends = data['google_trends']
        print("\n📈 Google Trends:")
        print(f"  Current Interest: {trends['current_interest']}")
        print(f"  Average Interest: {trends['avg_interest']:.1f}")
        print(f"  Trend: {trends['trend']}")
        print(f"  Momentum: {trends['momentum']:+.3f}")
    
    # Insider Trading
    if data['insider_trading']:
        insider = data['insider_trading']
        print("\n💼 Insider Trading:")
        print(f"  Net Buying: {insider['net_buying']}")
        print(f"  Transaction Count: {insider['transaction_count']}")
    
    # Institutional Holdings
    if data['institutional_holdings']:
        institutional = data['institutional_holdings']
        print("\n🏦 Institutional Holdings:")
        print(f"  Ownership: {institutional['total_institutional_ownership']:.1f}%")
        print(f"  Number of Institutions: {institutional['num_institutions']}")
    
    # Engineered features
    print("\n🔧 Engineered Features:")
    features = alt_data.engineer_features(data)
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")


async def demo_trending_tickers():
    """Demonstrate trending ticker detection"""
    print("\n" + "=" * 80)
    print("Trending Tickers Demo")
    print("=" * 80)
    
    config = {
        'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
        'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    }
    
    reddit = RedditSentimentAnalyzer(config)
    
    print("\n🔥 Trending on r/wallstreetbets")
    print("-" * 80)
    
    trending = reddit.get_trending_tickers('wallstreetbets', limit=10)
    
    if trending:
        for i, ticker_data in enumerate(trending, 1):
            print(f"{i:2d}. ${ticker_data['ticker']:5s} - "
                  f"{ticker_data['count']:3d} mentions, "
                  f"{ticker_data['upvotes']:5d} upvotes")
    else:
        print("No trending tickers found (API not configured)")


async def demo_sentiment_streaming():
    """Demonstrate real-time sentiment streaming"""
    print("\n" + "=" * 80)
    print("Real-Time Sentiment Streaming Demo")
    print("=" * 80)
    
    config = {
        'twitter_bearer_token': os.getenv('TWITTER_BEARER_TOKEN'),
        'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
        'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'newsapi_key': os.getenv('NEWSAPI_KEY'),
    }
    
    symbol = 'AAPL'
    aggregator = SentimentAggregator(config)
    
    print(f"\n📡 Streaming sentiment for {symbol} (press Ctrl+C to stop)")
    print("-" * 80)
    
    update_count = 0
    max_updates = 5  # Limit for demo
    
    async def on_update(result):
        nonlocal update_count
        update_count += 1
        
        timestamp = result.timestamp.strftime('%H:%M:%S')
        print(f"[{timestamp}] Sentiment: {result.overall_sentiment:+.3f} "
              f"(confidence: {result.confidence:.3f}, volume: {result.total_volume})")
        
        if update_count >= max_updates:
            raise asyncio.CancelledError("Demo complete")
    
    try:
        await aggregator.stream_sentiment(symbol, on_update)
    except asyncio.CancelledError:
        print("\nStreaming stopped")


async def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("ARA AI - Sentiment Analysis Demo")
    print("=" * 80)
    print("\nThis demo showcases the sentiment analysis capabilities.")
    print("Note: API keys are required for full functionality.")
    print("Set environment variables: TWITTER_BEARER_TOKEN, REDDIT_CLIENT_ID, etc.")
    
    try:
        # Run demos
        await demo_individual_sources()
        await demo_aggregated_sentiment()
        await demo_alternative_data()
        await demo_trending_tickers()
        
        # Uncomment to test streaming (will run for a while)
        # await demo_sentiment_streaming()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError in demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
