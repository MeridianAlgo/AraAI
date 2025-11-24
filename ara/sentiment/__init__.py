"""
Sentiment Analysis Module

This module provides sentiment analysis capabilities for financial assets
using multiple data sources including Twitter, Reddit, and news articles.
"""

from ara.sentiment.base_analyzer import SentimentAnalyzer, SentimentResult, SentimentScore
from ara.sentiment.twitter_analyzer import TwitterSentimentAnalyzer
from ara.sentiment.reddit_analyzer import RedditSentimentAnalyzer
from ara.sentiment.news_analyzer import NewsSentimentAnalyzer
from ara.sentiment.aggregator import SentimentAggregator
from ara.sentiment.alternative_data import AlternativeDataProvider

__all__ = [
    'SentimentAnalyzer',
    'SentimentResult',
    'SentimentScore',
    'TwitterSentimentAnalyzer',
    'RedditSentimentAnalyzer',
    'NewsSentimentAnalyzer',
    'SentimentAggregator',
    'AlternativeDataProvider',
]
