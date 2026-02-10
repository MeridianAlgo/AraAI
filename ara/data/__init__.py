"""
Data providers for ARA AI
Supports stocks, cryptocurrencies, and forex
"""

from ara.data.aggregator import DataAggregator, ProviderStats, RateLimiter
from ara.data.base_provider import BaseDataProvider
from ara.data.cache import CacheManager, LRUCache, RedisCache, cached
from ara.data.crypto_provider import (
    BinanceProvider,
    CoinbaseProvider,
    CryptoDataAggregator,
    CryptoExchangeProvider,
    KrakenProvider,
)
from ara.data.defi_provider import DeFiDataProvider
from ara.data.onchain_provider import OnChainMetricsProvider
from ara.data.validation import (
    DataCleaner,
    DataQualityReport,
    DataQualityScorer,
    DataValidator,
    ImputationStrategy,
    OutlierMethod,
    ValidationConfig,
)

__all__ = [
    "BaseDataProvider",
    "CryptoExchangeProvider",
    "BinanceProvider",
    "CoinbaseProvider",
    "KrakenProvider",
    "CryptoDataAggregator",
    "OnChainMetricsProvider",
    "DeFiDataProvider",
    "CacheManager",
    "LRUCache",
    "RedisCache",
    "cached",
    "DataValidator",
    "DataCleaner",
    "DataQualityScorer",
    "DataQualityReport",
    "ValidationConfig",
    "ImputationStrategy",
    "OutlierMethod",
    "DataAggregator",
    "RateLimiter",
    "ProviderStats",
]
