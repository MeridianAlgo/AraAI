"""
Multi-provider data aggregator with failover and quality scoring
Combines data from multiple sources with conflict resolution
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

from ara.core.interfaces import IDataProvider
from ara.core.exceptions import DataProviderError
from ara.data.cache import CacheManager
from ara.data.validation import (
    DataValidator,
    DataCleaner,
    ValidationConfig,
)
from ara.utils import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API requests
    """

    def __init__(
        self, requests_per_second: float = 5.0, burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter

        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size (default: 2x rate)
        """
        self.rate = requests_per_second
        self.burst_size = burst_size or int(requests_per_second * 2)
        self.tokens = float(self.burst_size)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens (wait if necessary)

        Args:
            tokens: Number of tokens to acquire
        """
        async with self._lock:
            while True:
                now = time.time()
                elapsed = now - self.last_update

                # Add tokens based on elapsed time
                self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Wait for tokens to replenish
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            "rate": self.rate,
            "burst_size": self.burst_size,
            "available_tokens": self.tokens,
        }


class ProviderStats:
    """
    Track statistics for a data provider
    """

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.requests = 0
        self.successes = 0
        self.failures = 0
        self.total_latency = 0.0
        self.last_success = None
        self.last_failure = None
        self.consecutive_failures = 0

    def record_success(self, latency: float) -> None:
        """Record successful request"""
        self.requests += 1
        self.successes += 1
        self.total_latency += latency
        self.last_success = datetime.now()
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record failed request"""
        self.requests += 1
        self.failures += 1
        self.last_failure = datetime.now()
        self.consecutive_failures += 1

    def get_success_rate(self) -> float:
        """Get success rate (0-1)"""
        if self.requests == 0:
            return 0.0
        return self.successes / self.requests

    def get_avg_latency(self) -> float:
        """Get average latency in seconds"""
        if self.successes == 0:
            return 0.0
        return self.total_latency / self.successes

    def is_healthy(self, max_consecutive_failures: int = 3) -> bool:
        """Check if provider is healthy"""
        return self.consecutive_failures < max_consecutive_failures

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        return {
            "provider": self.provider_name,
            "requests": self.requests,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.get_success_rate(),
            "avg_latency": self.get_avg_latency(),
            "consecutive_failures": self.consecutive_failures,
            "is_healthy": self.is_healthy(),
            "last_success": self.last_success,
            "last_failure": self.last_failure,
        }


class DataAggregator:
    """
    Aggregates data from multiple providers with automatic failover,
    quality scoring, and conflict resolution
    """

    def __init__(
        self,
        providers: List[IDataProvider],
        cache_manager: Optional[CacheManager] = None,
        validation_config: Optional[ValidationConfig] = None,
        rate_limit: float = 5.0,
        enable_failover: bool = True,
        enable_aggregation: bool = False,
        primary_provider: Optional[str] = None,
    ):
        """
        Initialize data aggregator

        Args:
            providers: List of data providers
            cache_manager: Cache manager instance
            validation_config: Validation configuration
            rate_limit: Requests per second per provider
            enable_failover: Enable automatic failover
            enable_aggregation: Enable multi-source aggregation
            primary_provider: Preferred provider name
        """
        self.providers = providers
        self.cache_manager = cache_manager or CacheManager()
        self.validation_config = validation_config or ValidationConfig()
        self.enable_failover = enable_failover
        self.enable_aggregation = enable_aggregation
        self.primary_provider = primary_provider

        # Initialize components
        self.validator = DataValidator(self.validation_config)
        self.cleaner = DataCleaner(self.validation_config)

        # Rate limiters per provider
        self.rate_limiters: Dict[str, RateLimiter] = {}
        for provider in providers:
            provider_name = provider.get_provider_name()
            self.rate_limiters[provider_name] = RateLimiter(
                requests_per_second=rate_limit
            )

        # Provider statistics
        self.provider_stats: Dict[str, ProviderStats] = {}
        for provider in providers:
            provider_name = provider.get_provider_name()
            self.provider_stats[provider_name] = ProviderStats(provider_name)

        logger.info(
            f"Initialized DataAggregator with {len(providers)} providers",
            providers=[p.get_provider_name() for p in providers],
            failover=enable_failover,
            aggregation=enable_aggregation,
        )

    async def fetch_historical(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True,
        clean_data: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical data with failover and quality scoring

        Args:
            symbol: Asset symbol
            period: Time period
            interval: Data interval
            use_cache: Use cached data if available
            clean_data: Clean and validate data

        Returns:
            DataFrame with historical data

        Raises:
            DataProviderError: If all providers fail
        """
        # Check cache first
        if use_cache:
            cache_key = f"historical:{symbol}:{period}:{interval}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                logger.info(f"Cache hit for {symbol}", symbol=symbol)
                return cached_data

        # Fetch data
        if self.enable_aggregation:
            df = await self._fetch_and_aggregate(symbol, period, interval)
        else:
            df = await self._fetch_with_failover(symbol, period, interval)

        # Clean and validate
        if clean_data:
            df, quality_report = self.cleaner.clean(df)

            if not quality_report.passed_validation:
                logger.warning(
                    f"Data quality below threshold for {symbol}",
                    symbol=symbol,
                    quality=quality_report.quality_score,
                )

            # Add quality score to dataframe
            df.attrs["quality_score"] = quality_report.quality_score
            df.attrs["quality_report"] = quality_report

        # Cache result
        if use_cache:
            cache_key = f"historical:{symbol}:{period}:{interval}"
            # Cache for 1 hour (L1) and 24 hours (L2)
            self.cache_manager.set(cache_key, df, l1_ttl=3600, l2_ttl=86400)

        return df

    async def fetch_realtime(
        self, symbol: str, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch real-time data with failover

        Args:
            symbol: Asset symbol
            use_cache: Use cached data if available

        Returns:
            Dict with real-time data

        Raises:
            DataProviderError: If all providers fail
        """
        # Check cache first (short TTL for real-time)
        if use_cache:
            cache_key = f"realtime:{symbol}"
            cached_data = self.cache_manager.get(cache_key)
            if cached_data is not None:
                return cached_data

        # Get provider order
        providers = self._get_provider_order()

        # Try each provider
        for provider in providers:
            provider_name = provider.get_provider_name()

            # Check if provider is healthy
            if not self.provider_stats[provider_name].is_healthy():
                logger.warning(
                    f"Skipping unhealthy provider: {provider_name}",
                    provider=provider_name,
                )
                continue

            try:
                # Rate limit
                await self.rate_limiters[provider_name].acquire()

                # Fetch data
                start_time = time.time()
                data = await provider.fetch_realtime(symbol)
                latency = time.time() - start_time

                # Record success
                self.provider_stats[provider_name].record_success(latency)

                logger.info(
                    f"Fetched realtime from {provider_name}",
                    symbol=symbol,
                    provider=provider_name,
                    latency=latency,
                )

                # Cache result (10 seconds)
                if use_cache:
                    cache_key = f"realtime:{symbol}"
                    self.cache_manager.set(cache_key, data, l1_ttl=10, l2_ttl=60)

                return data

            except Exception as e:
                self.provider_stats[provider_name].record_failure()
                logger.warning(
                    f"Provider {provider_name} failed: {e}",
                    symbol=symbol,
                    provider=provider_name,
                    error=str(e),
                )

                if not self.enable_failover:
                    raise

                continue

        raise DataProviderError(
            f"All providers failed to fetch realtime data for {symbol}",
            {"providers": [p.get_provider_name() for p in providers]},
        )

    async def _fetch_with_failover(
        self, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        """
        Fetch from providers with automatic failover

        Args:
            symbol: Asset symbol
            period: Time period
            interval: Data interval

        Returns:
            DataFrame from first successful provider

        Raises:
            DataProviderError: If all providers fail
        """
        providers = self._get_provider_order()

        for provider in providers:
            provider_name = provider.get_provider_name()

            # Check if provider is healthy
            if not self.provider_stats[provider_name].is_healthy():
                logger.warning(
                    f"Skipping unhealthy provider: {provider_name}",
                    provider=provider_name,
                )
                continue

            try:
                # Rate limit
                await self.rate_limiters[provider_name].acquire()

                # Fetch data
                start_time = time.time()
                df = await provider.fetch_historical(symbol, period, interval)
                latency = time.time() - start_time

                # Validate
                report = self.validator.validate(df)

                # Record success
                self.provider_stats[provider_name].record_success(latency)

                # Add metadata
                df.attrs["source"] = provider_name
                df.attrs["quality_score"] = report.quality_score

                logger.info(
                    f"Fetched from {provider_name}",
                    symbol=symbol,
                    provider=provider_name,
                    rows=len(df),
                    quality=report.quality_score,
                    latency=latency,
                )

                return df

            except Exception as e:
                self.provider_stats[provider_name].record_failure()
                logger.warning(
                    f"Provider {provider_name} failed: {e}",
                    symbol=symbol,
                    provider=provider_name,
                    error=str(e),
                )

                if not self.enable_failover:
                    raise

                continue

        raise DataProviderError(
            f"All providers failed to fetch data for {symbol}",
            {"providers": [p.get_provider_name() for p in providers]},
        )

    async def _fetch_and_aggregate(
        self, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        """
        Fetch from all providers and aggregate with conflict resolution

        Args:
            symbol: Asset symbol
            period: Time period
            interval: Data interval

        Returns:
            Aggregated DataFrame

        Raises:
            DataProviderError: If no providers succeed
        """
        # Fetch from all providers concurrently
        tasks = []
        for provider in self.providers:
            task = self._safe_fetch(provider, symbol, period, interval)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        valid_sources = {}
        for i, result in enumerate(results):
            provider = self.providers[i]
            provider_name = provider.get_provider_name()

            if isinstance(result, pd.DataFrame):
                # Validate
                report = self.validator.validate(result)
                result.attrs["source"] = provider_name
                result.attrs["quality_score"] = report.quality_score
                valid_sources[provider_name] = result

                logger.info(
                    f"Fetched from {provider_name} for aggregation",
                    provider=provider_name,
                    rows=len(result),
                    quality=report.quality_score,
                )
            else:
                logger.warning(
                    f"Provider {provider_name} failed",
                    provider=provider_name,
                    error=str(result),
                )

        if not valid_sources:
            raise DataProviderError(f"No providers returned valid data for {symbol}")

        # If only one source, return it
        if len(valid_sources) == 1:
            return list(valid_sources.values())[0]

        # Aggregate multiple sources
        aggregated = self._resolve_conflicts(valid_sources)

        logger.info(
            f"Aggregated data from {len(valid_sources)} sources",
            symbol=symbol,
            sources=list(valid_sources.keys()),
        )

        return aggregated

    async def _safe_fetch(
        self, provider: IDataProvider, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        """
        Safely fetch data from provider with rate limiting

        Args:
            provider: Data provider
            symbol: Asset symbol
            period: Time period
            interval: Data interval

        Returns:
            DataFrame from provider
        """
        provider_name = provider.get_provider_name()

        try:
            # Rate limit
            await self.rate_limiters[provider_name].acquire()

            # Fetch
            start_time = time.time()
            df = await provider.fetch_historical(symbol, period, interval)
            latency = time.time() - start_time

            # Record success
            self.provider_stats[provider_name].record_success(latency)

            return df

        except Exception:
            self.provider_stats[provider_name].record_failure()
            raise

    def _resolve_conflicts(self, sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Resolve conflicts between multiple data sources
        Uses weighted average based on quality scores

        Args:
            sources: Dict mapping source names to DataFrames

        Returns:
            Aggregated DataFrame
        """
        # Align all dataframes by timestamp
        aligned_dfs = []
        for source_name, df in sources.items():
            df_copy = df.copy()
            df_copy["_source"] = source_name
            df_copy["_quality"] = df.attrs.get("quality_score", 0.5)
            aligned_dfs.append(df_copy)

        # Concatenate all dataframes
        combined = pd.concat(aligned_dfs, axis=0)

        # Group by timestamp and aggregate
        result_data = []
        for timestamp, group in combined.groupby(level=0):
            # Calculate weighted average for each column
            row = {"timestamp": timestamp}

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in group.columns:
                    continue

                # Weighted average by quality score
                values = group[col].values
                weights = group["_quality"].values

                if len(values) > 0 and not all(pd.isna(values)):
                    # Remove NaN values
                    valid_mask = ~pd.isna(values)
                    valid_values = values[valid_mask]
                    valid_weights = weights[valid_mask]

                    if len(valid_values) > 0:
                        weighted_avg = np.average(valid_values, weights=valid_weights)
                        row[col] = weighted_avg

            result_data.append(row)

        # Create result dataframe
        result_df = pd.DataFrame(result_data)
        result_df["Date"] = pd.to_datetime(result_df["timestamp"])
        result_df = result_df.set_index("Date")
        result_df = result_df.drop("timestamp", axis=1)

        # Add metadata
        result_df.attrs["source"] = "aggregated"
        result_df.attrs["sources"] = list(sources.keys())
        result_df.attrs["quality_score"] = 1.0  # Aggregated data is high quality

        return result_df

    def _get_provider_order(self) -> List[IDataProvider]:
        """
        Get providers in priority order
        Primary provider first, then by success rate

        Returns:
            List of providers in priority order
        """
        # Start with primary provider if specified
        ordered = []

        if self.primary_provider:
            for provider in self.providers:
                if provider.get_provider_name() == self.primary_provider:
                    ordered.append(provider)
                    break

        # Sort remaining by success rate
        remaining = [
            p for p in self.providers if p.get_provider_name() != self.primary_provider
        ]

        remaining.sort(
            key=lambda p: self.provider_stats[p.get_provider_name()].get_success_rate(),
            reverse=True,
        )

        ordered.extend(remaining)

        return ordered

    def get_provider_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all providers

        Returns:
            List of provider statistics
        """
        stats = []
        for provider_name, provider_stats in self.provider_stats.items():
            stats.append(provider_stats.get_stats())

        # Sort by success rate
        stats.sort(key=lambda x: x["success_rate"], reverse=True)

        return stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict with cache stats
        """
        return self.cache_manager.get_stats()

    def get_rate_limiter_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics

        Returns:
            Dict with rate limiter stats per provider
        """
        stats = {}
        for provider_name, limiter in self.rate_limiters.items():
            stats[provider_name] = limiter.get_stats()

        return stats

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.cache_manager.clear()
        logger.info("Cleared cache")

    def reset_stats(self) -> None:
        """Reset all provider statistics"""
        for provider_name in self.provider_stats:
            self.provider_stats[provider_name] = ProviderStats(provider_name)

        logger.info("Reset provider statistics")


# Import numpy for aggregation
import numpy as np
