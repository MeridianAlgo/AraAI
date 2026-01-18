"""
Cryptocurrency exchange data providers
Supports multiple exchanges with unified interface
"""

import ccxt
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta

from ara.data.base_provider import BaseDataProvider
from ara.core.interfaces import AssetType
from ara.core.exceptions import DataProviderError
from ara.utils import get_logger, timed

logger = get_logger(__name__)


class CryptoExchangeProvider(BaseDataProvider):
    """
    Base class for cryptocurrency exchange providers
    Uses CCXT library for unified exchange access
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=f"crypto_{exchange_id}", asset_type=AssetType.CRYPTO)

        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret

        # Initialize exchange
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "enableRateLimit": True,
                    **kwargs,
                }
            )
        except AttributeError:
            raise DataProviderError(
                f"Exchange not supported: {exchange_id}",
                {"supported_exchanges": ccxt.exchanges},
            )

    @timed("crypto_fetch_historical")
    async def fetch_historical(
        self, symbol: str, period: str = "2y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from exchange

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            period: Time period
            interval: Candle interval

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert symbol format
            symbol = self._normalize_symbol(symbol)

            # Convert period to milliseconds
            since = self._period_to_timestamp(period)

            # Convert interval to exchange format
            timeframe = self._normalize_interval(interval)

            # Fetch data
            async def _fetch():
                # Load markets if not loaded
                if not self.exchange.markets:
                    await self.exchange.load_markets()

                # Fetch OHLCV data
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, since=since, limit=1000
                )

                return ohlcv

            ohlcv_data = await self.fetch_with_retry(_fetch)

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            # Convert timestamp to datetime
            df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.drop("timestamp", axis=1)
            df = df.set_index("Date")

            # Validate
            self.validate_dataframe(df)

            logger.info(
                f"Fetched {len(df)} candles from {self.exchange_id}",
                symbol=symbol,
                period=period,
                interval=interval,
                rows=len(df),
            )

            return df

        except Exception as e:
            raise DataProviderError(
                f"Failed to fetch historical data from {self.exchange_id}",
                {"symbol": symbol, "error": str(e)},
            )

    @timed("crypto_fetch_realtime")
    async def fetch_realtime(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real-time ticker data

        Args:
            symbol: Trading pair

        Returns:
            Dict with current price, volume, etc.
        """
        try:
            symbol = self._normalize_symbol(symbol)

            async def _fetch():
                if not self.exchange.markets:
                    await self.exchange.load_markets()

                ticker = await self.exchange.fetch_ticker(symbol)
                return ticker

            ticker = await self.fetch_with_retry(_fetch)

            return {
                "symbol": symbol,
                "price": ticker["last"],
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "volume": ticker["baseVolume"],
                "volume_quote": ticker["quoteVolume"],
                "high_24h": ticker["high"],
                "low_24h": ticker["low"],
                "change_24h": ticker["change"],
                "change_pct_24h": ticker["percentage"],
                "timestamp": datetime.fromtimestamp(ticker["timestamp"] / 1000),
                "exchange": self.exchange_id,
            }

        except Exception as e:
            raise DataProviderError(
                f"Failed to fetch realtime data from {self.exchange_id}",
                {"symbol": symbol, "error": str(e)},
            )

    async def stream_data(self, symbol: str, callback: Callable[[Dict], None]) -> None:
        """
        Stream real-time data via WebSocket

        Args:
            symbol: Trading pair
            callback: Function to call with each update
        """
        try:
            symbol = self._normalize_symbol(symbol)

            # Check if exchange supports WebSocket
            if not self.exchange.has["watchTicker"]:
                raise DataProviderError(
                    f"{self.exchange_id} does not support WebSocket streaming"
                )

            logger.info(f"Starting WebSocket stream for {symbol} on {self.exchange_id}")

            while True:
                try:
                    ticker = await self.exchange.watch_ticker(symbol)

                    data = {
                        "symbol": symbol,
                        "price": ticker["last"],
                        "volume": ticker["baseVolume"],
                        "timestamp": datetime.fromtimestamp(ticker["timestamp"] / 1000),
                        "exchange": self.exchange_id,
                    }

                    callback(data)

                except Exception as e:
                    logger.error(
                        f"WebSocket error: {e}",
                        symbol=symbol,
                        exchange=self.exchange_id,
                    )
                    await asyncio.sleep(5)  # Wait before reconnecting

        except Exception as e:
            raise DataProviderError(
                f"Failed to stream data from {self.exchange_id}",
                {"symbol": symbol, "error": str(e)},
            )

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported trading pairs"""
        try:
            if not self.exchange.markets:
                # Synchronous load for this method
                asyncio.run(self.exchange.load_markets())

            return list(self.exchange.markets.keys())

        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            return []

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to exchange format

        Args:
            symbol: Symbol in various formats

        Returns:
            Normalized symbol (e.g., 'BTC/USDT')
        """
        # Remove common suffixes
        symbol = symbol.upper()
        symbol = symbol.replace("-USD", "/USDT")
        symbol = symbol.replace("USD", "/USDT")

        # Ensure slash format
        if "/" not in symbol and len(symbol) > 3:
            # Try to split common pairs
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                symbol = f"{base}/USDT"
            elif symbol.endswith("BTC"):
                base = symbol[:-3]
                symbol = f"{base}/BTC"
            elif symbol.endswith("ETH"):
                base = symbol[:-3]
                symbol = f"{base}/ETH"

        return symbol

    def _normalize_interval(self, interval: str) -> str:
        """
        Normalize interval to exchange format

        Args:
            interval: Interval string (e.g., '1d', '1h')

        Returns:
            Exchange-specific interval format
        """
        # CCXT uses standard format: 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M
        interval_map = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1hour": "1h",
            "4hour": "4h",
            "1day": "1d",
            "1week": "1w",
            "1month": "1M",
        }

        return interval_map.get(interval, interval)

    def _period_to_timestamp(self, period: str) -> int:
        """
        Convert period string to timestamp

        Args:
            period: Period string (e.g., '1y', '6mo')

        Returns:
            Timestamp in milliseconds
        """
        now = datetime.now()

        period_map = {
            "1d": timedelta(days=1),
            "5d": timedelta(days=5),
            "1mo": timedelta(days=30),
            "3mo": timedelta(days=90),
            "6mo": timedelta(days=180),
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
            "5y": timedelta(days=1825),
            "10y": timedelta(days=3650),
        }

        delta = period_map.get(period, timedelta(days=730))
        since = now - delta

        return int(since.timestamp() * 1000)


class BinanceProvider(CryptoExchangeProvider):
    """Binance exchange provider"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__("binance", api_key, api_secret)


class CoinbaseProvider(CryptoExchangeProvider):
    """Coinbase exchange provider"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__("coinbase", api_key, api_secret)


class KrakenProvider(CryptoExchangeProvider):
    """Kraken exchange provider"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__("kraken", api_key, api_secret)


class CryptoDataAggregator:
    """
    Aggregates data from multiple cryptocurrency exchanges
    Provides unified interface with automatic failover and data quality scoring
    """

    def __init__(
        self,
        providers: Optional[List[CryptoExchangeProvider]] = None,
        primary_provider: Optional[str] = None,
    ):
        """
        Initialize aggregator with multiple providers

        Args:
            providers: List of exchange providers
            primary_provider: Preferred provider name (e.g., 'binance')
        """
        self.providers = providers or []
        self.primary_provider = primary_provider

        # Initialize default providers if none provided
        if not self.providers:
            self.providers = [BinanceProvider(), CoinbaseProvider(), KrakenProvider()]

        logger.info(
            f"Initialized CryptoDataAggregator with {len(self.providers)} providers",
            providers=[p.exchange_id for p in self.providers],
        )

    async def fetch_historical(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d",
        use_aggregation: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch historical data with automatic failover

        Args:
            symbol: Trading pair
            period: Time period
            interval: Candle interval
            use_aggregation: If True, fetch from all providers and aggregate

        Returns:
            DataFrame with OHLCV data
        """
        if use_aggregation:
            return await self._fetch_and_aggregate(symbol, period, interval)

        # Try primary provider first
        providers = self._get_provider_order()

        for provider in providers:
            try:
                logger.info(
                    f"Attempting to fetch from {provider.exchange_id}",
                    symbol=symbol,
                    provider=provider.exchange_id,
                )

                df = await provider.fetch_historical(symbol, period, interval)

                # Add source column
                df["source"] = provider.exchange_id
                df["quality_score"] = self._calculate_quality_score(df)

                logger.info(
                    f"Successfully fetched from {provider.exchange_id}",
                    symbol=symbol,
                    rows=len(df),
                    quality=df["quality_score"].mean(),
                )

                return df

            except Exception as e:
                logger.warning(
                    f"Failed to fetch from {provider.exchange_id}: {e}",
                    symbol=symbol,
                    provider=provider.exchange_id,
                    error=str(e),
                )
                continue

        raise DataProviderError(
            f"All providers failed to fetch data for {symbol}",
            {"providers": [p.exchange_id for p in self.providers]},
        )

    async def _fetch_and_aggregate(
        self, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        """
        Fetch from all providers and aggregate with conflict resolution

        Args:
            symbol: Trading pair
            period: Time period
            interval: Candle interval

        Returns:
            Aggregated DataFrame
        """
        tasks = []
        for provider in self.providers:
            task = self._safe_fetch(provider, symbol, period, interval)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        valid_dfs = []
        for i, result in enumerate(results):
            if isinstance(result, pd.DataFrame):
                result["source"] = self.providers[i].exchange_id
                result["quality_score"] = self._calculate_quality_score(result)
                valid_dfs.append(result)
            else:
                logger.warning(
                    f"Provider {self.providers[i].exchange_id} failed",
                    error=str(result),
                )

        if not valid_dfs:
            raise DataProviderError(f"No providers returned valid data for {symbol}")

        # Aggregate data
        aggregated = self._aggregate_dataframes(valid_dfs)

        logger.info(
            f"Aggregated data from {len(valid_dfs)} providers",
            symbol=symbol,
            providers=[df["source"].iloc[0] for df in valid_dfs],
        )

        return aggregated

    async def _safe_fetch(
        self, provider: CryptoExchangeProvider, symbol: str, period: str, interval: str
    ) -> pd.DataFrame:
        """Safely fetch data from provider"""
        try:
            return await provider.fetch_historical(symbol, period, interval)
        except Exception as e:
            logger.warning(f"Provider {provider.exchange_id} failed: {e}")
            raise

    def _aggregate_dataframes(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate multiple dataframes with conflict resolution

        Args:
            dfs: List of DataFrames from different providers

        Returns:
            Aggregated DataFrame
        """
        if len(dfs) == 1:
            return dfs[0]

        # Align all dataframes by timestamp
        aligned_dfs = []
        for df in dfs:
            df_copy = df.copy()
            df_copy["timestamp"] = df_copy.index
            aligned_dfs.append(df_copy)

        # Merge on timestamp
        merged = aligned_dfs[0]
        for df in aligned_dfs[1:]:
            merged = pd.merge(
                merged,
                df,
                on="timestamp",
                how="outer",
                suffixes=("", f'_{df["source"].iloc[0]}'),
            )

        # Resolve conflicts using weighted average based on quality scores
        result_df = pd.DataFrame(index=merged["timestamp"])

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            # Find all columns for this metric
            col_variants = [c for c in merged.columns if c.startswith(col)]

            if len(col_variants) == 1:
                result_df[col] = merged[col_variants[0]]
            else:
                # Weighted average based on quality scores
                values = []
                weights = []

                for variant in col_variants:
                    if variant == col:
                        quality_col = "quality_score"
                    else:
                        source = variant.split("_")[-1]
                        quality_col = f"quality_score_{source}"

                    if quality_col in merged.columns:
                        values.append(merged[variant])
                        weights.append(merged[quality_col])

                # Calculate weighted average
                if values and weights:
                    weighted_sum = sum(v * w for v, w in zip(values, weights))
                    weight_sum = sum(weights)
                    result_df[col] = weighted_sum / weight_sum
                else:
                    result_df[col] = merged[col_variants[0]]

        # Add metadata
        result_df["source"] = "aggregated"
        result_df["quality_score"] = 1.0
        result_df.index = pd.to_datetime(result_df.index)
        result_df.index.name = "Date"

        return result_df

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate data quality score (0-1)

        Args:
            df: DataFrame to score

        Returns:
            Quality score between 0 and 1
        """
        score = 1.0

        # Penalize missing data
        missing_ratio = df.isna().sum().sum() / (len(df) * len(df.columns))
        score -= missing_ratio * 0.3

        # Penalize zero volumes
        zero_volume_ratio = (df["Volume"] == 0).sum() / len(df)
        score -= zero_volume_ratio * 0.2

        # Check for suspicious patterns (e.g., repeated values)
        for col in ["Close", "Volume"]:
            if col in df.columns:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    score -= 0.1

        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

    def _get_provider_order(self) -> List[CryptoExchangeProvider]:
        """
        Get providers in priority order

        Returns:
            List of providers with primary first
        """
        if not self.primary_provider:
            return self.providers

        # Move primary provider to front
        ordered = []
        for provider in self.providers:
            if provider.exchange_id == self.primary_provider:
                ordered.insert(0, provider)
            else:
                ordered.append(provider)

        return ordered

    async def fetch_realtime(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real-time data with failover

        Args:
            symbol: Trading pair

        Returns:
            Dict with current price data
        """
        providers = self._get_provider_order()

        for provider in providers:
            try:
                return await provider.fetch_realtime(symbol)
            except Exception as e:
                logger.warning(
                    f"Failed to fetch realtime from {provider.exchange_id}: {e}"
                )
                continue

        raise DataProviderError(
            f"All providers failed to fetch realtime data for {symbol}"
        )

    def get_supported_symbols(self) -> List[str]:
        """
        Get union of all supported symbols across providers

        Returns:
            List of supported symbols
        """
        all_symbols = set()

        for provider in self.providers:
            try:
                symbols = provider.get_supported_symbols()
                all_symbols.update(symbols)
            except Exception as e:
                logger.warning(
                    f"Failed to get symbols from {provider.exchange_id}: {e}"
                )

        return sorted(list(all_symbols))

    def get_major_cryptocurrencies(self) -> List[str]:
        """
        Get list of 50+ major cryptocurrencies

        Returns:
            List of major crypto symbols
        """
        # Top 50+ cryptocurrencies by market cap
        major_cryptos = [
            # Top 10
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "XRP/USDT",
            "ADA/USDT",
            "DOGE/USDT",
            "SOL/USDT",
            "TRX/USDT",
            "DOT/USDT",
            "MATIC/USDT",
            # 11-20
            "LTC/USDT",
            "SHIB/USDT",
            "AVAX/USDT",
            "UNI/USDT",
            "LINK/USDT",
            "ATOM/USDT",
            "XMR/USDT",
            "ETC/USDT",
            "BCH/USDT",
            "XLM/USDT",
            # 21-30
            "ALGO/USDT",
            "FIL/USDT",
            "VET/USDT",
            "ICP/USDT",
            "HBAR/USDT",
            "APT/USDT",
            "NEAR/USDT",
            "QNT/USDT",
            "GRT/USDT",
            "AAVE/USDT",
            # 31-40
            "SAND/USDT",
            "MANA/USDT",
            "AXS/USDT",
            "THETA/USDT",
            "FTM/USDT",
            "EOS/USDT",
            "EGLD/USDT",
            "XTZ/USDT",
            "RUNE/USDT",
            "ZEC/USDT",
            # 41-50
            "CAKE/USDT",
            "MKR/USDT",
            "NEO/USDT",
            "KCS/USDT",
            "DASH/USDT",
            "COMP/USDT",
            "ENJ/USDT",
            "BAT/USDT",
            "ZIL/USDT",
            "WAVES/USDT",
            # 51-60 (bonus)
            "CHZ/USDT",
            "SUSHI/USDT",
            "HOT/USDT",
            "ONT/USDT",
            "ICX/USDT",
            "QTUM/USDT",
            "OMG/USDT",
            "ZRX/USDT",
            "SNX/USDT",
            "CRV/USDT",
        ]

        return major_cryptos
