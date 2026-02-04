"""
Currency converter with real-time exchange rates
"""

import asyncio
from datetime import datetime
from typing import Dict, List
import yfinance as yf
from ara.currency.models import Currency, ConversionResult, ExchangeRate
from ara.core.exceptions import DataProviderError
from ara.utils import get_logger

logger = get_logger(__name__)


class CurrencyConverter:
    """
    Currency converter with real-time exchange rates
    Supports 10+ major currencies with caching
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize currency converter

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.cache_ttl = cache_ttl
        self._rate_cache: Dict[tuple, ExchangeRate] = {}
        self._cache_timestamps: Dict[tuple, datetime] = {}

    def _get_forex_symbol(self, from_currency: Currency, to_currency: Currency) -> str:
        """
        Get forex symbol for currency pair

        Args:
            from_currency: Source currency
            to_currency: Target currency

        Returns:
            Forex symbol (e.g., 'EURUSD=X')
        """
        return f"{from_currency.value}{to_currency.value}=X"

    def _is_cache_valid(self, from_currency: Currency, to_currency: Currency) -> bool:
        """
        Check if cached rate is still valid

        Args:
            from_currency: Source currency
            to_currency: Target currency

        Returns:
            True if cache is valid
        """
        cache_key = (from_currency, to_currency)

        if cache_key not in self._cache_timestamps:
            return False

        cache_time = self._cache_timestamps[cache_key]
        age = (datetime.now() - cache_time).total_seconds()

        return age < self.cache_ttl

    async def get_exchange_rate(
        self, from_currency: Currency, to_currency: Currency, use_cache: bool = True
    ) -> ExchangeRate:
        """
        Get real-time exchange rate between two currencies

        Args:
            from_currency: Source currency
            to_currency: Target currency
            use_cache: Whether to use cached rates

        Returns:
            ExchangeRate object

        Raises:
            DataProviderError: If rate cannot be fetched
        """
        # Same currency
        if from_currency == to_currency:
            return ExchangeRate(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=1.0,
                timestamp=datetime.now(),
                source="identity",
            )

        # Check cache
        cache_key = (from_currency, to_currency)
        if use_cache and self._is_cache_valid(from_currency, to_currency):
            logger.debug(f"Using cached rate for {from_currency.value}/{to_currency.value}")
            return self._rate_cache[cache_key]

        # Fetch from Yahoo Finance
        try:
            forex_symbol = self._get_forex_symbol(from_currency, to_currency)

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, forex_symbol)
            info = await loop.run_in_executor(None, lambda: ticker.info)

            # Get current rate
            rate = info.get("regularMarketPrice") or info.get("previousClose")

            if rate is None:
                raise DataProviderError(
                    f"No rate available for {from_currency.value}/{to_currency.value}"
                )

            # Get bid/ask if available
            bid = info.get("bid")
            ask = info.get("ask")

            exchange_rate = ExchangeRate(
                from_currency=from_currency,
                to_currency=to_currency,
                rate=float(rate),
                timestamp=datetime.now(),
                source="yahoo_finance",
                bid=float(bid) if bid else None,
                ask=float(ask) if ask else None,
            )

            # Cache the rate
            self._rate_cache[cache_key] = exchange_rate
            self._cache_timestamps[cache_key] = datetime.now()

            logger.info(
                f"Fetched exchange rate: {from_currency.value}/{to_currency.value} = {rate}",
                from_currency=from_currency.value,
                to_currency=to_currency.value,
                rate=rate,
            )

            return exchange_rate

        except Exception as e:
            logger.error(
                f"Failed to fetch exchange rate: {e}",
                from_currency=from_currency.value,
                to_currency=to_currency.value,
                error=str(e),
            )
            raise DataProviderError(
                f"Failed to fetch exchange rate for {from_currency.value}/{to_currency.value}",
                {"error": str(e)},
            )

    async def convert(
        self,
        amount: float,
        from_currency: Currency,
        to_currency: Currency,
        use_cache: bool = True,
    ) -> ConversionResult:
        """
        Convert amount from one currency to another

        Args:
            amount: Amount to convert
            from_currency: Source currency
            to_currency: Target currency
            use_cache: Whether to use cached rates

        Returns:
            ConversionResult object
        """
        exchange_rate = await self.get_exchange_rate(
            from_currency, to_currency, use_cache=use_cache
        )

        converted_amount = amount * exchange_rate.rate

        return ConversionResult(
            original_amount=amount,
            original_currency=from_currency,
            converted_amount=converted_amount,
            target_currency=to_currency,
            exchange_rate=exchange_rate.rate,
            timestamp=exchange_rate.timestamp,
            source=exchange_rate.source,
        )

    async def convert_multiple(
        self,
        amounts: Dict[Currency, float],
        target_currency: Currency,
        use_cache: bool = True,
    ) -> Dict[Currency, ConversionResult]:
        """
        Convert multiple amounts to target currency

        Args:
            amounts: Dictionary of currency -> amount
            target_currency: Target currency
            use_cache: Whether to use cached rates

        Returns:
            Dictionary of currency -> ConversionResult
        """
        tasks = []
        currencies = []

        for currency, amount in amounts.items():
            task = self.convert(amount, currency, target_currency, use_cache)
            tasks.append(task)
            currencies.append(currency)

        results = await asyncio.gather(*tasks)

        return {currency: result for currency, result in zip(currencies, results)}

    async def get_cross_rates(
        self,
        base_currency: Currency,
        target_currencies: List[Currency],
        use_cache: bool = True,
    ) -> Dict[Currency, ExchangeRate]:
        """
        Get exchange rates from base currency to multiple targets

        Args:
            base_currency: Base currency
            target_currencies: List of target currencies
            use_cache: Whether to use cached rates

        Returns:
            Dictionary of currency -> ExchangeRate
        """
        tasks = []

        for target_currency in target_currencies:
            if target_currency != base_currency:
                task = self.get_exchange_rate(base_currency, target_currency, use_cache=use_cache)
                tasks.append(task)
            else:
                # Identity rate
                async def identity_rate():
                    return ExchangeRate(
                        from_currency=base_currency,
                        to_currency=base_currency,
                        rate=1.0,
                        timestamp=datetime.now(),
                        source="identity",
                    )

                tasks.append(identity_rate())

        results = await asyncio.gather(*tasks)

        return {currency: result for currency, result in zip(target_currencies, results)}

    def clear_cache(self):
        """Clear the exchange rate cache"""
        self._rate_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Exchange rate cache cleared")

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        total_entries = len(self._rate_cache)
        valid_entries = sum(
            1 for key in self._rate_cache.keys() if self._is_cache_valid(key[0], key[1])
        )

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": total_entries - valid_entries,
            "cache_ttl": self.cache_ttl,
        }
