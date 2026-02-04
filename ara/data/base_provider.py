"""
Base data provider with common functionality
"""

import asyncio
from typing import Dict, List, Any, Optional
import pandas as pd

from ara.core.interfaces import IDataProvider, AssetType
from ara.core.exceptions import DataProviderError
from ara.utils import get_logger

logger = get_logger(__name__)


class BaseDataProvider(IDataProvider):
    """
    Base class for data providers
    Implements common functionality and retry logic
    """

    def __init__(self, name: str, asset_type: AssetType, max_retries: int = 3, timeout: int = 30):
        self.name = name
        self.asset_type = asset_type
        self.max_retries = max_retries
        self.timeout = timeout
        self._supported_symbols: Optional[List[str]] = None

    async def fetch_with_retry(self, fetch_func, *args, **kwargs) -> Any:
        """
        Execute fetch function with retry logic

        Args:
            fetch_func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from fetch_func

        Raises:
            DataProviderError: If all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = await asyncio.wait_for(fetch_func(*args, **kwargs), timeout=self.timeout)
                return result

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    f"{self.name} timeout on attempt {attempt + 1}",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"{self.name} error on attempt {attempt + 1}: {e}",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )

            # Exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = 2**attempt
                await asyncio.sleep(wait_time)

        raise DataProviderError(
            f"{self.name} failed after {self.max_retries} attempts",
            {"last_error": str(last_error)},
        )

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLCV dataframe

        Args:
            df: DataFrame to validate

        Returns:
            True if valid

        Raises:
            DataProviderError: If validation fails
        """
        required_columns = ["open", "high", "low", "close", "volume"]

        # Check columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataProviderError(
                f"Missing required columns: {missing_columns}",
                {"columns": list(df.columns)},
            )

        # Check for empty dataframe
        if len(df) == 0:
            raise DataProviderError("Empty dataframe returned")

        # Check for NaN values in critical columns
        for col in ["close", "volume"]:
            if df[col].isna().all():
                raise DataProviderError(f"All values are NaN in column: {col}")

        # Check for negative prices
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if (df[col] < 0).any():
                raise DataProviderError(f"Negative values found in column: {col}")

        # Check High >= Low
        if (df["high"] < df["low"]).any():
            raise DataProviderError("High price is less than Low price")

        return True

    def get_provider_name(self) -> str:
        """Return provider name"""
        return self.name

    def get_asset_type(self) -> AssetType:
        """Return asset type"""
        return self.asset_type

    def get_supported_symbols(self) -> List[str]:
        """Return list of supported symbols"""
        if self._supported_symbols is None:
            return []
        return self._supported_symbols

    # Abstract methods to be implemented by subclasses
    async def fetch_historical(
        self, symbol: str, period: str = "2y", interval: str = "1d"
    ) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement fetch_historical()")

    async def fetch_realtime(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement fetch_realtime()")

    async def stream_data(self, symbol: str, callback: callable) -> None:
        raise NotImplementedError("Subclasses must implement stream_data()")
