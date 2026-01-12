"""
Alternative Data Provider

Integrates alternative data sources including Google Trends, insider trading,
and institutional holdings for enhanced predictions.
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
import logging

from ara.core.exceptions import AraAIException

logger = logging.getLogger(__name__)


class AlternativeDataError(AraAIException):
    """Alternative data related errors"""

    pass


class AlternativeDataProvider:
    """
    Provides alternative data for financial analysis.

    Features:
    - Google Trends search interest
    - SEC EDGAR insider trading data
    - Institutional holdings (13F filings)
    - Alternative data feature engineering
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize alternative data provider.

        Args:
            config: Configuration dictionary
                - enable_google_trends: Enable Google Trends (default: True)
                - enable_insider_trading: Enable insider trading data (default: True)
                - enable_institutional: Enable institutional holdings (default: True)
        """
        self.config = config or {}
        self.enable_google_trends = self.config.get("enable_google_trends", True)
        self.enable_insider_trading = self.config.get("enable_insider_trading", True)
        self.enable_institutional = self.config.get("enable_institutional", True)

    async def get_alternative_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get all alternative data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with alternative data
        """
        data = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "google_trends": None,
            "insider_trading": None,
            "institutional_holdings": None,
        }

        # Fetch data from enabled sources
        tasks = []

        if self.enable_google_trends:
            tasks.append(self._fetch_google_trends(symbol))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        if self.enable_insider_trading:
            tasks.append(self._fetch_insider_trading(symbol))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        if self.enable_institutional:
            tasks.append(self._fetch_institutional_holdings(symbol))
        else:
            tasks.append(asyncio.sleep(0, result=None))

        # Gather all data
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data["google_trends"] = (
            results[0] if not isinstance(results[0], Exception) else None
        )
        data["insider_trading"] = (
            results[1] if not isinstance(results[1], Exception) else None
        )
        data["institutional_holdings"] = (
            results[2] if not isinstance(results[2], Exception) else None
        )

        return data

    async def _fetch_google_trends(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch Google Trends search interest data.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with trends data
        """
        try:
            from pytrends.request import TrendReq

            # Initialize pytrends
            pytrends = TrendReq(hl="en-US", tz=360)

            # Build payload
            await asyncio.to_thread(
                pytrends.build_payload, [symbol], timeframe="today 3-m", geo="US"
            )

            # Get interest over time
            interest_df = await asyncio.to_thread(pytrends.interest_over_time)

            if interest_df.empty or symbol not in interest_df.columns:
                return None

            # Calculate metrics
            interest_values = interest_df[symbol].values

            return {
                "current_interest": (
                    int(interest_values[-1]) if len(interest_values) > 0 else 0
                ),
                "avg_interest": (
                    float(interest_values.mean()) if len(interest_values) > 0 else 0.0
                ),
                "max_interest": (
                    int(interest_values.max()) if len(interest_values) > 0 else 0
                ),
                "trend": self._calculate_trend(interest_values),
                "momentum": self._calculate_momentum(interest_values),
                "data_points": len(interest_values),
            }

        except ImportError:
            logger.warning(
                "pytrends library not installed. Install with: pip install pytrends"
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching Google Trends for {symbol}: {e}")
            return None

    async def _fetch_insider_trading(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch insider trading data from SEC EDGAR.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with insider trading data
        """
        try:
            import aiohttp

            # Use SEC EDGAR API
            # Note: This is a simplified implementation
            # In production, you'd want to use a dedicated SEC data provider

            url = f"https://data.sec.gov/submissions/CIK{symbol}.json"
            headers = {"User-Agent": "ARA-AI-Sentiment-Analyzer/1.0"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return None
                    await response.json()

            # Parse insider transactions (Form 4 filings)
            # This is a placeholder - actual implementation would parse Form 4 data

            return {
                "recent_transactions": [],
                "net_buying": 0,  # Positive = insiders buying, negative = selling
                "transaction_count": 0,
                "total_value": 0,
                "note": "Insider trading data requires additional SEC EDGAR parsing",
            }

        except ImportError:
            logger.warning(
                "aiohttp library not installed. Install with: pip install aiohttp"
            )
            return None
        except Exception as e:
            logger.debug(f"Error fetching insider trading for {symbol}: {e}")
            return None

    async def _fetch_institutional_holdings(
        self, symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch institutional holdings data (13F filings).

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with institutional holdings data
        """
        try:
            # Note: This is a placeholder implementation
            # In production, you'd want to use a dedicated financial data provider
            # that offers parsed 13F data (e.g., WhaleWisdom, Fintel, etc.)

            return {
                "total_institutional_ownership": 0.0,  # Percentage
                "num_institutions": 0,
                "top_holders": [],
                "recent_changes": [],
                "note": "Institutional holdings data requires a dedicated data provider",
            }

        except Exception as e:
            logger.debug(f"Error fetching institutional holdings for {symbol}: {e}")
            return None

    def _calculate_trend(self, values: list) -> str:
        """
        Calculate trend direction from time series.

        Args:
            values: List of values

        Returns:
            Trend direction: 'up', 'down', or 'flat'
        """
        if len(values) < 2:
            return "flat"

        import numpy as np

        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Determine trend
        if slope > 0.5:
            return "up"
        elif slope < -0.5:
            return "down"
        else:
            return "flat"

    def _calculate_momentum(self, values: list) -> float:
        """
        Calculate momentum from time series.

        Args:
            values: List of values

        Returns:
            Momentum score (-1 to +1)
        """
        if len(values) < 2:
            return 0.0

        import numpy as np

        # Compare recent vs older values
        split = len(values) // 2
        recent_avg = np.mean(values[split:])
        older_avg = np.mean(values[:split])

        if older_avg == 0:
            return 0.0

        # Calculate percentage change
        momentum = (recent_avg - older_avg) / older_avg

        # Normalize to -1 to +1
        return float(np.clip(momentum, -1.0, 1.0))

    def engineer_features(self, alternative_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Engineer features from alternative data.

        Args:
            alternative_data: Alternative data dictionary

        Returns:
            Dictionary of engineered features
        """
        features = {}

        # Google Trends features
        if alternative_data.get("google_trends"):
            trends = alternative_data["google_trends"]
            features["trends_current"] = trends.get("current_interest", 0) / 100.0
            features["trends_avg"] = trends.get("avg_interest", 0) / 100.0
            features["trends_momentum"] = trends.get("momentum", 0)

            # Trend direction encoding
            trend_dir = trends.get("trend", "flat")
            features["trends_up"] = 1.0 if trend_dir == "up" else 0.0
            features["trends_down"] = 1.0 if trend_dir == "down" else 0.0

        # Insider trading features
        if alternative_data.get("insider_trading"):
            insider = alternative_data["insider_trading"]
            features["insider_net_buying"] = insider.get("net_buying", 0)
            features["insider_transaction_count"] = insider.get("transaction_count", 0)
            features["insider_total_value"] = insider.get("total_value", 0)

        # Institutional holdings features
        if alternative_data.get("institutional_holdings"):
            institutional = alternative_data["institutional_holdings"]
            features["institutional_ownership"] = (
                institutional.get("total_institutional_ownership", 0) / 100.0
            )
            features["institutional_count"] = institutional.get("num_institutions", 0)

        return features

    async def get_features(self, symbol: str) -> Dict[str, float]:
        """
        Get engineered features from alternative data.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary of features
        """
        alternative_data = await self.get_alternative_data(symbol)
        return self.engineer_features(alternative_data)
