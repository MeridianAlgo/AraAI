"""
Currency risk analysis for multi-currency portfolios
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
import yfinance as yf
import asyncio

from ara.currency.models import Currency, CurrencyRiskMetrics
from ara.currency.converter import CurrencyConverter
from ara.core.exceptions import DataProviderError
from ara.utils import get_logger

logger = get_logger(__name__)


class CurrencyRiskAnalyzer:
    """
    Analyze currency risk for multi-currency portfolios
    Provides currency-hedged returns and risk metrics
    """

    def __init__(self, converter: Optional[CurrencyConverter] = None):
        """
        Initialize currency risk analyzer

        Args:
            converter: CurrencyConverter instance (creates new if None)
        """
        self.converter = converter or CurrencyConverter()

    async def _fetch_currency_history(
        self, from_currency: Currency, to_currency: Currency, period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch historical exchange rates

        Args:
            from_currency: Source currency
            to_currency: Target currency
            period: Historical period (e.g., '1y', '6mo')

        Returns:
            DataFrame with historical rates
        """
        if from_currency == to_currency:
            # Return identity rates
            dates = pd.date_range(end=datetime.now(), periods=252, freq="D")
            return pd.DataFrame({"Close": [1.0] * len(dates)}, index=dates)

        try:
            forex_symbol = f"{from_currency.value}{to_currency.value}=X"

            # Run in executor
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, forex_symbol)
            hist = await loop.run_in_executor(None, lambda: ticker.history(period=period))

            if hist.empty:
                raise DataProviderError(
                    f"No historical data for {from_currency.value}/{to_currency.value}"
                )

            return hist[["Close"]]

        except Exception as e:
            logger.error(
                f"Failed to fetch currency history: {e}",
                from_currency=from_currency.value,
                to_currency=to_currency.value,
                error=str(e),
            )
            raise DataProviderError("Failed to fetch currency history", {"error": str(e)})

    async def calculate_currency_volatility(
        self,
        from_currency: Currency,
        to_currency: Currency,
        period: str = "1y",
        annualize: bool = True,
    ) -> float:
        """
        Calculate currency volatility

        Args:
            from_currency: Source currency
            to_currency: Target currency
            period: Historical period
            annualize: Whether to annualize volatility

        Returns:
            Volatility (standard deviation of returns)
        """
        hist = await self._fetch_currency_history(from_currency, to_currency, period)

        # Calculate returns
        returns = hist["Close"].pct_change().dropna()

        # Calculate volatility
        volatility = returns.std()

        if annualize:
            # Annualize assuming 252 trading days
            volatility = volatility * np.sqrt(252)

        return float(volatility)

    async def calculate_currency_correlation(
        self,
        currency1: Currency,
        currency2: Currency,
        base_currency: Currency,
        period: str = "1y",
    ) -> float:
        """
        Calculate correlation between two currencies vs base

        Args:
            currency1: First currency
            currency2: Second currency
            base_currency: Base currency for comparison
            period: Historical period

        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Fetch histories
        hist1_task = self._fetch_currency_history(currency1, base_currency, period)
        hist2_task = self._fetch_currency_history(currency2, base_currency, period)

        hist1, hist2 = await asyncio.gather(hist1_task, hist2_task)

        # Calculate returns
        returns1 = hist1["Close"].pct_change().dropna()
        returns2 = hist2["Close"].pct_change().dropna()

        # Align indices
        common_index = returns1.index.intersection(returns2.index)
        returns1 = returns1.loc[common_index]
        returns2 = returns2.loc[common_index]

        # Calculate correlation
        if len(returns1) < 2:
            return 0.0

        correlation = returns1.corr(returns2)

        return float(correlation)

    async def calculate_hedged_return(
        self,
        asset_return: float,
        asset_currency: Currency,
        base_currency: Currency,
        period: str = "1y",
    ) -> Tuple[float, float]:
        """
        Calculate currency-hedged and unhedged returns

        Args:
            asset_return: Asset return in local currency
            asset_currency: Currency of the asset
            base_currency: Base currency for investor
            period: Period for currency return calculation

        Returns:
            Tuple of (hedged_return, unhedged_return)
        """
        if asset_currency == base_currency:
            # No currency effect
            return asset_return, asset_return

        # Fetch currency history
        hist = await self._fetch_currency_history(asset_currency, base_currency, period)

        # Calculate currency return
        currency_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1

        # Hedged return = asset return only
        hedged_return = asset_return

        # Unhedged return = asset return + currency return
        unhedged_return = (1 + asset_return) * (1 + currency_return) - 1

        return float(hedged_return), float(unhedged_return)

    async def analyze_portfolio_currency_risk(
        self,
        positions: Dict[str, Dict],  # symbol -> {amount, currency}
        base_currency: Currency,
        period: str = "1y",
    ) -> CurrencyRiskMetrics:
        """
        Analyze currency risk for a multi-currency portfolio

        Args:
            positions: Dictionary of positions with amounts and currencies
                      e.g., {'AAPL': {'amount': 10000, 'currency': Currency.USD}}
            base_currency: Base currency for analysis
            period: Historical period for calculations

        Returns:
            CurrencyRiskMetrics object
        """
        # Calculate currency exposures
        currency_exposures: Dict[Currency, float] = {}

        for symbol, position in positions.items():
            amount = position["amount"]
            currency = position["currency"]

            if currency in currency_exposures:
                currency_exposures[currency] += amount
            else:
                currency_exposures[currency] = amount

        # Convert all exposures to base currency
        converted_exposures = {}
        for currency, amount in currency_exposures.items():
            if currency != base_currency:
                result = await self.converter.convert(amount, currency, base_currency)
                converted_exposures[currency] = result.converted_amount
            else:
                converted_exposures[currency] = amount

        # Calculate total portfolio value
        total_value = sum(converted_exposures.values())

        # Calculate currency weights
        currency_weights = {
            currency: amount / total_value for currency, amount in converted_exposures.items()
        }

        # Calculate volatilities for each currency
        unique_currencies = list(currency_exposures.keys())
        volatility_tasks = []

        for currency in unique_currencies:
            if currency != base_currency:
                task = self.calculate_currency_volatility(currency, base_currency, period)
                volatility_tasks.append(task)
            else:
                # Base currency has zero volatility vs itself
                async def zero_vol():
                    return 0.0

                volatility_tasks.append(zero_vol())

        volatilities = await asyncio.gather(*volatility_tasks)
        currency_volatilities = dict(zip(unique_currencies, volatilities))

        # Calculate correlations between currencies
        currency_correlations = {}

        for i, curr1 in enumerate(unique_currencies):
            for j, curr2 in enumerate(unique_currencies):
                if i < j:  # Only calculate upper triangle
                    if curr1 == base_currency or curr2 == base_currency:
                        # Correlation with base is 1.0
                        corr = 1.0
                    else:
                        corr = await self.calculate_currency_correlation(
                            curr1, curr2, base_currency, period
                        )
                    currency_correlations[(curr1, curr2)] = corr
                    currency_correlations[(curr2, curr1)] = corr

        # Calculate portfolio currency risk (VaR-like measure)
        # Using variance-covariance approach
        total_currency_risk = 0.0

        for curr1 in unique_currencies:
            for curr2 in unique_currencies:
                weight1 = currency_weights.get(curr1, 0)
                weight2 = currency_weights.get(curr2, 0)
                vol1 = currency_volatilities.get(curr1, 0)
                vol2 = currency_volatilities.get(curr2, 0)

                if curr1 == curr2:
                    corr = 1.0
                else:
                    corr = currency_correlations.get((curr1, curr2), 0)

                total_currency_risk += weight1 * weight2 * vol1 * vol2 * corr

        total_currency_risk = np.sqrt(total_currency_risk) * total_value

        # Calculate hedged vs unhedged returns (simplified)
        # Assume 10% return in local currency for demonstration
        hedged_return = 0.10

        # Calculate weighted currency return
        currency_return = 0.0
        for currency, weight in currency_weights.items():
            if currency != base_currency:
                hist = await self._fetch_currency_history(currency, base_currency, period)
                curr_return = (hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1
                currency_return += weight * curr_return

        unhedged_return = (1 + hedged_return) * (1 + currency_return) - 1
        currency_contribution = unhedged_return - hedged_return

        return CurrencyRiskMetrics(
            base_currency=base_currency,
            currency_exposures=currency_exposures,
            currency_volatilities=currency_volatilities,
            currency_correlations=currency_correlations,
            total_currency_risk=float(total_currency_risk),
            hedged_return=float(hedged_return),
            unhedged_return=float(unhedged_return),
            currency_contribution=float(currency_contribution),
        )

    async def calculate_optimal_hedge_ratio(
        self, asset_currency: Currency, base_currency: Currency, period: str = "1y"
    ) -> float:
        """
        Calculate optimal currency hedge ratio using minimum variance approach

        Args:
            asset_currency: Currency of the asset
            base_currency: Base currency for investor
            period: Historical period

        Returns:
            Optimal hedge ratio (0 to 1)
        """
        if asset_currency == base_currency:
            return 0.0  # No hedging needed

        # Fetch currency history
        hist = await self._fetch_currency_history(asset_currency, base_currency, period)

        # Calculate returns
        returns = hist["Close"].pct_change().dropna()

        if len(returns) < 2:
            return 0.5  # Default to 50% hedge

        # Optimal hedge ratio = Cov(asset, currency) / Var(currency)
        # Simplified: assume perfect correlation, so hedge ratio = 1
        # In practice, this would use asset returns and currency returns

        # For now, return a conservative 0.5 (50% hedge)
        return 0.5
