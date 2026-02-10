"""
Pairs Trading Opportunity Analyzer

Identifies pairs trading opportunities based on correlation analysis,
cointegration testing, and spread analysis.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PairsTradingOpportunity:
    """Identified pairs trading opportunity"""

    asset1: str
    asset2: str
    correlation: float
    cointegration_score: float
    current_spread: float
    mean_spread: float
    std_spread: float
    z_score: float
    signal: str  # 'long_spread', 'short_spread', 'neutral'
    confidence: float
    entry_threshold: float
    exit_threshold: float


@dataclass
class SpreadAnalysis:
    """Analysis of the spread between two assets"""

    spread: pd.Series
    mean: float
    std: float
    current_value: float
    z_score: float
    half_life: Optional[float]


class PairsTradingAnalyzer:
    """
    Analyzes pairs trading opportunities based on:
    - High correlation (> 0.8)
    - Cointegration testing
    - Spread mean reversion
    - Z-score signals
    """

    def __init__(
        self,
        correlation_threshold: float = 0.8,
        entry_z_score: float = 2.0,
        exit_z_score: float = 0.5,
        lookback_days: int = 60,
    ):
        """
        Initialize PairsTradingAnalyzer

        Args:
            correlation_threshold: Minimum correlation for pairs consideration
            entry_z_score: Z-score threshold for trade entry
            exit_z_score: Z-score threshold for trade exit
            lookback_days: Days of historical data for analysis
        """
        self.correlation_threshold = correlation_threshold
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        self.lookback_days = lookback_days

        logger.info(
            f"Initialized PairsTradingAnalyzer with correlation threshold {correlation_threshold}"
        )

    def identify_pairs_opportunities(
        self, data: Dict[str, pd.Series], min_correlation: Optional[float] = None
    ) -> List[PairsTradingOpportunity]:
        """
        Identify all pairs trading opportunities from a set of assets

        Args:
            data: Dictionary mapping asset symbols to price series
            min_correlation: Override default correlation threshold

        Returns:
            List of identified pairs trading opportunities
        """
        if min_correlation is None:
            min_correlation = self.correlation_threshold

        opportunities = []
        assets = list(data.keys())

        # Test all pairs
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset1 = assets[i]
                asset2 = assets[j]

                try:
                    opportunity = self.analyze_pair(data[asset1], data[asset2], asset1, asset2)

                    if opportunity and opportunity.correlation >= min_correlation:
                        opportunities.append(opportunity)

                except Exception as e:
                    logger.warning(f"Failed to analyze pair {asset1}-{asset2}: {e}")
                    continue

        # Sort by confidence
        opportunities.sort(key=lambda x: x.confidence, reverse=True)

        logger.info(f"Identified {len(opportunities)} pairs trading opportunities")

        return opportunities

    def analyze_pair(
        self, data1: pd.Series, data2: pd.Series, asset1_name: str, asset2_name: str
    ) -> Optional[PairsTradingOpportunity]:
        """
        Analyze a specific pair for trading opportunity

        Args:
            data1: First asset price series
            data2: Second asset price series
            asset1_name: Name of first asset
            asset2_name: Name of second asset

        Returns:
            PairsTradingOpportunity if suitable, None otherwise
        """
        # Align data
        aligned_data = pd.DataFrame({"asset1": data1, "asset2": data2}).dropna()

        if len(aligned_data) < self.lookback_days:
            return None

        # Use recent data for analysis
        recent_data = aligned_data.tail(self.lookback_days)

        # Calculate correlation
        correlation = recent_data["asset1"].corr(recent_data["asset2"])

        if abs(correlation) < self.correlation_threshold:
            return None

        # Test for cointegration
        cointegration_score = self._test_cointegration(recent_data["asset1"], recent_data["asset2"])

        # Calculate spread
        spread_analysis = self._analyze_spread(recent_data["asset1"], recent_data["asset2"])

        # Determine trading signal
        signal = self._generate_signal(spread_analysis.z_score)

        # Calculate confidence
        confidence = self._calculate_confidence(correlation, cointegration_score, spread_analysis)

        opportunity = PairsTradingOpportunity(
            asset1=asset1_name,
            asset2=asset2_name,
            correlation=correlation,
            cointegration_score=cointegration_score,
            current_spread=spread_analysis.current_value,
            mean_spread=spread_analysis.mean,
            std_spread=spread_analysis.std,
            z_score=spread_analysis.z_score,
            signal=signal,
            confidence=confidence,
            entry_threshold=self.entry_z_score,
            exit_threshold=self.exit_z_score,
        )

        return opportunity

    def _test_cointegration(self, series1: pd.Series, series2: pd.Series) -> float:
        """
        Test for cointegration between two series

        Uses Engle-Granger two-step method approximation

        Args:
            series1: First time series
            series2: Second time series

        Returns:
            Cointegration score (0-1, higher is better)
        """
        # Simple linear regression to find hedge ratio
        X = series1.values.reshape(-1, 1)
        y = series2.values

        # Calculate hedge ratio using least squares
        hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]

        # Calculate spread
        spread = series2 - hedge_ratio * series1

        # Test if spread is mean-reverting using Hurst exponent approximation
        # A Hurst exponent < 0.5 indicates mean reversion
        hurst = self._calculate_hurst_exponent(spread)

        # Convert to score (0-1, where 1 is best)
        if hurst < 0.5:
            score = 1.0 - (hurst / 0.5)
        else:
            score = 0.0

        return score

    def _calculate_hurst_exponent(self, series: pd.Series) -> float:
        """
        Calculate Hurst exponent to test for mean reversion

        Args:
            series: Time series

        Returns:
            Hurst exponent (< 0.5 indicates mean reversion)
        """
        lags = range(2, min(20, len(series) // 2))
        tau = []

        for lag in lags:
            # Calculate standard deviation of differences
            std = np.std(np.subtract(series[lag:].values, series[:-lag].values))
            tau.append(std)

        # Linear regression on log-log plot
        if len(tau) < 2:
            return 0.5  # Neutral value

        log_lags = np.log(list(lags))
        log_tau = np.log(tau)

        # Fit line
        poly = np.polyfit(log_lags, log_tau, 1)
        hurst = poly[0]

        return hurst

    def _analyze_spread(self, series1: pd.Series, series2: pd.Series) -> SpreadAnalysis:
        """
        Analyze the spread between two series

        Args:
            series1: First time series
            series2: Second time series

        Returns:
            SpreadAnalysis object
        """
        # Normalize series to same scale
        norm1 = (series1 - series1.mean()) / series1.std()
        norm2 = (series2 - series2.mean()) / series2.std()

        # Calculate spread
        spread = norm1 - norm2

        # Calculate statistics
        mean_spread = spread.mean()
        std_spread = spread.std()
        current_spread = spread.iloc[-1]

        # Calculate z-score
        z_score = (current_spread - mean_spread) / std_spread if std_spread > 0 else 0

        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)

        return SpreadAnalysis(
            spread=spread,
            mean=mean_spread,
            std=std_spread,
            current_value=current_spread,
            z_score=z_score,
            half_life=half_life,
        )

    def _calculate_half_life(self, spread: pd.Series) -> Optional[float]:
        """
        Calculate half-life of mean reversion

        Args:
            spread: Spread time series

        Returns:
            Half-life in days, or None if not mean-reverting
        """
        # Lag the spread
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align
        aligned = pd.DataFrame({"lag": spread_lag, "diff": spread_diff}).dropna()

        if len(aligned) < 10:
            return None

        # Regression: spread_diff = alpha + beta * spread_lag
        X = aligned["lag"].values.reshape(-1, 1)
        y = aligned["diff"].values

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0][0]

            if beta >= 0:
                return None  # Not mean-reverting

            half_life = -np.log(2) / beta

            # Sanity check
            if half_life < 0 or half_life > 365:
                return None

            return half_life

        except Exception:
            return None

    def _generate_signal(self, z_score: float) -> str:
        """
        Generate trading signal based on z-score

        Args:
            z_score: Current z-score of spread

        Returns:
            Trading signal: 'long_spread', 'short_spread', or 'neutral'
        """
        if z_score > self.entry_z_score:
            return "short_spread"  # Spread too high, short it
        elif z_score < -self.entry_z_score:
            return "long_spread"  # Spread too low, long it
        else:
            return "neutral"

    def _calculate_confidence(
        self,
        correlation: float,
        cointegration_score: float,
        spread_analysis: SpreadAnalysis,
    ) -> float:
        """
        Calculate confidence in the pairs trading opportunity

        Args:
            correlation: Correlation coefficient
            cointegration_score: Cointegration test score
            spread_analysis: Spread analysis results

        Returns:
            Confidence score (0-1)
        """
        # Weight different factors
        corr_weight = 0.3
        coint_weight = 0.4
        zscore_weight = 0.3

        # Correlation component (higher is better)
        corr_component = abs(correlation) * corr_weight

        # Cointegration component
        coint_component = cointegration_score * coint_weight

        # Z-score component (higher absolute value is better for entry)
        zscore_component = min(abs(spread_analysis.z_score) / 3.0, 1.0) * zscore_weight

        confidence = corr_component + coint_component + zscore_component

        # Bonus for mean-reverting spread
        if spread_analysis.half_life and spread_analysis.half_life < 30:
            confidence *= 1.1

        return min(confidence, 1.0)

    def calculate_position_sizes(
        self,
        opportunity: PairsTradingOpportunity,
        capital: float,
        risk_per_trade: float = 0.02,
    ) -> Tuple[float, float]:
        """
        Calculate position sizes for a pairs trade

        Args:
            opportunity: Pairs trading opportunity
            capital: Total capital available
            risk_per_trade: Fraction of capital to risk per trade

        Returns:
            Tuple of (position_size_asset1, position_size_asset2)
        """
        # Calculate capital to allocate
        trade_capital = capital * risk_per_trade

        # Split equally between the two legs
        position_size = trade_capital / 2

        return (position_size, position_size)
