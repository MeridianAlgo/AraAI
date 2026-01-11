"""
Correlation Analyzer for Multi-Asset Analysis

Implements rolling correlation calculation, correlation breakdown detection,
and lead-lag relationship analysis.
"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation analysis between two assets"""

    asset1: str
    asset2: str
    correlation: float
    window_days: int
    start_date: datetime
    end_date: datetime
    is_significant: bool
    p_value: float


@dataclass
class CorrelationBreakdown:
    """Detected correlation breakdown event"""

    asset1: str
    asset2: str
    previous_correlation: float
    current_correlation: float
    change: float
    detection_date: datetime
    window_days: int


@dataclass
class LeadLagRelationship:
    """Lead-lag relationship between two assets"""

    leading_asset: str
    lagging_asset: str
    optimal_lag_days: int
    correlation_at_lag: float
    confidence: float


class CorrelationAnalyzer:
    """
    Analyzes correlations between multiple assets with support for:
    - Rolling correlation calculation (7-365 days)
    - Correlation breakdown detection (change > 0.3)
    - Lead-lag relationship detection
    - Statistical significance testing
    """

    def __init__(
        self,
        min_window: int = 7,
        max_window: int = 365,
        breakdown_threshold: float = 0.3,
        significance_level: float = 0.05,
    ):
        """
        Initialize CorrelationAnalyzer

        Args:
            min_window: Minimum rolling window size in days
            max_window: Maximum rolling window size in days
            breakdown_threshold: Threshold for detecting correlation breakdowns
            significance_level: P-value threshold for statistical significance
        """
        self.min_window = min_window
        self.max_window = max_window
        self.breakdown_threshold = breakdown_threshold
        self.significance_level = significance_level

        logger.info(
            f"Initialized CorrelationAnalyzer with windows {min_window}-{max_window} days"
        )

    def calculate_rolling_correlation(
        self, data1: pd.Series, data2: pd.Series, window: int = 30
    ) -> pd.Series:
        """
        Calculate rolling correlation between two time series

        Args:
            data1: First time series (indexed by date)
            data2: Second time series (indexed by date)
            window: Rolling window size in days

        Returns:
            Series of rolling correlations
        """
        if window < self.min_window or window > self.max_window:
            raise ValueError(
                f"Window must be between {self.min_window} and {self.max_window} days"
            )

        # Align the two series
        aligned_data = pd.DataFrame({"asset1": data1, "asset2": data2}).dropna()

        if len(aligned_data) < window:
            raise ValueError(
                f"Insufficient data: need at least {window} points, got {len(aligned_data)}"
            )

        # Calculate rolling correlation
        rolling_corr = (
            aligned_data["asset1"].rolling(window=window).corr(aligned_data["asset2"])
        )

        return rolling_corr

    def calculate_correlation_matrix(
        self, data: Dict[str, pd.Series], window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets

        Args:
            data: Dictionary mapping asset symbols to price series
            window: Optional rolling window size. If None, uses full period

        Returns:
            Correlation matrix as DataFrame
        """
        # Create DataFrame from all series
        df = pd.DataFrame(data).dropna()

        if len(df) == 0:
            raise ValueError("No overlapping data points found")

        # Calculate correlation matrix
        if window is None:
            corr_matrix = df.corr()
        else:
            # For rolling window, return the most recent correlation
            corr_matrix = df.tail(window).corr()

        return corr_matrix

    def detect_correlation_breakdowns(
        self,
        data1: pd.Series,
        data2: pd.Series,
        asset1_name: str,
        asset2_name: str,
        short_window: int = 30,
        long_window: int = 90,
    ) -> List[CorrelationBreakdown]:
        """
        Detect correlation breakdowns (changes > threshold)

        Args:
            data1: First time series
            data2: Second time series
            asset1_name: Name of first asset
            asset2_name: Name of second asset
            short_window: Short-term correlation window
            long_window: Long-term correlation window

        Returns:
            List of detected correlation breakdowns
        """
        # Calculate short and long-term correlations
        short_corr = self.calculate_rolling_correlation(data1, data2, short_window)
        long_corr = self.calculate_rolling_correlation(data1, data2, long_window)

        # Align the two correlation series
        aligned = pd.DataFrame({"short": short_corr, "long": long_corr}).dropna()

        # Calculate correlation change
        aligned["change"] = aligned["short"] - aligned["long"]

        # Detect breakdowns where change exceeds threshold
        breakdowns = []
        for date, row in aligned.iterrows():
            if abs(row["change"]) > self.breakdown_threshold:
                breakdown = CorrelationBreakdown(
                    asset1=asset1_name,
                    asset2=asset2_name,
                    previous_correlation=row["long"],
                    current_correlation=row["short"],
                    change=row["change"],
                    detection_date=date,
                    window_days=short_window,
                )
                breakdowns.append(breakdown)

        logger.info(
            f"Detected {len(breakdowns)} correlation breakdowns between "
            f"{asset1_name} and {asset2_name}"
        )

        return breakdowns

    def detect_lead_lag_relationship(
        self,
        data1: pd.Series,
        data2: pd.Series,
        asset1_name: str,
        asset2_name: str,
        max_lag: int = 10,
    ) -> Optional[LeadLagRelationship]:
        """
        Detect lead-lag relationships between two assets

        Args:
            data1: First time series
            data2: Second time series
            asset1_name: Name of first asset
            asset2_name: Name of second asset
            max_lag: Maximum lag to test (in days)

        Returns:
            LeadLagRelationship if significant relationship found, None otherwise
        """
        # Align the two series
        aligned_data = pd.DataFrame({"asset1": data1, "asset2": data2}).dropna()

        if len(aligned_data) < max_lag + 30:
            logger.warning(
                f"Insufficient data for lead-lag analysis: need at least "
                f"{max_lag + 30} points"
            )
            return None

        # Test correlations at different lags
        lag_correlations = {}

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr = aligned_data["asset1"].corr(aligned_data["asset2"])
            elif lag > 0:
                # asset1 leads asset2 by 'lag' days
                corr = (
                    aligned_data["asset1"]
                    .iloc[:-lag]
                    .corr(aligned_data["asset2"].iloc[lag:])
                )
            else:
                # asset2 leads asset1 by 'abs(lag)' days
                corr = (
                    aligned_data["asset1"]
                    .iloc[-lag:]
                    .corr(aligned_data["asset2"].iloc[:lag])
                )

            lag_correlations[lag] = corr

        # Find the lag with maximum absolute correlation
        optimal_lag = max(lag_correlations.items(), key=lambda x: abs(x[1]))

        # Determine leading and lagging assets
        if optimal_lag[0] > 0:
            leading_asset = asset1_name
            lagging_asset = asset2_name
            lag_days = optimal_lag[0]
        elif optimal_lag[0] < 0:
            leading_asset = asset2_name
            lagging_asset = asset1_name
            lag_days = abs(optimal_lag[0])
        else:
            # No lead-lag relationship, simultaneous correlation
            return None

        # Calculate confidence based on correlation strength
        confidence = abs(optimal_lag[1])

        # Only return if correlation is significant
        if confidence < 0.5:
            return None

        relationship = LeadLagRelationship(
            leading_asset=leading_asset,
            lagging_asset=lagging_asset,
            optimal_lag_days=lag_days,
            correlation_at_lag=optimal_lag[1],
            confidence=confidence,
        )

        logger.info(
            f"Detected lead-lag: {leading_asset} leads {lagging_asset} "
            f"by {lag_days} days (corr={optimal_lag[1]:.3f})"
        )

        return relationship

    def calculate_correlation_stability(
        self,
        data1: pd.Series,
        data2: pd.Series,
        window: int = 30,
        lookback_periods: int = 12,
    ) -> Dict[str, float]:
        """
        Calculate correlation stability metrics

        Args:
            data1: First time series
            data2: Second time series
            window: Rolling window size
            lookback_periods: Number of periods to analyze

        Returns:
            Dictionary with stability metrics
        """
        rolling_corr = self.calculate_rolling_correlation(data1, data2, window)

        # Get the last N correlation values
        recent_corrs = rolling_corr.dropna().tail(lookback_periods)

        if len(recent_corrs) < 2:
            return {
                "mean_correlation": 0.0,
                "std_correlation": 0.0,
                "stability_score": 0.0,
            }

        mean_corr = recent_corrs.mean()
        std_corr = recent_corrs.std()

        # Stability score: higher when std is low and mean is high
        stability_score = abs(mean_corr) * (1 - min(std_corr, 1.0))

        return {
            "mean_correlation": float(mean_corr),
            "std_correlation": float(std_corr),
            "stability_score": float(stability_score),
            "min_correlation": float(recent_corrs.min()),
            "max_correlation": float(recent_corrs.max()),
        }

    def analyze_correlation_regime(
        self, data1: pd.Series, data2: pd.Series, window: int = 30
    ) -> str:
        """
        Classify the current correlation regime

        Args:
            data1: First time series
            data2: Second time series
            window: Rolling window size

        Returns:
            Correlation regime classification
        """
        rolling_corr = self.calculate_rolling_correlation(data1, data2, window)
        current_corr = rolling_corr.iloc[-1]

        if pd.isna(current_corr):
            return "unknown"

        if current_corr > 0.7:
            return "strong_positive"
        elif current_corr > 0.3:
            return "moderate_positive"
        elif current_corr > -0.3:
            return "weak_or_no_correlation"
        elif current_corr > -0.7:
            return "moderate_negative"
        else:
            return "strong_negative"
