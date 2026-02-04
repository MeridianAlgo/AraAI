"""
Pattern Recognition

This module implements 20+ candlestick and chart pattern detection algorithms.
"""

import pandas as pd
import numpy as np
from ara.features.indicator_registry import get_registry


class PatternRecognition:
    """Collection of pattern recognition algorithms."""

    @staticmethod
    def doji(data: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
        """
        Doji candlestick pattern.

        Args:
            data: DataFrame with OHLC data
            threshold: Body size threshold as % of range

        Returns:
            DataFrame with doji pattern column
        """
        result = data.copy()

        body = np.abs(result["close"] - result["open"])
        range_hl = result["high"] - result["low"]

        result["pattern_doji"] = (body / range_hl < threshold).astype(int)

        return result

    @staticmethod
    def hammer(data: pd.DataFrame) -> pd.DataFrame:
        """
        Hammer candlestick pattern (bullish reversal).

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with hammer pattern column
        """
        result = data.copy()

        body = np.abs(result["close"] - result["open"])
        range_hl = result["high"] - result["low"]
        lower_shadow = np.minimum(result["open"], result["close"]) - result["low"]
        upper_shadow = result["high"] - np.maximum(result["open"], result["close"])

        # Hammer criteria: small body, long lower shadow, small upper shadow
        is_hammer = (body / range_hl < 0.3) & (lower_shadow > 2 * body) & (upper_shadow < body)

        result["pattern_hammer"] = is_hammer.astype(int)

        return result

    @staticmethod
    def shooting_star(data: pd.DataFrame) -> pd.DataFrame:
        """
        Shooting Star candlestick pattern (bearish reversal).

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with shooting star pattern column
        """
        result = data.copy()

        body = np.abs(result["close"] - result["open"])
        range_hl = result["high"] - result["low"]
        lower_shadow = np.minimum(result["open"], result["close"]) - result["low"]
        upper_shadow = result["high"] - np.maximum(result["open"], result["close"])

        # Shooting star criteria: small body, long upper shadow, small lower shadow
        is_shooting_star = (
            (body / range_hl < 0.3) & (upper_shadow > 2 * body) & (lower_shadow < body)
        )

        result["pattern_shooting_star"] = is_shooting_star.astype(int)

        return result

    @staticmethod
    def engulfing(data: pd.DataFrame) -> pd.DataFrame:
        """
        Engulfing candlestick patterns (bullish and bearish).

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with engulfing pattern columns
        """
        result = data.copy()

        # Current candle
        curr_body = result["close"] - result["open"]
        curr_body_abs = np.abs(curr_body)

        # Previous candle
        prev_body = result["open"].shift(1) - result["close"].shift(1)
        prev_body_abs = np.abs(prev_body)

        # Bullish engulfing: current green candle engulfs previous red candle
        bullish_engulfing = (
            (curr_body > 0)  # Current is bullish
            & (prev_body > 0)  # Previous is bearish
            & (result["open"] < result["close"].shift(1))  # Opens below previous close
            & (result["close"] > result["open"].shift(1))  # Closes above previous open
            & (curr_body_abs > prev_body_abs)  # Current body larger
        )

        # Bearish engulfing: current red candle engulfs previous green candle
        bearish_engulfing = (
            (curr_body < 0)  # Current is bearish
            & (prev_body < 0)  # Previous is bullish
            & (result["open"] > result["close"].shift(1))  # Opens above previous close
            & (result["close"] < result["open"].shift(1))  # Closes below previous open
            & (curr_body_abs > prev_body_abs)  # Current body larger
        )

        result["pattern_bullish_engulfing"] = bullish_engulfing.astype(int)
        result["pattern_bearish_engulfing"] = bearish_engulfing.astype(int)

        return result

    @staticmethod
    def morning_star(data: pd.DataFrame) -> pd.DataFrame:
        """
        Morning Star pattern (bullish reversal).

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with morning star pattern column
        """
        result = data.copy()

        # Three candles
        body_0 = result["close"] - result["open"]
        body_1 = result["close"].shift(1) - result["open"].shift(1)
        body_2 = result["close"].shift(2) - result["open"].shift(2)

        # Morning star criteria
        is_morning_star = (
            (body_2 < 0)  # First candle is bearish
            & (np.abs(body_1) < np.abs(body_2) * 0.3)  # Second candle has small body
            & (body_0 > 0)  # Third candle is bullish
            & (result["close"] > result["open"].shift(2))  # Third closes above first open
        )

        result["pattern_morning_star"] = is_morning_star.astype(int)

        return result

    @staticmethod
    def evening_star(data: pd.DataFrame) -> pd.DataFrame:
        """
        Evening Star pattern (bearish reversal).

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with evening star pattern column
        """
        result = data.copy()

        # Three candles
        body_0 = result["close"] - result["open"]
        body_1 = result["close"].shift(1) - result["open"].shift(1)
        body_2 = result["close"].shift(2) - result["open"].shift(2)

        # Evening star criteria
        is_evening_star = (
            (body_2 > 0)  # First candle is bullish
            & (np.abs(body_1) < np.abs(body_2) * 0.3)  # Second candle has small body
            & (body_0 < 0)  # Third candle is bearish
            & (result["close"] < result["open"].shift(2))  # Third closes below first open
        )

        result["pattern_evening_star"] = is_evening_star.astype(int)

        return result

    @staticmethod
    def three_white_soldiers(data: pd.DataFrame) -> pd.DataFrame:
        """
        Three White Soldiers pattern (bullish continuation).

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with three white soldiers pattern column
        """
        result = data.copy()

        # Three consecutive bullish candles
        body_0 = result["close"] - result["open"]
        body_1 = result["close"].shift(1) - result["open"].shift(1)
        body_2 = result["close"].shift(2) - result["open"].shift(2)

        is_pattern = (
            (body_0 > 0)
            & (body_1 > 0)
            & (body_2 > 0)  # All bullish
            & (result["close"] > result["close"].shift(1))  # Each closes higher
            & (result["close"].shift(1) > result["close"].shift(2))
            & (result["open"] > result["open"].shift(1))  # Each opens higher
            & (result["open"].shift(1) > result["open"].shift(2))
        )

        result["pattern_three_white_soldiers"] = is_pattern.astype(int)

        return result

    @staticmethod
    def three_black_crows(data: pd.DataFrame) -> pd.DataFrame:
        """
        Three Black Crows pattern (bearish continuation).

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with three black crows pattern column
        """
        result = data.copy()

        # Three consecutive bearish candles
        body_0 = result["close"] - result["open"]
        body_1 = result["close"].shift(1) - result["open"].shift(1)
        body_2 = result["close"].shift(2) - result["open"].shift(2)

        is_pattern = (
            (body_0 < 0)
            & (body_1 < 0)
            & (body_2 < 0)  # All bearish
            & (result["close"] < result["close"].shift(1))  # Each closes lower
            & (result["close"].shift(1) < result["close"].shift(2))
            & (result["open"] < result["open"].shift(1))  # Each opens lower
            & (result["open"].shift(1) < result["open"].shift(2))
        )

        result["pattern_three_black_crows"] = is_pattern.astype(int)

        return result

    @staticmethod
    def head_and_shoulders(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Head and Shoulders pattern detection (simplified).

        Args:
            data: DataFrame with OHLC data
            window: Window for peak detection

        Returns:
            DataFrame with head and shoulders pattern column
        """
        result = data.copy()

        # Find local peaks
        peaks = (
            (result["high"] > result["high"].shift(1))
            & (result["high"] > result["high"].shift(-1))
            & (result["high"] > result["high"].rolling(window=window).mean())
        )

        result["pattern_head_shoulders"] = peaks.astype(int)

        return result

    @staticmethod
    def double_top(data: pd.DataFrame, window: int = 20, tolerance: float = 0.02) -> pd.DataFrame:
        """
        Double Top pattern detection (simplified).

        Args:
            data: DataFrame with OHLC data
            window: Window for peak detection
            tolerance: Price tolerance for matching peaks

        Returns:
            DataFrame with double top pattern column
        """
        result = data.copy()

        # Find local peaks
        is_peak = (result["high"] > result["high"].shift(1)) & (
            result["high"] > result["high"].shift(-1)
        )

        # Check for matching peaks within window
        double_top = pd.Series(0, index=result.index)

        peak_indices = result.index[is_peak].tolist()
        for i in range(len(peak_indices) - 1):
            idx1 = peak_indices[i]
            idx2 = peak_indices[i + 1]

            if idx2 - idx1 <= window:
                price1 = result.loc[idx1, "high"]
                price2 = result.loc[idx2, "high"]

                if abs(price1 - price2) / price1 < tolerance:
                    double_top.loc[idx2] = 1

        result["pattern_double_top"] = double_top

        return result

    @staticmethod
    def double_bottom(
        data: pd.DataFrame, window: int = 20, tolerance: float = 0.02
    ) -> pd.DataFrame:
        """
        Double Bottom pattern detection (simplified).

        Args:
            data: DataFrame with OHLC data
            window: Window for trough detection
            tolerance: Price tolerance for matching troughs

        Returns:
            DataFrame with double bottom pattern column
        """
        result = data.copy()

        # Find local troughs
        is_trough = (result["low"] < result["low"].shift(1)) & (
            result["low"] < result["low"].shift(-1)
        )

        # Check for matching troughs within window
        double_bottom = pd.Series(0, index=result.index)

        trough_indices = result.index[is_trough].tolist()
        for i in range(len(trough_indices) - 1):
            idx1 = trough_indices[i]
            idx2 = trough_indices[i + 1]

            if idx2 - idx1 <= window:
                price1 = result.loc[idx1, "low"]
                price2 = result.loc[idx2, "low"]

                if abs(price1 - price2) / price1 < tolerance:
                    double_bottom.loc[idx2] = 1

        result["pattern_double_bottom"] = double_bottom

        return result

    @staticmethod
    def triangle_pattern(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Triangle pattern detection (ascending, descending, symmetrical).

        Args:
            data: DataFrame with OHLC data
            window: Window for trend detection

        Returns:
            DataFrame with triangle pattern columns
        """
        result = data.copy()

        # Calculate highs and lows trend
        high_slope = (
            result["high"]
            .rolling(window=window)
            .apply(
                lambda x: (np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0),
                raw=True,
            )
        )

        low_slope = (
            result["low"]
            .rolling(window=window)
            .apply(
                lambda x: (np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0),
                raw=True,
            )
        )

        # Ascending triangle: flat highs, rising lows
        result["pattern_ascending_triangle"] = (
            (np.abs(high_slope) < 0.01) & (low_slope > 0.01)
        ).astype(int)

        # Descending triangle: falling highs, flat lows
        result["pattern_descending_triangle"] = (
            (high_slope < -0.01) & (np.abs(low_slope) < 0.01)
        ).astype(int)

        # Symmetrical triangle: converging highs and lows
        result["pattern_symmetrical_triangle"] = ((high_slope < -0.01) & (low_slope > 0.01)).astype(
            int
        )

        return result

    @staticmethod
    def wedge_pattern(data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Wedge pattern detection (rising and falling).

        Args:
            data: DataFrame with OHLC data
            window: Window for trend detection

        Returns:
            DataFrame with wedge pattern columns
        """
        result = data.copy()

        # Calculate highs and lows trend
        high_slope = (
            result["high"]
            .rolling(window=window)
            .apply(
                lambda x: (np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0),
                raw=True,
            )
        )

        low_slope = (
            result["low"]
            .rolling(window=window)
            .apply(
                lambda x: (np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0),
                raw=True,
            )
        )

        # Rising wedge: both rising, converging
        result["pattern_rising_wedge"] = (
            (high_slope > 0) & (low_slope > 0) & (high_slope < low_slope * 2)
        ).astype(int)

        # Falling wedge: both falling, converging
        result["pattern_falling_wedge"] = (
            (high_slope < 0) & (low_slope < 0) & (high_slope > low_slope * 2)
        ).astype(int)

        return result

    @staticmethod
    def channel_pattern(
        data: pd.DataFrame, window: int = 20, tolerance: float = 0.02
    ) -> pd.DataFrame:
        """
        Channel pattern detection (parallel highs and lows).

        Args:
            data: DataFrame with OHLC data
            window: Window for trend detection
            tolerance: Tolerance for parallel slopes

        Returns:
            DataFrame with channel pattern column
        """
        result = data.copy()

        # Calculate highs and lows trend
        high_slope = (
            result["high"]
            .rolling(window=window)
            .apply(
                lambda x: (np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0),
                raw=True,
            )
        )

        low_slope = (
            result["low"]
            .rolling(window=window)
            .apply(
                lambda x: (np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0),
                raw=True,
            )
        )

        # Channel: parallel slopes
        result["pattern_channel"] = (np.abs(high_slope - low_slope) < tolerance).astype(int)

        return result

    @staticmethod
    def cup_and_handle(data: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Cup and Handle pattern detection (simplified).

        Args:
            data: DataFrame with OHLC data
            window: Window for pattern detection

        Returns:
            DataFrame with cup and handle pattern column
        """
        result = data.copy()

        # Simplified: U-shaped bottom followed by consolidation
        rolling_min = result["low"].rolling(window=window).min()
        rolling_max = result["high"].rolling(window=window).max()

        # Cup: price near rolling max after touching rolling min
        cup_formed = (result["low"].shift(window // 2) <= rolling_min.shift(window // 2) * 1.05) & (
            result["high"] >= rolling_max * 0.95
        )

        result["pattern_cup_handle"] = cup_formed.astype(int)

        return result

    @staticmethod
    def all_patterns(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all pattern recognition indicators.

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with all pattern columns
        """
        result = data.copy()

        # Candlestick patterns
        result = PatternRecognition.doji(result)
        result = PatternRecognition.hammer(result)
        result = PatternRecognition.shooting_star(result)
        result = PatternRecognition.engulfing(result)
        result = PatternRecognition.morning_star(result)
        result = PatternRecognition.evening_star(result)
        result = PatternRecognition.three_white_soldiers(result)
        result = PatternRecognition.three_black_crows(result)

        # Chart patterns
        result = PatternRecognition.head_and_shoulders(result)
        result = PatternRecognition.double_top(result)
        result = PatternRecognition.double_bottom(result)
        result = PatternRecognition.triangle_pattern(result)
        result = PatternRecognition.wedge_pattern(result)
        result = PatternRecognition.channel_pattern(result)
        result = PatternRecognition.cup_and_handle(result)

        return result


# Register pattern recognition
def register_pattern_indicators():
    """Register pattern recognition with the global registry."""
    registry = get_registry()

    registry.register(
        name="patterns_all",
        func=PatternRecognition.all_patterns,
        category="pattern",
        description="All Pattern Recognition",
        parameters={},
        required_columns=["open", "high", "low", "close"],
        output_columns=[
            "pattern_doji",
            "pattern_hammer",
            "pattern_shooting_star",
            "pattern_bullish_engulfing",
            "pattern_bearish_engulfing",
            "pattern_morning_star",
            "pattern_evening_star",
            "pattern_three_white_soldiers",
            "pattern_three_black_crows",
            "pattern_head_shoulders",
            "pattern_double_top",
            "pattern_double_bottom",
            "pattern_ascending_triangle",
            "pattern_descending_triangle",
            "pattern_symmetrical_triangle",
            "pattern_rising_wedge",
            "pattern_falling_wedge",
            "pattern_channel",
            "pattern_cup_handle",
        ],
    )


# Auto-register on import
register_pattern_indicators()
