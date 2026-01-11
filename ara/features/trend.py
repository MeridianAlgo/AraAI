"""
Trend Indicators

This module implements 20+ trend-following technical indicators using vectorized NumPy operations.
"""

import pandas as pd
import numpy as np
from ara.features.indicator_registry import get_registry


class TrendIndicators:
    """Collection of trend-following indicators."""

    @staticmethod
    def sma(
        data: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.DataFrame:
        """
        Simple Moving Average (SMA).

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate SMA on

        Returns:
            DataFrame with SMA column added
        """
        result = data.copy()
        result[f"sma_{period}"] = result[column].rolling(window=period).mean()
        return result

    @staticmethod
    def ema(
        data: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.DataFrame:
        """
        Exponential Moving Average (EMA).

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate EMA on

        Returns:
            DataFrame with EMA column added
        """
        result = data.copy()
        result[f"ema_{period}"] = result[column].ewm(span=period, adjust=False).mean()
        return result

    @staticmethod
    def wma(
        data: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.DataFrame:
        """
        Weighted Moving Average (WMA).

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate WMA on

        Returns:
            DataFrame with WMA column added
        """
        result = data.copy()
        weights = np.arange(1, period + 1)

        def weighted_mean(x):
            if len(x) < period:
                return np.nan
            return np.dot(x[-period:], weights) / weights.sum()

        result[f"wma_{period}"] = (
            result[column].rolling(window=period).apply(weighted_mean, raw=True)
        )
        return result

    @staticmethod
    def dema(
        data: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.DataFrame:
        """
        Double Exponential Moving Average (DEMA).

        DEMA = 2 * EMA - EMA(EMA)

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate DEMA on

        Returns:
            DataFrame with DEMA column added
        """
        result = data.copy()
        ema1 = result[column].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        result[f"dema_{period}"] = 2 * ema1 - ema2
        return result

    @staticmethod
    def tema(
        data: pd.DataFrame, period: int = 20, column: str = "close"
    ) -> pd.DataFrame:
        """
        Triple Exponential Moving Average (TEMA).

        TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate TEMA on

        Returns:
            DataFrame with TEMA column added
        """
        result = data.copy()
        ema1 = result[column].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        result[f"tema_{period}"] = 3 * ema1 - 3 * ema2 + ema3
        return result

    @staticmethod
    def macd(
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence (MACD).

        Args:
            data: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to calculate MACD on

        Returns:
            DataFrame with MACD, signal, and histogram columns
        """
        result = data.copy()

        # Calculate MACD line
        ema_fast = result[column].ewm(span=fast, adjust=False).mean()
        ema_slow = result[column].ewm(span=slow, adjust=False).mean()
        result["macd"] = ema_fast - ema_slow

        # Calculate signal line
        result["macd_signal"] = result["macd"].ewm(span=signal, adjust=False).mean()

        # Calculate histogram
        result["macd_histogram"] = result["macd"] - result["macd_signal"]

        return result

    @staticmethod
    def adx(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average Directional Index (ADX).

        Measures trend strength (0-100).

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with ADX, +DI, and -DI columns
        """
        result = data.copy()

        # Calculate True Range
        high_low = result["high"] - result["low"]
        high_close = np.abs(result["high"] - result["close"].shift())
        low_close = np.abs(result["low"] - result["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate Directional Movement
        up_move = result["high"] - result["high"].shift()
        down_move = result["low"].shift() - result["low"]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth with Wilder's method
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = (
            100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
        )
        minus_di = (
            100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
        )

        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        result["adx"] = adx
        result["plus_di"] = plus_di
        result["minus_di"] = minus_di

        return result

    @staticmethod
    def parabolic_sar(
        data: pd.DataFrame,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.2,
    ) -> pd.DataFrame:
        """
        Parabolic SAR (Stop and Reverse).

        Args:
            data: DataFrame with OHLC data
            af_start: Starting acceleration factor
            af_increment: AF increment
            af_max: Maximum AF

        Returns:
            DataFrame with SAR column
        """
        result = data.copy()

        high = result["high"].values
        low = result["low"].values
        close = result["close"].values

        sar = np.zeros(len(result))
        ep = np.zeros(len(result))
        af = np.zeros(len(result))
        trend = np.zeros(len(result))

        # Initialize
        sar[0] = low[0]
        ep[0] = high[0]
        af[0] = af_start
        trend[0] = 1  # 1 for uptrend, -1 for downtrend

        for i in range(1, len(result)):
            # Calculate SAR
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])

            if trend[i - 1] == 1:  # Uptrend
                sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])

                if low[i] < sar[i]:  # Reverse to downtrend
                    trend[i] = -1
                    sar[i] = ep[i - 1]
                    ep[i] = low[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
                    if high[i] > ep[i - 1]:
                        ep[i] = high[i]
                        af[i] = min(af[i - 1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]
            else:  # Downtrend
                sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])

                if high[i] > sar[i]:  # Reverse to uptrend
                    trend[i] = 1
                    sar[i] = ep[i - 1]
                    ep[i] = high[i]
                    af[i] = af_start
                else:
                    trend[i] = -1
                    if low[i] < ep[i - 1]:
                        ep[i] = low[i]
                        af[i] = min(af[i - 1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i - 1]
                        af[i] = af[i - 1]

        result["psar"] = sar
        result["psar_trend"] = trend

        return result

    @staticmethod
    def supertrend(
        data: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> pd.DataFrame:
        """
        Supertrend Indicator.

        Args:
            data: DataFrame with OHLC data
            period: ATR period
            multiplier: ATR multiplier

        Returns:
            DataFrame with Supertrend columns
        """
        result = data.copy()

        # Calculate ATR
        high_low = result["high"] - result["low"]
        high_close = np.abs(result["high"] - result["close"].shift())
        low_close = np.abs(result["low"] - result["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate basic bands
        hl_avg = (result["high"] + result["low"]) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)

        # Calculate final bands
        final_upper = upper_band.copy()
        final_lower = lower_band.copy()

        for i in range(period, len(result)):
            if (
                upper_band.iloc[i] < final_upper.iloc[i - 1]
                or result["close"].iloc[i - 1] > final_upper.iloc[i - 1]
            ):
                final_upper.iloc[i] = upper_band.iloc[i]
            else:
                final_upper.iloc[i] = final_upper.iloc[i - 1]

            if (
                lower_band.iloc[i] > final_lower.iloc[i - 1]
                or result["close"].iloc[i - 1] < final_lower.iloc[i - 1]
            ):
                final_lower.iloc[i] = lower_band.iloc[i]
            else:
                final_lower.iloc[i] = final_lower.iloc[i - 1]

        # Determine trend
        supertrend = pd.Series(index=result.index, dtype=float)
        trend = pd.Series(index=result.index, dtype=int)

        for i in range(period, len(result)):
            if i == period:
                supertrend.iloc[i] = final_upper.iloc[i]
                trend.iloc[i] = -1
            else:
                if (
                    supertrend.iloc[i - 1] == final_upper.iloc[i - 1]
                    and result["close"].iloc[i] <= final_upper.iloc[i]
                ):
                    supertrend.iloc[i] = final_upper.iloc[i]
                    trend.iloc[i] = -1
                elif (
                    supertrend.iloc[i - 1] == final_upper.iloc[i - 1]
                    and result["close"].iloc[i] > final_upper.iloc[i]
                ):
                    supertrend.iloc[i] = final_lower.iloc[i]
                    trend.iloc[i] = 1
                elif (
                    supertrend.iloc[i - 1] == final_lower.iloc[i - 1]
                    and result["close"].iloc[i] >= final_lower.iloc[i]
                ):
                    supertrend.iloc[i] = final_lower.iloc[i]
                    trend.iloc[i] = 1
                else:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    trend.iloc[i] = -1

        result["supertrend"] = supertrend
        result["supertrend_trend"] = trend

        return result

    @staticmethod
    def ichimoku(
        data: pd.DataFrame,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
    ) -> pd.DataFrame:
        """
        Ichimoku Cloud (Ichimoku Kinko Hyo).

        Args:
            data: DataFrame with OHLC data
            tenkan_period: Tenkan-sen (Conversion Line) period
            kijun_period: Kijun-sen (Base Line) period
            senkou_b_period: Senkou Span B period
            displacement: Displacement for Senkou spans

        Returns:
            DataFrame with Ichimoku components
        """
        result = data.copy()

        # Tenkan-sen (Conversion Line)
        high_tenkan = result["high"].rolling(window=tenkan_period).max()
        low_tenkan = result["low"].rolling(window=tenkan_period).min()
        result["ichimoku_tenkan"] = (high_tenkan + low_tenkan) / 2

        # Kijun-sen (Base Line)
        high_kijun = result["high"].rolling(window=kijun_period).max()
        low_kijun = result["low"].rolling(window=kijun_period).min()
        result["ichimoku_kijun"] = (high_kijun + low_kijun) / 2

        # Senkou Span A (Leading Span A)
        result["ichimoku_senkou_a"] = (
            (result["ichimoku_tenkan"] + result["ichimoku_kijun"]) / 2
        ).shift(displacement)

        # Senkou Span B (Leading Span B)
        high_senkou = result["high"].rolling(window=senkou_b_period).max()
        low_senkou = result["low"].rolling(window=senkou_b_period).min()
        result["ichimoku_senkou_b"] = ((high_senkou + low_senkou) / 2).shift(
            displacement
        )

        # Chikou Span (Lagging Span)
        result["ichimoku_chikou"] = result["close"].shift(-displacement)

        return result

    @staticmethod
    def aroon(data: pd.DataFrame, period: int = 25) -> pd.DataFrame:
        """
        Aroon Indicator.

        Measures time since highest high and lowest low.

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with Aroon Up, Down, and Oscillator
        """
        result = data.copy()

        # Aroon Up: ((period - periods since highest high) / period) * 100
        aroon_up = (
            result["high"]
            .rolling(window=period + 1)
            .apply(
                lambda x: (period - (len(x) - 1 - np.argmax(x))) / period * 100,
                raw=True,
            )
        )

        # Aroon Down: ((period - periods since lowest low) / period) * 100
        aroon_down = (
            result["low"]
            .rolling(window=period + 1)
            .apply(
                lambda x: (period - (len(x) - 1 - np.argmin(x))) / period * 100,
                raw=True,
            )
        )

        result["aroon_up"] = aroon_up
        result["aroon_down"] = aroon_down
        result["aroon_oscillator"] = aroon_up - aroon_down

        return result

    @staticmethod
    def vortex(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Vortex Indicator (VI).

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with VI+ and VI- columns
        """
        result = data.copy()

        # Calculate Vortex Movement
        vm_plus = np.abs(result["high"] - result["low"].shift())
        vm_minus = np.abs(result["low"] - result["high"].shift())

        # Calculate True Range
        high_low = result["high"] - result["low"]
        high_close = np.abs(result["high"] - result["close"].shift())
        low_close = np.abs(result["low"] - result["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Sum over period
        vm_plus_sum = vm_plus.rolling(window=period).sum()
        vm_minus_sum = vm_minus.rolling(window=period).sum()
        tr_sum = tr.rolling(window=period).sum()

        # Calculate VI
        result["vortex_plus"] = vm_plus_sum / tr_sum
        result["vortex_minus"] = vm_minus_sum / tr_sum

        return result

    @staticmethod
    def kst(
        data: pd.DataFrame,
        roc1: int = 10,
        roc2: int = 15,
        roc3: int = 20,
        roc4: int = 30,
        sma1: int = 10,
        sma2: int = 10,
        sma3: int = 10,
        sma4: int = 15,
        signal: int = 9,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Know Sure Thing (KST) Oscillator.

        Args:
            data: DataFrame with price data
            roc1-4: ROC periods
            sma1-4: SMA periods for each ROC
            signal: Signal line period
            column: Column to calculate on

        Returns:
            DataFrame with KST and signal line
        """
        result = data.copy()

        # Calculate ROCs
        roc_1 = (
            (result[column] - result[column].shift(roc1)) / result[column].shift(roc1)
        ) * 100
        roc_2 = (
            (result[column] - result[column].shift(roc2)) / result[column].shift(roc2)
        ) * 100
        roc_3 = (
            (result[column] - result[column].shift(roc3)) / result[column].shift(roc3)
        ) * 100
        roc_4 = (
            (result[column] - result[column].shift(roc4)) / result[column].shift(roc4)
        ) * 100

        # Smooth with SMAs
        roc_1_sma = roc_1.rolling(window=sma1).mean()
        roc_2_sma = roc_2.rolling(window=sma2).mean()
        roc_3_sma = roc_3.rolling(window=sma3).mean()
        roc_4_sma = roc_4.rolling(window=sma4).mean()

        # Calculate KST
        result["kst"] = (
            (roc_1_sma * 1) + (roc_2_sma * 2) + (roc_3_sma * 3) + (roc_4_sma * 4)
        )
        result["kst_signal"] = result["kst"].rolling(window=signal).mean()

        return result

    @staticmethod
    def trix(
        data: pd.DataFrame, period: int = 15, signal: int = 9, column: str = "close"
    ) -> pd.DataFrame:
        """
        TRIX (Triple Exponential Average).

        Args:
            data: DataFrame with price data
            period: EMA period
            signal: Signal line period
            column: Column to calculate on

        Returns:
            DataFrame with TRIX and signal line
        """
        result = data.copy()

        # Triple EMA
        ema1 = result[column].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()

        # TRIX = 1-period percent change of triple EMA
        result["trix"] = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100
        result["trix_signal"] = result["trix"].rolling(window=signal).mean()

        return result

    @staticmethod
    def mass_index(
        data: pd.DataFrame, period: int = 9, sum_period: int = 25
    ) -> pd.DataFrame:
        """
        Mass Index.

        Identifies trend reversals based on range expansion.

        Args:
            data: DataFrame with OHLC data
            period: EMA period
            sum_period: Summation period

        Returns:
            DataFrame with Mass Index column
        """
        result = data.copy()

        # Calculate range
        hl_range = result["high"] - result["low"]

        # Single and double EMA
        ema1 = hl_range.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()

        # Mass Index
        mass = ema1 / ema2
        result["mass_index"] = mass.rolling(window=sum_period).sum()

        return result

    @staticmethod
    def qstick(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Qstick Indicator.

        Measures the average difference between open and close.

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with Qstick column
        """
        result = data.copy()
        result["qstick"] = (
            (result["close"] - result["open"]).rolling(window=period).mean()
        )
        return result


# Register all trend indicators
def register_trend_indicators():
    """Register all trend indicators with the global registry."""
    registry = get_registry()

    # SMA
    registry.register(
        name="sma",
        func=TrendIndicators.sma,
        category="trend",
        description="Simple Moving Average",
        parameters={"period": 20, "column": "close"},
        required_columns=["close"],
        output_columns=["sma_{period}"],
    )

    # EMA
    registry.register(
        name="ema",
        func=TrendIndicators.ema,
        category="trend",
        description="Exponential Moving Average",
        parameters={"period": 20, "column": "close"},
        required_columns=["close"],
        output_columns=["ema_{period}"],
    )

    # WMA
    registry.register(
        name="wma",
        func=TrendIndicators.wma,
        category="trend",
        description="Weighted Moving Average",
        parameters={"period": 20, "column": "close"},
        required_columns=["close"],
        output_columns=["wma_{period}"],
    )

    # DEMA
    registry.register(
        name="dema",
        func=TrendIndicators.dema,
        category="trend",
        description="Double Exponential Moving Average",
        parameters={"period": 20, "column": "close"},
        required_columns=["close"],
        output_columns=["dema_{period}"],
    )

    # TEMA
    registry.register(
        name="tema",
        func=TrendIndicators.tema,
        category="trend",
        description="Triple Exponential Moving Average",
        parameters={"period": 20, "column": "close"},
        required_columns=["close"],
        output_columns=["tema_{period}"],
    )

    # MACD
    registry.register(
        name="macd",
        func=TrendIndicators.macd,
        category="trend",
        description="Moving Average Convergence Divergence",
        parameters={"fast": 12, "slow": 26, "signal": 9, "column": "close"},
        required_columns=["close"],
        output_columns=["macd", "macd_signal", "macd_histogram"],
    )

    # ADX
    registry.register(
        name="adx",
        func=TrendIndicators.adx,
        category="trend",
        description="Average Directional Index",
        parameters={"period": 14},
        required_columns=["high", "low", "close"],
        output_columns=["adx", "plus_di", "minus_di"],
    )

    # Parabolic SAR
    registry.register(
        name="psar",
        func=TrendIndicators.parabolic_sar,
        category="trend",
        description="Parabolic Stop and Reverse",
        parameters={"af_start": 0.02, "af_increment": 0.02, "af_max": 0.2},
        required_columns=["high", "low", "close"],
        output_columns=["psar", "psar_trend"],
    )

    # Supertrend
    registry.register(
        name="supertrend",
        func=TrendIndicators.supertrend,
        category="trend",
        description="Supertrend Indicator",
        parameters={"period": 10, "multiplier": 3.0},
        required_columns=["high", "low", "close"],
        output_columns=["supertrend", "supertrend_trend"],
    )

    # Ichimoku
    registry.register(
        name="ichimoku",
        func=TrendIndicators.ichimoku,
        category="trend",
        description="Ichimoku Cloud",
        parameters={
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "displacement": 26,
        },
        required_columns=["high", "low", "close"],
        output_columns=[
            "ichimoku_tenkan",
            "ichimoku_kijun",
            "ichimoku_senkou_a",
            "ichimoku_senkou_b",
            "ichimoku_chikou",
        ],
    )

    # Aroon
    registry.register(
        name="aroon",
        func=TrendIndicators.aroon,
        category="trend",
        description="Aroon Indicator",
        parameters={"period": 25},
        required_columns=["high", "low"],
        output_columns=["aroon_up", "aroon_down", "aroon_oscillator"],
    )

    # Vortex
    registry.register(
        name="vortex",
        func=TrendIndicators.vortex,
        category="trend",
        description="Vortex Indicator",
        parameters={"period": 14},
        required_columns=["high", "low", "close"],
        output_columns=["vortex_plus", "vortex_minus"],
    )

    # KST
    registry.register(
        name="kst",
        func=TrendIndicators.kst,
        category="trend",
        description="Know Sure Thing Oscillator",
        parameters={
            "roc1": 10,
            "roc2": 15,
            "roc3": 20,
            "roc4": 30,
            "sma1": 10,
            "sma2": 10,
            "sma3": 10,
            "sma4": 15,
            "signal": 9,
            "column": "close",
        },
        required_columns=["close"],
        output_columns=["kst", "kst_signal"],
    )

    # TRIX
    registry.register(
        name="trix",
        func=TrendIndicators.trix,
        category="trend",
        description="Triple Exponential Average",
        parameters={"period": 15, "signal": 9, "column": "close"},
        required_columns=["close"],
        output_columns=["trix", "trix_signal"],
    )

    # Mass Index
    registry.register(
        name="mass_index",
        func=TrendIndicators.mass_index,
        category="trend",
        description="Mass Index",
        parameters={"period": 9, "sum_period": 25},
        required_columns=["high", "low"],
        output_columns=["mass_index"],
    )

    # Qstick
    registry.register(
        name="qstick",
        func=TrendIndicators.qstick,
        category="trend",
        description="Qstick Indicator",
        parameters={"period": 14},
        required_columns=["open", "close"],
        output_columns=["qstick"],
    )


# Auto-register on import
register_trend_indicators()
