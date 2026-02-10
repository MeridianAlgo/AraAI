"""
Momentum Indicators

This module implements 20+ momentum oscillators using vectorized NumPy operations.
"""

import numpy as np
import pandas as pd

from ara.features.indicator_registry import get_registry


class MomentumIndicators:
    """Collection of momentum oscillators."""

    @staticmethod
    def rsi(data: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
        """
        Relative Strength Index (RSI).

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate RSI on

        Returns:
            DataFrame with RSI column added
        """
        result = data.copy()

        # Calculate price changes
        delta = result[column].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1 / period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1 / period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        return result

    @staticmethod
    def stochastic(
        data: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
        stoch_type: str = "slow",
    ) -> pd.DataFrame:
        """
        Stochastic Oscillator (Fast, Slow, Full).

        Args:
            data: DataFrame with OHLC data
            k_period: %K period
            d_period: %D period (signal line)
            smooth_k: Smoothing period for %K
            stoch_type: 'fast', 'slow', or 'full'

        Returns:
            DataFrame with %K and %D columns
        """
        result = data.copy()

        # Calculate raw %K (Fast Stochastic)
        lowest_low = result["low"].rolling(window=k_period).min()
        highest_high = result["high"].rolling(window=k_period).max()

        fast_k = 100 * (result["close"] - lowest_low) / (highest_high - lowest_low)

        if stoch_type == "fast":
            result["stoch_k"] = fast_k
            result["stoch_d"] = fast_k.rolling(window=d_period).mean()
        elif stoch_type == "slow":
            result["stoch_k"] = fast_k.rolling(window=smooth_k).mean()
            result["stoch_d"] = result["stoch_k"].rolling(window=d_period).mean()
        else:  # full
            result["stoch_k"] = fast_k.rolling(window=smooth_k).mean()
            result["stoch_d"] = result["stoch_k"].rolling(window=d_period).mean()

        return result

    @staticmethod
    def williams_r(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Williams %R.

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with Williams %R column
        """
        result = data.copy()

        highest_high = result["high"].rolling(window=period).max()
        lowest_low = result["low"].rolling(window=period).min()

        result["williams_r"] = -100 * (highest_high - result["close"]) / (highest_high - lowest_low)

        return result

    @staticmethod
    def cci(data: pd.DataFrame, period: int = 20, constant: float = 0.015) -> pd.DataFrame:
        """
        Commodity Channel Index (CCI).

        Args:
            data: DataFrame with OHLC data
            period: Lookback period
            constant: Scaling constant (typically 0.015)

        Returns:
            DataFrame with CCI column
        """
        result = data.copy()

        # Calculate Typical Price
        tp = (result["high"] + result["low"] + result["close"]) / 3

        # Calculate SMA of Typical Price
        sma_tp = tp.rolling(window=period).mean()

        # Calculate Mean Deviation
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)

        # Calculate CCI
        result["cci"] = (tp - sma_tp) / (constant * mad)

        return result

    @staticmethod
    def roc(data: pd.DataFrame, period: int = 12, column: str = "close") -> pd.DataFrame:
        """
        Rate of Change (ROC).

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate ROC on

        Returns:
            DataFrame with ROC column
        """
        result = data.copy()

        result[f"roc_{period}"] = (
            (result[column] - result[column].shift(period)) / result[column].shift(period)
        ) * 100

        return result

    @staticmethod
    def momentum(data: pd.DataFrame, period: int = 10, column: str = "close") -> pd.DataFrame:
        """
        Momentum (MOM).

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate momentum on

        Returns:
            DataFrame with Momentum column
        """
        result = data.copy()

        result[f"momentum_{period}"] = result[column] - result[column].shift(period)

        return result

    @staticmethod
    def ultimate_oscillator(
        data: pd.DataFrame,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
        weight1: float = 4.0,
        weight2: float = 2.0,
        weight3: float = 1.0,
    ) -> pd.DataFrame:
        """
        Ultimate Oscillator.

        Args:
            data: DataFrame with OHLC data
            period1, period2, period3: Three periods
            weight1, weight2, weight3: Weights for each period

        Returns:
            DataFrame with Ultimate Oscillator column
        """
        result = data.copy()

        # Calculate Buying Pressure
        bp = result["close"] - pd.concat([result["low"], result["close"].shift()], axis=1).min(
            axis=1
        )

        # Calculate True Range
        high_low = result["high"] - result["low"]
        high_close = np.abs(result["high"] - result["close"].shift())
        low_close = np.abs(result["low"] - result["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate averages for each period
        avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
        avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
        avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()

        # Calculate Ultimate Oscillator
        result["ultimate_osc"] = (
            100
            * ((weight1 * avg1) + (weight2 * avg2) + (weight3 * avg3))
            / (weight1 + weight2 + weight3)
        )

        return result

    @staticmethod
    def tsi(
        data: pd.DataFrame,
        long_period: int = 25,
        short_period: int = 13,
        signal_period: int = 13,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        True Strength Index (TSI).

        Args:
            data: DataFrame with price data
            long_period: Long EMA period
            short_period: Short EMA period
            signal_period: Signal line period
            column: Column to calculate TSI on

        Returns:
            DataFrame with TSI and signal line
        """
        result = data.copy()

        # Calculate price momentum
        momentum = result[column].diff()

        # Double smooth momentum
        momentum_ema1 = momentum.ewm(span=long_period, adjust=False).mean()
        momentum_ema2 = momentum_ema1.ewm(span=short_period, adjust=False).mean()

        # Double smooth absolute momentum
        abs_momentum_ema1 = momentum.abs().ewm(span=long_period, adjust=False).mean()
        abs_momentum_ema2 = abs_momentum_ema1.ewm(span=short_period, adjust=False).mean()

        # Calculate TSI
        result["tsi"] = 100 * (momentum_ema2 / abs_momentum_ema2)
        result["tsi_signal"] = result["tsi"].ewm(span=signal_period, adjust=False).mean()

        return result

    @staticmethod
    def ppo(
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Percentage Price Oscillator (PPO).

        Similar to MACD but expressed as percentage.

        Args:
            data: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to calculate PPO on

        Returns:
            DataFrame with PPO, signal, and histogram
        """
        result = data.copy()

        # Calculate EMAs
        ema_fast = result[column].ewm(span=fast, adjust=False).mean()
        ema_slow = result[column].ewm(span=slow, adjust=False).mean()

        # Calculate PPO
        result["ppo"] = ((ema_fast - ema_slow) / ema_slow) * 100
        result["ppo_signal"] = result["ppo"].ewm(span=signal, adjust=False).mean()
        result["ppo_histogram"] = result["ppo"] - result["ppo_signal"]

        return result

    @staticmethod
    def awesome_oscillator(data: pd.DataFrame, fast: int = 5, slow: int = 34) -> pd.DataFrame:
        """
        Awesome Oscillator (AO).

        Args:
            data: DataFrame with OHLC data
            fast: Fast SMA period
            slow: Slow SMA period

        Returns:
            DataFrame with AO column
        """
        result = data.copy()

        # Calculate median price
        median_price = (result["high"] + result["low"]) / 2

        # Calculate AO
        result["awesome_osc"] = (
            median_price.rolling(window=fast).mean() - median_price.rolling(window=slow).mean()
        )

        return result

    @staticmethod
    def kama(
        data: pd.DataFrame,
        period: int = 10,
        fast_ema: int = 2,
        slow_ema: int = 30,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Kaufman's Adaptive Moving Average (KAMA).

        Args:
            data: DataFrame with price data
            period: Efficiency Ratio period
            fast_ema: Fast EMA constant
            slow_ema: Slow EMA constant
            column: Column to calculate KAMA on

        Returns:
            DataFrame with KAMA column
        """
        result = data.copy()
        prices = result[column].values

        # Calculate Efficiency Ratio
        change = np.abs(prices - np.roll(prices, period))
        volatility = (
            pd.Series(np.abs(np.diff(prices, prepend=prices[0])))
            .rolling(window=period)
            .sum()
            .values
        )

        er = change / volatility
        er = np.nan_to_num(er)

        # Calculate Smoothing Constant
        fast_sc = 2 / (fast_ema + 1)
        slow_sc = 2 / (slow_ema + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # Calculate KAMA
        kama = np.zeros_like(prices)
        kama[0] = prices[0]

        for i in range(1, len(prices)):
            kama[i] = kama[i - 1] + sc[i] * (prices[i] - kama[i - 1])

        result["kama"] = kama

        return result

    @staticmethod
    def cmo(data: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
        """
        Chande Momentum Oscillator (CMO).

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate CMO on

        Returns:
            DataFrame with CMO column
        """
        result = data.copy()

        # Calculate price changes
        delta = result[column].diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Sum over period
        sum_gains = gains.rolling(window=period).sum()
        sum_losses = losses.rolling(window=period).sum()

        # Calculate CMO
        result["cmo"] = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)

        return result

    @staticmethod
    def dpo(data: pd.DataFrame, period: int = 20, column: str = "close") -> pd.DataFrame:
        """
        Detrended Price Oscillator (DPO).

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate DPO on

        Returns:
            DataFrame with DPO column
        """
        result = data.copy()

        # Calculate SMA
        sma = result[column].rolling(window=period).mean()

        # Shift SMA back by (period/2 + 1)
        shift = int(period / 2) + 1
        sma_shifted = sma.shift(shift)

        # Calculate DPO
        result["dpo"] = result[column] - sma_shifted

        return result

    @staticmethod
    def pmo(
        data: pd.DataFrame,
        period1: int = 35,
        period2: int = 20,
        signal: int = 10,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Price Momentum Oscillator (PMO).

        Args:
            data: DataFrame with price data
            period1: First smoothing period
            period2: Second smoothing period
            signal: Signal line period
            column: Column to calculate PMO on

        Returns:
            DataFrame with PMO and signal line
        """
        result = data.copy()

        # Calculate ROC
        roc = ((result[column] / result[column].shift(1)) - 1) * 100

        # Double smooth
        pmo = roc.ewm(span=period1, adjust=False).mean()
        pmo = pmo.ewm(span=period2, adjust=False).mean() * 10

        result["pmo"] = pmo
        result["pmo_signal"] = pmo.ewm(span=signal, adjust=False).mean()

        return result

    @staticmethod
    def rvi(data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """
        Relative Vigor Index (RVI).

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with RVI and signal line
        """
        result = data.copy()

        # Calculate numerator and denominator
        numerator = (
            result["close"]
            - result["open"]
            + 2 * (result["close"].shift(1) - result["open"].shift(1))
            + 2 * (result["close"].shift(2) - result["open"].shift(2))
            + (result["close"].shift(3) - result["open"].shift(3))
        ) / 6

        denominator = (
            result["high"]
            - result["low"]
            + 2 * (result["high"].shift(1) - result["low"].shift(1))
            + 2 * (result["high"].shift(2) - result["low"].shift(2))
            + (result["high"].shift(3) - result["low"].shift(3))
        ) / 6

        # Calculate RVI
        rvi = numerator.rolling(window=period).sum() / denominator.rolling(window=period).sum()

        result["rvi"] = rvi
        result["rvi_signal"] = (rvi + 2 * rvi.shift(1) + 2 * rvi.shift(2) + rvi.shift(3)) / 6

        return result

    @staticmethod
    def inertia(
        data: pd.DataFrame,
        period: int = 20,
        rvi_period: int = 14,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Inertia Indicator.

        Args:
            data: DataFrame with price data
            period: RVI period
            rvi_period: Regression period
            column: Column to calculate on

        Returns:
            DataFrame with Inertia column
        """
        result = data.copy()

        # Calculate RVI first
        result = MomentumIndicators.rvi(result, period=period)

        # Apply linear regression to RVI
        result["inertia"] = (
            result["rvi"]
            .rolling(window=rvi_period)
            .apply(
                lambda x: (
                    np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == rvi_period else np.nan
                ),
                raw=True,
            )
        )

        return result


# Register all momentum indicators
def register_momentum_indicators():
    """Register all momentum indicators with the global registry."""
    registry = get_registry()

    # RSI
    registry.register(
        name="rsi",
        func=MomentumIndicators.rsi,
        category="momentum",
        description="Relative Strength Index",
        parameters={"period": 14, "column": "close"},
        required_columns=["close"],
        output_columns=["rsi_{period}"],
    )

    # Stochastic
    registry.register(
        name="stochastic",
        func=MomentumIndicators.stochastic,
        category="momentum",
        description="Stochastic Oscillator",
        parameters={"k_period": 14, "d_period": 3, "smooth_k": 3, "stoch_type": "slow"},
        required_columns=["high", "low", "close"],
        output_columns=["stoch_k", "stoch_d"],
    )

    # Williams %R
    registry.register(
        name="williams_r",
        func=MomentumIndicators.williams_r,
        category="momentum",
        description="Williams %R",
        parameters={"period": 14},
        required_columns=["high", "low", "close"],
        output_columns=["williams_r"],
    )

    # CCI
    registry.register(
        name="cci",
        func=MomentumIndicators.cci,
        category="momentum",
        description="Commodity Channel Index",
        parameters={"period": 20, "constant": 0.015},
        required_columns=["high", "low", "close"],
        output_columns=["cci"],
    )

    # ROC
    registry.register(
        name="roc",
        func=MomentumIndicators.roc,
        category="momentum",
        description="Rate of Change",
        parameters={"period": 12, "column": "close"},
        required_columns=["close"],
        output_columns=["roc_{period}"],
    )

    # Momentum
    registry.register(
        name="momentum",
        func=MomentumIndicators.momentum,
        category="momentum",
        description="Momentum",
        parameters={"period": 10, "column": "close"},
        required_columns=["close"],
        output_columns=["momentum_{period}"],
    )

    # Ultimate Oscillator
    registry.register(
        name="ultimate_osc",
        func=MomentumIndicators.ultimate_oscillator,
        category="momentum",
        description="Ultimate Oscillator",
        parameters={
            "period1": 7,
            "period2": 14,
            "period3": 28,
            "weight1": 4.0,
            "weight2": 2.0,
            "weight3": 1.0,
        },
        required_columns=["high", "low", "close"],
        output_columns=["ultimate_osc"],
    )

    # TSI
    registry.register(
        name="tsi",
        func=MomentumIndicators.tsi,
        category="momentum",
        description="True Strength Index",
        parameters={
            "long_period": 25,
            "short_period": 13,
            "signal_period": 13,
            "column": "close",
        },
        required_columns=["close"],
        output_columns=["tsi", "tsi_signal"],
    )

    # PPO
    registry.register(
        name="ppo",
        func=MomentumIndicators.ppo,
        category="momentum",
        description="Percentage Price Oscillator",
        parameters={"fast": 12, "slow": 26, "signal": 9, "column": "close"},
        required_columns=["close"],
        output_columns=["ppo", "ppo_signal", "ppo_histogram"],
    )

    # Awesome Oscillator
    registry.register(
        name="awesome_osc",
        func=MomentumIndicators.awesome_oscillator,
        category="momentum",
        description="Awesome Oscillator",
        parameters={"fast": 5, "slow": 34},
        required_columns=["high", "low"],
        output_columns=["awesome_osc"],
    )

    # KAMA
    registry.register(
        name="kama",
        func=MomentumIndicators.kama,
        category="momentum",
        description="Kaufman's Adaptive Moving Average",
        parameters={"period": 10, "fast_ema": 2, "slow_ema": 30, "column": "close"},
        required_columns=["close"],
        output_columns=["kama"],
    )

    # CMO
    registry.register(
        name="cmo",
        func=MomentumIndicators.cmo,
        category="momentum",
        description="Chande Momentum Oscillator",
        parameters={"period": 14, "column": "close"},
        required_columns=["close"],
        output_columns=["cmo"],
    )

    # DPO
    registry.register(
        name="dpo",
        func=MomentumIndicators.dpo,
        category="momentum",
        description="Detrended Price Oscillator",
        parameters={"period": 20, "column": "close"},
        required_columns=["close"],
        output_columns=["dpo"],
    )

    # PMO
    registry.register(
        name="pmo",
        func=MomentumIndicators.pmo,
        category="momentum",
        description="Price Momentum Oscillator",
        parameters={"period1": 35, "period2": 20, "signal": 10, "column": "close"},
        required_columns=["close"],
        output_columns=["pmo", "pmo_signal"],
    )

    # RVI
    registry.register(
        name="rvi",
        func=MomentumIndicators.rvi,
        category="momentum",
        description="Relative Vigor Index",
        parameters={"period": 10},
        required_columns=["open", "high", "low", "close"],
        output_columns=["rvi", "rvi_signal"],
    )

    # Inertia
    registry.register(
        name="inertia",
        func=MomentumIndicators.inertia,
        category="momentum",
        description="Inertia Indicator",
        parameters={"period": 20, "rvi_period": 14, "column": "close"},
        required_columns=["open", "high", "low", "close"],
        output_columns=["inertia"],
    )


# Auto-register on import
register_momentum_indicators()
