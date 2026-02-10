"""
Volume Indicators

This module implements 15+ volume-based indicators using vectorized NumPy operations.
"""

import numpy as np
import pandas as pd

from ara.features.indicator_registry import get_registry


class VolumeIndicators:
    """Collection of volume-based indicators."""

    @staticmethod
    def obv(data: pd.DataFrame) -> pd.DataFrame:
        """
        On-Balance Volume (OBV).

        Args:
            data: DataFrame with price and volume data

        Returns:
            DataFrame with OBV column
        """
        result = data.copy()

        # Calculate price direction
        price_change = result["close"].diff()

        # Calculate OBV
        obv = np.where(
            price_change > 0,
            result["volume"],
            np.where(price_change < 0, -result["volume"], 0),
        )

        result["obv"] = pd.Series(obv, index=result.index).cumsum()

        return result

    @staticmethod
    def vwap(data: pd.DataFrame) -> pd.DataFrame:
        """
        Volume Weighted Average Price (VWAP).

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with VWAP column
        """
        result = data.copy()

        # Calculate typical price
        typical_price = (result["high"] + result["low"] + result["close"]) / 3

        # Calculate VWAP (cumulative for intraday, rolling for daily)
        if len(result) > 100:  # Assume daily data, use rolling
            period = 20
            result["vwap"] = (typical_price * result["volume"]).rolling(
                window=period
            ).sum() / result["volume"].rolling(window=period).sum()
        else:  # Intraday data, use cumulative
            result["vwap"] = (typical_price * result["volume"]).cumsum() / result["volume"].cumsum()

        return result

    @staticmethod
    def mfi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Money Flow Index (MFI).

        Args:
            data: DataFrame with OHLCV data
            period: Lookback period

        Returns:
            DataFrame with MFI column
        """
        result = data.copy()

        # Calculate typical price
        typical_price = (result["high"] + result["low"] + result["close"]) / 3

        # Calculate money flow
        money_flow = typical_price * result["volume"]

        # Separate positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = money_flow.where(price_change > 0, 0)
        negative_flow = money_flow.where(price_change < 0, 0)

        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfr = positive_mf / negative_mf

        # Calculate MFI
        result["mfi"] = 100 - (100 / (1 + mfr))

        return result

    @staticmethod
    def ad_line(data: pd.DataFrame) -> pd.DataFrame:
        """
        Accumulation/Distribution Line (A/D).

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with A/D line column
        """
        result = data.copy()

        # Calculate Money Flow Multiplier
        mfm = ((result["close"] - result["low"]) - (result["high"] - result["close"])) / (
            result["high"] - result["low"]
        )

        # Handle division by zero
        mfm = mfm.fillna(0)

        # Calculate Money Flow Volume
        mfv = mfm * result["volume"]

        # Calculate A/D Line
        result["ad_line"] = mfv.cumsum()

        return result

    @staticmethod
    def chaikin_money_flow(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Chaikin Money Flow (CMF).

        Args:
            data: DataFrame with OHLCV data
            period: Lookback period

        Returns:
            DataFrame with CMF column
        """
        result = data.copy()

        # Calculate Money Flow Multiplier
        mfm = ((result["close"] - result["low"]) - (result["high"] - result["close"])) / (
            result["high"] - result["low"]
        )
        mfm = mfm.fillna(0)

        # Calculate Money Flow Volume
        mfv = mfm * result["volume"]

        # Calculate CMF
        result["cmf"] = (
            mfv.rolling(window=period).sum() / result["volume"].rolling(window=period).sum()
        )

        return result

    @staticmethod
    def volume_roc(data: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """
        Volume Rate of Change.

        Args:
            data: DataFrame with volume data
            period: Lookback period

        Returns:
            DataFrame with Volume ROC column
        """
        result = data.copy()

        result["volume_roc"] = (
            (result["volume"] - result["volume"].shift(period)) / result["volume"].shift(period)
        ) * 100

        return result

    @staticmethod
    def force_index(data: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """
        Force Index.

        Args:
            data: DataFrame with price and volume data
            period: EMA period

        Returns:
            DataFrame with Force Index column
        """
        result = data.copy()

        # Calculate raw force
        force = result["close"].diff() * result["volume"]

        # Smooth with EMA
        result["force_index"] = force.ewm(span=period, adjust=False).mean()

        return result

    @staticmethod
    def ease_of_movement(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Ease of Movement (EMV).

        Args:
            data: DataFrame with OHLCV data
            period: SMA period

        Returns:
            DataFrame with EMV column
        """
        result = data.copy()

        # Calculate distance moved
        distance = ((result["high"] + result["low"]) / 2) - (
            (result["high"].shift() + result["low"].shift()) / 2
        )

        # Calculate box ratio
        box_ratio = (result["volume"] / 1000000) / (result["high"] - result["low"])

        # Calculate EMV
        emv = distance / box_ratio

        # Smooth with SMA
        result["emv"] = emv.rolling(window=period).mean()

        return result

    @staticmethod
    def volume_oscillator(data: pd.DataFrame, fast: int = 5, slow: int = 10) -> pd.DataFrame:
        """
        Volume Oscillator.

        Args:
            data: DataFrame with volume data
            fast: Fast EMA period
            slow: Slow EMA period

        Returns:
            DataFrame with Volume Oscillator column
        """
        result = data.copy()

        fast_ema = result["volume"].ewm(span=fast, adjust=False).mean()
        slow_ema = result["volume"].ewm(span=slow, adjust=False).mean()

        result["volume_osc"] = ((fast_ema - slow_ema) / slow_ema) * 100

        return result

    @staticmethod
    def negative_volume_index(data: pd.DataFrame) -> pd.DataFrame:
        """
        Negative Volume Index (NVI).

        Args:
            data: DataFrame with price and volume data

        Returns:
            DataFrame with NVI column
        """
        result = data.copy()

        nvi = pd.Series(index=result.index, dtype=float)
        nvi.iloc[0] = 1000  # Starting value

        for i in range(1, len(result)):
            if result["volume"].iloc[i] < result["volume"].iloc[i - 1]:
                # Volume decreased, update NVI
                price_change = (result["close"].iloc[i] - result["close"].iloc[i - 1]) / result[
                    "close"
                ].iloc[i - 1]
                nvi.iloc[i] = nvi.iloc[i - 1] * (1 + price_change)
            else:
                # Volume increased or same, keep previous value
                nvi.iloc[i] = nvi.iloc[i - 1]

        result["nvi"] = nvi

        return result

    @staticmethod
    def positive_volume_index(data: pd.DataFrame) -> pd.DataFrame:
        """
        Positive Volume Index (PVI).

        Args:
            data: DataFrame with price and volume data

        Returns:
            DataFrame with PVI column
        """
        result = data.copy()

        pvi = pd.Series(index=result.index, dtype=float)
        pvi.iloc[0] = 1000  # Starting value

        for i in range(1, len(result)):
            if result["volume"].iloc[i] > result["volume"].iloc[i - 1]:
                # Volume increased, update PVI
                price_change = (result["close"].iloc[i] - result["close"].iloc[i - 1]) / result[
                    "close"
                ].iloc[i - 1]
                pvi.iloc[i] = pvi.iloc[i - 1] * (1 + price_change)
            else:
                # Volume decreased or same, keep previous value
                pvi.iloc[i] = pvi.iloc[i - 1]

        result["pvi"] = pvi

        return result

    @staticmethod
    def volume_weighted_macd(
        data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """
        Volume Weighted MACD.

        Args:
            data: DataFrame with price and volume data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with Volume Weighted MACD columns
        """
        result = data.copy()

        # Calculate volume-weighted price
        vw_price = result["close"] * result["volume"]

        # Calculate EMAs
        fast_ema = (
            vw_price.ewm(span=fast, adjust=False).mean()
            / result["volume"].ewm(span=fast, adjust=False).mean()
        )
        slow_ema = (
            vw_price.ewm(span=slow, adjust=False).mean()
            / result["volume"].ewm(span=slow, adjust=False).mean()
        )

        # Calculate MACD
        result["vw_macd"] = fast_ema - slow_ema
        result["vw_macd_signal"] = result["vw_macd"].ewm(span=signal, adjust=False).mean()
        result["vw_macd_histogram"] = result["vw_macd"] - result["vw_macd_signal"]

        return result

    @staticmethod
    def klinger_oscillator(
        data: pd.DataFrame, fast: int = 34, slow: int = 55, signal: int = 13
    ) -> pd.DataFrame:
        """
        Klinger Oscillator.

        Args:
            data: DataFrame with OHLCV data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with Klinger Oscillator columns
        """
        result = data.copy()

        # Calculate typical price
        tp = (result["high"] + result["low"] + result["close"]) / 3

        # Calculate trend
        trend = np.where(tp > tp.shift(1), 1, -1)

        # Calculate volume force
        dm = result["high"] - result["low"]
        cm = dm.cumsum()
        vf = result["volume"] * trend * 100 * ((dm / cm) - 1)

        # Calculate Klinger Oscillator
        fast_ema = vf.ewm(span=fast, adjust=False).mean()
        slow_ema = vf.ewm(span=slow, adjust=False).mean()

        result["klinger_osc"] = fast_ema - slow_ema
        result["klinger_signal"] = result["klinger_osc"].ewm(span=signal, adjust=False).mean()

        return result

    @staticmethod
    def price_volume_trend(data: pd.DataFrame) -> pd.DataFrame:
        """
        Price Volume Trend (PVT).

        Args:
            data: DataFrame with price and volume data

        Returns:
            DataFrame with PVT column
        """
        result = data.copy()

        # Calculate price change percentage
        price_change_pct = result["close"].pct_change()

        # Calculate PVT
        pvt = (price_change_pct * result["volume"]).cumsum()

        result["pvt"] = pvt

        return result

    @staticmethod
    def elder_ray_index(data: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """
        Elder Ray Index (Bull Power and Bear Power).

        Args:
            data: DataFrame with OHLC data
            period: EMA period

        Returns:
            DataFrame with Bull Power and Bear Power columns
        """
        result = data.copy()

        # Calculate EMA
        ema = result["close"].ewm(span=period, adjust=False).mean()

        # Calculate Bull and Bear Power
        result["bull_power"] = result["high"] - ema
        result["bear_power"] = result["low"] - ema

        return result


# Register all volume indicators
def register_volume_indicators():
    """Register all volume indicators with the global registry."""
    registry = get_registry()

    # OBV
    registry.register(
        name="obv",
        func=VolumeIndicators.obv,
        category="volume",
        description="On-Balance Volume",
        parameters={},
        required_columns=["close", "volume"],
        output_columns=["obv"],
    )

    # VWAP
    registry.register(
        name="vwap",
        func=VolumeIndicators.vwap,
        category="volume",
        description="Volume Weighted Average Price",
        parameters={},
        required_columns=["high", "low", "close", "volume"],
        output_columns=["vwap"],
    )

    # MFI
    registry.register(
        name="mfi",
        func=VolumeIndicators.mfi,
        category="volume",
        description="Money Flow Index",
        parameters={"period": 14},
        required_columns=["high", "low", "close", "volume"],
        output_columns=["mfi"],
    )

    # A/D Line
    registry.register(
        name="ad_line",
        func=VolumeIndicators.ad_line,
        category="volume",
        description="Accumulation/Distribution Line",
        parameters={},
        required_columns=["high", "low", "close", "volume"],
        output_columns=["ad_line"],
    )

    # CMF
    registry.register(
        name="cmf",
        func=VolumeIndicators.chaikin_money_flow,
        category="volume",
        description="Chaikin Money Flow",
        parameters={"period": 20},
        required_columns=["high", "low", "close", "volume"],
        output_columns=["cmf"],
    )

    # Volume ROC
    registry.register(
        name="volume_roc",
        func=VolumeIndicators.volume_roc,
        category="volume",
        description="Volume Rate of Change",
        parameters={"period": 12},
        required_columns=["volume"],
        output_columns=["volume_roc"],
    )

    # Force Index
    registry.register(
        name="force_index",
        func=VolumeIndicators.force_index,
        category="volume",
        description="Force Index",
        parameters={"period": 13},
        required_columns=["close", "volume"],
        output_columns=["force_index"],
    )

    # EMV
    registry.register(
        name="emv",
        func=VolumeIndicators.ease_of_movement,
        category="volume",
        description="Ease of Movement",
        parameters={"period": 14},
        required_columns=["high", "low", "volume"],
        output_columns=["emv"],
    )

    # Volume Oscillator
    registry.register(
        name="volume_osc",
        func=VolumeIndicators.volume_oscillator,
        category="volume",
        description="Volume Oscillator",
        parameters={"fast": 5, "slow": 10},
        required_columns=["volume"],
        output_columns=["volume_osc"],
    )

    # NVI
    registry.register(
        name="nvi",
        func=VolumeIndicators.negative_volume_index,
        category="volume",
        description="Negative Volume Index",
        parameters={},
        required_columns=["close", "volume"],
        output_columns=["nvi"],
    )

    # PVI
    registry.register(
        name="pvi",
        func=VolumeIndicators.positive_volume_index,
        category="volume",
        description="Positive Volume Index",
        parameters={},
        required_columns=["close", "volume"],
        output_columns=["pvi"],
    )

    # Volume Weighted MACD
    registry.register(
        name="vw_macd",
        func=VolumeIndicators.volume_weighted_macd,
        category="volume",
        description="Volume Weighted MACD",
        parameters={"fast": 12, "slow": 26, "signal": 9},
        required_columns=["close", "volume"],
        output_columns=["vw_macd", "vw_macd_signal", "vw_macd_histogram"],
    )

    # Klinger Oscillator
    registry.register(
        name="klinger_osc",
        func=VolumeIndicators.klinger_oscillator,
        category="volume",
        description="Klinger Oscillator",
        parameters={"fast": 34, "slow": 55, "signal": 13},
        required_columns=["high", "low", "close", "volume"],
        output_columns=["klinger_osc", "klinger_signal"],
    )

    # PVT
    registry.register(
        name="pvt",
        func=VolumeIndicators.price_volume_trend,
        category="volume",
        description="Price Volume Trend",
        parameters={},
        required_columns=["close", "volume"],
        output_columns=["pvt"],
    )

    # Elder Ray Index
    registry.register(
        name="elder_ray",
        func=VolumeIndicators.elder_ray_index,
        category="volume",
        description="Elder Ray Index",
        parameters={"period": 13},
        required_columns=["high", "low", "close"],
        output_columns=["bull_power", "bear_power"],
    )


# Auto-register on import
register_volume_indicators()
