"""
Volatility Indicators

This module implements 15+ volatility indicators using vectorized NumPy operations.
"""

import numpy as np
import pandas as pd

from ara.features.indicator_registry import get_registry


class VolatilityIndicators:
    """Collection of volatility indicators."""

    @staticmethod
    def bollinger_bands(
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Bollinger Bands.

        Args:
            data: DataFrame with price data
            period: SMA period
            std_dev: Number of standard deviations
            column: Column to calculate on

        Returns:
            DataFrame with upper, middle, and lower bands
        """
        result = data.copy()

        # Calculate middle band (SMA)
        middle = result[column].rolling(window=period).mean()

        # Calculate standard deviation
        std = result[column].rolling(window=period).std()

        # Calculate bands
        result["bb_upper"] = middle + (std_dev * std)
        result["bb_middle"] = middle
        result["bb_lower"] = middle - (std_dev * std)
        result["bb_width"] = result["bb_upper"] - result["bb_lower"]
        result["bb_percent"] = (result[column] - result["bb_lower"]) / (
            result["bb_upper"] - result["bb_lower"]
        )

        return result

    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average True Range (ATR).

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with ATR column
        """
        result = data.copy()

        # Calculate True Range
        high_low = result["high"] - result["low"]
        high_close = np.abs(result["high"] - result["close"].shift())
        low_close = np.abs(result["low"] - result["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Calculate ATR using Wilder's smoothing
        result["atr"] = tr.ewm(alpha=1 / period, adjust=False).mean()

        return result

    @staticmethod
    def keltner_channels(
        data: pd.DataFrame,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> pd.DataFrame:
        """
        Keltner Channels.

        Args:
            data: DataFrame with OHLC data
            ema_period: EMA period for middle line
            atr_period: ATR period
            multiplier: ATR multiplier

        Returns:
            DataFrame with Keltner Channel bands
        """
        result = data.copy()

        # Calculate middle line (EMA)
        middle = result["close"].ewm(span=ema_period, adjust=False).mean()

        # Calculate ATR
        high_low = result["high"] - result["low"]
        high_close = np.abs(result["high"] - result["close"].shift())
        low_close = np.abs(result["low"] - result["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

        # Calculate bands
        result["keltner_upper"] = middle + (multiplier * atr)
        result["keltner_middle"] = middle
        result["keltner_lower"] = middle - (multiplier * atr)

        return result

    @staticmethod
    def donchian_channels(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Donchian Channels.

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with Donchian Channel bands
        """
        result = data.copy()

        result["donchian_upper"] = result["high"].rolling(window=period).max()
        result["donchian_lower"] = result["low"].rolling(window=period).min()
        result["donchian_middle"] = (result["donchian_upper"] + result["donchian_lower"]) / 2

        return result

    @staticmethod
    def historical_volatility(
        data: pd.DataFrame,
        period: int = 20,
        annualize: bool = True,
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Historical Volatility (Standard Deviation of Returns).

        Args:
            data: DataFrame with price data
            period: Lookback period
            annualize: Annualize the volatility (multiply by sqrt(252))
            column: Column to calculate on

        Returns:
            DataFrame with historical volatility column
        """
        result = data.copy()

        # Calculate log returns
        log_returns = np.log(result[column] / result[column].shift(1))

        # Calculate rolling standard deviation
        volatility = log_returns.rolling(window=period).std()

        # Annualize if requested
        if annualize:
            volatility = volatility * np.sqrt(252)

        result["historical_volatility"] = volatility * 100  # Convert to percentage

        return result

    @staticmethod
    def chaikin_volatility(
        data: pd.DataFrame, period: int = 10, roc_period: int = 10
    ) -> pd.DataFrame:
        """
        Chaikin Volatility.

        Args:
            data: DataFrame with OHLC data
            period: EMA period
            roc_period: Rate of change period

        Returns:
            DataFrame with Chaikin Volatility column
        """
        result = data.copy()

        # Calculate high-low range
        hl_range = result["high"] - result["low"]

        # Calculate EMA of range
        ema_range = hl_range.ewm(span=period, adjust=False).mean()

        # Calculate rate of change
        result["chaikin_volatility"] = (
            (ema_range - ema_range.shift(roc_period)) / ema_range.shift(roc_period)
        ) * 100

        return result

    @staticmethod
    def std_dev_bands(
        data: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        ma_type: str = "sma",
        column: str = "close",
    ) -> pd.DataFrame:
        """
        Standard Deviation Bands.

        Args:
            data: DataFrame with price data
            period: Moving average period
            std_dev: Number of standard deviations
            ma_type: Type of moving average ('sma' or 'ema')
            column: Column to calculate on

        Returns:
            DataFrame with standard deviation bands
        """
        result = data.copy()

        # Calculate moving average
        if ma_type == "sma":
            ma = result[column].rolling(window=period).mean()
        else:  # ema
            ma = result[column].ewm(span=period, adjust=False).mean()

        # Calculate standard deviation
        std = result[column].rolling(window=period).std()

        # Calculate bands
        result["std_upper"] = ma + (std_dev * std)
        result["std_middle"] = ma
        result["std_lower"] = ma - (std_dev * std)

        return result

    @staticmethod
    def ulcer_index(data: pd.DataFrame, period: int = 14, column: str = "close") -> pd.DataFrame:
        """
        Ulcer Index.

        Measures downside volatility.

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate on

        Returns:
            DataFrame with Ulcer Index column
        """
        result = data.copy()

        # Calculate percentage drawdown from highest high
        highest = result[column].rolling(window=period).max()
        drawdown = ((result[column] - highest) / highest) * 100

        # Calculate Ulcer Index
        result["ulcer_index"] = np.sqrt((drawdown**2).rolling(window=period).mean())

        return result

    @staticmethod
    def true_range(data: pd.DataFrame) -> pd.DataFrame:
        """
        True Range.

        Args:
            data: DataFrame with OHLC data

        Returns:
            DataFrame with True Range column
        """
        result = data.copy()

        high_low = result["high"] - result["low"]
        high_close = np.abs(result["high"] - result["close"].shift())
        low_close = np.abs(result["low"] - result["close"].shift())

        result["true_range"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        return result

    @staticmethod
    def natr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Normalized Average True Range (NATR).

        ATR expressed as percentage of close price.

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with NATR column
        """
        result = data.copy()

        # Calculate ATR
        high_low = result["high"] - result["low"]
        high_close = np.abs(result["high"] - result["close"].shift())
        low_close = np.abs(result["low"] - result["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()

        # Normalize by close price
        result["natr"] = (atr / result["close"]) * 100

        return result

    @staticmethod
    def rvi_volatility(data: pd.DataFrame, period: int = 10, column: str = "close") -> pd.DataFrame:
        """
        Relative Volatility Index (RVI).

        Args:
            data: DataFrame with price data
            period: Lookback period
            column: Column to calculate on

        Returns:
            DataFrame with RVI column
        """
        result = data.copy()

        # Calculate standard deviation
        std = result[column].rolling(window=period).std()

        # Separate positive and negative std changes
        std_change = std.diff()
        pos_std = std_change.where(std_change > 0, 0)
        neg_std = -std_change.where(std_change < 0, 0)

        # Calculate EMA of positive and negative changes
        pos_avg = pos_std.ewm(alpha=1 / period, adjust=False).mean()
        neg_avg = neg_std.ewm(alpha=1 / period, adjust=False).mean()

        # Calculate RVI
        result["rvi_volatility"] = 100 * pos_avg / (pos_avg + neg_avg)

        return result

    @staticmethod
    def price_channels(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Price Channels (High/Low Channels).

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with price channel bands
        """
        result = data.copy()

        result["price_channel_upper"] = result["high"].rolling(window=period).max()
        result["price_channel_lower"] = result["low"].rolling(window=period).min()
        result["price_channel_middle"] = (
            result["price_channel_upper"] + result["price_channel_lower"]
        ) / 2

        return result

    @staticmethod
    def chandelier_exit(
        data: pd.DataFrame,
        period: int = 22,
        multiplier: float = 3.0,
        use_close: bool = True,
    ) -> pd.DataFrame:
        """
        Chandelier Exit.

        Args:
            data: DataFrame with OHLC data
            period: ATR period
            multiplier: ATR multiplier
            use_close: Use close instead of high/low

        Returns:
            DataFrame with Chandelier Exit levels
        """
        result = data.copy()

        # Calculate ATR
        high_low = result["high"] - result["low"]
        high_close = np.abs(result["high"] - result["close"].shift())
        low_close = np.abs(result["low"] - result["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()

        if use_close:
            highest = result["close"].rolling(window=period).max()
            lowest = result["close"].rolling(window=period).min()
        else:
            highest = result["high"].rolling(window=period).max()
            lowest = result["low"].rolling(window=period).min()

        result["chandelier_long"] = highest - (multiplier * atr)
        result["chandelier_short"] = lowest + (multiplier * atr)

        return result

    @staticmethod
    def mass_index_volatility(
        data: pd.DataFrame, period: int = 9, sum_period: int = 25
    ) -> pd.DataFrame:
        """
        Mass Index (Volatility version).

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
        result["mass_index_vol"] = mass.rolling(window=sum_period).sum()

        return result

    @staticmethod
    def garman_klass_volatility(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Garman-Klass Volatility Estimator.

        More efficient volatility estimator using OHLC data.

        Args:
            data: DataFrame with OHLC data
            period: Lookback period

        Returns:
            DataFrame with Garman-Klass volatility
        """
        result = data.copy()

        # Calculate log ratios
        log_hl = np.log(result["high"] / result["low"])
        log_co = np.log(result["close"] / result["open"])

        # Garman-Klass formula
        gk = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)

        # Calculate rolling average and annualize
        result["garman_klass_vol"] = np.sqrt(gk.rolling(window=period).mean() * 252) * 100

        return result


# Register all volatility indicators
def register_volatility_indicators():
    """Register all volatility indicators with the global registry."""
    registry = get_registry()

    # Bollinger Bands
    registry.register(
        name="bollinger_bands",
        func=VolatilityIndicators.bollinger_bands,
        category="volatility",
        description="Bollinger Bands",
        parameters={"period": 20, "std_dev": 2.0, "column": "close"},
        required_columns=["close"],
        output_columns=["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_percent"],
    )

    # ATR
    registry.register(
        name="atr",
        func=VolatilityIndicators.atr,
        category="volatility",
        description="Average True Range",
        parameters={"period": 14},
        required_columns=["high", "low", "close"],
        output_columns=["atr"],
    )

    # Keltner Channels
    registry.register(
        name="keltner_channels",
        func=VolatilityIndicators.keltner_channels,
        category="volatility",
        description="Keltner Channels",
        parameters={"ema_period": 20, "atr_period": 10, "multiplier": 2.0},
        required_columns=["high", "low", "close"],
        output_columns=["keltner_upper", "keltner_middle", "keltner_lower"],
    )

    # Donchian Channels
    registry.register(
        name="donchian_channels",
        func=VolatilityIndicators.donchian_channels,
        category="volatility",
        description="Donchian Channels",
        parameters={"period": 20},
        required_columns=["high", "low"],
        output_columns=["donchian_upper", "donchian_lower", "donchian_middle"],
    )

    # Historical Volatility
    registry.register(
        name="historical_volatility",
        func=VolatilityIndicators.historical_volatility,
        category="volatility",
        description="Historical Volatility",
        parameters={"period": 20, "annualize": True, "column": "close"},
        required_columns=["close"],
        output_columns=["historical_volatility"],
    )

    # Chaikin Volatility
    registry.register(
        name="chaikin_volatility",
        func=VolatilityIndicators.chaikin_volatility,
        category="volatility",
        description="Chaikin Volatility",
        parameters={"period": 10, "roc_period": 10},
        required_columns=["high", "low"],
        output_columns=["chaikin_volatility"],
    )

    # Standard Deviation Bands
    registry.register(
        name="std_dev_bands",
        func=VolatilityIndicators.std_dev_bands,
        category="volatility",
        description="Standard Deviation Bands",
        parameters={"period": 20, "std_dev": 2.0, "ma_type": "sma", "column": "close"},
        required_columns=["close"],
        output_columns=["std_upper", "std_middle", "std_lower"],
    )

    # Ulcer Index
    registry.register(
        name="ulcer_index",
        func=VolatilityIndicators.ulcer_index,
        category="volatility",
        description="Ulcer Index",
        parameters={"period": 14, "column": "close"},
        required_columns=["close"],
        output_columns=["ulcer_index"],
    )

    # True Range
    registry.register(
        name="true_range",
        func=VolatilityIndicators.true_range,
        category="volatility",
        description="True Range",
        parameters={},
        required_columns=["high", "low", "close"],
        output_columns=["true_range"],
    )

    # NATR
    registry.register(
        name="natr",
        func=VolatilityIndicators.natr,
        category="volatility",
        description="Normalized Average True Range",
        parameters={"period": 14},
        required_columns=["high", "low", "close"],
        output_columns=["natr"],
    )

    # RVI Volatility
    registry.register(
        name="rvi_volatility",
        func=VolatilityIndicators.rvi_volatility,
        category="volatility",
        description="Relative Volatility Index",
        parameters={"period": 10, "column": "close"},
        required_columns=["close"],
        output_columns=["rvi_volatility"],
    )

    # Price Channels
    registry.register(
        name="price_channels",
        func=VolatilityIndicators.price_channels,
        category="volatility",
        description="Price Channels",
        parameters={"period": 20},
        required_columns=["high", "low"],
        output_columns=[
            "price_channel_upper",
            "price_channel_lower",
            "price_channel_middle",
        ],
    )

    # Chandelier Exit
    registry.register(
        name="chandelier_exit",
        func=VolatilityIndicators.chandelier_exit,
        category="volatility",
        description="Chandelier Exit",
        parameters={"period": 22, "multiplier": 3.0, "use_close": True},
        required_columns=["high", "low", "close"],
        output_columns=["chandelier_long", "chandelier_short"],
    )

    # Mass Index Volatility
    registry.register(
        name="mass_index_vol",
        func=VolatilityIndicators.mass_index_volatility,
        category="volatility",
        description="Mass Index (Volatility)",
        parameters={"period": 9, "sum_period": 25},
        required_columns=["high", "low"],
        output_columns=["mass_index_vol"],
    )

    # Garman-Klass Volatility
    registry.register(
        name="garman_klass_vol",
        func=VolatilityIndicators.garman_klass_volatility,
        category="volatility",
        description="Garman-Klass Volatility",
        parameters={"period": 20},
        required_columns=["open", "high", "low", "close"],
        output_columns=["garman_klass_vol"],
    )


# Auto-register on import
register_volatility_indicators()
