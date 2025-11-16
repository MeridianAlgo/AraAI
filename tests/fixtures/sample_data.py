"""
Sample data generators for testing.

Provides realistic market data for different asset types and scenarios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def generate_stock_data(
    symbol: str = "AAPL",
    days: int = 252,
    start_price: float = 150.0,
    volatility: float = 0.02,
    trend: float = 0.001,
    start_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Generate realistic stock market data.
    
    Args:
        symbol: Stock symbol
        days: Number of trading days
        start_price: Starting price
        volatility: Daily volatility (std dev)
        trend: Daily trend (drift)
        start_date: Starting date (defaults to 1 year ago)
        
    Returns:
        DataFrame with OHLCV data
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
        
    # Generate dates (skip weekends)
    dates = []
    current_date = start_date
    while len(dates) < days:
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            dates.append(current_date)
        current_date += timedelta(days=1)
        
    # Generate price series with geometric Brownian motion
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(trend, volatility, days)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Add some intraday volatility
        daily_vol = volatility * 0.5
        high = close * (1 + abs(np.random.normal(0, daily_vol)))
        low = close * (1 - abs(np.random.normal(0, daily_vol)))
        open_price = prices[i-1] if i > 0 else start_price
        
        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume (higher on volatile days)
        base_volume = 50000000
        volume_factor = 1 + abs(returns[i]) * 10
        volume = int(base_volume * volume_factor)
        
        data.append({
            "date": date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": symbol
        })
        
    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)
    return df


def generate_crypto_data(
    symbol: str = "BTC-USD",
    days: int = 365,
    start_price: float = 40000.0,
    volatility: float = 0.04,
    trend: float = 0.002,
    start_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Generate realistic cryptocurrency data (24/7 market).
    
    Args:
        symbol: Crypto symbol
        days: Number of days
        start_price: Starting price
        volatility: Daily volatility (higher than stocks)
        trend: Daily trend
        start_date: Starting date
        
    Returns:
        DataFrame with OHLCV data
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
        
    # Crypto trades 24/7, so include all days
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    # Generate price series with higher volatility
    np.random.seed(43)
    returns = np.random.normal(trend, volatility, days)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        daily_vol = volatility * 0.5
        high = close * (1 + abs(np.random.normal(0, daily_vol)))
        low = close * (1 - abs(np.random.normal(0, daily_vol)))
        open_price = prices[i-1] if i > 0 else start_price
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Crypto has higher volume variability
        base_volume = 1000000000
        volume_factor = 1 + abs(returns[i]) * 20
        volume = int(base_volume * volume_factor)
        
        data.append({
            "date": date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": symbol
        })
        
    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)
    return df


def generate_forex_data(
    symbol: str = "EURUSD",
    days: int = 252,
    start_price: float = 1.10,
    volatility: float = 0.005,
    trend: float = 0.0001,
    start_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Generate realistic forex data.
    
    Args:
        symbol: Forex pair
        days: Number of trading days
        start_price: Starting exchange rate
        volatility: Daily volatility (lower than stocks)
        trend: Daily trend
        start_date: Starting date
        
    Returns:
        DataFrame with OHLC data
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
        
    # Forex trades 24/5 (skip weekends)
    dates = []
    current_date = start_date
    while len(dates) < days:
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)
        
    # Generate price series with low volatility
    np.random.seed(44)
    returns = np.random.normal(trend, volatility, days)
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        daily_vol = volatility * 0.5
        high = close * (1 + abs(np.random.normal(0, daily_vol)))
        low = close * (1 - abs(np.random.normal(0, daily_vol)))
        open_price = prices[i-1] if i > 0 else start_price
        
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Forex doesn't have traditional volume
        volume = 0
        
        data.append({
            "date": date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": symbol
        })
        
    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)
    return df


def generate_features_data(
    days: int = 100,
    n_features: int = 50
) -> pd.DataFrame:
    """
    Generate sample feature data for ML models.
    
    Args:
        days: Number of samples
        n_features: Number of features
        
    Returns:
        DataFrame with feature data
    """
    np.random.seed(45)
    
    # Generate correlated features
    data = {}
    for i in range(n_features):
        if i == 0:
            # Base feature
            data[f"feature_{i}"] = np.random.randn(days)
        else:
            # Correlated with previous features
            correlation = 0.3
            data[f"feature_{i}"] = (
                correlation * data[f"feature_{i-1}"] +
                (1 - correlation) * np.random.randn(days)
            )
            
    return pd.DataFrame(data)


def generate_market_regime_data(
    regime: str = "bull",
    days: int = 100,
    start_price: float = 100.0
) -> pd.DataFrame:
    """
    Generate data for specific market regime.
    
    Args:
        regime: Market regime (bull, bear, sideways, high_volatility)
        days: Number of days
        start_price: Starting price
        
    Returns:
        DataFrame with OHLCV data
    """
    regime_params = {
        "bull": {"trend": 0.003, "volatility": 0.015},
        "bear": {"trend": -0.003, "volatility": 0.020},
        "sideways": {"trend": 0.0, "volatility": 0.010},
        "high_volatility": {"trend": 0.0, "volatility": 0.050}
    }
    
    params = regime_params.get(regime, regime_params["sideways"])
    
    return generate_stock_data(
        symbol="TEST",
        days=days,
        start_price=start_price,
        trend=params["trend"],
        volatility=params["volatility"]
    )


def generate_prediction_scenarios() -> dict:
    """
    Generate various prediction scenarios for testing.
    
    Returns:
        Dictionary of scenario name to data
    """
    scenarios = {
        "strong_uptrend": generate_stock_data(
            days=100, trend=0.005, volatility=0.015
        ),
        "strong_downtrend": generate_stock_data(
            days=100, trend=-0.005, volatility=0.015
        ),
        "high_volatility": generate_stock_data(
            days=100, trend=0.0, volatility=0.05
        ),
        "low_volatility": generate_stock_data(
            days=100, trend=0.0, volatility=0.005
        ),
        "trend_reversal": pd.concat([
            generate_stock_data(days=50, trend=0.003, volatility=0.015),
            generate_stock_data(days=50, trend=-0.003, volatility=0.015)
        ]),
    }
    
    return scenarios
