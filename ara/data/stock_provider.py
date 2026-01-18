"""
Yahoo Finance data provider for stocks and forex
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Any, List
from ara.data.base_provider import BaseDataProvider
from ara.core.interfaces import AssetType
from ara.core.exceptions import DataProviderError

class YahooFinanceProvider(BaseDataProvider):
    """Data provider using yfinance library"""
    
    def __init__(self, asset_type: AssetType = AssetType.STOCK):
        super().__init__(name="yfinance", asset_type=asset_type)

    async def fetch_historical(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance"""
        try:
            # Normalize symbol for YF (e.g. EURUSD -> EURUSD=X)
            yf_symbol = symbol
            if self.asset_type == AssetType.FOREX and "=" not in symbol:
                yf_symbol = f"{symbol}=X"
            
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise DataProviderError(f"No data found for {symbol}")
                
            # Standardize columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            
            # Map 'adj close' to 'close' if it exists and standard 'close' doesn't or as preferred
            if 'adj close' in df.columns:
                df['close'] = df['adj close']
                
            # Keep only standard columns to avoid issues with Dividends/Splits
            standard_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in standard_cols if c in df.columns]]
            
            return df
        except Exception as e:
            raise DataProviderError(f"yfinance fetch failed for {symbol}: {str(e)}")

    async def fetch_realtime(self, symbol: str) -> Dict[str, Any]:
        """Fetch current price"""
        df = await self.fetch_historical(symbol, period="1d")
        if df.empty:
            raise DataProviderError(f"Could not get realtime data for {symbol}")
            
        last_row = df.iloc[-1]
        return {
            "symbol": symbol,
            "price": float(last_row["close"]),
            "timestamp": df.index[-1]
        }
