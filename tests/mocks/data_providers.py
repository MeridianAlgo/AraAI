"""
Mock data providers for testing.

These mocks provide fast, deterministic data for testing without
requiring network access or external API calls.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from ara.core.interfaces import IDataProvider


class MockDataProvider(IDataProvider):
    """Mock data provider for testing."""
    
    def __init__(self):
        self._data: Dict[str, pd.DataFrame] = {}
        self._realtime_data: Dict[str, Dict] = {}
        self._call_count = 0
        self._should_fail = False
        
    def get_provider_name(self) -> str:
        """Get provider name."""
        return "MockProvider"
        
    def get_asset_type(self) -> str:
        """Get asset type."""
        return "stock"
        
    def set_data(self, symbol: str, data: pd.DataFrame):
        """Set mock data for a symbol."""
        self._data[symbol] = data.copy()
        
    def set_realtime_data(self, symbol: str, data: Dict):
        """Set mock real-time data."""
        self._realtime_data[symbol] = data
        
    def set_should_fail(self, should_fail: bool):
        """Configure whether the provider should fail."""
        self._should_fail = should_fail
        
    async def fetch_historical(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Fetch historical data (mock)."""
        self._call_count += 1
        
        if self._should_fail:
            raise Exception(f"Mock provider failed for {symbol}")
            
        if symbol not in self._data:
            raise ValueError(f"No mock data for symbol: {symbol}")
            
        return self._data[symbol].copy()
        
    async def fetch_realtime(self, symbol: str) -> Dict:
        """Fetch real-time data (mock)."""
        self._call_count += 1
        
        if self._should_fail:
            raise Exception(f"Mock provider failed for {symbol}")
            
        if symbol in self._realtime_data:
            return self._realtime_data[symbol].copy()
            
        # Return last row of historical data if available
        if symbol in self._data:
            last_row = self._data[symbol].iloc[-1]
            return {
                "symbol": symbol,
                "price": float(last_row["close"]),
                "timestamp": datetime.now(),
                "volume": float(last_row["volume"])
            }
            
        raise ValueError(f"No mock data for symbol: {symbol}")
        
    async def stream_data(self, symbol: str, callback: Callable):
        """Stream data (mock - not implemented)."""
        pass
        
    def get_supported_symbols(self) -> List[str]:
        """Return list of supported symbols."""
        return list(self._data.keys())
        
    def get_call_count(self) -> int:
        """Get number of times provider was called."""
        return self._call_count
        
    def reset_call_count(self):
        """Reset call counter."""
        self._call_count = 0


class MockCryptoProvider(MockDataProvider):
    """Mock cryptocurrency data provider."""
    
    def __init__(self):
        super().__init__()
        self._exchange_name = "MockExchange"
        self._onchain_metrics = {}
        
    def set_onchain_metrics(self, symbol: str, metrics: Dict):
        """Set mock on-chain metrics."""
        self._onchain_metrics[symbol] = metrics
        
    async def fetch_onchain_metrics(self, symbol: str) -> Dict:
        """Fetch on-chain metrics (mock)."""
        if symbol in self._onchain_metrics:
            return self._onchain_metrics[symbol].copy()
            
        # Return default mock metrics
        return {
            "active_addresses": 1000000,
            "transaction_volume": 50000000000,
            "hash_rate": 200000000,
            "nvt_ratio": 50.0,
            "mvrv_ratio": 2.5
        }


class MockForexProvider(MockDataProvider):
    """Mock forex data provider."""
    
    def __init__(self):
        super().__init__()
        
    async def fetch_exchange_rate(self, base: str, quote: str) -> float:
        """Fetch exchange rate (mock)."""
        symbol = f"{base}{quote}"
        if symbol in self._data:
            return float(self._data[symbol].iloc[-1]["close"])
        return 1.0


class FailingDataProvider(IDataProvider):
    """Data provider that always fails - for testing error handling."""
    
    async def fetch_historical(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        raise ConnectionError("Mock connection error")
        
    async def fetch_realtime(self, symbol: str) -> Dict:
        raise ConnectionError("Mock connection error")
        
    async def stream_data(self, symbol: str, callback: Callable):
        raise ConnectionError("Mock connection error")
        
    def get_supported_symbols(self) -> List[str]:
        return []


class SlowDataProvider(MockDataProvider):
    """Data provider with artificial delays - for testing timeouts."""
    
    def __init__(self, delay_seconds: float = 5.0):
        super().__init__()
        self.delay_seconds = delay_seconds
        
    async def fetch_historical(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        import asyncio
        await asyncio.sleep(self.delay_seconds)
        return await super().fetch_historical(symbol, period)
