"""
Tests for data aggregator, caching, and validation
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ara.data import (
    DataAggregator,
    CacheManager,
    LRUCache,
    DataValidator,
    DataCleaner,
    ValidationConfig,
    ImputationStrategy,
    OutlierMethod,
    RateLimiter
)
from ara.core.interfaces import IDataProvider, AssetType
from ara.core.exceptions import DataProviderError, ValidationError


# Mock provider for testing
class MockProvider(IDataProvider):
    """Mock data provider for testing"""
    
    def __init__(self, name: str, should_fail: bool = False, latency: float = 0.1):
        self.name = name
        self.should_fail = should_fail
        self.latency = latency
        self.call_count = 0
    
    async def fetch_historical(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        self.call_count += 1
        await asyncio.sleep(self.latency)
        
        if self.should_fail:
            raise DataProviderError(f"{self.name} failed")
        
        # Generate mock data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        data = {
            'Open': np.random.uniform(100, 110, 30),
            'High': np.random.uniform(110, 120, 30),
            'Low': np.random.uniform(90, 100, 30),
            'Close': np.random.uniform(100, 110, 30),
            'Volume': np.random.uniform(1000, 2000, 30)
        }
        df = pd.DataFrame(data, index=dates)
        df.index.name = 'Date'
        return df
    
    async def fetch_realtime(self, symbol: str) -> dict:
        self.call_count += 1
        await asyncio.sleep(self.latency)
        
        if self.should_fail:
            raise DataProviderError(f"{self.name} failed")
        
        return {
            'symbol': symbol,
            'price': 100.0,
            'volume': 1000.0,
            'timestamp': datetime.now()
        }
    
    async def stream_data(self, symbol: str, callback):
        pass
    
    def get_provider_name(self) -> str:
        return self.name
    
    def get_asset_type(self) -> AssetType:
        return AssetType.CRYPTO
    
    def get_supported_symbols(self) -> list:
        return ['BTC/USDT', 'ETH/USDT']


class TestLRUCache:
    """Test LRU cache functionality"""
    
    def test_basic_operations(self):
        """Test basic get/set operations"""
        cache = LRUCache(max_size=10, default_ttl=60)
        
        # Set and get
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        # Get non-existent key
        assert cache.get('key2') is None
    
    def test_expiration(self):
        """Test TTL expiration"""
        cache = LRUCache(max_size=10, default_ttl=1)
        
        cache.set('key1', 'value1', ttl=1)
        assert cache.get('key1') == 'value1'
        
        # Wait for expiration
        import time
        time.sleep(1.1)
        assert cache.get('key1') is None
    
    def test_lru_eviction(self):
        """Test LRU eviction"""
        cache = LRUCache(max_size=1, default_ttl=60)  # 1MB max
        
        # Add items until eviction
        large_value = 'x' * (1024 * 1024)  # 1MB
        cache.set('key1', large_value)
        
        # This should trigger eviction
        cache.set('key2', large_value)
        
        # key1 should be evicted
        assert cache.get('key1') is None
        assert cache.get('key2') == large_value
    
    def test_stats(self):
        """Test cache statistics"""
        cache = LRUCache(max_size=10, default_ttl=60)
        
        cache.set('key1', 'value1')
        cache.get('key1')  # Hit
        cache.get('key2')  # Miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5


class TestCacheManager:
    """Test cache manager functionality"""
    
    def test_l1_cache(self):
        """Test L1 cache operations"""
        manager = CacheManager(l1_size_mb=10, l2_enabled=False)
        
        manager.set('key1', 'value1')
        assert manager.get('key1') == 'value1'
    
    def test_cache_miss(self):
        """Test cache miss"""
        manager = CacheManager(l2_enabled=False)
        assert manager.get('nonexistent') is None
    
    def test_clear(self):
        """Test cache clearing"""
        manager = CacheManager(l2_enabled=False)
        
        manager.set('key1', 'value1')
        manager.clear()
        assert manager.get('key1') is None


class TestDataValidator:
    """Test data validation functionality"""
    
    def test_valid_data(self):
        """Test validation of good data"""
        # Create valid data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        data = {
            'Open': np.random.uniform(100, 110, 50),
            'High': np.random.uniform(110, 120, 50),
            'Low': np.random.uniform(90, 100, 50),
            'Close': np.random.uniform(100, 110, 50),
            'Volume': np.random.uniform(1000, 2000, 50)
        }
        df = pd.DataFrame(data, index=dates)
        
        validator = DataValidator()
        report = validator.validate(df)
        
        assert report.quality_score > 0.7
        assert report.passed_validation
    
    def test_missing_data_detection(self):
        """Test missing data detection"""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        data = {
            'Open': [100.0] * 50,
            'High': [110.0] * 50,
            'Low': [90.0] * 50,
            'Close': [100.0] * 50,
            'Volume': [1000.0] * 50
        }
        df = pd.DataFrame(data, index=dates)
        
        # Add missing values
        df.loc[df.index[0:10], 'Close'] = np.nan
        
        validator = DataValidator()
        report = validator.validate(df)
        
        assert 'Close' in report.missing_data
        assert report.missing_data['Close'] == 10
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        data = {
            'Open': [100.0] * 50,
            'High': [110.0] * 50,
            'Low': [90.0] * 50,
            'Close': [100.0] * 50,
            'Volume': [1000.0] * 50
        }
        df = pd.DataFrame(data, index=dates)
        
        # Add outliers
        df.loc[df.index[0], 'Close'] = 1000.0  # Extreme outlier
        
        config = ValidationConfig(outlier_method=OutlierMethod.IQR)
        validator = DataValidator(config)
        report = validator.validate(df)
        
        assert 'Close' in report.outliers_detected


class TestDataCleaner:
    """Test data cleaning functionality"""
    
    def test_missing_data_imputation(self):
        """Test missing data imputation"""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        data = {
            'Open': [100.0] * 50,
            'High': [110.0] * 50,
            'Low': [90.0] * 50,
            'Close': [100.0] * 50,
            'Volume': [1000.0] * 50
        }
        df = pd.DataFrame(data, index=dates)
        
        # Add missing values
        df.loc[df.index[10:15], 'Close'] = np.nan
        
        config = ValidationConfig(imputation_strategy=ImputationStrategy.LINEAR_INTERPOLATE)
        cleaner = DataCleaner(config)
        
        df_clean, report = cleaner.clean(df)
        
        # Check that missing values are filled
        assert df_clean['Close'].isna().sum() == 0
    
    def test_consistency_fixing(self):
        """Test consistency issue fixing"""
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        data = {
            'Open': [100.0] * 10,
            'High': [90.0] * 10,  # Invalid: High < Low
            'Low': [110.0] * 10,
            'Close': [100.0] * 10,
            'Volume': [1000.0] * 10
        }
        df = pd.DataFrame(data, index=dates)
        
        cleaner = DataCleaner()
        df_clean, report = cleaner.clean(df)
        
        # Check that High >= Low
        assert (df_clean['High'] >= df_clean['Low']).all()


class TestRateLimiter:
    """Test rate limiter functionality"""
    
    def test_rate_limiting(self):
        """Test that rate limiting works"""
        async def run_test():
            import time
            limiter = RateLimiter(requests_per_second=2.0, burst_size=2)
            
            # Make 4 requests (burst allows 2, then rate limiting kicks in)
            start = time.time()
            
            await limiter.acquire()  # Immediate (burst)
            await limiter.acquire()  # Immediate (burst)
            await limiter.acquire()  # Wait ~0.5s
            await limiter.acquire()  # Wait ~0.5s
            
            elapsed = time.time() - start
            
            # Should take at least 0.8 seconds (2 immediate + 2 at 2/sec)
            assert elapsed >= 0.8
        
        asyncio.run(run_test())


class TestDataAggregator:
    """Test data aggregator functionality"""
    
    def test_basic_fetch(self):
        """Test basic data fetching"""
        async def run_test():
            provider = MockProvider('test_provider')
            aggregator = DataAggregator(
                providers=[provider],
                cache_manager=CacheManager(l2_enabled=False)
            )
            
            df = await aggregator.fetch_historical('BTC/USDT', period='1mo')
            
            assert len(df) > 0
            assert provider.call_count == 1
        
        asyncio.run(run_test())
    
    def test_failover(self):
        """Test automatic failover"""
        async def run_test():
            provider1 = MockProvider('provider1', should_fail=True)
            provider2 = MockProvider('provider2', should_fail=False)
            
            aggregator = DataAggregator(
                providers=[provider1, provider2],
                enable_failover=True,
                cache_manager=CacheManager(l2_enabled=False)
            )
            
            df = await aggregator.fetch_historical('BTC/USDT', period='1mo')
            
            assert len(df) > 0
            assert provider1.call_count == 1  # Tried first
            assert provider2.call_count == 1  # Succeeded
        
        asyncio.run(run_test())
    
    def test_all_providers_fail(self):
        """Test when all providers fail"""
        async def run_test():
            provider1 = MockProvider('provider1', should_fail=True)
            provider2 = MockProvider('provider2', should_fail=True)
            
            aggregator = DataAggregator(
                providers=[provider1, provider2],
                enable_failover=True,
                cache_manager=CacheManager(l2_enabled=False)
            )
            
            with pytest.raises(DataProviderError):
                await aggregator.fetch_historical('BTC/USDT', period='1mo')
        
        asyncio.run(run_test())
    
    def test_caching(self):
        """Test caching functionality"""
        async def run_test():
            provider = MockProvider('test_provider')
            aggregator = DataAggregator(
                providers=[provider],
                cache_manager=CacheManager(l2_enabled=False)
            )
            
            # First fetch
            df1 = await aggregator.fetch_historical('BTC/USDT', period='1mo', use_cache=True)
            assert provider.call_count == 1
            
            # Second fetch (should use cache)
            df2 = await aggregator.fetch_historical('BTC/USDT', period='1mo', use_cache=True)
            assert provider.call_count == 1  # No additional call
        
        asyncio.run(run_test())
    
    def test_realtime_fetch(self):
        """Test real-time data fetching"""
        async def run_test():
            provider = MockProvider('test_provider')
            aggregator = DataAggregator(
                providers=[provider],
                cache_manager=CacheManager(l2_enabled=False)
            )
            
            data = await aggregator.fetch_realtime('BTC/USDT')
            
            assert data['symbol'] == 'BTC/USDT'
            assert data['price'] > 0
        
        asyncio.run(run_test())
    
    def test_provider_stats(self):
        """Test provider statistics tracking"""
        async def run_test():
            provider = MockProvider('test_provider')
            aggregator = DataAggregator(
                providers=[provider],
                cache_manager=CacheManager(l2_enabled=False)
            )
            
            # Make some requests
            await aggregator.fetch_historical('BTC/USDT', period='1mo', use_cache=False)
            await aggregator.fetch_historical('ETH/USDT', period='1mo', use_cache=False)
            
            stats = aggregator.get_provider_stats()
            
            assert len(stats) == 1
            assert stats[0]['provider'] == 'test_provider'
            assert stats[0]['requests'] == 2
            assert stats[0]['successes'] == 2
            assert stats[0]['success_rate'] == 1.0
        
        asyncio.run(run_test())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
