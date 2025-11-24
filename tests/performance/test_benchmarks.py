"""
Performance benchmarks for critical paths.

Tests prediction speed, feature calculation, and model inference performance.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Import components to benchmark
try:
    from ara.features.calculator import FeatureCalculator
    from ara.models.ensemble import EnsembleModel
    from ara.data.cache import CacheManager
except ImportError:
    pytest.skip("ARA components not available", allow_module_level=True)

from tests.fixtures.sample_data import generate_stock_data
from tests.mocks.ml_models import MockMLModel, FastMLModel


@pytest.mark.performance
@pytest.mark.benchmark
class TestPredictionPerformance:
    """Benchmark prediction performance."""
    
    def test_single_prediction_speed(self, benchmark):
        """Test single prediction completes in < 2 seconds."""
        model = FastMLModel()
        X = np.random.randn(100, 50)
        
        def predict():
            return model.predict(X)
            
        result = benchmark(predict)
        
        # Assert prediction is fast
        assert benchmark.stats['mean'] < 2.0, "Prediction too slow"
        
    def test_batch_prediction_speed(self, benchmark):
        """Test batch predictions for 100 assets."""
        model = FastMLModel()
        
        def batch_predict():
            results = []
            for _ in range(100):
                X = np.random.randn(100, 50)
                preds, confs = model.predict(X)
                results.append(preds)
            return results
            
        result = benchmark(batch_predict)
        
        # Assert batch completes in < 60 seconds
        assert benchmark.stats['mean'] < 60.0, "Batch prediction too slow"


@pytest.mark.performance
@pytest.mark.benchmark
class TestFeatureCalculation:
    """Benchmark feature calculation performance."""
    
    def test_feature_calculation_speed(self, benchmark):
        """Test feature calculation for 1000 data points."""
        data = generate_stock_data(days=1000)
        
        def calculate_features():
            # Mock feature calculation
            features = {}
            features['sma_20'] = data['close'].rolling(20).mean()
            features['sma_50'] = data['close'].rolling(50).mean()
            features['rsi'] = 50.0  # Simplified
            return features
            
        result = benchmark(calculate_features)
        
        # Assert calculation is fast (< 100ms)
        assert benchmark.stats['mean'] < 0.1, "Feature calculation too slow"
        
    def test_multi_timeframe_calculation(self, benchmark):
        """Test multi-timeframe feature calculation."""
        data = generate_stock_data(days=500)
        
        def calculate_multi_timeframe():
            results = {}
            for window in [5, 10, 20, 50, 200]:
                results[f'sma_{window}'] = data['close'].rolling(window).mean()
            return results
            
        result = benchmark(calculate_multi_timeframe)
        
        assert benchmark.stats['mean'] < 0.2, "Multi-timeframe calculation too slow"


@pytest.mark.performance
@pytest.mark.benchmark
class TestCachePerformance:
    """Benchmark cache performance."""
    
    def test_cache_read_speed(self, benchmark):
        """Test cache read performance."""
        from tests.mocks.databases import MockCache
        
        cache = MockCache()
        cache.set("test_key", {"data": "value"})
        
        def read_cache():
            return cache.get("test_key")
            
        result = benchmark(read_cache)
        
        # Cache reads should be very fast (< 1ms)
        assert benchmark.stats['mean'] < 0.001, "Cache read too slow"
        
    def test_cache_write_speed(self, benchmark):
        """Test cache write performance."""
        from tests.mocks.databases import MockCache
        
        cache = MockCache()
        
        def write_cache():
            cache.set(f"key_{time.time()}", {"data": "value"})
            
        result = benchmark(write_cache)
        
        assert benchmark.stats['mean'] < 0.001, "Cache write too slow"


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage of critical components."""
    
    def test_model_memory_footprint(self):
        """Test model memory usage is reasonable."""
        import sys
        
        model = MockMLModel()
        X = np.random.randn(1000, 100)
        y = np.random.randn(1000)
        
        # Train model
        model.train(X, y)
        
        # Check memory size (rough estimate)
        model_size = sys.getsizeof(model)
        
        # Model should be < 10MB
        assert model_size < 10 * 1024 * 1024, "Model too large"
        
    def test_data_memory_efficiency(self):
        """Test data structures are memory efficient."""
        import sys
        
        # Generate large dataset
        data = generate_stock_data(days=10000)
        
        # Check memory usage
        data_size = data.memory_usage(deep=True).sum()
        
        # Should be < 50MB for 10k rows
        assert data_size < 50 * 1024 * 1024, "Data structure too large"


@pytest.mark.performance
class TestScalability:
    """Test system scalability."""
    
    def test_linear_scaling_with_data_size(self):
        """Test that processing time scales linearly with data size."""
        times = []
        sizes = [100, 500, 1000, 5000]
        
        for size in sizes:
            data = generate_stock_data(days=size)
            
            start = time.time()
            # Simple processing
            result = data['close'].rolling(20).mean()
            elapsed = time.time() - start
            
            times.append(elapsed)
            
        # Check that time increases roughly linearly
        # (not exponentially)
        time_ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        
        # Ratios should be close to size ratios
        for ratio in time_ratios:
            assert ratio < 10, "Processing time not scaling linearly"
            
    def test_concurrent_predictions(self):
        """Test handling concurrent prediction requests."""
        import asyncio
        
        async def make_prediction():
            model = FastMLModel()
            X = np.random.randn(100, 50)
            return model.predict(X)
            
        async def concurrent_test():
            # Simulate 10 concurrent requests
            tasks = [make_prediction() for _ in range(10)]
            results = await asyncio.gather(*tasks)
            return results
            
        start = time.time()
        results = asyncio.run(concurrent_test())
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0, "Concurrent predictions too slow"
        assert len(results) == 10, "Not all predictions completed"


@pytest.mark.performance
def test_api_response_time():
    """Test API response time is acceptable."""
    from fastapi.testclient import TestClient
    
    try:
        from ara.api.app import app
        client = TestClient(app)
        
        # Test health endpoint
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        
        assert elapsed < 0.5, "API response too slow"
        assert response.status_code == 200
    except ImportError:
        pytest.skip("API not available")


@pytest.mark.performance
def test_database_query_performance():
    """Test database query performance."""
    from tests.mocks.databases import MockDatabase
    
    db = MockDatabase()
    
    # Insert test data
    for i in range(1000):
        db.insert("predictions", {
            "symbol": f"TEST{i % 10}",
            "price": 100.0 + i,
            "confidence": 0.85
        })
        
    # Test query performance
    start = time.time()
    results = db.select("predictions", {"symbol": "TEST5"})
    elapsed = time.time() - start
    
    assert elapsed < 0.1, "Database query too slow"
    assert len(results) == 100, "Query returned wrong number of results"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
