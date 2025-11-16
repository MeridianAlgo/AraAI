"""
Performance Optimization Demo
Demonstrates GPU acceleration, model quantization, ONNX export, caching, and scaling
"""

import numpy as np
import asyncio
from pathlib import Path

# Import performance utilities
from ara.utils.performance import (
    GPUAccelerator,
    ModelQuantizer,
    ONNXExporter,
    BatchProcessor,
    ParallelFeatureCalculator
)
from ara.utils.cache_optimizer import (
    CacheHitRateMonitor,
    IntelligentCacheWarmer,
    LazyLoader
)
from ara.utils.scaling import (
    StatelessAPIHandler,
    DistributedCache,
    LoadBalancer
)
from ara.utils.performance_integration import (
    get_performance_manager,
    get_scaling_manager
)
from ara.data.cache import CacheManager


def demo_gpu_acceleration():
    """Demonstrate GPU acceleration"""
    print("\n" + "="*60)
    print("GPU ACCELERATION DEMO")
    print("="*60)
    
    gpu = GPUAccelerator()
    
    print(f"\nGPU Device: {gpu.device}")
    print(f"Device Name: {gpu.device_name}")
    print(f"GPU Available: {gpu.is_available}")
    
    if gpu.is_available:
        print(f"\nMixed Precision Enabled: {gpu.enable_mixed_precision()}")
        
        # Get memory stats
        mem_stats = gpu.get_memory_stats()
        print(f"\nMemory Stats:")
        for key, value in mem_stats.items():
            print(f"  {key}: {value}")


def demo_model_quantization():
    """Demonstrate model quantization"""
    print("\n" + "="*60)
    print("MODEL QUANTIZATION DEMO")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(100, 50)
                self.fc2 = nn.Linear(50, 10)
                self.fc3 = nn.Linear(10, 1)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)
        
        model = SimpleModel()
        
        # Measure original size
        original_size = ModelQuantizer.measure_model_size(model)
        print(f"\nOriginal Model:")
        print(f"  Parameters: {original_size['parameter_count']:,}")
        print(f"  Size: {original_size['total_size_mb']:.2f} MB")
        
        # Quantize to FP16
        model_fp16 = ModelQuantizer.quantize_to_fp16(model)
        fp16_size = ModelQuantizer.measure_model_size(model_fp16)
        print(f"\nFP16 Quantized Model:")
        print(f"  Size: {fp16_size['total_size_mb']:.2f} MB")
        print(f"  Reduction: {(1 - fp16_size['total_size_mb']/original_size['total_size_mb'])*100:.1f}%")
        
        # Quantize to INT8
        model_int8 = ModelQuantizer.quantize_to_int8(model)
        int8_size = ModelQuantizer.measure_model_size(model_int8)
        print(f"\nINT8 Quantized Model:")
        print(f"  Size: {int8_size['total_size_mb']:.2f} MB")
        print(f"  Reduction: {(1 - int8_size['total_size_mb']/original_size['total_size_mb'])*100:.1f}%")
        
    except ImportError:
        print("\nPyTorch not available, skipping quantization demo")


def demo_onnx_export():
    """Demonstrate ONNX export"""
    print("\n" + "="*60)
    print("ONNX EXPORT DEMO")
    print("="*60)
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleModel()
        model.eval()
        
        # Export to ONNX
        output_path = Path("models/demo_model.onnx")
        success = ONNXExporter.export_to_onnx(
            model,
            input_shape=(1, 10),
            output_path=output_path
        )
        
        if success:
            print(f"\nModel exported to: {output_path}")
            print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
            
            # Try to load and run inference
            try:
                session = ONNXExporter.load_onnx_model(output_path)
                
                # Test inference
                test_input = np.random.randn(1, 10).astype(np.float32)
                output = ONNXExporter.onnx_inference(session, test_input)
                
                print(f"\nONNX Inference Test:")
                print(f"  Input shape: {test_input.shape}")
                print(f"  Output shape: {output.shape}")
                print(f"  Output value: {output[0][0]:.4f}")
                
            except Exception as e:
                print(f"\nONNX inference test failed: {e}")
        
    except ImportError:
        print("\nPyTorch not available, skipping ONNX demo")


def demo_batch_processing():
    """Demonstrate batch processing"""
    print("\n" + "="*60)
    print("BATCH PROCESSING DEMO")
    print("="*60)
    
    # Create batch processor
    batch_processor = BatchProcessor(batch_size=10)
    
    # Create dummy data
    items = list(range(100))
    
    # Define processing function
    def process_batch(batch):
        # Simulate some processing
        return [x * 2 for x in batch]
    
    # Process in batches
    print(f"\nProcessing {len(items)} items in batches of {batch_processor.batch_size}...")
    results = batch_processor.process_batch(items, process_batch)
    
    print(f"Results: {results[:10]}... (showing first 10)")
    
    # Get stats
    stats = batch_processor.get_stats()
    print(f"\nBatch Processing Stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def demo_parallel_features():
    """Demonstrate parallel feature calculation"""
    print("\n" + "="*60)
    print("PARALLEL FEATURE CALCULATION DEMO")
    print("="*60)
    
    # Create parallel calculator
    calculator = ParallelFeatureCalculator(n_workers=4)
    
    # Create dummy data
    data_list = [np.random.randn(100, 10) for _ in range(20)]
    
    # Define feature function
    def calculate_features(data):
        # Simulate feature calculation
        return {
            'mean': data.mean(),
            'std': data.std(),
            'max': data.max(),
            'min': data.min()
        }
    
    # Calculate features in parallel
    print(f"\nCalculating features for {len(data_list)} datasets...")
    results = calculator.calculate_features_parallel(data_list, calculate_features)
    
    print(f"\nSample result:")
    print(results[0])


def demo_cache_optimization():
    """Demonstrate cache optimization"""
    print("\n" + "="*60)
    print("CACHE OPTIMIZATION DEMO")
    print("="*60)
    
    # Create cache manager
    cache_manager = CacheManager(
        l1_size_mb=50,
        l1_ttl=60,
        l2_enabled=False  # Disable Redis for demo
    )
    
    # Create cache monitor
    monitor = CacheHitRateMonitor()
    
    # Simulate cache operations
    print("\nSimulating cache operations...")
    
    for i in range(100):
        key = f"key_{i % 20}"  # 20 unique keys, repeated
        
        # Try to get from cache
        value = cache_manager.get(key)
        
        if value is None:
            # Cache miss
            monitor.record_miss(key)
            # Simulate fetching data
            value = f"value_{i}"
            cache_manager.set(key, value)
        else:
            # Cache hit
            monitor.record_hit(key)
    
    # Get cache stats
    cache_stats = cache_manager.get_stats()
    print(f"\nCache Stats:")
    print(f"  L1 Hit Rate: {cache_stats['l1']['hit_rate']:.2%}")
    print(f"  L1 Size: {cache_stats['l1']['size_mb']:.2f} MB")
    print(f"  L1 Items: {cache_stats['l1']['items']}")
    
    # Get monitor stats
    print(f"\nMonitor Stats:")
    print(f"  Overall Hit Rate: {monitor.get_hit_rate():.2%}")
    
    # Get top keys
    top_keys = monitor.get_top_keys(n=5, by='hits')
    print(f"\nTop 5 Keys by Hits:")
    for key, stats in top_keys:
        print(f"  {key}: {stats['hits']} hits, {stats['hit_rate']:.2%} hit rate")
    
    # Get recommendations
    recommendations = monitor.get_optimization_recommendations()
    if recommendations:
        print(f"\nOptimization Recommendations:")
        for rec in recommendations:
            print(f"  [{rec['severity'].upper()}] {rec['message']}")


async def demo_cache_warming():
    """Demonstrate intelligent cache warming"""
    print("\n" + "="*60)
    print("CACHE WARMING DEMO")
    print("="*60)
    
    # Create cache manager and monitor
    cache_manager = CacheManager(l2_enabled=False)
    monitor = CacheHitRateMonitor()
    
    # Create cache warmer
    warmer = IntelligentCacheWarmer(cache_manager, monitor)
    
    # Register warming tasks
    async def fetch_popular_data():
        # Simulate fetching popular data
        await asyncio.sleep(0.1)
        return {"data": "popular_data"}
    
    warmer.register_warming_task(
        "popular_key",
        fetch_popular_data,
        priority=10
    )
    
    print("\nRegistered cache warming task")
    
    # Warm cache once
    await warmer._warm_cache()
    
    # Check if data is in cache
    value = cache_manager.get("popular_key")
    print(f"\nCache warming result: {value}")
    
    # Get warming stats
    stats = warmer.get_warming_stats()
    print(f"\nWarming Stats:")
    print(f"  Enabled: {stats['enabled']}")
    print(f"  Task Count: {stats['task_count']}")


def demo_lazy_loading():
    """Demonstrate lazy loading"""
    print("\n" + "="*60)
    print("LAZY LOADING DEMO")
    print("="*60)
    
    # Create lazy loader
    loader = LazyLoader()
    
    # Register loaders
    def load_model_a():
        print("  Loading Model A...")
        return {"name": "Model A", "size": "100MB"}
    
    def load_model_b():
        print("  Loading Model B...")
        return {"name": "Model B", "size": "200MB"}
    
    loader.register("model_a", load_model_a)
    loader.register("model_b", load_model_b)
    
    print("\nRegistered 2 models for lazy loading")
    print(f"Loaded: {loader.get_stats()['loaded_count']}")
    
    # Access model A (will trigger loading)
    print("\nAccessing Model A...")
    model_a = loader.get("model_a")
    print(f"Got: {model_a}")
    print(f"Loaded: {loader.get_stats()['loaded_count']}")
    
    # Access model A again (already loaded)
    print("\nAccessing Model A again...")
    model_a = loader.get("model_a")
    print(f"Got: {model_a} (from cache)")
    
    # Unload to free memory
    print("\nUnloading Model A...")
    loader.unload("model_a")
    print(f"Loaded: {loader.get_stats()['loaded_count']}")


def demo_load_balancing():
    """Demonstrate load balancing"""
    print("\n" + "="*60)
    print("LOAD BALANCING DEMO")
    print("="*60)
    
    # Create load balancer
    lb = LoadBalancer(strategy='round_robin')
    
    # Define backends
    backends = [
        "http://server1:8000",
        "http://server2:8000",
        "http://server3:8000"
    ]
    
    print(f"\nBackends: {backends}")
    print(f"Strategy: {lb.strategy}")
    
    # Simulate requests
    print("\nSimulating 10 requests:")
    for i in range(10):
        backend = lb.select_backend(backends)
        print(f"  Request {i+1} -> {backend}")


def demo_performance_manager():
    """Demonstrate unified performance manager"""
    print("\n" + "="*60)
    print("PERFORMANCE MANAGER DEMO")
    print("="*60)
    
    # Get performance manager
    config = {
        'batch_size': 32,
        'n_workers': 4,
        'load_balancer_strategy': 'round_robin'
    }
    
    pm = get_performance_manager(config)
    
    print("\nPerformance Manager initialized")
    print(f"GPU Available: {pm.gpu_accelerator.is_available}")
    print(f"Batch Size: {pm.batch_processor.batch_size}")
    
    # Print performance report
    pm.print_performance_report()


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print(" "*15 + "PERFORMANCE OPTIMIZATION DEMOS")
    print("="*70)
    
    # Run demos
    demo_gpu_acceleration()
    demo_model_quantization()
    demo_onnx_export()
    demo_batch_processing()
    demo_parallel_features()
    demo_cache_optimization()
    
    # Async demos
    asyncio.run(demo_cache_warming())
    
    demo_lazy_loading()
    demo_load_balancing()
    demo_performance_manager()
    
    print("\n" + "="*70)
    print(" "*20 + "ALL DEMOS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
