"""
Performance Integration Module
Integrates all performance optimizations into a unified system
"""

from typing import Optional, Dict, Any
from pathlib import Path

from ara.utils import get_logger
from ara.utils.performance import (
    GPUAccelerator,
    ModelQuantizer,
    ONNXExporter,
    BatchProcessor,
    ParallelFeatureCalculator,
    PerformanceProfiler
)
from ara.utils.cache_optimizer import (
    CacheHitRateMonitor,
    IntelligentCacheWarmer,
    ConnectionPool,
    LazyLoader
)
from ara.utils.scaling import (
    StatelessAPIHandler,
    DistributedCache,
    WorkerPool,
    ServiceDiscovery,
    LoadBalancer
)

logger = get_logger(__name__)


class PerformanceManager:
    """
    Unified performance management system
    Coordinates all performance optimization components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance manager
        
        Args:
            config: Configuration dict
        """
        self.config = config or {}
        
        # GPU Acceleration
        self.gpu_accelerator = GPUAccelerator()
        
        # Batch Processing
        batch_size = self.config.get('batch_size', 32)
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        
        # Parallel Processing
        n_workers = self.config.get('n_workers', None)
        self.parallel_calculator = ParallelFeatureCalculator(n_workers=n_workers)
        
        # Performance Profiling
        self.profiler = PerformanceProfiler()
        
        # Cache Monitoring
        self.cache_monitor = CacheHitRateMonitor()
        
        # Lazy Loading
        self.lazy_loader = LazyLoader()
        
        # Load Balancing
        lb_strategy = self.config.get('load_balancer_strategy', 'round_robin')
        self.load_balancer = LoadBalancer(strategy=lb_strategy)
        
        logger.info("Initialized PerformanceManager")
    
    def optimize_model_for_inference(self, model, quantize: str = 'fp16'):
        """
        Optimize model for inference
        
        Args:
            model: PyTorch model
            quantize: Quantization type ('fp16', 'int8', or None)
            
        Returns:
            Optimized model
        """
        logger.info(f"Optimizing model for inference (quantize={quantize})")
        
        # Move to GPU if available
        if self.gpu_accelerator.is_available:
            model = self.gpu_accelerator.move_to_device(model)
        
        # Quantize if requested
        if quantize == 'fp16':
            model = ModelQuantizer.quantize_to_fp16(model)
        elif quantize == 'int8':
            model = ModelQuantizer.quantize_to_int8(model)
        
        # Set to eval mode
        model.eval()
        
        logger.info("Model optimization complete")
        return model
    
    def export_model_to_onnx(
        self,
        model,
        input_shape: tuple,
        output_path: Path
    ) -> bool:
        """
        Export model to ONNX format
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            output_path: Output path
            
        Returns:
            True if successful
        """
        return ONNXExporter.export_to_onnx(
            model,
            input_shape,
            output_path
        )
    
    def batch_predict(self, items: list, predict_func: callable) -> list:
        """
        Perform batch predictions
        
        Args:
            items: List of items to predict
            predict_func: Prediction function
            
        Returns:
            List of predictions
        """
        return self.batch_processor.process_batch(items, predict_func)
    
    def parallel_calculate_features(self, data_list: list, feature_func: callable) -> list:
        """
        Calculate features in parallel
        
        Args:
            data_list: List of data items
            feature_func: Feature calculation function
            
        Returns:
            List of features
        """
        return self.parallel_calculator.calculate_features_parallel(
            data_list,
            feature_func
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics
        
        Returns:
            Dict with all performance stats
        """
        return {
            'gpu': self.gpu_accelerator.get_memory_stats(),
            'batch_processor': self.batch_processor.get_stats(),
            'cache_monitor': {
                'hit_rate': self.cache_monitor.get_hit_rate(),
                'recommendations': self.cache_monitor.get_optimization_recommendations()
            },
            'lazy_loader': self.lazy_loader.get_stats(),
            'load_balancer': self.load_balancer.get_stats(),
            'profiler': self.profiler.get_stats()
        }
    
    def print_performance_report(self) -> None:
        """Print comprehensive performance report"""
        stats = self.get_performance_stats()
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        # GPU Stats
        print("\n[GPU Acceleration]")
        gpu_stats = stats['gpu']
        print(f"  Device: {gpu_stats.get('device', 'N/A')}")
        print(f"  Available: {gpu_stats.get('available', False)}")
        if 'allocated_mb' in gpu_stats:
            print(f"  Memory Allocated: {gpu_stats['allocated_mb']:.2f} MB")
            print(f"  Memory Reserved: {gpu_stats['reserved_mb']:.2f} MB")
        
        # Batch Processing Stats
        print("\n[Batch Processing]")
        batch_stats = stats['batch_processor']
        print(f"  Batch Size: {batch_stats['batch_size']}")
        print(f"  Batches Processed: {batch_stats['batches_processed']}")
        print(f"  Items Processed: {batch_stats['items_processed']}")
        print(f"  Throughput: {batch_stats['throughput']:.2f} items/s")
        
        # Cache Stats
        print("\n[Cache Performance]")
        cache_stats = stats['cache_monitor']
        print(f"  Hit Rate: {cache_stats['hit_rate']:.2%}")
        if cache_stats['recommendations']:
            print("  Recommendations:")
            for rec in cache_stats['recommendations']:
                print(f"    - [{rec['severity'].upper()}] {rec['message']}")
        
        # Lazy Loader Stats
        print("\n[Lazy Loading]")
        lazy_stats = stats['lazy_loader']
        print(f"  Loaded Items: {lazy_stats['loaded_count']}")
        print(f"  Registered Items: {lazy_stats['registered_count']}")
        
        # Load Balancer Stats
        print("\n[Load Balancing]")
        lb_stats = stats['load_balancer']
        print(f"  Strategy: {lb_stats['strategy']}")
        print(f"  Total Connections: {lb_stats['total_connections']}")
        
        # Profiler Stats
        print("\n[Function Profiling]")
        profiler_stats = stats['profiler']
        if profiler_stats:
            for func_name, metrics in sorted(
                profiler_stats.items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )[:5]:  # Top 5
                print(f"  {func_name}:")
                print(f"    Calls: {metrics['count']}")
                print(f"    Total: {metrics['total']:.3f}s")
                print(f"    Mean: {metrics['mean']:.3f}s")
        
        print("\n" + "="*60)


class ScalingManager:
    """
    Unified scaling management system
    Coordinates horizontal scaling components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize scaling manager
        
        Args:
            config: Configuration dict
        """
        self.config = config or {}
        
        # Stateless API
        self.api_handler = StatelessAPIHandler()
        
        # Distributed Cache
        cluster_nodes = self.config.get('redis_cluster_nodes')
        redis_password = self.config.get('redis_password')
        self.distributed_cache = DistributedCache(
            cluster_nodes=cluster_nodes,
            password=redis_password
        )
        
        # Worker Pool
        broker_url = self.config.get('celery_broker', 'redis://localhost:6379/0')
        backend_url = self.config.get('celery_backend', 'redis://localhost:6379/1')
        max_workers = self.config.get('max_workers', 4)
        self.worker_pool = WorkerPool(
            broker_url=broker_url,
            backend_url=backend_url,
            max_workers=max_workers
        )
        
        # Service Discovery
        registry_url = self.config.get('registry_url', 'redis://localhost:6379/2')
        self.service_discovery = ServiceDiscovery(
            registry_backend='redis',
            registry_url=registry_url
        )
        
        logger.info("Initialized ScalingManager")
    
    def register_service(
        self,
        service_name: str,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a service instance
        
        Args:
            service_name: Service name
            host: Service host
            port: Service port
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        return self.service_discovery.register_service(
            service_name,
            host,
            port,
            metadata
        )
    
    def discover_service(self, service_name: str) -> list:
        """
        Discover service instances
        
        Args:
            service_name: Service name
            
        Returns:
            List of service instances
        """
        return self.service_discovery.discover_service(service_name)
    
    def submit_async_task(self, task_name: str, *args, **kwargs) -> Optional[str]:
        """
        Submit an async task
        
        Args:
            task_name: Task name
            *args: Task arguments
            **kwargs: Task keyword arguments
            
        Returns:
            Task ID
        """
        return self.worker_pool.submit_task(task_name, *args, **kwargs)
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get task result
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds
            
        Returns:
            Task result
        """
        return self.worker_pool.get_task_result(task_id, timeout)
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """
        Get scaling statistics
        
        Returns:
            Dict with scaling stats
        """
        return {
            'distributed_cache': self.distributed_cache.get_cluster_info(),
            'service_discovery': {
                'backend': 'redis',
                'url': self.config.get('registry_url')
            }
        }


# Global instances (can be initialized once and reused)
_performance_manager: Optional[PerformanceManager] = None
_scaling_manager: Optional[ScalingManager] = None


def get_performance_manager(config: Optional[Dict[str, Any]] = None) -> PerformanceManager:
    """
    Get or create global performance manager
    
    Args:
        config: Configuration dict
        
    Returns:
        PerformanceManager instance
    """
    global _performance_manager
    
    if _performance_manager is None:
        _performance_manager = PerformanceManager(config)
    
    return _performance_manager


def get_scaling_manager(config: Optional[Dict[str, Any]] = None) -> ScalingManager:
    """
    Get or create global scaling manager
    
    Args:
        config: Configuration dict
        
    Returns:
        ScalingManager instance
    """
    global _scaling_manager
    
    if _scaling_manager is None:
        _scaling_manager = ScalingManager(config)
    
    return _scaling_manager
