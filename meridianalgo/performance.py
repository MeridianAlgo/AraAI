"""
Performance monitoring and optimization for Ara AI
"""

import time
import psutil
import threading
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """
    Monitor system performance and resource usage
    """
    
    def __init__(self, log_dir=".ara_cache"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.performance_log = self.log_dir / "performance.json"
        self.monitoring = False
        self.monitor_thread = None
        
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'prediction_times': [],
            'cache_hit_rate': 0,
            'model_accuracy': []
        }
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics['cpu_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'value': cpu_percent
                })
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'percent': memory.percent,
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3)
                })
                
                # GPU usage (if available)
                gpu_usage = self._get_gpu_usage()
                if gpu_usage:
                    self.metrics['gpu_usage'].append({
                        'timestamp': datetime.now().isoformat(),
                        'usage': gpu_usage
                    })
                
                # Keep only recent data (last 100 entries)
                for key in ['cpu_usage', 'memory_usage', 'gpu_usage']:
                    if len(self.metrics[key]) > 100:
                        self.metrics[key] = self.metrics[key][-100:]
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)
    
    def _get_gpu_usage(self):
        """Get GPU usage if available"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    'memory_allocated': torch.cuda.memory_allocated() / (1024**3),
                    'memory_reserved': torch.cuda.memory_reserved() / (1024**3),
                    'utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                }
        except Exception:
            pass
        return None
    
    def record_prediction_time(self, symbol, prediction_time, model_type="ensemble"):
        """Record prediction timing"""
        self.metrics['prediction_times'].append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'time_seconds': prediction_time,
            'model_type': model_type
        })
        
        # Keep only recent predictions
        if len(self.metrics['prediction_times']) > 1000:
            self.metrics['prediction_times'] = self.metrics['prediction_times'][-1000:]
    
    def record_accuracy(self, symbol, accuracy, error_rate):
        """Record model accuracy"""
        self.metrics['model_accuracy'].append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'accuracy': accuracy,
            'error_rate': error_rate
        })
        
        # Keep only recent accuracy data
        if len(self.metrics['model_accuracy']) > 500:
            self.metrics['model_accuracy'] = self.metrics['model_accuracy'][-500:]
    
    def get_performance_summary(self):
        """Get performance summary"""
        try:
            summary = {
                'monitoring_active': self.monitoring,
                'data_points': {
                    'cpu_samples': len(self.metrics['cpu_usage']),
                    'memory_samples': len(self.metrics['memory_usage']),
                    'gpu_samples': len(self.metrics['gpu_usage']),
                    'predictions': len(self.metrics['prediction_times']),
                    'accuracy_records': len(self.metrics['model_accuracy'])
                }
            }
            
            # CPU statistics
            if self.metrics['cpu_usage']:
                cpu_values = [entry['value'] for entry in self.metrics['cpu_usage']]
                summary['cpu_stats'] = {
                    'average': sum(cpu_values) / len(cpu_values),
                    'max': max(cpu_values),
                    'min': min(cpu_values),
                    'current': cpu_values[-1] if cpu_values else 0
                }
            
            # Memory statistics
            if self.metrics['memory_usage']:
                memory_values = [entry['percent'] for entry in self.metrics['memory_usage']]
                summary['memory_stats'] = {
                    'average_percent': sum(memory_values) / len(memory_values),
                    'max_percent': max(memory_values),
                    'current_percent': memory_values[-1] if memory_values else 0,
                    'current_available_gb': self.metrics['memory_usage'][-1]['available_gb'] if self.metrics['memory_usage'] else 0
                }
            
            # Prediction timing statistics
            if self.metrics['prediction_times']:
                times = [entry['time_seconds'] for entry in self.metrics['prediction_times']]
                summary['prediction_stats'] = {
                    'average_time': sum(times) / len(times),
                    'fastest': min(times),
                    'slowest': max(times),
                    'total_predictions': len(times)
                }
            
            # Accuracy statistics
            if self.metrics['model_accuracy']:
                accuracies = [entry['accuracy'] for entry in self.metrics['model_accuracy']]
                errors = [entry['error_rate'] for entry in self.metrics['model_accuracy']]
                summary['accuracy_stats'] = {
                    'average_accuracy': sum(accuracies) / len(accuracies),
                    'best_accuracy': max(accuracies),
                    'average_error': sum(errors) / len(errors),
                    'lowest_error': min(errors)
                }
            
            return summary
            
        except Exception as e:
            return {'error': f"Failed to generate summary: {e}"}
    
    def save_performance_log(self):
        """Save performance metrics to file"""
        try:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics,
                'summary': self.get_performance_summary()
            }
            
            with open(self.performance_log, 'w') as f:
                json.dump(log_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Failed to save performance log: {e}")
    
    def load_performance_log(self):
        """Load performance metrics from file"""
        try:
            if self.performance_log.exists():
                with open(self.performance_log, 'r') as f:
                    log_data = json.load(f)
                    self.metrics = log_data.get('metrics', self.metrics)
                    return log_data.get('summary', {})
            return {}
        except Exception as e:
            print(f"Failed to load performance log: {e}")
            return {}

class ResourceOptimizer:
    """
    Optimize resource usage and performance
    """
    
    def __init__(self):
        self.optimization_settings = {
            'max_cpu_usage': 80,  # Maximum CPU usage percentage
            'max_memory_usage': 85,  # Maximum memory usage percentage
            'cache_cleanup_threshold': 90,  # Memory threshold for cache cleanup
            'model_batch_size': 32,  # Default batch size for ML models
            'prediction_timeout': 30  # Timeout for predictions in seconds
        }
    
    def optimize_for_system(self):
        """Optimize settings based on system capabilities"""
        try:
            # Get system information
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Adjust settings based on system specs
            if memory_gb < 4:
                # Low memory system
                self.optimization_settings.update({
                    'model_batch_size': 16,
                    'cache_cleanup_threshold': 75,
                    'max_memory_usage': 75
                })
            elif memory_gb > 16:
                # High memory system
                self.optimization_settings.update({
                    'model_batch_size': 64,
                    'cache_cleanup_threshold': 95,
                    'max_memory_usage': 90
                })
            
            if cpu_count < 4:
                # Low CPU system
                self.optimization_settings['max_cpu_usage'] = 70
            elif cpu_count > 8:
                # High CPU system
                self.optimization_settings['max_cpu_usage'] = 90
            
            return self.optimization_settings
            
        except Exception as e:
            print(f"System optimization failed: {e}")
            return self.optimization_settings
    
    def check_resource_usage(self):
        """Check current resource usage and suggest optimizations"""
        try:
            suggestions = []
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.optimization_settings['max_cpu_usage']:
                suggestions.append({
                    'type': 'cpu',
                    'message': f"High CPU usage: {cpu_percent:.1f}%",
                    'suggestion': "Consider reducing model complexity or batch size"
                })
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.optimization_settings['max_memory_usage']:
                suggestions.append({
                    'type': 'memory',
                    'message': f"High memory usage: {memory.percent:.1f}%",
                    'suggestion': "Consider clearing cache or reducing data retention"
                })
            
            # Check available disk space
            disk = psutil.disk_usage('.')
            if disk.percent > 90:
                suggestions.append({
                    'type': 'disk',
                    'message': f"Low disk space: {disk.percent:.1f}% used",
                    'suggestion': "Clean up old cache files and logs"
                })
            
            return suggestions
            
        except Exception as e:
            return [{'type': 'error', 'message': f"Resource check failed: {e}"}]
    
    def optimize_model_settings(self, model_type="ensemble"):
        """Get optimized settings for ML models"""
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            settings = {
                'batch_size': self.optimization_settings['model_batch_size'],
                'num_workers': min(4, cpu_count),
                'pin_memory': memory_gb > 8,
                'use_gpu': False  # Will be overridden by GPU manager
            }
            
            # Model-specific optimizations
            if model_type == "lstm":
                if memory_gb < 4:
                    settings.update({
                        'hidden_size': 64,
                        'num_layers': 2,
                        'batch_size': 16
                    })
                elif memory_gb > 16:
                    settings.update({
                        'hidden_size': 256,
                        'num_layers': 4,
                        'batch_size': 64
                    })
                else:
                    settings.update({
                        'hidden_size': 128,
                        'num_layers': 3,
                        'batch_size': 32
                    })
            
            elif model_type == "ensemble":
                if cpu_count < 4:
                    settings.update({
                        'n_estimators': 100,
                        'n_jobs': 2
                    })
                else:
                    settings.update({
                        'n_estimators': 200,
                        'n_jobs': -1
                    })
            
            return settings
            
        except Exception as e:
            print(f"Model optimization failed: {e}")
            return {'batch_size': 32, 'num_workers': 2}
    
    def cleanup_resources(self, cache_manager=None):
        """Clean up resources to free memory"""
        try:
            cleanup_actions = []
            
            # Clean up cache if provided
            if cache_manager:
                try:
                    cache_manager.cleanup_old_cache()
                    cleanup_actions.append("Cleaned old cache files")
                except Exception as e:
                    cleanup_actions.append(f"Cache cleanup failed: {e}")
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            cleanup_actions.append(f"Garbage collected {collected} objects")
            
            # Clear PyTorch cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    cleanup_actions.append("Cleared GPU cache")
            except Exception:
                pass
            
            return cleanup_actions
            
        except Exception as e:
            return [f"Cleanup failed: {e}"]

class BenchmarkRunner:
    """
    Run performance benchmarks
    """
    
    def __init__(self):
        self.benchmark_results = {}
    
    def benchmark_prediction_speed(self, ara_instance, symbols=['AAPL', 'TSLA', 'MSFT']):
        """Benchmark prediction speed"""
        try:
            results = {
                'symbols_tested': len(symbols),
                'individual_times': {},
                'average_time': 0,
                'fastest': float('inf'),
                'slowest': 0
            }
            
            total_time = 0
            
            for symbol in symbols:
                start_time = time.time()
                
                try:
                    result = ara_instance.predict(symbol, days=5, use_cache=False)
                    prediction_time = time.time() - start_time
                    
                    results['individual_times'][symbol] = prediction_time
                    total_time += prediction_time
                    
                    results['fastest'] = min(results['fastest'], prediction_time)
                    results['slowest'] = max(results['slowest'], prediction_time)
                    
                except Exception as e:
                    results['individual_times'][symbol] = f"Error: {e}"
            
            if total_time > 0:
                results['average_time'] = total_time / len(symbols)
            
            self.benchmark_results['prediction_speed'] = results
            return results
            
        except Exception as e:
            return {'error': f"Benchmark failed: {e}"}
    
    def benchmark_memory_usage(self, ara_instance):
        """Benchmark memory usage during operations"""
        try:
            import tracemalloc
            
            # Start memory tracing
            tracemalloc.start()
            
            # Initial memory
            initial_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            # Perform operations
            ara_instance.predict('AAPL', days=5)
            
            # Peak memory
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            final_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            
            results = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': final_memory - initial_memory,
                'peak_traced_mb': peak / (1024**2),
                'current_traced_mb': current / (1024**2)
            }
            
            self.benchmark_results['memory_usage'] = results
            return results
            
        except Exception as e:
            return {'error': f"Memory benchmark failed: {e}"}
    
    def get_system_benchmark(self):
        """Get system performance benchmark"""
        try:
            # CPU benchmark
            start_time = time.time()
            # Simple CPU-intensive task
            result = sum(i * i for i in range(100000))
            cpu_time = time.time() - start_time
            
            # Memory benchmark
            memory_info = psutil.virtual_memory()
            
            # Disk benchmark
            start_time = time.time()
            test_file = Path('.benchmark_test')
            test_data = b'0' * (1024 * 1024)  # 1MB
            test_file.write_bytes(test_data)
            write_time = time.time() - start_time
            
            start_time = time.time()
            _ = test_file.read_bytes()
            read_time = time.time() - start_time
            
            test_file.unlink()  # Clean up
            
            results = {
                'cpu_benchmark_seconds': cpu_time,
                'memory_total_gb': memory_info.total / (1024**3),
                'memory_available_gb': memory_info.available / (1024**3),
                'disk_write_speed_mb_s': 1 / write_time if write_time > 0 else 0,
                'disk_read_speed_mb_s': 1 / read_time if read_time > 0 else 0,
                'cpu_count': psutil.cpu_count(),
                'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            }
            
            self.benchmark_results['system'] = results
            return results
            
        except Exception as e:
            return {'error': f"System benchmark failed: {e}"}
    
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'benchmarks': self.benchmark_results,
                'recommendations': []
            }
            
            # Generate recommendations based on results
            if 'prediction_speed' in self.benchmark_results:
                avg_time = self.benchmark_results['prediction_speed'].get('average_time', 0)
                if avg_time > 10:
                    report['recommendations'].append("Consider enabling GPU acceleration for faster predictions")
                elif avg_time < 2:
                    report['recommendations'].append("Excellent prediction speed - system is well optimized")
            
            if 'memory_usage' in self.benchmark_results:
                memory_increase = self.benchmark_results['memory_usage'].get('memory_increase_mb', 0)
                if memory_increase > 500:
                    report['recommendations'].append("High memory usage detected - consider reducing model complexity")
            
            if 'system' in self.benchmark_results:
                cpu_count = self.benchmark_results['system'].get('cpu_count', 0)
                if cpu_count < 4:
                    report['recommendations'].append("Limited CPU cores - consider cloud computing for better performance")
            
            return report
            
        except Exception as e:
            return {'error': f"Report generation failed: {e}"}