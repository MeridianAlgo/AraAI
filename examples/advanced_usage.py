#!/usr/bin/env python3
"""
Advanced Ara AI Usage Example
"""

from meridianalgo import AraAI
from meridianalgo.utils import GPUManager, CacheManager, AccuracyTracker

def main():
    """Advanced usage examples"""
    
    print(" Ara AI Advanced Usage Examples")
    print("=" * 40)
    
    # Initialize full system
    ara = AraAI(verbose=True)
    
    # Check system capabilities
    print("\n1. System Information:")
    system_info = ara.get_system_info()
    print(f"Device: {system_info['device']}")
    print(f"GPU Info: {system_info.get('gpu_info', {})}")
    
    # Advanced prediction
    print("\n2. Advanced Prediction:")
    result = ara.predict('TSLA', days=7, use_cache=True)
    if result:
        print(f"TSLA prediction generated with {len(result['predictions'])} days")
    
    # Component usage
    print("\n3. Individual Components:")
    
    # GPU Manager
    gpu_manager = GPUManager()
    gpu_info = gpu_manager.detect_gpu_vendor()
    print(f"GPU Support: {gpu_info}")
    
    # Cache Manager
    cache_manager = CacheManager()
    cache_stats = cache_manager.get_cache_stats()
    print(f"Cache Stats: {cache_stats}")
    
    # Accuracy Tracker
    tracker = AccuracyTracker()
    accuracy_stats = tracker.get_accuracy_stats()
    print(f"Accuracy Stats: {accuracy_stats}")

if __name__ == "__main__":
    main()
