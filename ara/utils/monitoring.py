"""
Performance monitoring and metrics collection
Provides decorators and utilities for tracking system performance
"""

import time
import functools
import inspect
from typing import Callable, Any, Dict, Optional
from collections import defaultdict
import threading

from ara.utils.logging import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Collects and aggregates performance metrics
    Thread-safe metrics storage
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._timings: Dict[str, list] = defaultdict(list)

    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric"""
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric"""
        with self._lock:
            self._gauges[name] = value

    def record_histogram(self, name: str, value: float) -> None:
        """Record a histogram value"""
        with self._lock:
            self._histograms[name].append(value)
            # Keep only last 1000 values
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

    def record_timing(self, name: str, duration_ms: float) -> None:
        """Record a timing measurement"""
        with self._lock:
            self._timings[name].append(duration_ms)
            # Keep only last 1000 values
            if len(self._timings[name]) > 1000:
                self._timings[name] = self._timings[name][-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            metrics = {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {},
                "timings": {},
            }

            # Calculate histogram statistics
            for name, values in self._histograms.items():
                if values:
                    metrics["histograms"][name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                    }

            # Calculate timing statistics
            for name, values in self._timings.items():
                if values:
                    sorted_values = sorted(values)
                    p50_idx = int(len(sorted_values) * 0.50)
                    p95_idx = int(len(sorted_values) * 0.95)
                    p99_idx = int(len(sorted_values) * 0.99)

                    metrics["timings"][name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "p50": sorted_values[p50_idx],
                        "p95": sorted_values[p95_idx],
                        "p99": sorted_values[p99_idx],
                    }

            return metrics

    def reset(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timings.clear()


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> Dict[str, Any]:
    """Get global metrics"""
    return _metrics.get_metrics()


def increment_counter(name: str, value: int = 1) -> None:
    """Increment a global counter"""
    _metrics.increment_counter(name, value)


def set_gauge(name: str, value: float) -> None:
    """Set a global gauge"""
    _metrics.set_gauge(name, value)


def record_histogram(name: str, value: float) -> None:
    """Record a global histogram value"""
    _metrics.record_histogram(name, value)


def timed(metric_name: Optional[str] = None):
    """
    Decorator to time function execution

    Args:
        metric_name: Name for the timing metric (uses function name if None)

    Example:
        @timed("prediction_time")
        def predict(data):
            ...
    """

    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                _metrics.record_timing(name, duration_ms)
                logger.debug(f"Function {name} completed", duration_ms=duration_ms)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start_time) * 1000
                _metrics.record_timing(name, duration_ms)
                logger.debug(f"Function {name} completed", duration_ms=duration_ms)

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def counted(metric_name: Optional[str] = None):
    """
    Decorator to count function calls

    Args:
        metric_name: Name for the counter metric (uses function name if None)

    Example:
        @counted("api_calls")
        def api_endpoint():
            ...
    """

    def decorator(func: Callable) -> Callable:
        name = metric_name or f"{func.__module__}.{func.__name__}_calls"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _metrics.increment_counter(name)
            return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            _metrics.increment_counter(name)
            return await func(*args, **kwargs)

        if functools.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


class Timer:
    """
    Context manager for timing code blocks

    Example:
        with Timer("data_processing") as t:
            process_data()
        print(f"Took {t.duration_ms}ms")
    """

    def __init__(self, name: str, log: bool = True):
        self.name = name
        self.log = log
        self.start_time = None
        self.duration_ms = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.time() - self.start_time) * 1000
        _metrics.record_timing(self.name, self.duration_ms)

        if self.log:
            logger.debug(f"Timer {self.name} completed", duration_ms=self.duration_ms)
