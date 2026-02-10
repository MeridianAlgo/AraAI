"""
Prometheus metrics exporter
Provides metrics collection and export for Prometheus monitoring
"""

import inspect
import time
from functools import wraps
from typing import Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from ara.utils.logging import get_logger

logger = get_logger(__name__)


class PrometheusMetrics:
    """
    Prometheus metrics collector
    Provides counters, gauges, histograms, and summaries
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize Prometheus metrics

        Args:
            registry: Prometheus registry (creates new if None)
        """
        self.registry = registry or CollectorRegistry()

        # API Metrics
        self.api_requests_total = Counter(
            "ara_api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.api_request_duration = Histogram(
            "ara_api_request_duration_seconds",
            "API request duration",
            ["method", "endpoint"],
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        )

        # Prediction Metrics
        self.predictions_total = Counter(
            "ara_predictions_total",
            "Total predictions made",
            ["asset_type", "symbol"],
            registry=self.registry,
        )

        self.prediction_duration = Histogram(
            "ara_prediction_duration_seconds",
            "Prediction generation duration",
            ["asset_type"],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )

        self.prediction_confidence = Histogram(
            "ara_prediction_confidence",
            "Prediction confidence scores",
            ["asset_type"],
            registry=self.registry,
            buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )

        self.prediction_accuracy = Gauge(
            "ara_prediction_accuracy",
            "Current prediction accuracy",
            ["asset_type", "timeframe"],
            registry=self.registry,
        )

        # Model Metrics
        self.model_inference_duration = Histogram(
            "ara_model_inference_duration_seconds",
            "Model inference duration",
            ["model_name"],
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
        )

        self.model_training_duration = Histogram(
            "ara_model_training_duration_seconds",
            "Model training duration",
            ["model_name"],
            registry=self.registry,
            buckets=(10, 30, 60, 300, 600, 1800, 3600),
        )

        self.active_models = Gauge(
            "ara_active_models",
            "Number of active models",
            ["model_type"],
            registry=self.registry,
        )

        # Data Provider Metrics
        self.data_fetch_total = Counter(
            "ara_data_fetch_total",
            "Total data fetches",
            ["provider", "status"],
            registry=self.registry,
        )

        self.data_fetch_duration = Histogram(
            "ara_data_fetch_duration_seconds",
            "Data fetch duration",
            ["provider"],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        )

        self.data_quality_score = Gauge(
            "ara_data_quality_score",
            "Data quality score",
            ["provider", "symbol"],
            registry=self.registry,
        )

        # Cache Metrics
        self.cache_hits_total = Counter(
            "ara_cache_hits_total",
            "Total cache hits",
            ["cache_level"],
            registry=self.registry,
        )

        self.cache_misses_total = Counter(
            "ara_cache_misses_total",
            "Total cache misses",
            ["cache_level"],
            registry=self.registry,
        )

        self.cache_size = Gauge(
            "ara_cache_size_bytes",
            "Cache size in bytes",
            ["cache_level"],
            registry=self.registry,
        )

        # Feature Engineering Metrics
        self.feature_calculation_duration = Histogram(
            "ara_feature_calculation_duration_seconds",
            "Feature calculation duration",
            ["feature_type"],
            registry=self.registry,
            buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0),
        )

        self.features_calculated_total = Counter(
            "ara_features_calculated_total",
            "Total features calculated",
            ["feature_type"],
            registry=self.registry,
        )

        # System Metrics
        self.errors_total = Counter(
            "ara_errors_total",
            "Total errors",
            ["error_type", "component"],
            registry=self.registry,
        )

        self.active_requests = Gauge(
            "ara_active_requests", "Number of active requests", registry=self.registry
        )

        self.system_health = Gauge(
            "ara_system_health",
            "System health status (1=healthy, 0=unhealthy)",
            ["component"],
            registry=self.registry,
        )

        # Backtesting Metrics
        self.backtests_total = Counter(
            "ara_backtests_total",
            "Total backtests run",
            ["symbol"],
            registry=self.registry,
        )

        self.backtest_duration = Histogram(
            "ara_backtest_duration_seconds",
            "Backtest duration",
            registry=self.registry,
            buckets=(1, 5, 10, 30, 60, 300, 600),
        )

        # Portfolio Metrics
        self.portfolio_optimizations_total = Counter(
            "ara_portfolio_optimizations_total",
            "Total portfolio optimizations",
            registry=self.registry,
        )

        self.portfolio_value = Gauge(
            "ara_portfolio_value",
            "Portfolio value",
            ["portfolio_id"],
            registry=self.registry,
        )

    def export_metrics(self) -> bytes:
        """
        Export metrics in Prometheus format

        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """
        Get Prometheus content type

        Returns:
            Content type string
        """
        return CONTENT_TYPE_LATEST


# Global metrics instance
_prometheus_metrics: Optional[PrometheusMetrics] = None


def get_prometheus_metrics() -> PrometheusMetrics:
    """
    Get global Prometheus metrics instance

    Returns:
        PrometheusMetrics instance
    """
    global _prometheus_metrics
    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics()
    return _prometheus_metrics


def track_prediction(asset_type: str = "stock"):
    """
    Decorator to track prediction metrics

    Args:
        asset_type: Type of asset being predicted
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_prometheus_metrics()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Track metrics
                duration = time.time() - start_time
                metrics.prediction_duration.labels(asset_type=asset_type).observe(duration)
                metrics.predictions_total.labels(
                    asset_type=asset_type, symbol=kwargs.get("symbol", "unknown")
                ).inc()

                # Track confidence if available
                if hasattr(result, "confidence"):
                    confidence = (
                        result.confidence.overall
                        if hasattr(result.confidence, "overall")
                        else result.confidence
                    )
                    metrics.prediction_confidence.labels(asset_type=asset_type).observe(confidence)

                return result
            except Exception as e:
                metrics.errors_total.labels(
                    error_type=type(e).__name__, component="prediction"
                ).inc()
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = get_prometheus_metrics()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                # Track metrics
                duration = time.time() - start_time
                metrics.prediction_duration.labels(asset_type=asset_type).observe(duration)
                metrics.predictions_total.labels(
                    asset_type=asset_type, symbol=kwargs.get("symbol", "unknown")
                ).inc()

                # Track confidence if available
                if hasattr(result, "confidence"):
                    confidence = (
                        result.confidence.overall
                        if hasattr(result.confidence, "overall")
                        else result.confidence
                    )
                    metrics.prediction_confidence.labels(asset_type=asset_type).observe(confidence)

                return result
            except Exception as e:
                metrics.errors_total.labels(
                    error_type=type(e).__name__, component="prediction"
                ).inc()
                raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def track_model_inference(model_name: str):
    """
    Decorator to track model inference metrics

    Args:
        model_name: Name of the model
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_prometheus_metrics()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.model_inference_duration.labels(model_name=model_name).observe(duration)
                return result
            except Exception as e:
                metrics.errors_total.labels(
                    error_type=type(e).__name__, component="model_inference"
                ).inc()
                raise

        return wrapper

    return decorator


def track_data_fetch(provider: str):
    """
    Decorator to track data fetch metrics

    Args:
        provider: Name of the data provider
    """

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_prometheus_metrics()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.data_fetch_duration.labels(provider=provider).observe(duration)
                metrics.data_fetch_total.labels(provider=provider, status="success").inc()
                return result
            except Exception as e:
                metrics.data_fetch_total.labels(provider=provider, status="error").inc()
                metrics.errors_total.labels(
                    error_type=type(e).__name__, component="data_fetch"
                ).inc()
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = get_prometheus_metrics()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.data_fetch_duration.labels(provider=provider).observe(duration)
                metrics.data_fetch_total.labels(provider=provider, status="success").inc()
                return result
            except Exception as e:
                metrics.data_fetch_total.labels(provider=provider, status="error").inc()
                metrics.errors_total.labels(
                    error_type=type(e).__name__, component="data_fetch"
                ).inc()
                raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
