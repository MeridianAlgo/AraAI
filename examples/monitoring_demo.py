"""
Monitoring and Observability Demo
Demonstrates how to use ARA AI's monitoring features
"""

import asyncio
import time
from ara.utils.prometheus_metrics import (
    get_prometheus_metrics,
    track_prediction,
    track_model_inference,
    track_data_fetch
)
from ara.utils.tracing import init_tracing, trace, start_span
from ara.utils.error_tracking import (
    init_error_tracking,
    capture_exception,
    capture_message,
    track_errors
)
from ara.utils.monitoring import timed, counted, Timer, get_metrics
from ara.utils.logging import get_logger

logger = get_logger(__name__)


# Example 1: Prometheus Metrics
def demo_prometheus_metrics():
    """Demonstrate Prometheus metrics collection"""
    print("\n" + "="*70)
    print("Demo 1: Prometheus Metrics")
    print("="*70)
    
    metrics = get_prometheus_metrics()
    
    # Simulate some predictions
    for i in range(5):
        # Track prediction
        metrics.predictions_total.labels(
            asset_type="stock",
            symbol="AAPL"
        ).inc()
        
        # Track prediction duration
        duration = 1.5 + (i * 0.1)
        metrics.prediction_duration.labels(asset_type="stock").observe(duration)
        
        # Track confidence
        confidence = 0.75 + (i * 0.02)
        metrics.prediction_confidence.labels(asset_type="stock").observe(confidence)
    
    # Track model inference
    metrics.model_inference_duration.labels(model_name="transformer").observe(0.05)
    metrics.model_inference_duration.labels(model_name="ensemble").observe(0.03)
    
    # Track data fetches
    metrics.data_fetch_total.labels(provider="yfinance", status="success").inc()
    metrics.data_fetch_duration.labels(provider="yfinance").observe(0.5)
    
    # Track cache hits
    metrics.cache_hits_total.labels(cache_level="L1").inc(10)
    metrics.cache_misses_total.labels(cache_level="L1").inc(2)
    
    print("\n✓ Metrics recorded")
    print("  - 5 predictions tracked")
    print("  - 2 model inferences tracked")
    print("  - 1 data fetch tracked")
    print("  - Cache metrics tracked")
    print("\nView metrics at: http://localhost:8000/health/metrics")


# Example 2: Decorators for Automatic Tracking
@track_prediction(asset_type="stock")
async def predict_stock(symbol: str):
    """Example prediction function with automatic tracking"""
    await asyncio.sleep(1.5)  # Simulate prediction
    return {
        "symbol": symbol,
        "prediction": 150.0,
        "confidence": {"overall": 0.85}
    }


@track_model_inference(model_name="transformer")
def run_model_inference(data):
    """Example model inference with automatic tracking"""
    time.sleep(0.05)  # Simulate inference
    return [0.1, 0.2, 0.3]


@track_data_fetch(provider="yfinance")
async def fetch_market_data(symbol: str):
    """Example data fetch with automatic tracking"""
    await asyncio.sleep(0.5)  # Simulate API call
    return {"symbol": symbol, "price": 150.0}


async def demo_decorators():
    """Demonstrate automatic tracking with decorators"""
    print("\n" + "="*70)
    print("Demo 2: Automatic Tracking with Decorators")
    print("="*70)
    
    # Run tracked functions
    result = await predict_stock("AAPL")
    print(f"\n✓ Prediction tracked: {result['symbol']}")
    
    inference_result = run_model_inference([1, 2, 3])
    print(f"✓ Model inference tracked: {len(inference_result)} outputs")
    
    data = await fetch_market_data("AAPL")
    print(f"✓ Data fetch tracked: {data['symbol']}")


# Example 3: Distributed Tracing
@trace("prediction_workflow")
async def prediction_workflow(symbol: str):
    """Example workflow with distributed tracing"""
    
    # Span 1: Data fetching
    with start_span("fetch_data", symbol=symbol):
        await asyncio.sleep(0.5)
        data = {"symbol": symbol, "price": 150.0}
    
    # Span 2: Feature calculation
    with start_span("calculate_features", symbol=symbol):
        await asyncio.sleep(0.3)
        features = [1, 2, 3, 4, 5]
    
    # Span 3: Model inference
    with start_span("model_inference", model="transformer"):
        await asyncio.sleep(0.2)
        prediction = 155.0
    
    return {"symbol": symbol, "prediction": prediction}


async def demo_tracing():
    """Demonstrate distributed tracing"""
    print("\n" + "="*70)
    print("Demo 3: Distributed Tracing")
    print("="*70)
    
    # Initialize tracing (optional - requires OpenTelemetry)
    # init_tracing(otlp_endpoint="http://localhost:4317")
    
    result = await prediction_workflow("AAPL")
    print(f"\n✓ Workflow traced: {result['symbol']}")
    print("  - Data fetch span")
    print("  - Feature calculation span")
    print("  - Model inference span")
    print("\nView traces at: http://localhost:16686 (Jaeger UI)")


# Example 4: Error Tracking
@track_errors
def risky_operation():
    """Example function that might fail"""
    import random
    if random.random() < 0.3:
        raise ValueError("Random error occurred")
    return "Success"


def demo_error_tracking():
    """Demonstrate error tracking"""
    print("\n" + "="*70)
    print("Demo 4: Error Tracking")
    print("="*70)
    
    # Initialize error tracking (optional - requires Sentry)
    # init_error_tracking(dsn="your-sentry-dsn")
    
    # Try risky operation
    for i in range(5):
        try:
            result = risky_operation()
            print(f"  Attempt {i+1}: {result}")
        except Exception as e:
            print(f"  Attempt {i+1}: Error caught and tracked")
            # Error is automatically tracked by @track_errors decorator
    
    # Manual error capture
    try:
        raise RuntimeError("Manual error example")
    except Exception as e:
        capture_exception(
            e,
            context={
                "user_id": "demo_user",
                "operation": "demo"
            }
        )
        print("\n✓ Manual error captured with context")
    
    # Capture message
    capture_message(
        "Important event occurred",
        level="info",
        context={"event_type": "demo"}
    )
    print("✓ Message captured")
    print("\nView errors at: https://sentry.io (if configured)")


# Example 5: Performance Monitoring
@timed("demo_function")
def slow_function():
    """Example function with timing"""
    time.sleep(0.5)
    return "Done"


@counted("demo_calls")
def counted_function():
    """Example function with call counting"""
    return "Called"


def demo_performance_monitoring():
    """Demonstrate performance monitoring"""
    print("\n" + "="*70)
    print("Demo 5: Performance Monitoring")
    print("="*70)
    
    # Timed function
    for i in range(3):
        result = slow_function()
    print("\n✓ Timed function called 3 times")
    
    # Counted function
    for i in range(10):
        counted_function()
    print("✓ Counted function called 10 times")
    
    # Manual timing with context manager
    with Timer("manual_operation") as t:
        time.sleep(0.2)
    print(f"✓ Manual timing: {t.duration_ms:.2f}ms")
    
    # Get all metrics
    metrics = get_metrics()
    print("\n[Metrics Summary]")
    print(f"  Counters: {len(metrics.get('counters', {}))}")
    print(f"  Gauges: {len(metrics.get('gauges', {}))}")
    print(f"  Timings: {len(metrics.get('timings', {}))}")


# Example 6: Structured Logging
def demo_structured_logging():
    """Demonstrate structured logging"""
    print("\n" + "="*70)
    print("Demo 6: Structured Logging")
    print("="*70)
    
    # Add context
    logger.add_context(user_id="demo_user", session_id="abc123")
    
    # Log with structured data
    logger.info(
        "Prediction started",
        symbol="AAPL",
        model="transformer",
        confidence=0.85
    )
    print("\n✓ Structured log with context")
    
    # Log warning
    logger.warning(
        "Low confidence prediction",
        symbol="TSLA",
        confidence=0.65,
        threshold=0.70
    )
    print("✓ Warning logged")
    
    # Log error
    try:
        raise ValueError("Example error")
    except Exception as e:
        logger.exception(
            "Error during prediction",
            symbol="AAPL",
            error_type=type(e).__name__
        )
        print("✓ Exception logged with traceback")
    
    # Clear context
    logger.clear_context()
    print("✓ Context cleared")


async def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("ARA AI Monitoring and Observability Demo")
    print("="*70)
    
    # Run demos
    demo_prometheus_metrics()
    await demo_decorators()
    await demo_tracing()
    demo_error_tracking()
    demo_performance_monitoring()
    demo_structured_logging()
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Start the API: python run_api.py")
    print("2. View metrics: http://localhost:8000/health/metrics")
    print("3. View detailed health: http://localhost:8000/health/detailed")
    print("4. Start monitoring stack: cd ara/monitoring && docker-compose up")
    print("5. View Grafana dashboards: http://localhost:3000")
    print("6. View Jaeger traces: http://localhost:16686")
    print("\nFor more information, see ara/monitoring/README.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
