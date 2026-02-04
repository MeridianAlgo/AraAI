"""
Health check and monitoring endpoints
"""

from fastapi import APIRouter, Response
from datetime import datetime
from typing import Dict, Any
import psutil
import sys

from ara.api.models import HealthResponse
from ara.utils.prometheus_metrics import get_prometheus_metrics
from ara.utils.monitoring import get_metrics
from ara import __version__

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Check if API is running",
)
async def health_check():
    """Basic health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(),
        services={"api": "operational"},
    )


@router.get(
    "/detailed",
    summary="Detailed health check",
    description="Detailed health check with component status",
)
async def detailed_health_check():
    """Detailed health check with all components"""

    # Check system resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    # Get metrics
    metrics = get_metrics()

    # Determine overall health
    health_status = "healthy"
    if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
        health_status = "degraded"

    return {
        "status": health_status,
        "version": __version__,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": metrics.get("uptime", 0),
        "components": {
            "api": {
                "status": "operational",
                "requests_total": metrics.get("counters", {}).get("api_requests", 0),
            },
            "prediction_engine": {
                "status": "operational",
                "predictions_total": metrics.get("counters", {}).get("predictions", 0),
            },
            "data_providers": {
                "status": "operational",
                "fetches_total": metrics.get("counters", {}).get("data_fetches", 0),
            },
            "cache": {
                "status": "operational",
                "hit_rate": _calculate_cache_hit_rate(metrics),
            },
            "models": {
                "status": "operational",
                "active_models": metrics.get("gauges", {}).get("active_models", 0),
            },
        },
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024 * 1024 * 1024),
            "python_version": sys.version,
        },
        "performance": {
            "avg_prediction_time_ms": _get_avg_timing(metrics, "prediction"),
            "avg_api_response_time_ms": _get_avg_timing(metrics, "api_request"),
        },
    }


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if service is ready to accept requests",
)
async def readiness_check():
    """
    Readiness check for Kubernetes/orchestration
    Returns 200 if ready, 503 if not ready
    """
    # Check critical components
    try:
        # TODO: Add actual checks for database, cache, etc.
        ready = True

        if ready:
            return {"status": "ready", "timestamp": datetime.now().isoformat()}
        else:
            return Response(
                content='{"status": "not_ready"}',
                status_code=503,
                media_type="application/json",
            )
    except Exception as e:
        return Response(
            content=f'{{"status": "not_ready", "error": "{str(e)}"}}',
            status_code=503,
            media_type="application/json",
        )


@router.get("/live", summary="Liveness check", description="Check if service is alive")
async def liveness_check():
    """
    Liveness check for Kubernetes/orchestration
    Returns 200 if alive, 503 if dead
    """
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Export metrics in Prometheus format",
)
async def prometheus_metrics():
    """Export Prometheus metrics"""
    metrics = get_prometheus_metrics()
    return Response(content=metrics.export_metrics(), media_type=metrics.get_content_type())


def _calculate_cache_hit_rate(metrics: Dict[str, Any]) -> float:
    """Calculate cache hit rate"""
    counters = metrics.get("counters", {})
    hits = counters.get("cache_hits", 0)
    misses = counters.get("cache_misses", 0)
    total = hits + misses

    if total == 0:
        return 0.0

    return (hits / total) * 100


def _get_avg_timing(metrics: Dict[str, Any], name: str) -> float:
    """Get average timing for a metric"""
    timings = metrics.get("timings", {})
    timing_data = timings.get(name, {})
    return timing_data.get("mean", 0.0)
