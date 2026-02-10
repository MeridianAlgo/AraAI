"""
Grafana dashboard configurations
Provides JSON configurations for Grafana dashboards
"""

import json
from typing import Any, Dict, List


def get_system_metrics_dashboard() -> Dict[str, Any]:
    """
    Get system metrics dashboard configuration

    Returns:
        Grafana dashboard JSON
    """
    return {
        "dashboard": {
            "title": "ARA AI - System Metrics",
            "tags": ["ara-ai", "system"],
            "timezone": "browser",
            "schemaVersion": 16,
            "version": 0,
            "refresh": "10s",
            "panels": [
                {
                    "id": 1,
                    "title": "API Request Rate",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "rate(ara_api_requests_total[5m])",
                            "legendFormat": "{{method}} {{endpoint}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "reqps", "label": "Requests/sec"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 2,
                    "title": "API Response Time (p95)",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(ara_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "{{method}} {{endpoint}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "s", "label": "Duration"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 3,
                    "title": "Active Requests",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "ara_active_requests",
                            "legendFormat": "Active Requests",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "short", "label": "Count"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 4,
                    "title": "Error Rate",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": "rate(ara_errors_total[5m])",
                            "legendFormat": "{{error_type}} - {{component}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "ops", "label": "Errors/sec"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 5,
                    "title": "CPU Usage",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 8, "x": 0, "y": 16},
                    "targets": [
                        {
                            "expr": "process_cpu_seconds_total",
                            "legendFormat": "CPU",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "percent", "label": "CPU %"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 6,
                    "title": "Memory Usage",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 8, "x": 8, "y": 16},
                    "targets": [
                        {
                            "expr": "process_resident_memory_bytes",
                            "legendFormat": "Memory",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "bytes", "label": "Memory"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 7,
                    "title": "Cache Hit Rate",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 8, "x": 16, "y": 16},
                    "targets": [
                        {
                            "expr": "rate(ara_cache_hits_total[5m]) / (rate(ara_cache_hits_total[5m]) + rate(ara_cache_misses_total[5m])) * 100",
                            "legendFormat": "{{cache_level}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "percent", "label": "Hit Rate %"},
                        {"format": "short"},
                    ],
                },
            ],
        }
    }


def get_prediction_accuracy_dashboard() -> Dict[str, Any]:
    """
    Get prediction accuracy monitoring dashboard

    Returns:
        Grafana dashboard JSON
    """
    return {
        "dashboard": {
            "title": "ARA AI - Prediction Accuracy",
            "tags": ["ara-ai", "predictions", "accuracy"],
            "timezone": "browser",
            "schemaVersion": 16,
            "version": 0,
            "refresh": "30s",
            "panels": [
                {
                    "id": 1,
                    "title": "Prediction Accuracy by Asset Type",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "ara_prediction_accuracy",
                            "legendFormat": "{{asset_type}} - {{timeframe}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {
                            "format": "percentunit",
                            "label": "Accuracy",
                            "min": 0,
                            "max": 1,
                        },
                        {"format": "short"},
                    ],
                    "alert": {
                        "conditions": [
                            {
                                "evaluator": {"params": [0.70], "type": "lt"},
                                "operator": {"type": "and"},
                                "query": {"params": ["A", "5m", "now"]},
                                "reducer": {"params": [], "type": "avg"},
                                "type": "query",
                            }
                        ],
                        "name": "Low Prediction Accuracy",
                    },
                },
                {
                    "id": 2,
                    "title": "Prediction Rate",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "targets": [
                        {
                            "expr": "rate(ara_predictions_total[5m])",
                            "legendFormat": "{{asset_type}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "ops", "label": "Predictions/sec"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 3,
                    "title": "Prediction Duration (p95)",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(ara_prediction_duration_seconds_bucket[5m]))",
                            "legendFormat": "{{asset_type}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "s", "label": "Duration"},
                        {"format": "short"},
                    ],
                    "alert": {
                        "conditions": [
                            {
                                "evaluator": {"params": [5.0], "type": "gt"},
                                "operator": {"type": "and"},
                                "query": {"params": ["A", "5m", "now"]},
                                "reducer": {"params": [], "type": "avg"},
                                "type": "query",
                            }
                        ],
                        "name": "Slow Predictions",
                    },
                },
                {
                    "id": 4,
                    "title": "Prediction Confidence Distribution",
                    "type": "heatmap",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": "rate(ara_prediction_confidence_bucket[5m])",
                            "legendFormat": "{{asset_type}}",
                            "refId": "A",
                        }
                    ],
                },
                {
                    "id": 5,
                    "title": "Top Predicted Symbols",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                    "targets": [
                        {
                            "expr": "topk(10, sum by (symbol) (rate(ara_predictions_total[1h])))",
                            "format": "table",
                            "refId": "A",
                        }
                    ],
                },
                {
                    "id": 6,
                    "title": "Prediction Errors",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                    "targets": [
                        {
                            "expr": 'rate(ara_errors_total{component="prediction"}[5m])',
                            "legendFormat": "{{error_type}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "ops", "label": "Errors/sec"},
                        {"format": "short"},
                    ],
                },
            ],
        }
    }


def get_model_performance_dashboard() -> Dict[str, Any]:
    """
    Get model performance tracking dashboard

    Returns:
        Grafana dashboard JSON
    """
    return {
        "dashboard": {
            "title": "ARA AI - Model Performance",
            "tags": ["ara-ai", "models", "ml"],
            "timezone": "browser",
            "schemaVersion": 16,
            "version": 0,
            "refresh": "30s",
            "panels": [
                {
                    "id": 1,
                    "title": "Model Inference Time",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(ara_model_inference_duration_seconds_bucket[5m]))",
                            "legendFormat": "{{model_name}} (p95)",
                            "refId": "A",
                        },
                        {
                            "expr": "histogram_quantile(0.50, rate(ara_model_inference_duration_seconds_bucket[5m]))",
                            "legendFormat": "{{model_name}} (p50)",
                            "refId": "B",
                        },
                    ],
                    "yaxes": [
                        {"format": "s", "label": "Duration"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 2,
                    "title": "Active Models",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "targets": [
                        {
                            "expr": "ara_active_models",
                            "legendFormat": "{{model_type}}",
                            "refId": "A",
                        }
                    ],
                },
                {
                    "id": 3,
                    "title": "Model Training Duration",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "ara_model_training_duration_seconds",
                            "legendFormat": "{{model_name}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "s", "label": "Duration"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 4,
                    "title": "Feature Calculation Time",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(ara_feature_calculation_duration_seconds_bucket[5m]))",
                            "legendFormat": "{{feature_type}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "s", "label": "Duration"},
                        {"format": "short"},
                    ],
                },
            ],
        }
    }


def get_api_performance_dashboard() -> Dict[str, Any]:
    """
    Get API performance dashboard

    Returns:
        Grafana dashboard JSON
    """
    return {
        "dashboard": {
            "title": "ARA AI - API Performance",
            "tags": ["ara-ai", "api", "performance"],
            "timezone": "browser",
            "schemaVersion": 16,
            "version": 0,
            "refresh": "10s",
            "panels": [
                {
                    "id": 1,
                    "title": "Request Rate by Endpoint",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "rate(ara_api_requests_total[5m])",
                            "legendFormat": "{{endpoint}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "reqps", "label": "Requests/sec"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 2,
                    "title": "Response Time Percentiles",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.99, rate(ara_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "p99",
                            "refId": "A",
                        },
                        {
                            "expr": "histogram_quantile(0.95, rate(ara_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "p95",
                            "refId": "B",
                        },
                        {
                            "expr": "histogram_quantile(0.50, rate(ara_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "p50",
                            "refId": "C",
                        },
                    ],
                    "yaxes": [
                        {"format": "s", "label": "Duration"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 3,
                    "title": "Error Rate by Status Code",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": 'rate(ara_api_requests_total{status=~"4..|5.."}[5m])',
                            "legendFormat": "{{status}} - {{endpoint}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "ops", "label": "Errors/sec"},
                        {"format": "short"},
                    ],
                },
                {
                    "id": 4,
                    "title": "Data Provider Performance",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(ara_data_fetch_duration_seconds_bucket[5m]))",
                            "legendFormat": "{{provider}}",
                            "refId": "A",
                        }
                    ],
                    "yaxes": [
                        {"format": "s", "label": "Duration"},
                        {"format": "short"},
                    ],
                },
            ],
        }
    }


def export_dashboard_to_file(dashboard: Dict[str, Any], filename: str):
    """
    Export dashboard configuration to JSON file

    Args:
        dashboard: Dashboard configuration
        filename: Output filename
    """
    with open(filename, "w") as f:
        json.dump(dashboard, f, indent=2)


def get_all_dashboards() -> List[Dict[str, Any]]:
    """
    Get all dashboard configurations

    Returns:
        List of dashboard configurations
    """
    return [
        get_system_metrics_dashboard(),
        get_prediction_accuracy_dashboard(),
        get_model_performance_dashboard(),
        get_api_performance_dashboard(),
    ]
