# Monitoring and Observability

This module provides comprehensive monitoring and observability features for the ARA AI prediction system.

## Features

### 1. Prometheus Metrics

Export metrics in Prometheus format for monitoring:

- **API Metrics**: Request rate, duration, error rate
- **Prediction Metrics**: Prediction count, duration, confidence, accuracy
- **Model Metrics**: Inference time, training duration, active models
- **Data Provider Metrics**: Fetch duration, quality scores
- **Cache Metrics**: Hit rate, size
- **System Metrics**: Errors, active requests, health status

**Usage:**

```python
from ara.utils.prometheus_metrics import get_prometheus_metrics, track_prediction

# Get metrics instance
metrics = get_prometheus_metrics()

# Track predictions with decorator
@track_prediction(asset_type="stock")
async def predict(symbol: str):
    # Your prediction logic
    pass

# Export metrics (in API endpoint)
metrics_data = metrics.export_metrics()
```

**Endpoint:**

```
GET /health/metrics
```

### 2. Distributed Tracing (OpenTelemetry)

Track request flows across components:

**Installation:**

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

**Usage:**

```python
from ara.utils.tracing import init_tracing, trace, start_span

# Initialize tracing
init_tracing(
    service_name="ara-ai",
    otlp_endpoint="http://localhost:4317"
)

# Trace function execution
@trace("prediction_workflow")
async def predict(symbol: str):
    # Your logic
    pass

# Manual span creation
with start_span("data_fetch", symbol=symbol) as span:
    data = await fetch_data(symbol)
```

### 3. Error Tracking (Sentry)

Automatic error capture and reporting:

**Installation:**

```bash
pip install sentry-sdk
```

**Usage:**

```python
from ara.utils.error_tracking import init_error_tracking, capture_exception

# Initialize error tracking
init_error_tracking(
    dsn="https://your-sentry-dsn@sentry.io/project",
    environment="production"
)

# Automatic error capture with decorator
from ara.utils.error_tracking import track_errors

@track_errors
async def risky_operation():
    # Your logic
    pass

# Manual error capture
try:
    risky_operation()
except Exception as e:
    capture_exception(e, context={"user_id": "123"})
```

### 4. Health Check Endpoints

Multiple health check endpoints for monitoring:

**Endpoints:**

- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health with component status
- `GET /health/ready` - Readiness check (Kubernetes)
- `GET /health/live` - Liveness check (Kubernetes)
- `GET /health/metrics` - Prometheus metrics

**Example Response (Detailed):**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00",
  "components": {
    "api": {"status": "operational"},
    "prediction_engine": {"status": "operational"},
    "data_providers": {"status": "operational"},
    "cache": {"status": "operational", "hit_rate": 85.5},
    "models": {"status": "operational", "active_models": 12}
  },
  "system": {
    "cpu_percent": 45.2,
    "memory_percent": 62.1,
    "disk_percent": 35.8
  },
  "performance": {
    "avg_prediction_time_ms": 1234.5,
    "avg_api_response_time_ms": 156.7
  }
}
```

### 5. Grafana Dashboards

Pre-configured Grafana dashboards for visualization:

**Available Dashboards:**

1. **System Metrics** - API performance, resource usage, cache metrics
2. **Prediction Accuracy** - Accuracy tracking, prediction rates, confidence
3. **Model Performance** - Inference time, training duration, feature calculation
4. **API Performance** - Request rates, response times, error rates

**Export Dashboards:**

```python
from ara.monitoring import get_all_dashboards, export_dashboard_to_file

# Get all dashboards
dashboards = get_all_dashboards()

# Export to files
for i, dashboard in enumerate(dashboards):
    export_dashboard_to_file(dashboard, f"dashboard_{i}.json")
```

**Import to Grafana:**

1. Open Grafana UI
2. Go to Dashboards → Import
3. Upload the JSON file
4. Select Prometheus data source
5. Click Import

## Setup Guide

### 1. Prometheus Setup

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

**prometheus.yml:**

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ara-ai'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/health/metrics'
```

### 2. OpenTelemetry Collector Setup

**docker-compose.yml:**

```yaml
services:
  otel-collector:
    image: otel/opentelemetry-collector:latest
    ports:
      - "4317:4317"  # OTLP gRPC
      - "4318:4318"  # OTLP HTTP
    volumes:
      - ./otel-collector-config.yml:/etc/otel-collector-config.yml
    command: ["--config=/etc/otel-collector-config.yml"]

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Jaeger collector
```

**otel-collector-config.yml:**

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [jaeger]
```

### 3. Sentry Setup

**Environment Variables:**

```bash
export SENTRY_DSN="https://your-key@sentry.io/project-id"
export SENTRY_ENVIRONMENT="production"
```

**In Application:**

```python
from ara.utils.error_tracking import init_error_tracking
import os

init_error_tracking(
    dsn=os.getenv("SENTRY_DSN"),
    environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
    release="1.0.0"
)
```

## Monitoring Best Practices

### 1. Alerting Rules

Create alerts for critical metrics:

**Prometheus Alert Rules:**

```yaml
groups:
  - name: ara_ai_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(ara_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      - alert: LowPredictionAccuracy
        expr: ara_prediction_accuracy < 0.70
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Prediction accuracy below threshold"

      - alert: SlowPredictions
        expr: histogram_quantile(0.95, rate(ara_prediction_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Predictions taking too long"
```

### 2. Log Aggregation

Use structured logging for better analysis:

```python
from ara.utils.logging import get_logger

logger = get_logger(__name__)

# Add context
logger.add_context(user_id="123", request_id="abc")

# Log with structured data
logger.info(
    "Prediction completed",
    symbol="AAPL",
    confidence=0.85,
    duration_ms=1234
)
```

### 3. Performance Monitoring

Track key performance indicators:

- API response time (p50, p95, p99)
- Prediction accuracy (daily, weekly, monthly)
- Model inference time
- Cache hit rate
- Error rate by component

### 4. Capacity Planning

Monitor resource usage:

- CPU utilization
- Memory usage
- Disk space
- Network bandwidth
- Database connections

## Troubleshooting

### Metrics Not Appearing

1. Check Prometheus is scraping: `http://localhost:9090/targets`
2. Verify metrics endpoint: `http://localhost:8000/health/metrics`
3. Check Prometheus logs for errors

### Traces Not Showing

1. Verify OTLP collector is running
2. Check collector logs
3. Ensure correct endpoint in configuration
4. Verify OpenTelemetry packages are installed

### High Memory Usage

1. Check cache size: Monitor `ara_cache_size_bytes`
2. Review active models: Check `ara_active_models`
3. Look for memory leaks in logs
4. Consider reducing cache TTL

## Integration Examples

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ara-ai
  labels:
    app: ara-ai
spec:
  ports:
    - port: 8000
      name: http
  selector:
    app: ara-ai

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ara-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ara-ai
  template:
    metadata:
      labels:
        app: ara-ai
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/health/metrics"
    spec:
      containers:
        - name: ara-ai
          image: ara-ai:latest
          ports:
            - containerPort: 8000
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          env:
            - name: SENTRY_DSN
              valueFrom:
                secretKeyRef:
                  name: ara-secrets
                  key: sentry-dsn
            - name: OTLP_ENDPOINT
              value: "http://otel-collector:4317"
```

### Docker Compose Full Stack

```yaml
version: '3.8'

services:
  ara-ai:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SENTRY_DSN=${SENTRY_DSN}
      - OTLP_ENDPOINT=http://otel-collector:4317
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  otel-collector:
    image: otel/opentelemetry-collector:latest
    ports:
      - "4317:4317"
    volumes:
      - ./otel-config.yml:/etc/otel-config.yml
    command: ["--config=/etc/otel-config.yml"]

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
```

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Sentry Documentation](https://docs.sentry.io/)
