# ARA AI Monitoring Setup Guide

Complete guide for setting up monitoring and observability for ARA AI.

## Overview

ARA AI uses a comprehensive monitoring stack:
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **AlertManager**: Alert routing and management
- **Loki**: Log aggregation (optional)
- **Jaeger**: Distributed tracing (optional)

## Quick Start

```bash
# Setup monitoring stack
python -m ara.monitoring.setup_monitoring

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# AlertManager: http://localhost:9093
```

## Prometheus Setup

### Installation

#### Docker Compose

```yaml
# Already included in docker-compose.yml
prometheus:
  image: prom/prometheus:latest
  ports:
    - "9090:9090"
  volumes:
    - ./ara/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    - prometheus-data:/prometheus
```

#### Kubernetes

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

### Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ara-api'
    static_configs:
      - targets: ['ara-api:8000']
    metrics_path: '/metrics'

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Key Metrics

```python
# In your application
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])

# Prediction metrics
prediction_count = Counter('predictions_total', 'Total predictions', ['symbol', 'asset_type'])
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration', ['symbol'])
prediction_confidence = Gauge('prediction_confidence', 'Prediction confidence', ['symbol'])

# Model metrics
model_load_time = Histogram('model_load_duration_seconds', 'Model load time', ['model_name'])
model_inference_time = Histogram('model_inference_duration_seconds', 'Model inference time', ['model_name'])

# Cache metrics
cache_hits = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
cache_misses = Counter('cache_misses_total', 'Cache misses', ['cache_type'])
```

## Grafana Setup

### Installation

#### Docker Compose

```yaml
grafana:
  image: grafana/grafana:latest
  ports:
    - "3000:3000"
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
    - GF_USERS_ALLOW_SIGN_UP=false
  volumes:
    - grafana-data:/var/lib/grafana
```

#### Kubernetes

```bash
# Already included with Prometheus Operator
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

### Add Prometheus Data Source

1. Login to Grafana (http://localhost:3000)
2. Go to Configuration → Data Sources
3. Add Prometheus data source
4. URL: http://prometheus:9090
5. Save & Test

### Import Dashboards

```bash
# Import pre-built dashboards
python -m ara.monitoring.grafana_dashboards --import

# Or manually import dashboard IDs:
# - Node Exporter: 1860
# - PostgreSQL: 9628
# - Redis: 11835
```

### Custom Dashboards

#### API Performance Dashboard

```json
{
  "dashboard": {
    "title": "ARA AI API Performance",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time (p95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## AlertManager Setup

### Configuration

Create `alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@ara-ai.com'
  smtp_auth_username: 'alerts@ara-ai.com'
  smtp_auth_password: 'your-password'

route:
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'team-email'
  routes:
    - match:
        severity: critical
      receiver: 'team-pager'
    - match:
        severity: warning
      receiver: 'team-email'

receivers:
  - name: 'team-email'
    email_configs:
      - to: 'team@ara-ai.com'
        headers:
          Subject: '[ARA AI] {{ .GroupLabels.alertname }}'

  - name: 'team-pager'
    pagerduty_configs:
      - service_key: 'your-pagerduty-key'

  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts'
        title: 'ARA AI Alert'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

### Alert Rules

Create `alert_rules.yml`:

```yaml
groups:
  - name: ara_api_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # High response time
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "P95 response time is {{ $value }}s"

      # Low prediction accuracy
      - alert: LowPredictionAccuracy
        expr: prediction_accuracy < 0.70
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Prediction accuracy below threshold"
          description: "Accuracy is {{ $value | humanizePercentage }}"

      # High memory usage
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"

      # Database connection issues
      - alert: DatabaseConnectionFailed
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failed"
          description: "Cannot connect to PostgreSQL"

      # Redis connection issues
      - alert: RedisConnectionFailed
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis connection failed"
          description: "Cannot connect to Redis"
```

## Log Aggregation (Optional)

### Loki Setup

```yaml
# docker-compose.yml
loki:
  image: grafana/loki:latest
  ports:
    - "3100:3100"
  volumes:
    - ./loki-config.yml:/etc/loki/local-config.yaml
    - loki-data:/loki

promtail:
  image: grafana/promtail:latest
  volumes:
    - /var/log:/var/log
    - ./promtail-config.yml:/etc/promtail/config.yml
  command: -config.file=/etc/promtail/config.yml
```

### Loki Configuration

```yaml
# loki-config.yml
auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1

schema_config:
  configs:
    - from: 2020-10-24
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks
```

## Distributed Tracing (Optional)

### Jaeger Setup

```yaml
# docker-compose.yml
jaeger:
  image: jaegertracing/all-in-one:latest
  ports:
    - "5775:5775/udp"
    - "6831:6831/udp"
    - "6832:6832/udp"
    - "5778:5778"
    - "16686:16686"
    - "14268:14268"
    - "14250:14250"
    - "9411:9411"
  environment:
    - COLLECTOR_ZIPKIN_HOST_PORT=:9411
```

### Application Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Use in code
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("predict"):
    # Your prediction code
    pass
```

## Health Checks

### Application Health Endpoint

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "checks": {
            "database": check_database(),
            "redis": check_redis(),
            "models": check_models()
        }
    }
```

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

## Monitoring Checklist

- [ ] Prometheus installed and configured
- [ ] Grafana installed with dashboards
- [ ] AlertManager configured with notification channels
- [ ] Alert rules defined
- [ ] Health checks implemented
- [ ] Log aggregation setup (optional)
- [ ] Distributed tracing setup (optional)
- [ ] Monitoring documented
- [ ] Team trained on monitoring tools
- [ ] Runbooks created for common alerts

## Support

For monitoring setup help:
- Documentation: https://docs.ara-ai.com/monitoring
- GitHub Issues: https://github.com/yourusername/ara-ai/issues
- Email: support@ara-ai.com
