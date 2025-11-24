# ARA AI Scaling Guide

Guide for scaling ARA AI Prediction System from single server to multi-server deployment.

## Table of Contents

1. [Scaling Overview](#scaling-overview)
2. [Vertical Scaling](#vertical-scaling)
3. [Horizontal Scaling](#horizontal-scaling)
4. [Database Scaling](#database-scaling)
5. [Caching Strategy](#caching-strategy)
6. [Load Balancing](#load-balancing)
7. [Performance Monitoring](#performance-monitoring)
8. [Cost Optimization](#cost-optimization)

## Scaling Overview

### When to Scale

Scale your deployment when you observe:
- **CPU Usage** > 70% sustained
- **Memory Usage** > 80% sustained
- **Response Time** > 2 seconds (p95)
- **Request Rate** > 1000 req/min per server
- **Database Connections** > 80% of max

### Scaling Strategies

1. **Vertical Scaling**: Increase resources (CPU, RAM) of existing servers
2. **Horizontal Scaling**: Add more servers/pods
3. **Database Scaling**: Read replicas, sharding
4. **Caching**: Redis cluster, CDN
5. **Async Processing**: Background workers for heavy tasks

## Vertical Scaling

### Single Server Upgrade

#### 1. Increase Server Resources

**AWS EC2 Example**:
```bash
# Stop instance
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Change instance type
aws ec2 modify-instance-attribute \
  --instance-id i-1234567890abcdef0 \
  --instance-type t3.2xlarge

# Start instance
aws ec2 start-instances --instance-ids i-1234567890abcdef0
```

**Recommended Instance Types**:
- Small: t3.large (2 vCPU, 8 GB RAM) - 100 req/min
- Medium: t3.xlarge (4 vCPU, 16 GB RAM) - 500 req/min
- Large: t3.2xlarge (8 vCPU, 32 GB RAM) - 1000 req/min
- X-Large: c5.4xlarge (16 vCPU, 32 GB RAM) - 2000 req/min

#### 2. Optimize Application Settings

```bash
# Edit .env file
API_WORKERS=16  # Increase workers (2x CPU cores)
MAX_WORKERS=32  # Increase max workers
CACHE_SIZE_MB=4096  # Increase cache size
```

#### 3. Enable GPU Acceleration

```bash
# Install CUDA drivers
sudo apt-get install nvidia-driver-525 nvidia-cuda-toolkit

# Update .env
ENABLE_GPU=true
GPU_MEMORY_FRACTION=0.8
```

### Docker Resource Limits

```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 16G
```

## Horizontal Scaling

### Multi-Server Deployment

#### 1. Kubernetes Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ara-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ara-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

Apply:
```bash
kubectl apply -f hpa.yaml
```

#### 2. Manual Scaling

```bash
# Scale to 10 replicas
kubectl scale deployment ara-api --replicas=10 -n ara-ai

# Verify scaling
kubectl get pods -n ara-ai
```

#### 3. Docker Swarm Scaling

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml ara

# Scale service
docker service scale ara_api=5
```

### Stateless Design

Ensure your application is stateless:

```python
# ❌ Bad: Storing state in memory
class PredictionEngine:
    def __init__(self):
        self.cache = {}  # Don't do this!

# ✅ Good: Using external cache
class PredictionEngine:
    def __init__(self, redis_client):
        self.cache = redis_client
```

## Database Scaling

### Read Replicas

#### 1. PostgreSQL Replication

**Primary Server** (postgresql.conf):
```ini
wal_level = replica
max_wal_senders = 3
wal_keep_size = 64
```

**Replica Server**:
```bash
# Create replica
pg_basebackup -h primary-host -D /var/lib/postgresql/data -U replication -P

# Configure standby.signal
touch /var/lib/postgresql/data/standby.signal
```

#### 2. Application Configuration

```python
# config.py
DATABASE_URLS = {
    'primary': 'postgresql://user:pass@primary:5432/ara_ai',
    'replica1': 'postgresql://user:pass@replica1:5432/ara_ai',
    'replica2': 'postgresql://user:pass@replica2:5432/ara_ai',
}

# Use primary for writes, replicas for reads
def get_db_connection(read_only=False):
    if read_only:
        return random.choice([
            DATABASE_URLS['replica1'],
            DATABASE_URLS['replica2']
        ])
    return DATABASE_URLS['primary']
```

### Connection Pooling

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

### Database Sharding

For very large datasets:

```python
# Shard by symbol
def get_shard(symbol: str) -> str:
    shard_id = hash(symbol) % NUM_SHARDS
    return f"postgresql://user:pass@shard{shard_id}:5432/ara_ai"
```

## Caching Strategy

### Redis Cluster

#### 1. Setup Redis Cluster

```bash
# Create 6 Redis instances (3 masters, 3 replicas)
for port in 7000 7001 7002 7003 7004 7005; do
  redis-server --port $port --cluster-enabled yes \
    --cluster-config-file nodes-${port}.conf \
    --cluster-node-timeout 5000 \
    --appendonly yes \
    --daemonize yes
done

# Create cluster
redis-cli --cluster create \
  127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002 \
  127.0.0.1:7003 127.0.0.1:7004 127.0.0.1:7005 \
  --cluster-replicas 1
```

#### 2. Application Configuration

```python
from redis.cluster import RedisCluster

# Connect to cluster
redis_client = RedisCluster(
    startup_nodes=[
        {"host": "redis1", "port": 7000},
        {"host": "redis2", "port": 7001},
        {"host": "redis3", "port": 7002},
    ],
    decode_responses=True
)
```

### Multi-Level Caching

```python
class CacheManager:
    def __init__(self):
        self.l1_cache = {}  # In-memory (100 MB)
        self.l2_cache = redis_client  # Redis (1 GB)
        self.l3_cache = db_client  # Database
    
    def get(self, key):
        # Try L1
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Try L2
        value = self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value
        
        # Try L3
        value = self.l3_cache.get(key)
        if value:
            self.l2_cache.set(key, value, ex=300)
            self.l1_cache[key] = value
            return value
        
        return None
```

### CDN for Static Assets

```nginx
# Nginx configuration
location /static/ {
    alias /var/www/static/;
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## Load Balancing

### Nginx Load Balancer

```nginx
upstream ara_api {
    least_conn;  # Use least connections algorithm
    
    server api1.ara-ai.com:8000 weight=3 max_fails=3 fail_timeout=30s;
    server api2.ara-ai.com:8000 weight=3 max_fails=3 fail_timeout=30s;
    server api3.ara-ai.com:8000 weight=2 max_fails=3 fail_timeout=30s;
    
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name api.ara-ai.com;
    
    location / {
        proxy_pass http://ara_api;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### AWS Application Load Balancer

```bash
# Create target group
aws elbv2 create-target-group \
  --name ara-api-targets \
  --protocol HTTP \
  --port 8000 \
  --vpc-id vpc-12345678 \
  --health-check-path /health

# Create load balancer
aws elbv2 create-load-balancer \
  --name ara-api-lb \
  --subnets subnet-12345678 subnet-87654321 \
  --security-groups sg-12345678

# Register targets
aws elbv2 register-targets \
  --target-group-arn arn:aws:elasticloadbalancing:... \
  --targets Id=i-1234567890abcdef0 Id=i-0fedcba0987654321
```

### Kubernetes Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: ara-api
spec:
  type: LoadBalancer
  selector:
    app: ara-api
  ports:
  - port: 80
    targetPort: 8000
  sessionAffinity: ClientIP  # Sticky sessions
```

## Performance Monitoring

### Metrics to Track

1. **Request Metrics**:
   - Requests per second
   - Response time (p50, p95, p99)
   - Error rate

2. **Resource Metrics**:
   - CPU usage per pod/server
   - Memory usage per pod/server
   - Network I/O

3. **Application Metrics**:
   - Prediction latency
   - Cache hit rate
   - Database query time
   - Model inference time

### Prometheus Queries

```promql
# Request rate
rate(http_requests_total[5m])

# Average response time
rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# CPU usage
rate(container_cpu_usage_seconds_total[5m])

# Memory usage
container_memory_usage_bytes / container_spec_memory_limit_bytes
```

### Grafana Dashboards

Import pre-built dashboards:
```bash
python -m ara.monitoring.setup_monitoring
```

## Cost Optimization

### 1. Right-Sizing

```bash
# Analyze resource usage
kubectl top pods -n ara-ai

# Adjust resource requests/limits
kubectl set resources deployment ara-api \
  --requests=cpu=500m,memory=1Gi \
  --limits=cpu=2000m,memory=4Gi \
  -n ara-ai
```

### 2. Spot Instances (AWS)

```yaml
# Kubernetes node group with spot instances
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ara-cluster
nodeGroups:
  - name: spot-workers
    instancesDistribution:
      instanceTypes: ["t3.large", "t3.xlarge"]
      onDemandBaseCapacity: 2
      onDemandPercentageAboveBaseCapacity: 0
      spotInstancePools: 2
    minSize: 3
    maxSize: 20
```

### 3. Auto-Scaling Policies

```yaml
# Scale down during off-peak hours
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ara-api-hpa
spec:
  minReplicas: 3  # Peak hours
  maxReplicas: 10
  # Use CronHPA for time-based scaling
```

### 4. Caching Optimization

```python
# Increase cache TTL for stable data
CACHE_TTL = {
    'historical_data': 3600,  # 1 hour
    'predictions': 300,  # 5 minutes
    'models': 86400,  # 24 hours
}
```

## Scaling Checklist

- [ ] Enable horizontal pod autoscaling
- [ ] Configure database read replicas
- [ ] Setup Redis cluster for caching
- [ ] Implement connection pooling
- [ ] Configure load balancer
- [ ] Setup monitoring and alerting
- [ ] Optimize database queries
- [ ] Enable CDN for static assets
- [ ] Implement rate limiting
- [ ] Setup backup and disaster recovery
- [ ] Document scaling procedures
- [ ] Test failover scenarios
- [ ] Optimize model loading
- [ ] Enable GPU acceleration (if needed)
- [ ] Configure auto-scaling policies

## Troubleshooting

### High Latency

1. Check database query performance
2. Verify cache hit rate
3. Monitor network latency
4. Check model inference time
5. Review load balancer configuration

### Memory Leaks

1. Monitor memory usage over time
2. Check for unclosed connections
3. Review cache eviction policies
4. Profile application memory
5. Restart pods periodically

### Database Bottlenecks

1. Add database indexes
2. Optimize slow queries
3. Increase connection pool size
4. Add read replicas
5. Consider database sharding

## Support

For scaling assistance:
- Documentation: https://docs.ara-ai.com/scaling
- GitHub Issues: https://github.com/yourusername/ara-ai/issues
- Email: support@ara-ai.com
