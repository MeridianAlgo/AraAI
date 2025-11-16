# ARA AI Deployment

Complete deployment resources for ARA AI Prediction System.

## Quick Links

- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) - Complete deployment instructions
- [Scaling Guide](docs/SCALING_GUIDE.md) - Scale from single server to multi-server
- [Kubernetes Guide](docs/KUBERNETES_GUIDE.md) - Kubernetes-specific deployment
- [Monitoring Setup](docs/MONITORING_SETUP.md) - Setup monitoring and observability
- [Disaster Recovery](docs/DISASTER_RECOVERY.md) - Backup and recovery procedures

## Directory Structure

```
deployment/
├── README.md                    # This file
├── Dockerfile                   # Production Docker image
├── docker-compose.yml          # Local development with Docker
├── config/                     # Environment configurations
│   ├── development.env         # Development settings
│   ├── staging.env            # Staging settings
│   └── production.env         # Production settings
├── kubernetes/                 # Kubernetes manifests
│   ├── namespace.yaml         # Namespace definition
│   ├── configmap.yaml         # Configuration
│   ├── secrets.yaml           # Secrets template
│   ├── deployment.yaml        # Deployments
│   ├── service.yaml           # Services
│   ├── ingress.yaml           # Ingress configuration
│   ├── pvc.yaml               # Persistent volumes
│   └── hpa.yaml               # Horizontal pod autoscaler
├── helm/                       # Helm charts
│   └── ara-ai/
│       ├── Chart.yaml         # Chart metadata
│       ├── values.yaml        # Default values
│       └── templates/         # Kubernetes templates
├── nginx/                      # Nginx configuration
│   └── nginx.conf             # Reverse proxy config
└── docs/                       # Deployment documentation
    ├── DEPLOYMENT_GUIDE.md
    ├── SCALING_GUIDE.md
    ├── KUBERNETES_GUIDE.md
    ├── MONITORING_SETUP.md
    └── DISASTER_RECOVERY.md
```

## Deployment Options

### 1. Single Server (Development/Small Scale)

Best for: Development, testing, small deployments (<100 req/min)

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp deployment/config/development.env .env
nano .env

# Start services
python run_api.py
```

[Full Guide →](docs/DEPLOYMENT_GUIDE.md#single-server-deployment)

### 2. Docker Compose (Local Development)

Best for: Local development, testing, small teams

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

[Full Guide →](docs/DEPLOYMENT_GUIDE.md#docker-deployment)

### 3. Kubernetes (Production)

Best for: Production, high availability, auto-scaling

```bash
# Deploy with kubectl
kubectl apply -f deployment/kubernetes/

# Or deploy with Helm
helm install ara-ai deployment/helm/ara-ai \
  --namespace ara-ai \
  --create-namespace
```

[Full Guide →](docs/KUBERNETES_GUIDE.md)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker 20.10+ (for containerized deployment)
- Kubernetes 1.25+ (for production deployment)
- PostgreSQL 15+ or SQLite
- Redis 7+

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/ara-ai.git
cd ara-ai
```

### 2. Choose Deployment Method

#### Option A: Local Development

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp deployment/config/development.env .env

# Run
python run_api.py
```

#### Option B: Docker Compose

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# Access API
curl http://localhost:8000/health
```

#### Option C: Kubernetes

```bash
# Create namespace
kubectl create namespace ara-ai

# Create secrets
kubectl create secret generic ara-secrets \
  --from-literal=POSTGRES_PASSWORD=your_password \
  --from-literal=JWT_SECRET_KEY=your_jwt_secret \
  --from-literal=ARA_MASTER_KEY=your_master_key \
  -n ara-ai

# Deploy
kubectl apply -f deployment/kubernetes/

# Check status
kubectl get pods -n ara-ai
```

### 3. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "days": 7}'
```

## Environment Configuration

### Development

```bash
ARA_ENV=development
ARA_LOG_LEVEL=DEBUG
ENABLE_DOCS=true
RATE_LIMIT_ENABLED=false
```

### Staging

```bash
ARA_ENV=staging
ARA_LOG_LEVEL=INFO
ENABLE_DOCS=true
RATE_LIMIT_ENABLED=true
```

### Production

```bash
ARA_ENV=production
ARA_LOG_LEVEL=WARNING
ENABLE_DOCS=false
RATE_LIMIT_ENABLED=true
```

[Full Configuration Guide →](docs/DEPLOYMENT_GUIDE.md#environment-configuration)

## Security

### Required Secrets

Generate secure secrets before deployment:

```bash
# JWT secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Master encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# PostgreSQL password
openssl rand -base64 32
```

### SSL/TLS Setup

```bash
# Using Let's Encrypt
sudo certbot --nginx -d api.ara-ai.com

# Or use your own certificates
# Place in deployment/nginx/ssl/
```

[Full Security Guide →](docs/DEPLOYMENT_GUIDE.md#security-setup)

## Monitoring

### Setup Monitoring Stack

```bash
# Automated setup
python -m ara.monitoring.setup_monitoring

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Key Metrics

- Request rate and latency
- Error rate
- Prediction accuracy
- Resource usage (CPU, memory)
- Cache hit rate
- Database performance

[Full Monitoring Guide →](docs/MONITORING_SETUP.md)

## Scaling

### Horizontal Scaling

```bash
# Kubernetes
kubectl scale deployment ara-api --replicas=10 -n ara-ai

# Enable autoscaling
kubectl autoscale deployment ara-api \
  --min=3 --max=10 \
  --cpu-percent=70 \
  -n ara-ai
```

### Vertical Scaling

```bash
# Increase resources
kubectl set resources deployment ara-api \
  --requests=cpu=2000m,memory=4Gi \
  --limits=cpu=4000m,memory=8Gi \
  -n ara-ai
```

[Full Scaling Guide →](docs/SCALING_GUIDE.md)

## Backup and Recovery

### Automated Backups

```bash
# Database backup (every 6 hours)
0 */6 * * * /usr/local/bin/backup-postgres.sh

# Model backup (daily)
0 2 * * * /usr/local/bin/backup-models.sh
```

### Recovery

```bash
# Complete system recovery
./deployment/scripts/complete-recovery.sh
```

[Full DR Guide →](docs/DISASTER_RECOVERY.md)

## Troubleshooting

### Common Issues

#### API Not Starting

```bash
# Check logs
docker-compose logs api
kubectl logs -n ara-ai deployment/ara-api

# Common causes:
# - Database connection failed
# - Missing environment variables
# - Port already in use
```

#### Database Connection Error

```bash
# Test connection
docker-compose exec postgres psql -U ara_user -d ara_ai

# Verify environment variables
echo $POSTGRES_HOST
echo $POSTGRES_PORT
```

#### High Memory Usage

```bash
# Check resource usage
docker stats
kubectl top pods -n ara-ai

# Solutions:
# - Reduce API_WORKERS
# - Increase memory limits
# - Enable caching
```

[Full Troubleshooting Guide →](docs/DEPLOYMENT_GUIDE.md#troubleshooting)

## Performance Tuning

### Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_predictions_symbol ON predictions(symbol);
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
```

### API Optimization

```bash
# Increase workers
API_WORKERS=8

# Enable caching
CACHE_TTL_SECONDS=300

# Use GPU
ENABLE_GPU=true
```

### Caching Strategy

```bash
# Redis configuration
maxmemory 2gb
maxmemory-policy allkeys-lru
```

## Maintenance

### Update Application

```bash
# Docker
docker-compose pull
docker-compose up -d

# Kubernetes
kubectl set image deployment/ara-api \
  ara-api=ara-ai:v1.1.0 \
  -n ara-ai
```

### Rollback

```bash
# Kubernetes
kubectl rollout undo deployment/ara-api -n ara-ai

# Docker Compose
docker-compose down
docker-compose up -d
```

## Support

### Documentation

- [Main Documentation](../docs/DOCUMENTATION_INDEX.md)
- [API Documentation](../docs/API_DOCUMENTATION.md)
- [User Manual](../docs/USER_MANUAL.md)

### Community

- GitHub Issues: https://github.com/yourusername/ara-ai/issues
- Discussions: https://github.com/yourusername/ara-ai/discussions
- Email: support@ara-ai.com

### Commercial Support

For enterprise support, SLA, and consulting:
- Email: enterprise@ara-ai.com
- Website: https://ara-ai.com/enterprise

## License

See [LICENSE](../LICENSE) file for details.
