# ARA AI Deployment Guide

Complete guide for deploying ARA AI Prediction System in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Single-Server Deployment](#single-server-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Environment Configuration](#environment-configuration)
6. [Security Setup](#security-setup)
7. [Monitoring Setup](#monitoring-setup)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum (Development)**:
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB
- OS: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

**Recommended (Production)**:
- CPU: 8+ cores
- RAM: 32 GB
- Storage: 200 GB SSD
- OS: Linux (Ubuntu 22.04 LTS)
- GPU: Optional (NVIDIA with CUDA support for faster predictions)

### Software Requirements

- Python 3.11+
- Docker 20.10+ (for containerized deployment)
- Docker Compose 2.0+ (for local development)
- Kubernetes 1.25+ (for production deployment)
- Helm 3.0+ (for Kubernetes deployment)
- PostgreSQL 15+ (for production database)
- Redis 7+ (for caching)
- Nginx (for reverse proxy)

## Single-Server Deployment

### Step 1: Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip \
    postgresql-15 redis-server nginx git

# macOS
brew install python@3.11 postgresql@15 redis nginx
```

### Step 2: Clone Repository

```bash
git clone https://github.com/yourusername/ara-ai.git
cd ara-ai
```

### Step 3: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Configure Database

```bash
# Create PostgreSQL database
sudo -u postgres psql
```

```sql
CREATE DATABASE ara_ai;
CREATE USER ara_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE ara_ai TO ara_user;
\q
```

### Step 6: Configure Environment

```bash
# Copy environment template
cp deployment/config/production.env .env

# Edit .env file with your settings
nano .env
```

Set the following required variables:
```
POSTGRES_PASSWORD=your_secure_password
JWT_SECRET_KEY=your_jwt_secret_key
ARA_MASTER_KEY=your_master_encryption_key
```

### Step 7: Initialize Database

```bash
# Run database migrations (if applicable)
python -m ara.utils.init_db
```

### Step 8: Start Services

```bash
# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Start API server
python run_api.py
```

### Step 9: Configure Nginx

```bash
# Copy Nginx configuration
sudo cp deployment/nginx/nginx.conf /etc/nginx/sites-available/ara-ai
sudo ln -s /etc/nginx/sites-available/ara-ai /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

### Step 10: Setup SSL Certificate

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d api.ara-ai.com
```

## Docker Deployment

### Step 1: Build Docker Image

```bash
# Build image
docker build -t ara-ai:latest .

# Verify image
docker images | grep ara-ai
```

### Step 2: Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### Step 3: Initialize Database

```bash
# Run migrations
docker-compose exec api python -m ara.utils.init_db
```

### Step 4: Access Services

- API: http://localhost:8000
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### Step 5: Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Kubernetes Deployment

### Step 1: Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installations
kubectl version --client
helm version
```

### Step 2: Configure kubectl

```bash
# Set up kubeconfig (example for GKE)
gcloud container clusters get-credentials ara-cluster --region us-central1

# Verify connection
kubectl cluster-info
```

### Step 3: Create Namespace

```bash
kubectl apply -f deployment/kubernetes/namespace.yaml
```

### Step 4: Create Secrets

```bash
# Create secrets from environment file
kubectl create secret generic ara-secrets \
  --from-literal=POSTGRES_PASSWORD=your_secure_password \
  --from-literal=ARA_MASTER_KEY=your_master_key \
  --from-literal=JWT_SECRET_KEY=your_jwt_secret \
  -n ara-ai
```

### Step 5: Deploy with kubectl

```bash
# Apply all configurations
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/pvc.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml
kubectl apply -f deployment/kubernetes/ingress.yaml
kubectl apply -f deployment/kubernetes/hpa.yaml

# Check deployment status
kubectl get all -n ara-ai
```

### Step 6: Deploy with Helm

```bash
# Install chart
helm install ara-ai deployment/helm/ara-ai \
  --namespace ara-ai \
  --create-namespace \
  --values deployment/helm/ara-ai/values.yaml

# Check status
helm status ara-ai -n ara-ai

# Upgrade deployment
helm upgrade ara-ai deployment/helm/ara-ai \
  --namespace ara-ai \
  --values deployment/helm/ara-ai/values.yaml
```

### Step 7: Verify Deployment

```bash
# Check pods
kubectl get pods -n ara-ai

# Check services
kubectl get svc -n ara-ai

# Check ingress
kubectl get ingress -n ara-ai

# View logs
kubectl logs -f deployment/ara-api -n ara-ai
```

### Step 8: Access Application

```bash
# Port forward for testing
kubectl port-forward svc/ara-api 8000:8000 -n ara-ai

# Access via ingress (after DNS setup)
curl https://api.ara-ai.com/health
```

## Environment Configuration

### Development Environment

```bash
# Use development configuration
cp deployment/config/development.env .env

# Key settings for development
ARA_ENV=development
ARA_LOG_LEVEL=DEBUG
ENABLE_DOCS=true
RATE_LIMIT_ENABLED=false
```

### Staging Environment

```bash
# Use staging configuration
cp deployment/config/staging.env .env

# Key settings for staging
ARA_ENV=staging
ARA_LOG_LEVEL=INFO
ENABLE_DOCS=true
RATE_LIMIT_ENABLED=true
```

### Production Environment

```bash
# Use production configuration
cp deployment/config/production.env .env

# Key settings for production
ARA_ENV=production
ARA_LOG_LEVEL=WARNING
ENABLE_DOCS=false
RATE_LIMIT_ENABLED=true
```

## Security Setup

### 1. Generate Secure Keys

```bash
# Generate JWT secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate master encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### 2. Configure Firewall

```bash
# Ubuntu/Debian with UFW
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# Check status
sudo ufw status
```

### 3. Setup SSL/TLS

```bash
# Using Let's Encrypt
sudo certbot --nginx -d api.ara-ai.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

### 4. Secure Database

```bash
# PostgreSQL security
sudo -u postgres psql

# Change default password
ALTER USER postgres WITH PASSWORD 'new_secure_password';

# Restrict connections (edit pg_hba.conf)
sudo nano /etc/postgresql/15/main/pg_hba.conf
```

### 5. Enable Security Features

```bash
# In .env file
ENABLE_RATE_LIMITING=true
ENABLE_AUDIT_LOGGING=true
ENABLE_INPUT_VALIDATION=true
```

## Monitoring Setup

### 1. Prometheus

```bash
# Install Prometheus
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/ara/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### 2. Grafana

```bash
# Install Grafana
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  grafana/grafana

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

### 3. Setup Dashboards

```bash
# Import pre-built dashboards
python -m ara.monitoring.setup_monitoring
```

### 4. Configure Alerts

```bash
# Edit alertmanager configuration
nano ara/monitoring/alertmanager.yml

# Restart alertmanager
docker restart alertmanager
```

## Troubleshooting

### Common Issues

#### 1. API Not Starting

```bash
# Check logs
docker-compose logs api

# Common causes:
# - Database connection failed
# - Missing environment variables
# - Port already in use

# Solutions:
# - Verify database is running: docker-compose ps
# - Check .env file
# - Change port in docker-compose.yml
```

#### 2. Database Connection Error

```bash
# Test database connection
docker-compose exec postgres psql -U ara_user -d ara_ai

# If connection fails:
# - Check POSTGRES_* environment variables
# - Verify postgres service is running
# - Check network connectivity
```

#### 3. High Memory Usage

```bash
# Check memory usage
docker stats

# Solutions:
# - Reduce API_WORKERS in .env
# - Increase container memory limits
# - Enable model caching
```

#### 4. Slow Predictions

```bash
# Check performance metrics
curl http://localhost:8000/metrics

# Solutions:
# - Enable Redis caching
# - Use GPU acceleration
# - Optimize model loading
# - Increase worker count
```

### Health Checks

```bash
# API health check
curl http://localhost:8000/health

# Database health check
docker-compose exec postgres pg_isready

# Redis health check
docker-compose exec redis redis-cli ping
```

### Logs

```bash
# View API logs
docker-compose logs -f api

# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f postgres
docker-compose logs -f redis
```

### Backup and Restore

```bash
# Backup database
docker-compose exec postgres pg_dump -U ara_user ara_ai > backup.sql

# Restore database
docker-compose exec -T postgres psql -U ara_user ara_ai < backup.sql

# Backup models
tar -czf models_backup.tar.gz models/

# Restore models
tar -xzf models_backup.tar.gz
```

## Performance Tuning

### 1. Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_predictions_symbol ON predictions(symbol);
CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);

-- Analyze tables
ANALYZE predictions;
```

### 2. Redis Configuration

```bash
# Edit redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```

### 3. API Optimization

```bash
# In .env file
API_WORKERS=8  # Set to number of CPU cores
CACHE_TTL_SECONDS=300
MAX_WORKERS=16
```

## Scaling

### Horizontal Scaling (Kubernetes)

```bash
# Scale API pods
kubectl scale deployment ara-api --replicas=5 -n ara-ai

# Enable autoscaling
kubectl autoscale deployment ara-api \
  --min=3 --max=10 \
  --cpu-percent=70 \
  -n ara-ai
```

### Vertical Scaling

```bash
# Increase resources in deployment.yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

## Support

For additional help:
- Documentation: https://docs.ara-ai.com
- GitHub Issues: https://github.com/yourusername/ara-ai/issues
- Email: support@ara-ai.com
