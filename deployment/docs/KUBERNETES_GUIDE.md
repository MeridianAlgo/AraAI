# ARA AI Kubernetes Deployment Guide

Complete guide for deploying ARA AI on Kubernetes clusters.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Cluster Setup](#cluster-setup)
3. [Deployment Steps](#deployment-steps)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Maintenance](#maintenance)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

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

### Kubernetes Cluster

You need a Kubernetes cluster (1.25+). Options:

1. **Local Development**: Minikube, Kind, Docker Desktop
2. **Cloud Providers**: GKE, EKS, AKS
3. **Self-Hosted**: kubeadm, k3s, RKE

## Cluster Setup

### Option 1: Minikube (Local Development)

```bash
# Install Minikube
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

# Start cluster
minikube start --cpus=4 --memory=8192 --disk-size=50g

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server

# Verify
kubectl cluster-info
```

### Option 2: Google Kubernetes Engine (GKE)

```bash
# Create cluster
gcloud container clusters create ara-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials ara-cluster --zone us-central1-a

# Verify
kubectl get nodes
```

### Option 3: Amazon EKS

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster
eksctl create cluster \
  --name ara-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed

# Verify
kubectl get nodes
```

### Option 4: Azure AKS

```bash
# Create resource group
az group create --name ara-rg --location eastus

# Create cluster
az aks create \
  --resource-group ara-rg \
  --name ara-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10 \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group ara-rg --name ara-cluster

# Verify
kubectl get nodes
```

## Deployment Steps

### Step 1: Create Namespace

```bash
kubectl apply -f deployment/kubernetes/namespace.yaml
```

### Step 2: Create Secrets

```bash
# Generate secure passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -base64 32)
MASTER_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Create secret
kubectl create secret generic ara-secrets \
  --from-literal=POSTGRES_PASSWORD=$POSTGRES_PASSWORD \
  --from-literal=JWT_SECRET_KEY=$JWT_SECRET \
  --from-literal=ARA_MASTER_KEY=$MASTER_KEY \
  -n ara-ai

# Verify
kubectl get secrets -n ara-ai
```

### Step 3: Create ConfigMap

```bash
kubectl apply -f deployment/kubernetes/configmap.yaml
```

### Step 4: Create Persistent Volumes

```bash
kubectl apply -f deployment/kubernetes/pvc.yaml

# Verify
kubectl get pvc -n ara-ai
```

### Step 5: Deploy Services

```bash
# Deploy all services
kubectl apply -f deployment/kubernetes/deployment.yaml
kubectl apply -f deployment/kubernetes/service.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=ara-api -n ara-ai --timeout=300s

# Check status
kubectl get pods -n ara-ai
kubectl get svc -n ara-ai
```

### Step 6: Setup Ingress

```bash
# Install Nginx Ingress Controller (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Deploy ingress
kubectl apply -f deployment/kubernetes/ingress.yaml

# Get ingress IP
kubectl get ingress -n ara-ai
```

### Step 7: Configure DNS

```bash
# Get external IP
EXTERNAL_IP=$(kubectl get ingress ara-ingress -n ara-ai -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Add DNS record
# api.ara-ai.com -> $EXTERNAL_IP
```

### Step 8: Setup SSL Certificate

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@ara-ai.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Certificate will be automatically created by ingress
```

### Step 9: Enable Autoscaling

```bash
kubectl apply -f deployment/kubernetes/hpa.yaml

# Verify
kubectl get hpa -n ara-ai
```

## Configuration

### Using Helm

#### Install

```bash
# Install with default values
helm install ara-ai deployment/helm/ara-ai \
  --namespace ara-ai \
  --create-namespace

# Install with custom values
helm install ara-ai deployment/helm/ara-ai \
  --namespace ara-ai \
  --create-namespace \
  --values custom-values.yaml
```

#### Upgrade

```bash
# Upgrade deployment
helm upgrade ara-ai deployment/helm/ara-ai \
  --namespace ara-ai \
  --values custom-values.yaml

# Rollback if needed
helm rollback ara-ai -n ara-ai
```

#### Uninstall

```bash
helm uninstall ara-ai -n ara-ai
```

### Custom Values

Create `custom-values.yaml`:

```yaml
api:
  replicaCount: 5
  resources:
    requests:
      memory: "4Gi"
      cpu: "2000m"
    limits:
      memory: "8Gi"
      cpu: "4000m"

postgres:
  persistence:
    size: 200Gi

redis:
  persistence:
    size: 20Gi

ingress:
  hosts:
    - host: api.your-domain.com
      paths:
        - path: /
          pathType: Prefix
```

## Monitoring

### Install Prometheus Stack

```bash
# Add Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

### Access Grafana

```bash
# Port forward
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Get admin password
kubectl get secret -n monitoring prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Access at http://localhost:3000
```

### Import Dashboards

```bash
# Import ARA AI dashboards
python -m ara.monitoring.setup_monitoring --kubernetes
```

## Maintenance

### Update Application

```bash
# Build new image
docker build -t ara-ai:v1.1.0 .

# Push to registry
docker tag ara-ai:v1.1.0 your-registry/ara-ai:v1.1.0
docker push your-registry/ara-ai:v1.1.0

# Update deployment
kubectl set image deployment/ara-api \
  ara-api=your-registry/ara-ai:v1.1.0 \
  -n ara-ai

# Check rollout status
kubectl rollout status deployment/ara-api -n ara-ai
```

### Rollback Deployment

```bash
# View rollout history
kubectl rollout history deployment/ara-api -n ara-ai

# Rollback to previous version
kubectl rollout undo deployment/ara-api -n ara-ai

# Rollback to specific revision
kubectl rollout undo deployment/ara-api --to-revision=2 -n ara-ai
```

### Backup Database

```bash
# Create backup job
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: postgres-backup
  namespace: ara-ai
spec:
  template:
    spec:
      containers:
      - name: backup
        image: postgres:15-alpine
        command:
        - /bin/sh
        - -c
        - |
          pg_dump -h ara-postgres -U ara_user ara_ai > /backup/backup-$(date +%Y%m%d-%H%M%S).sql
        env:
        - name: PGPASSWORD
          valueFrom:
            secretKeyRef:
              name: ara-secrets
              key: POSTGRES_PASSWORD
        volumeMounts:
        - name: backup
          mountPath: /backup
      volumes:
      - name: backup
        persistentVolumeClaim:
          claimName: backup-pvc
      restartPolicy: Never
EOF
```

### Scale Resources

```bash
# Scale replicas
kubectl scale deployment ara-api --replicas=10 -n ara-ai

# Update resource limits
kubectl set resources deployment ara-api \
  --requests=cpu=2000m,memory=4Gi \
  --limits=cpu=4000m,memory=8Gi \
  -n ara-ai
```

## Troubleshooting

### Check Pod Status

```bash
# List pods
kubectl get pods -n ara-ai

# Describe pod
kubectl describe pod <pod-name> -n ara-ai

# View logs
kubectl logs <pod-name> -n ara-ai

# Follow logs
kubectl logs -f <pod-name> -n ara-ai

# Previous logs (if pod crashed)
kubectl logs <pod-name> -n ara-ai --previous
```

### Debug Pod

```bash
# Execute shell in pod
kubectl exec -it <pod-name> -n ara-ai -- /bin/bash

# Run command in pod
kubectl exec <pod-name> -n ara-ai -- python -c "import ara; print(ara.__version__)"
```

### Check Events

```bash
# View events
kubectl get events -n ara-ai --sort-by='.lastTimestamp'

# Watch events
kubectl get events -n ara-ai --watch
```

### Network Issues

```bash
# Test service connectivity
kubectl run -it --rm debug --image=busybox --restart=Never -n ara-ai -- sh

# Inside pod:
wget -O- http://ara-api:8000/health
nslookup ara-api
```

### Resource Issues

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n ara-ai

# Check resource quotas
kubectl describe resourcequota -n ara-ai
```

### Common Issues

#### 1. ImagePullBackOff

```bash
# Check image name and registry
kubectl describe pod <pod-name> -n ara-ai

# Solution: Verify image exists and credentials are correct
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n ara-ai
```

#### 2. CrashLoopBackOff

```bash
# Check logs
kubectl logs <pod-name> -n ara-ai --previous

# Common causes:
# - Missing environment variables
# - Database connection failed
# - Application error

# Solution: Fix configuration and redeploy
```

#### 3. Pending Pods

```bash
# Check why pod is pending
kubectl describe pod <pod-name> -n ara-ai

# Common causes:
# - Insufficient resources
# - PVC not bound
# - Node selector mismatch

# Solution: Add more nodes or adjust resource requests
```

## Best Practices

1. **Use namespaces** for isolation
2. **Set resource limits** to prevent resource exhaustion
3. **Use liveness and readiness probes** for health checks
4. **Enable autoscaling** for dynamic workloads
5. **Use secrets** for sensitive data
6. **Implement monitoring** and alerting
7. **Regular backups** of data
8. **Use rolling updates** for zero-downtime deployments
9. **Implement network policies** for security
10. **Document** your deployment

## Support

For Kubernetes deployment help:
- Documentation: https://docs.ara-ai.com/kubernetes
- GitHub Issues: https://github.com/yourusername/ara-ai/issues
- Email: support@ara-ai.com
