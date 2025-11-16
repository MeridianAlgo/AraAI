# ARA AI Disaster Recovery Procedures

Complete disaster recovery and business continuity plan for ARA AI.

## Table of Contents

1. [Overview](#overview)
2. [Backup Strategy](#backup-strategy)
3. [Recovery Procedures](#recovery-procedures)
4. [Failover Procedures](#failover-procedures)
5. [Testing](#testing)
6. [Incident Response](#incident-response)

## Overview

### Recovery Objectives

- **RTO (Recovery Time Objective)**: 4 hours
- **RPO (Recovery Point Objective)**: 1 hour
- **Data Retention**: 90 days

### Critical Components

1. **Database**: PostgreSQL (highest priority)
2. **Models**: Trained ML models
3. **Configuration**: Environment variables, secrets
4. **Cache**: Redis (can be rebuilt)
5. **Logs**: Audit logs, application logs

## Backup Strategy

### Automated Backups

#### Database Backups

```bash
# Create backup script
cat > /usr/local/bin/backup-postgres.sh << 'EOF'
#!/bin/bash
set -e

BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/ara_ai_$TIMESTAMP.sql.gz"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump -h localhost -U ara_user ara_ai | gzip > $BACKUP_FILE

# Keep only last 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

# Upload to S3 (optional)
aws s3 cp $BACKUP_FILE s3://ara-backups/postgres/

echo "Backup completed: $BACKUP_FILE"
EOF

chmod +x /usr/local/bin/backup-postgres.sh

# Schedule with cron (every 6 hours)
crontab -e
# Add: 0 */6 * * * /usr/local/bin/backup-postgres.sh
```

#### Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: ara-ai
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
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
              TIMESTAMP=$(date +%Y%m%d_%H%M%S)
              pg_dump -h ara-postgres -U ara_user ara_ai | gzip > /backup/ara_ai_$TIMESTAMP.sql.gz
              # Upload to S3
              aws s3 cp /backup/ara_ai_$TIMESTAMP.sql.gz s3://ara-backups/postgres/
              # Cleanup old backups
              find /backup -name "*.sql.gz" -mtime +30 -delete
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: ara-secrets
                  key: POSTGRES_PASSWORD
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: access-key-id
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: aws-credentials
                  key: secret-access-key
            volumeMounts:
            - name: backup
              mountPath: /backup
          volumes:
          - name: backup
            persistentVolumeClaim:
              claimName: backup-pvc
          restartPolicy: OnFailure
```

#### Model Backups

```bash
# Backup models
cat > /usr/local/bin/backup-models.sh << 'EOF'
#!/bin/bash
set -e

BACKUP_DIR="/backups/models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/models_$TIMESTAMP.tar.gz"

# Create backup
tar -czf $BACKUP_FILE /app/models/

# Upload to S3
aws s3 cp $BACKUP_FILE s3://ara-backups/models/

# Keep only last 7 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Models backup completed: $BACKUP_FILE"
EOF

chmod +x /usr/local/bin/backup-models.sh

# Schedule daily
# crontab: 0 2 * * * /usr/local/bin/backup-models.sh
```

#### Configuration Backups

```bash
# Backup secrets and configs
kubectl get secrets -n ara-ai -o yaml > secrets-backup.yaml
kubectl get configmaps -n ara-ai -o yaml > configmaps-backup.yaml

# Encrypt and upload
gpg --encrypt --recipient admin@ara-ai.com secrets-backup.yaml
aws s3 cp secrets-backup.yaml.gpg s3://ara-backups/config/
```

### Backup Verification

```bash
# Test database backup
gunzip -c backup.sql.gz | psql -h localhost -U ara_user ara_ai_test

# Test model backup
tar -tzf models_backup.tar.gz

# Verify S3 backups
aws s3 ls s3://ara-backups/ --recursive
```

## Recovery Procedures

### Database Recovery

#### Full Database Restore

```bash
# Stop application
kubectl scale deployment ara-api --replicas=0 -n ara-ai

# Download backup from S3
aws s3 cp s3://ara-backups/postgres/ara_ai_20231115_120000.sql.gz .

# Drop existing database
psql -h localhost -U postgres -c "DROP DATABASE ara_ai;"
psql -h localhost -U postgres -c "CREATE DATABASE ara_ai;"

# Restore backup
gunzip -c ara_ai_20231115_120000.sql.gz | psql -h localhost -U ara_user ara_ai

# Verify restoration
psql -h localhost -U ara_user ara_ai -c "SELECT COUNT(*) FROM predictions;"

# Restart application
kubectl scale deployment ara-api --replicas=3 -n ara-ai
```

#### Point-in-Time Recovery

```bash
# Restore base backup
gunzip -c base_backup.sql.gz | psql -h localhost -U ara_user ara_ai

# Apply WAL files up to specific time
pg_restore --target-time='2023-11-15 12:00:00' \
  -h localhost -U ara_user -d ara_ai \
  /path/to/wal/files
```

### Model Recovery

```bash
# Download model backup
aws s3 cp s3://ara-backups/models/models_20231115_020000.tar.gz .

# Extract models
tar -xzf models_20231115_020000.tar.gz -C /app/

# Verify models
python -c "
from ara.models import load_model
model = load_model('models/demo_ensemble.json')
print('Model loaded successfully')
"

# Restart application
kubectl rollout restart deployment/ara-api -n ara-ai
```

### Configuration Recovery

```bash
# Download encrypted backup
aws s3 cp s3://ara-backups/config/secrets-backup.yaml.gpg .

# Decrypt
gpg --decrypt secrets-backup.yaml.gpg > secrets-backup.yaml

# Apply secrets
kubectl apply -f secrets-backup.yaml

# Verify
kubectl get secrets -n ara-ai
```

### Complete System Recovery

```bash
#!/bin/bash
# complete-recovery.sh

set -e

echo "Starting complete system recovery..."

# 1. Restore database
echo "Restoring database..."
aws s3 cp s3://ara-backups/postgres/latest.sql.gz .
gunzip -c latest.sql.gz | psql -h localhost -U ara_user ara_ai

# 2. Restore models
echo "Restoring models..."
aws s3 cp s3://ara-backups/models/latest.tar.gz .
tar -xzf latest.tar.gz -C /app/

# 3. Restore configuration
echo "Restoring configuration..."
aws s3 cp s3://ara-backups/config/secrets-backup.yaml.gpg .
gpg --decrypt secrets-backup.yaml.gpg > secrets-backup.yaml
kubectl apply -f secrets-backup.yaml

# 4. Deploy application
echo "Deploying application..."
kubectl apply -f deployment/kubernetes/

# 5. Wait for pods to be ready
echo "Waiting for pods..."
kubectl wait --for=condition=ready pod -l app=ara-api -n ara-ai --timeout=300s

# 6. Verify health
echo "Verifying health..."
kubectl exec -n ara-ai deployment/ara-api -- curl -f http://localhost:8000/health

echo "Recovery completed successfully!"
```

## Failover Procedures

### Database Failover

#### Automatic Failover (PostgreSQL Replication)

```bash
# Setup streaming replication
# On primary server (postgresql.conf):
wal_level = replica
max_wal_senders = 3
wal_keep_size = 64

# On standby server:
# 1. Create base backup
pg_basebackup -h primary -D /var/lib/postgresql/data -U replication -P

# 2. Create standby.signal
touch /var/lib/postgresql/data/standby.signal

# 3. Configure recovery (postgresql.auto.conf)
primary_conninfo = 'host=primary port=5432 user=replication'
```

#### Manual Failover

```bash
# Promote standby to primary
pg_ctl promote -D /var/lib/postgresql/data

# Update application to use new primary
kubectl set env deployment/ara-api \
  POSTGRES_HOST=new-primary-host \
  -n ara-ai

# Restart application
kubectl rollout restart deployment/ara-api -n ara-ai
```

### Application Failover

#### Multi-Region Deployment

```bash
# Deploy to secondary region
kubectl config use-context us-west-2
kubectl apply -f deployment/kubernetes/

# Update DNS to point to secondary region
# Route53 example:
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.ara-ai.com",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "Z1234567890ABC",
          "DNSName": "us-west-2-lb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'
```

## Testing

### Backup Testing

```bash
# Monthly backup test
# 1. Restore to test environment
./restore-to-test.sh

# 2. Verify data integrity
python -m ara.utils.verify_data

# 3. Run smoke tests
pytest tests/test_smoke.py

# 4. Document results
echo "Backup test completed: $(date)" >> backup-test-log.txt
```

### Disaster Recovery Drill

```bash
# Quarterly DR drill
# 1. Simulate failure
kubectl delete namespace ara-ai

# 2. Execute recovery
./complete-recovery.sh

# 3. Verify functionality
curl https://api.ara-ai.com/health
python -m ara.cli predict AAPL --days 7

# 4. Measure RTO
# Record time from failure to full recovery

# 5. Document lessons learned
```

## Incident Response

### Severity Levels

**P0 - Critical**:
- Complete system outage
- Data loss
- Security breach
- Response time: Immediate

**P1 - High**:
- Partial system outage
- Degraded performance
- Response time: 1 hour

**P2 - Medium**:
- Non-critical feature unavailable
- Response time: 4 hours

**P3 - Low**:
- Minor issues
- Response time: 24 hours

### Incident Response Procedure

#### 1. Detection

```bash
# Automated alerts via AlertManager
# Manual detection via monitoring dashboards
# User reports
```

#### 2. Assessment

```bash
# Check system status
kubectl get pods -n ara-ai
kubectl get events -n ara-ai --sort-by='.lastTimestamp'

# Check metrics
curl http://prometheus:9090/api/v1/query?query=up

# Check logs
kubectl logs -n ara-ai deployment/ara-api --tail=100
```

#### 3. Communication

```bash
# Create incident channel
# Slack: #incident-2023-11-15

# Notify stakeholders
# Email: incidents@ara-ai.com

# Update status page
# https://status.ara-ai.com
```

#### 4. Mitigation

```bash
# Rollback deployment
kubectl rollout undo deployment/ara-api -n ara-ai

# Scale resources
kubectl scale deployment ara-api --replicas=10 -n ara-ai

# Failover to backup region
./failover-to-backup.sh
```

#### 5. Resolution

```bash
# Apply fix
kubectl apply -f fix.yaml

# Verify resolution
./verify-health.sh

# Monitor for 1 hour
watch -n 60 'kubectl get pods -n ara-ai'
```

#### 6. Post-Mortem

```markdown
# Incident Post-Mortem

## Summary
- Date: 2023-11-15
- Duration: 2 hours
- Severity: P1
- Impact: 30% of requests failed

## Timeline
- 10:00 - Incident detected
- 10:05 - Team notified
- 10:15 - Root cause identified
- 11:30 - Fix deployed
- 12:00 - Incident resolved

## Root Cause
Database connection pool exhausted

## Resolution
Increased connection pool size from 20 to 50

## Action Items
- [ ] Implement connection pool monitoring
- [ ] Add alerts for connection pool usage
- [ ] Update runbook
- [ ] Conduct training on connection pool tuning
```

## Contact Information

### On-Call Rotation

- **Primary**: oncall-primary@ara-ai.com
- **Secondary**: oncall-secondary@ara-ai.com
- **Manager**: oncall-manager@ara-ai.com

### Escalation Path

1. On-call engineer (0-15 min)
2. Team lead (15-30 min)
3. Engineering manager (30-60 min)
4. CTO (60+ min)

## Runbooks

### Common Scenarios

#### Database Connection Issues

```bash
# 1. Check database status
kubectl exec -n ara-ai deployment/ara-postgres -- pg_isready

# 2. Check connections
kubectl exec -n ara-ai deployment/ara-postgres -- \
  psql -U ara_user -c "SELECT count(*) FROM pg_stat_activity;"

# 3. Kill idle connections
kubectl exec -n ara-ai deployment/ara-postgres -- \
  psql -U ara_user -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle';"

# 4. Restart database if needed
kubectl rollout restart deployment/ara-postgres -n ara-ai
```

#### High Memory Usage

```bash
# 1. Identify memory-hungry pods
kubectl top pods -n ara-ai --sort-by=memory

# 2. Check for memory leaks
kubectl exec -n ara-ai <pod-name> -- python -m memory_profiler

# 3. Restart affected pods
kubectl delete pod <pod-name> -n ara-ai

# 4. Scale horizontally if needed
kubectl scale deployment ara-api --replicas=5 -n ara-ai
```

#### Model Loading Failures

```bash
# 1. Check model files
kubectl exec -n ara-ai deployment/ara-api -- ls -lh /app/models/

# 2. Verify model integrity
kubectl exec -n ara-ai deployment/ara-api -- \
  python -c "from ara.models import load_model; load_model('models/demo_ensemble.json')"

# 3. Restore from backup if corrupted
./restore-models.sh

# 4. Restart application
kubectl rollout restart deployment/ara-api -n ara-ai
```

## Support

For disaster recovery assistance:
- Emergency Hotline: +1-XXX-XXX-XXXX
- Email: emergency@ara-ai.com
- Slack: #incidents
