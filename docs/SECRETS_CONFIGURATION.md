# Required GitHub Secrets Configuration

## Repository Secrets

Để CD pipeline hoạt động đúng, cần cấu hình các secrets sau trong GitHub Repository Settings > Secrets and variables > Actions:

### 🔐 Kubernetes Configuration
```
KUBE_CONFIG_STAGING          # Base64-encoded kubeconfig for staging cluster
KUBE_CONFIG_PRODUCTION       # Base64-encoded kubeconfig for production cluster
```

### 🗄️ Database Credentials

#### Staging Environment
```
MONGODB_ROOT_PASSWORD_STAGING    # MongoDB root password for staging
MONGODB_USERNAME_STAGING         # MongoDB application username for staging  
MONGODB_PASSWORD_STAGING         # MongoDB application password for staging
REDIS_PASSWORD_STAGING           # Redis password for staging
```

#### Production Environment
```
MONGODB_ROOT_PASSWORD_PRODUCTION # MongoDB root password for production
MONGODB_USERNAME_PRODUCTION      # MongoDB application username for production
MONGODB_PASSWORD_PRODUCTION      # MongoDB application password for production
REDIS_PASSWORD_PRODUCTION        # Redis password for production
```

### 🔑 Application Secrets
```
API_SECRET_KEY_STAGING          # JWT secret key for staging environment
API_SECRET_KEY_PRODUCTION       # JWT secret key for production environment
```

### 📢 Notifications (Optional)
```
SLACK_WEBHOOK_URL              # Slack webhook URL for deployment notifications
```

### 🐳 Container Registry
```
GITHUB_TOKEN                   # Automatically provided by GitHub Actions
```

## Environment Configuration

### Staging Environment
- **Namespace**: `staging`
- **Ingress Host**: `staging-api.face-verification.com`
- **Replicas**: 2
- **Resources**: Moderate (2 CPU, 4GB RAM)

### Production Environment  
- **Namespace**: `production`
- **Ingress Host**: `api.face-verification.com`
- **Replicas**: 3 (with autoscaling)
- **Resources**: High (4 CPU, 8GB RAM)

## How to Generate Secrets

### 1. Kubernetes Config (Base64-encoded)
```bash
# For staging cluster
kubectl config view --raw --minify --context=staging-context | base64 -w 0

# For production cluster  
kubectl config view --raw --minify --context=production-context | base64 -w 0
```

### 2. MongoDB Credentials
```bash
# Generate secure passwords
openssl rand -base64 32  # For MongoDB passwords
openssl rand -base64 24  # For application passwords
```

### 3. API Secret Keys
```bash
# Generate JWT secret keys
openssl rand -base64 64  # For staging
openssl rand -base64 64  # For production
```

### 4. Redis Passwords
```bash
# Generate Redis passwords
openssl rand -base64 32  # For staging
openssl rand -base64 32  # For production
```

## Slack Webhook Setup (Optional)

1. Go to your Slack workspace
2. Create a new app or use existing one
3. Enable Incoming Webhooks
4. Create webhook for #deployments channel
5. Copy webhook URL to `SLACK_WEBHOOK_URL` secret

## GitHub Environments

Configure the following environments in Repository Settings > Environments:

### staging
- **Protection rules**: None (auto-deploy)
- **Environment secrets**: Staging-specific overrides if needed

### production  
- **Protection rules**: 
  - Required reviewers (1-2 people)
  - Wait timer: 5 minutes
  - Deployment branch rule: `main` only
- **Environment secrets**: Production-specific overrides if needed

## Verification

After configuring secrets, test the pipeline:

1. **Test staging deployment**:
   ```bash
   git checkout develop
   git commit --allow-empty -m "test: trigger staging deployment"
   git push origin develop
   ```

2. **Test production deployment**:
   ```bash
   git checkout main
   git merge develop
   git push origin main
   ```

3. **Monitor deployment**:
   - Check GitHub Actions tab
   - Monitor staging: `https://staging-api.face-verification.com/health`
   - Monitor production: `https://api.face-verification.com/health`

## Troubleshooting

### Common Issues

1. **Invalid kubeconfig**: Verify base64 encoding
   ```bash
   echo "YOUR_BASE64_CONFIG" | base64 -d | kubectl --kubeconfig=/dev/stdin cluster-info
   ```

2. **Database connection failures**: Check credentials and network access

3. **Image not found**: Ensure CI pipeline completed successfully

4. **Ingress issues**: Verify DNS configuration and SSL certificates

### Debug Commands

```bash
# Check deployment status
kubectl get deployments -n staging
kubectl get deployments -n production

# Check pod logs
kubectl logs -n staging deployment/face-verification-staging
kubectl logs -n production deployment/face-verification-green

# Check services
kubectl get svc -n staging
kubectl get svc -n production

# Check ingress
kubectl get ingress -n staging
kubectl get ingress -n production
```

## Security Best Practices

1. **Rotate secrets regularly** (every 90 days)
2. **Use least privilege** for service accounts
3. **Enable audit logging** in Kubernetes
4. **Monitor secret access** in GitHub
5. **Use separate databases** for staging/production
6. **Enable network policies** to restrict traffic
7. **Regular security scans** of deployed images

## Backup and Recovery

### Database Backups
```bash
# MongoDB backup
kubectl exec -n production deployment/mongodb -- mongodump --archive | gzip > backup-$(date +%Y%m%d).gz

# Redis backup  
kubectl exec -n production deployment/redis -- redis-cli save
kubectl cp production/redis-pod:/data/dump.rdb ./redis-backup-$(date +%Y%m%d).rdb
```

### Application Rollback
```bash
# List deployment history
kubectl rollout history deployment/face-verification-production -n production

# Rollback to previous version
kubectl rollout undo deployment/face-verification-production -n production

# Rollback to specific revision
kubectl rollout undo deployment/face-verification-production -n production --to-revision=2
```
