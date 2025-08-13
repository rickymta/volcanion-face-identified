# 🚀 Quick Setup Guide for Ubuntu Server Deployment

This guide will help you quickly set up the Face Verification API on your Ubuntu server.

## 📋 Prerequisites

- Ubuntu Server 18.04+ (20.04 or 22.04 recommended)
- SSH access to the server
- Domain name pointed to your server (optional, for SSL)
- GitHub repository with admin access

## 🏗️ Server Setup

### 1. Run the Setup Script

```bash
# Clone the repository
git clone https://github.com/rickymta/volcanion-face-identified.git
cd volcanion-face-identified

# Make setup script executable
chmod +x scripts/setup-ubuntu-server.sh

# Run setup for staging environment
./scripts/setup-ubuntu-server.sh staging

# OR run setup for production environment
./scripts/setup-ubuntu-server.sh production
```

### 2. Configure GitHub Secrets

In your GitHub repository, go to **Settings > Secrets and Variables > Actions** and add:

#### Docker Hub Secrets
```
DOCKER_HUB_USERNAME=your_dockerhub_username
DOCKER_HUB_TOKEN=your_dockerhub_token
```

#### Staging Server Secrets
```
STAGING_SERVER_HOST=staging.yourdomain.com
STAGING_SERVER_USER=ubuntu
STAGING_SSH_PRIVATE_KEY=-----BEGIN OPENSSH PRIVATE KEY-----...

API_SECRET_KEY_STAGING=your_staging_secret_key
MONGODB_URL_STAGING=mongodb://localhost:27017
MONGODB_USERNAME_STAGING=admin
MONGODB_PASSWORD_STAGING=your_mongodb_password
REDIS_URL_STAGING=redis://localhost:6379
REDIS_PASSWORD_STAGING=your_redis_password
SSL_EMAIL=your_email@domain.com
```

#### Production Server Secrets
```
PRODUCTION_SERVER_HOST=api.yourdomain.com
PRODUCTION_SERVER_USER=ubuntu
PRODUCTION_SSH_PRIVATE_KEY=-----BEGIN OPENSSH PRIVATE KEY-----...

API_SECRET_KEY_PRODUCTION=your_production_secret_key
MONGODB_URL_PRODUCTION=mongodb://localhost:27017
MONGODB_USERNAME_PRODUCTION=admin
MONGODB_PASSWORD_PRODUCTION=your_mongodb_password
REDIS_URL_PRODUCTION=redis://localhost:6379
REDIS_PASSWORD_PRODUCTION=your_redis_password
```

### 3. Generate SSH Keys

```bash
# On your local machine, generate SSH key pair
ssh-keygen -t rsa -b 4096 -C "deployment@yourdomain.com"

# Copy public key to server
ssh-copy-id -i ~/.ssh/id_rsa.pub ubuntu@your-server-ip

# Copy private key content for GitHub secrets
cat ~/.ssh/id_rsa
```

## 🚀 Deployment Process

### Automatic Deployment

1. **Staging**: Push to `develop` branch
2. **Production**: Push to `main` branch

### Manual Deployment

1. Go to **Actions** tab in GitHub
2. Select **CD Pipeline - Ubuntu Server Deployment**
3. Click **Run workflow**
4. Choose environment and version

## 🔍 Verification

### Check Deployment Status

```bash
# System status
sudo system-status.sh

# Container status
docker ps

# Application health
curl http://localhost:8000/health  # Production
curl http://localhost:8001/health  # Staging
```

### Access Application

- **Staging**: `http://your-staging-server:8001`
- **Production**: `http://your-production-server:8000`
- **API Docs**: `http://your-server:port/docs`

## 🔧 Management Commands

### Service Management
```bash
# Start/stop services
sudo systemctl start face-verification-production
sudo systemctl stop face-verification-production
sudo systemctl status face-verification-production

# View logs
docker logs face-verification-production
docker-compose -f docker-compose.production.yml logs -f
```

### Manual Deployment
```bash
# Deploy latest version
sudo deploy-face-verification-production.sh

# Deploy specific version
sudo deploy-face-verification-production.sh rickymta/volcanion-face-identified:v1.1.0
```

### Monitoring
```bash
# System monitoring
htop
sudo system-status.sh

# Application monitoring
sudo monitor-face-verification-production.sh
```

## 🔒 SSL Setup (Optional)

### For Production with Domain

```bash
# Install certbot (if not already installed)
sudo apt install certbot python3-certbot-nginx

# Generate SSL certificate
sudo certbot --nginx -d api.yourdomain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

### Update Nginx Configuration

```bash
# Edit nginx configuration
sudo nano /etc/nginx/sites-available/face-verification-production

# Test and reload
sudo nginx -t
sudo systemctl reload nginx
```

## 🆘 Troubleshooting

### Common Issues

1. **Container won't start**
```bash
docker logs face-verification-production
sudo systemctl status face-verification-production
```

2. **Port conflicts**
```bash
sudo netstat -tulpn | grep :8000
sudo kill -9 <PID>
```

3. **Permission issues**
```bash
sudo chown -R $USER:$USER /opt/face-verification-production
```

4. **Disk space issues**
```bash
df -h
docker system prune -f
```

5. **Memory issues**
```bash
free -h
sudo systemctl restart face-verification-production
```

### Log Locations

- Application logs: `/opt/face-verification-{env}/logs/`
- System logs: `/var/log/face-verification/`
- Nginx logs: `/var/log/nginx/`
- Container logs: `docker logs <container_name>`

## 📞 Support

For additional help:

1. Check the full documentation in `DEPLOYMENT_INFO.md`
2. Review GitHub Actions logs for deployment issues
3. Monitor application logs for runtime issues
4. Use system monitoring tools for server issues

## 🎯 Quick Reference

### Environment URLs
- **Staging**: `http://staging-server:8001`
- **Production**: `http://production-server:8000`

### Key Directories
- **Application**: `/opt/face-verification-{env}/`
- **Backups**: `/opt/backups/`
- **Logs**: `/var/log/face-verification/`

### Important Files
- **Service**: `/etc/systemd/system/face-verification-{env}.service`
- **Nginx**: `/etc/nginx/sites-available/face-verification-{env}`
- **Monitoring**: `/usr/local/bin/monitor-face-verification-{env}.sh`
- **Deployment**: `/usr/local/bin/deploy-face-verification-{env}.sh`
