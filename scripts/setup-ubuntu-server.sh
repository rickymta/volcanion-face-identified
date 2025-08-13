#!/bin/bash

# Ubuntu Server Setup Script for Face Verification API
# This script prepares an Ubuntu server for deployment

set -e

echo "🚀 Setting up Ubuntu Server for Face Verification API Deployment"
echo "================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root. Please run as a regular user with sudo privileges."
   exit 1
fi

# Get environment type
ENVIRONMENT=${1:-staging}
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    error "Invalid environment. Use 'staging' or 'production'"
    echo "Usage: $0 [staging|production]"
    exit 1
fi

log "Setting up environment: $ENVIRONMENT"

# Update system
log "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install required packages
log "Installing required packages..."
sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    wget \
    unzip \
    git \
    htop \
    nano \
    ufw \
    fail2ban \
    nginx \
    certbot \
    python3-certbot-nginx

# Install Docker
log "Installing Docker..."
if ! command -v docker &> /dev/null; then
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Set up the repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Add user to docker group
    sudo usermod -aG docker $USER
    log "Docker installed successfully. You may need to log out and back in for group changes to take effect."
else
    log "Docker is already installed"
fi

# Install Docker Compose (standalone)
log "Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_VERSION="2.20.2"
    sudo curl -L "https://github.com/docker/compose/releases/download/v${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    log "Docker Compose installed successfully"
else
    log "Docker Compose is already installed"
fi

# Create application directories
log "Creating application directories..."
sudo mkdir -p /opt/face-verification-${ENVIRONMENT}
sudo mkdir -p /opt/backups
sudo mkdir -p /var/log/face-verification

# Set ownership
sudo chown -R $USER:$USER /opt/face-verification-${ENVIRONMENT}
sudo chown -R $USER:$USER /opt/backups
sudo chown -R $USER:$USER /var/log/face-verification

# Configure firewall
log "Configuring firewall..."
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (be careful with this)
sudo ufw allow ssh

# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow application ports
if [[ "$ENVIRONMENT" == "staging" ]]; then
    sudo ufw allow 8001/tcp  # Staging port
    log "Opened port 8001 for staging environment"
else
    sudo ufw allow 8000/tcp  # Production port
    log "Opened port 8000 for production environment"
fi

# Allow MongoDB and Redis (if using external access)
# sudo ufw allow 27017/tcp  # MongoDB
# sudo ufw allow 6379/tcp   # Redis

# Enable firewall
sudo ufw --force enable

# Configure fail2ban
log "Configuring fail2ban..."
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Create fail2ban jail for nginx
sudo tee /etc/fail2ban/jail.d/nginx.conf > /dev/null <<EOF
[nginx-http-auth]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log

[nginx-dos]
enabled = true
port = http,https
logpath = /var/log/nginx/access.log
maxretry = 300
findtime = 600
bantime = 600
EOF

sudo systemctl restart fail2ban

# Configure nginx
log "Configuring Nginx..."
if [[ "$ENVIRONMENT" == "staging" ]]; then
    NGINX_CONFIG="/etc/nginx/sites-available/face-verification-staging"
    PORT="8001"
else
    NGINX_CONFIG="/etc/nginx/sites-available/face-verification-production"
    PORT="8000"
fi

# Create nginx configuration
sudo tee $NGINX_CONFIG > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # Rate limiting
    limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Proxy to application
    location / {
        proxy_pass http://localhost:${PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    # Health check endpoint (bypass rate limiting)
    location /health {
        limit_req off;
        proxy_pass http://localhost:${PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Static files (if any)
    location /static/ {
        alias /opt/face-verification-${ENVIRONMENT}/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Log files
    access_log /var/log/nginx/face-verification-${ENVIRONMENT}-access.log;
    error_log /var/log/nginx/face-verification-${ENVIRONMENT}-error.log;
}
EOF

# Enable nginx site
sudo ln -sf $NGINX_CONFIG /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
sudo nginx -t

# Start and enable nginx
sudo systemctl enable nginx
sudo systemctl restart nginx

# Create systemd service for automatic container startup
log "Creating systemd service..."
SERVICE_FILE="/etc/systemd/system/face-verification-${ENVIRONMENT}.service"

sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Face Verification API - ${ENVIRONMENT}
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/face-verification-${ENVIRONMENT}
ExecStart=/usr/local/bin/docker-compose -f docker-compose.${ENVIRONMENT}.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.${ENVIRONMENT}.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable face-verification-${ENVIRONMENT}.service

# Setup log rotation
log "Setting up log rotation..."
sudo tee /etc/logrotate.d/face-verification-${ENVIRONMENT} > /dev/null <<EOF
/var/log/face-verification/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        docker kill -s USR1 face-verification-${ENVIRONMENT} 2>/dev/null || true
    endscript
}

/opt/face-verification-${ENVIRONMENT}/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
}
EOF

# Create monitoring script
log "Creating monitoring script..."
MONITOR_SCRIPT="/usr/local/bin/monitor-face-verification-${ENVIRONMENT}.sh"

sudo tee $MONITOR_SCRIPT > /dev/null <<'EOF'
#!/bin/bash

ENVIRONMENT=ENVIRONMENT_PLACEHOLDER
WORKDIR="/opt/face-verification-${ENVIRONMENT}"
LOG_FILE="/var/log/face-verification/monitor.log"

# Function to log with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
}

# Function to check container health
check_container_health() {
    CONTAINER_NAME="face-verification-${ENVIRONMENT}"
    
    if ! docker ps | grep -q $CONTAINER_NAME; then
        log_message "ERROR: Container $CONTAINER_NAME is not running"
        return 1
    fi
    
    # Check container health status
    HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "unknown")
    
    if [[ "$HEALTH_STATUS" != "healthy" ]]; then
        log_message "WARNING: Container $CONTAINER_NAME health status: $HEALTH_STATUS"
        return 1
    fi
    
    return 0
}

# Function to check application response
check_app_response() {
    PORT="8000"
    if [[ "$ENVIRONMENT" == "staging" ]]; then
        PORT="8001"
    fi
    
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/health || echo "000")
    
    if [[ "$HTTP_CODE" != "200" ]]; then
        log_message "ERROR: Application not responding (HTTP $HTTP_CODE)"
        return 1
    fi
    
    return 0
}

# Function to restart services if needed
restart_services() {
    log_message "Attempting to restart services..."
    
    cd $WORKDIR
    docker-compose -f docker-compose.${ENVIRONMENT}.yml down
    sleep 10
    docker-compose -f docker-compose.${ENVIRONMENT}.yml up -d
    
    log_message "Services restarted"
}

# Main monitoring logic
main() {
    log_message "Starting health check..."
    
    if ! check_container_health; then
        restart_services
        sleep 30
        
        if ! check_container_health; then
            log_message "CRITICAL: Container restart failed"
            exit 1
        fi
    fi
    
    if ! check_app_response; then
        restart_services
        sleep 30
        
        if ! check_app_response; then
            log_message "CRITICAL: Application restart failed"
            exit 1
        fi
    fi
    
    log_message "Health check completed successfully"
}

# Run main function
main
EOF

# Replace placeholder and make executable
sudo sed -i "s/ENVIRONMENT_PLACEHOLDER/$ENVIRONMENT/g" $MONITOR_SCRIPT
sudo chmod +x $MONITOR_SCRIPT

# Setup cron job for monitoring
log "Setting up monitoring cron job..."
(crontab -l 2>/dev/null; echo "*/5 * * * * $MONITOR_SCRIPT") | crontab -

# Create deployment script
log "Creating deployment script..."
DEPLOY_SCRIPT="/usr/local/bin/deploy-face-verification-${ENVIRONMENT}.sh"

sudo tee $DEPLOY_SCRIPT > /dev/null <<EOF
#!/bin/bash

# Deployment script for Face Verification API - ${ENVIRONMENT}

set -e

ENVIRONMENT="${ENVIRONMENT}"
WORKDIR="/opt/face-verification-\${ENVIRONMENT}"
BACKUP_DIR="/opt/backups/face-verification-\$(date +%Y%m%d-%H%M%S)"
IMAGE_NAME="\${1:-rickymta/volcanion-face-identified:latest}"

echo "🚀 Starting deployment of \$IMAGE_NAME to \$ENVIRONMENT environment"

# Create backup
echo "📦 Creating backup..."
mkdir -p \$BACKUP_DIR
if [ -d "\$WORKDIR" ]; then
    cp -r \$WORKDIR/* \$BACKUP_DIR/ 2>/dev/null || true
fi

# Pull new image
echo "📥 Pulling new image..."
docker pull \$IMAGE_NAME

# Update docker-compose file
cd \$WORKDIR
if [ -f "docker-compose.\${ENVIRONMENT}.yml" ]; then
    # Update image in docker-compose file
    sed -i "s|image: rickymta/volcanion-face-identified:.*|image: \$IMAGE_NAME|g" docker-compose.\${ENVIRONMENT}.yml
    
    # Deploy with zero downtime
    echo "🔄 Deploying with zero downtime..."
    docker-compose -f docker-compose.\${ENVIRONMENT}.yml up -d
    
    # Wait for health check
    echo "⏳ Waiting for health check..."
    sleep 30
    
    # Verify deployment
    if docker-compose -f docker-compose.\${ENVIRONMENT}.yml ps | grep -q "Up"; then
        echo "✅ Deployment successful!"
        
        # Cleanup old images
        echo "🧹 Cleaning up old images..."
        docker image prune -f
        
        echo "🎉 Deployment completed successfully!"
    else
        echo "❌ Deployment failed!"
        echo "📋 Container logs:"
        docker-compose -f docker-compose.\${ENVIRONMENT}.yml logs
        exit 1
    fi
else
    echo "❌ Docker compose file not found: docker-compose.\${ENVIRONMENT}.yml"
    exit 1
fi
EOF

sudo chmod +x $DEPLOY_SCRIPT

# Setup system monitoring
log "Installing system monitoring tools..."
sudo apt-get install -y htop iotop nethogs

# Create system status script
SYSTEM_STATUS_SCRIPT="/usr/local/bin/system-status.sh"

sudo tee $SYSTEM_STATUS_SCRIPT > /dev/null <<'EOF'
#!/bin/bash

echo "=== System Status ==="
echo "Date: $(date)"
echo "Uptime: $(uptime)"
echo ""

echo "=== Disk Usage ==="
df -h | grep -E "^/dev|Used"
echo ""

echo "=== Memory Usage ==="
free -h
echo ""

echo "=== CPU Usage ==="
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "CPU Usage: " 100 - $1 "%"}'
echo ""

echo "=== Docker Containers ==="
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

echo "=== Network Connections ==="
ss -tuln | grep -E ":(80|443|8000|8001|27017|6379)"
echo ""

echo "=== Recent Logs ==="
tail -n 10 /var/log/face-verification/monitor.log 2>/dev/null || echo "No monitor logs found"
EOF

sudo chmod +x $SYSTEM_STATUS_SCRIPT

# Create SSH hardening
log "Hardening SSH configuration..."
sudo cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup

sudo tee /etc/ssh/sshd_config.d/99-hardening.conf > /dev/null <<EOF
# SSH Hardening for Face Verification API Server
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthenticationMethods publickey
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
Protocol 2
X11Forwarding no
AllowTcpForwarding no
PermitTunnel no
EOF

sudo systemctl restart ssh

# Create environment-specific information file
log "Creating environment information..."
INFO_FILE="/opt/face-verification-${ENVIRONMENT}/DEPLOYMENT_INFO.md"

tee $INFO_FILE > /dev/null <<EOF
# Face Verification API - ${ENVIRONMENT} Environment

## Server Information
- **Environment**: ${ENVIRONMENT}
- **Setup Date**: $(date)
- **User**: $USER
- **Hostname**: $(hostname)
- **IP Address**: $(curl -s ifconfig.me || echo "Unable to determine")

## Application Details
- **Working Directory**: /opt/face-verification-${ENVIRONMENT}
- **Port**: $([ "$ENVIRONMENT" = "staging" ] && echo "8001" || echo "8000")
- **Service Name**: face-verification-${ENVIRONMENT}
- **Log Directory**: /var/log/face-verification

## Useful Commands

### Service Management
\`\`\`bash
# Start/stop service
sudo systemctl start face-verification-${ENVIRONMENT}
sudo systemctl stop face-verification-${ENVIRONMENT}
sudo systemctl status face-verification-${ENVIRONMENT}

# View containers
docker ps
docker-compose -f docker-compose.${ENVIRONMENT}.yml ps

# View logs
docker logs face-verification-${ENVIRONMENT}
docker-compose -f docker-compose.${ENVIRONMENT}.yml logs -f
\`\`\`

### Deployment
\`\`\`bash
# Deploy latest version
sudo deploy-face-verification-${ENVIRONMENT}.sh

# Deploy specific version
sudo deploy-face-verification-${ENVIRONMENT}.sh rickymta/volcanion-face-identified:v1.1.0
\`\`\`

### Monitoring
\`\`\`bash
# System status
sudo system-status.sh

# Monitor containers
sudo monitor-face-verification-${ENVIRONMENT}.sh

# Check application health
curl http://localhost:$([ "$ENVIRONMENT" = "staging" ] && echo "8001" || echo "8000")/health
\`\`\`

### Troubleshooting
\`\`\`bash
# Check disk space
df -h

# Check memory usage
free -h

# Check running processes
htop

# Check network connections
ss -tuln

# View system logs
sudo journalctl -f
\`\`\`

## Security Features
- ✅ Firewall configured (UFW)
- ✅ Fail2ban protection
- ✅ SSH hardening
- ✅ Nginx rate limiting
- ✅ Container security (non-root user)
- ✅ Log rotation configured

## Backup & Recovery
- **Backup Directory**: /opt/backups
- **Automatic Backups**: Created before each deployment
- **Log Rotation**: Configured for 30 days retention

## Support
For issues or questions, check the deployment logs:
- Application logs: /opt/face-verification-${ENVIRONMENT}/logs/
- System logs: /var/log/face-verification/
- Nginx logs: /var/log/nginx/
EOF

# Print summary
echo ""
echo "================================================================="
echo -e "${GREEN}🎉 Ubuntu Server Setup Complete!${NC}"
echo "================================================================="
echo ""
echo -e "${BLUE}Environment:${NC} $ENVIRONMENT"
echo -e "${BLUE}Working Directory:${NC} /opt/face-verification-${ENVIRONMENT}"
echo -e "${BLUE}Port:${NC} $([ "$ENVIRONMENT" = "staging" ] && echo "8001" || echo "8000")"
echo -e "${BLUE}Service:${NC} face-verification-${ENVIRONMENT}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Configure your GitHub repository secrets"
echo "2. Push code to trigger deployment"
echo "3. Monitor deployment with: sudo system-status.sh"
echo ""
echo -e "${YELLOW}Useful Commands:${NC}"
echo "- Deploy manually: sudo deploy-face-verification-${ENVIRONMENT}.sh"
echo "- Check status: sudo systemctl status face-verification-${ENVIRONMENT}"
echo "- View logs: docker logs face-verification-${ENVIRONMENT}"
echo "- System status: sudo system-status.sh"
echo ""
echo -e "${YELLOW}Documentation:${NC} /opt/face-verification-${ENVIRONMENT}/DEPLOYMENT_INFO.md"
echo ""
warn "Please log out and back in for Docker group changes to take effect!"
echo "================================================================="
