#!/bin/bash

# Discord Bot EC2 Deployment Script
# This script sets up an EC2 instance and deploys the Discord bot

set -e

echo "🚀 Starting Discord Bot EC2 Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on EC2
if [ ! -f /sys/hypervisor/uuid ]; then
    print_warning "This script is designed to run on EC2. Some features may not work on other systems."
fi

# Update system packages
print_status "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
print_status "Installing Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
print_status "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER

# Create application directory
print_status "Setting up application directory..."
mkdir -p ~/discord-bot
cd ~/discord-bot

# Create necessary directories
mkdir -p data logs

# Create .env file template
print_status "Creating environment file template..."
cat > .env << EOF
# Discord Bot Configuration
DISCORD_TOKEN=your_discord_token_here

# Dataset Configuration
DATASET_FILE=data/global_dataset.csv
REJECTED_DATASET_FILE=data/rejected_dataset.csv
CONFIG_PATH=data/server_config.json

# Bot Configuration
COMMAND_PREFIX=!
FLAG_COMMAND=!flag
EOF

print_warning "Please edit the .env file and add your Discord bot token!"
print_status "You can edit it with: nano .env"

# Create systemd service for auto-start
print_status "Creating systemd service..."
sudo tee /etc/systemd/system/discord-bot.service > /dev/null << EOF
[Unit]
Description=Discord Spam Bot
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/discord-bot
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl enable discord-bot.service

# Create monitoring script
print_status "Creating monitoring script..."
cat > monitor.sh << 'EOF'
#!/bin/bash

echo "=== Discord Bot Status ==="
docker-compose ps

echo -e "\n=== Bot Logs (last 20 lines) ==="
docker-compose logs --tail=20 discord-bot

echo -e "\n=== System Resources ==="
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}'

echo -e "\nMemory Usage:"
free -h

echo -e "\nDisk Usage:"
df -h /
EOF

chmod +x monitor.sh

# Create backup script
print_status "Creating backup script..."
cat > backup.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/home/ubuntu/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

echo "Creating backup: $BACKUP_DIR/discord-bot-backup-$DATE.tar.gz"

tar -czf $BACKUP_DIR/discord-bot-backup-$DATE.tar.gz \
    --exclude='*.log' \
    --exclude='__pycache__' \
    --exclude='.git' \
    .

echo "Backup created: $BACKUP_DIR/discord-bot-backup-$DATE.tar.gz"

# Keep only last 5 backups
ls -t $BACKUP_DIR/discord-bot-backup-*.tar.gz | tail -n +6 | xargs -r rm
EOF

chmod +x backup.sh

# Create update script
print_status "Creating update script..."
cat > update.sh << 'EOF'
#!/bin/bash

echo "Updating Discord Bot..."

# Stop the bot
docker-compose down

# Pull latest changes (if using git)
# git pull origin main

# Rebuild and start
docker-compose up -d --build

echo "Update completed!"
EOF

chmod +x update.sh

print_status "Deployment setup completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Edit .env file: nano .env"
echo "2. Add your Discord bot token to the .env file"
echo "3. Copy your bot files to this directory"
echo "4. Start the bot: docker-compose up -d"
echo "5. Monitor the bot: ./monitor.sh"
echo ""
echo "📁 Useful Commands:"
echo "  Start bot: docker-compose up -d"
echo "  Stop bot: docker-compose down"
echo "  View logs: docker-compose logs -f discord-bot"
echo "  Monitor: ./monitor.sh"
echo "  Backup: ./backup.sh"
echo "  Update: ./update.sh"
echo ""
echo "🔧 Service Management:"
echo "  Enable auto-start: sudo systemctl enable discord-bot"
echo "  Start service: sudo systemctl start discord-bot"
echo "  Check status: sudo systemctl status discord-bot" 