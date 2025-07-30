# Discord Spam Bot

An intelligent Discord bot that uses machine learning to detect and prevent spam messages. Features include content-based detection using nearest neighbor classification and admin approval workflows.

## Features

- **Content-Based Detection**: Uses nearest neighbor classification to identify spam based on message content
- **External Dataset Integration**: Trained on both internal deleted messages and external scam datasets
- **Admin Approval System**: Configurable approval workflows for message deletion
- **Multi-Server Support**: Handles multiple Discord servers with separate configurations
- **Real-time Monitoring**: Comprehensive logging and status reporting

## Quick Start (Local Development)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd discord_spam_bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your Discord bot token
   ```

4. **Run the bot**
   ```bash
   python bot.py
   ```

## EC2 Deployment Guide

### Prerequisites

- AWS EC2 instance (Ubuntu 20.04+ recommended)
- Discord bot token
- SSH access to your EC2 instance

### Step 1: Launch EC2 Instance

1. **Launch an EC2 instance**:
   - **AMI**: Ubuntu Server 20.04 LTS
   - **Instance Type**: t3.medium or larger (recommended for ML workloads)
   - **Storage**: At least 20GB
   - **Security Group**: Allow SSH (port 22) and HTTP (port 80) if needed

2. **Connect to your instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

### Step 2: Deploy the Bot

1. **Upload your bot files**:
   ```bash
   # From your local machine
   scp -r . ubuntu@your-ec2-ip:~/discord-bot/
   ```

2. **SSH into your EC2 instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Run the deployment script**:
   ```bash
   cd ~/discord-bot
   chmod +x deploy.sh
   ./deploy.sh
   ```

4. **Configure the bot**:
   ```bash
   nano .env
   # Add your Discord bot token
   ```

5. **Start the bot**:
   ```bash
   docker-compose up -d
   ```

### Step 3: Verify Deployment

1. **Check bot status**:
   ```bash
   ./monitor.sh
   ```

2. **View logs**:
   ```bash
   docker-compose logs -f discord-bot
   ```

3. **Test the bot** in your Discord server

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f discord-bot

# Stop
docker-compose down
```

### Using Docker directly

```bash
# Build image
docker build -t discord-spam-bot .

# Run container
docker run -d \
  --name discord-bot \
  --restart unless-stopped \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --env-file .env \
  discord-spam-bot
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Discord Bot Configuration
DISCORD_TOKEN=your_discord_bot_token_here

# Dataset Configuration
DATASET_FILE=data/global_dataset.csv
REJECTED_DATASET_FILE=data/rejected_dataset.csv
CONFIG_PATH=data/server_config.json

# Bot Configuration
COMMAND_PREFIX=!
FLAG_COMMAND=!flag
```

### Bot Commands

| Command | Description | Permission |
|---------|-------------|------------|
| `!setadmin #channel` | Set admin channel for approvals | Administrator |
| `!load_dataset` | Reload training datasets | Administrator |
| `!toggle_approval` | Toggle admin approval requirement | Administrator |
| `!status` | Show bot status and statistics | Administrator |

### Flagging Messages

Reply to any message with `!flag [reason]` to flag it for deletion (requires administrator permissions).

## Monitoring and Maintenance

### Useful Commands

```bash
# Check bot status
./monitor.sh

# View real-time logs
docker-compose logs -f discord-bot

# Create backup
./backup.sh

# Update bot
./update.sh

# Restart bot
docker-compose restart discord-bot
```

### Systemd Service

The bot runs as a systemd service for automatic startup:

```bash
# Enable auto-start
sudo systemctl enable discord-bot

# Start service
sudo systemctl start discord-bot

# Check status
sudo systemctl status discord-bot

# View logs
sudo journalctl -u discord-bot -f
```

## Performance

- **Content Detection**: ~5-20ms per message (depending on dataset size)
- **Memory Usage**: ~500MB-1GB (depending on dataset size)
- **CPU Usage**: Low during normal operation, spikes during training

## Troubleshooting

### Common Issues

1. **Bot not responding**:
   ```bash
   docker-compose logs discord-bot
   ```

2. **Permission errors**:
   ```bash
   sudo chown -R $USER:$USER data/
   ```

3. **Out of memory**:
   - Increase EC2 instance size
   - Reduce dataset size or use sampling

4. **Discord API errors**:
   - Check bot token in `.env`
   - Verify bot permissions in Discord

### Logs

- **Application logs**: `docker-compose logs discord-bot`
- **System logs**: `sudo journalctl -u discord-bot`
- **Docker logs**: `docker logs discord-spam-bot`

## Security Considerations

- **Bot Token**: Never commit your Discord bot token to version control
- **Firewall**: Configure security groups to only allow necessary ports
- **Updates**: Regularly update the bot and dependencies
- **Backups**: Use the provided backup script regularly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.