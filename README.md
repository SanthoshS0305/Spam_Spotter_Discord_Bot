# Spam Spotter Discord Bot

A Discord bot that monitors messages, matches them against a dataset, and can delete matching messages with optional admin approval.

## Features

- Load and parse message datasets (CSV format)
- Monitor Discord channels for matching messages
- Delete matching messages automatically or with admin approval
- Log deleted messages back to the dataset
- Configurable settings via environment variables
- Admin commands for control and monitoring

## Setup

### Option 1: Direct Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   DISCORD_TOKEN=your_bot_token
   DATASET_PATH=path/to/your/dataset.csv
   ADMIN_CHANNEL_ID=your_admin_channel_id
   ```
4. Create your dataset CSV file with the following columns:
   - author_id
   - content
   - timestamp
   - channel_id
   - server_id
   - status (optional)

### Option 2: Docker Installation

1. Install Docker on your system:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install docker.io
   sudo systemctl start docker
   sudo usermod -aG docker $USER
   # Log out and log back in for group changes to take effect
   ```

2. Build the Docker image:
   ```bash
   docker build -t discord-spam-bot .
   ```

3. Create a `.env` file with your configuration (same as above)

4. Run the container:
   ```bash
   docker run -d \
     --name discord-bot \
     --restart unless-stopped \
     --env-file .env \
     discord-spam-bot
   ```

5. Check container status:
   ```bash
   docker ps
   docker logs discord-bot
   ```

## Usage

1. Start the bot:
   ```bash
   # Direct installation
   python bot.py

   # Docker
   docker start discord-bot
   ```

2. Available Commands:
   - `!load_dataset` - Reload the message dataset
   - `!toggle_approval` - Toggle admin approval requirement
   - `!status` - Check bot status and settings

## Configuration

The bot can be configured through environment variables in the `.env` file:

- `DISCORD_TOKEN`: Your Discord bot token
- `DATASET_PATH`: Path to your message dataset CSV file
- `ADMIN_CHANNEL_ID`: ID of the channel for admin notifications
- `COMMAND_PREFIX`: Bot command prefix (default: !)

## Docker Commands

- **Build image:**
  ```bash
  docker build -t discord-spam-bot .
  ```

- **Run container:**
  ```bash
  docker run -d --name discord-bot --restart unless-stopped --env-file .env discord-spam-bot
  ```

- **Stop container:**
  ```bash
  docker stop discord-bot
  ```

- **View logs:**
  ```bash
  docker logs -f discord-bot
  ```

- **Remove container:**
  ```bash
  docker rm discord-bot
  ```

- **Check container status:**
  ```bash
  docker ps
  ```

## Security

- Never share your bot token
- Keep your dataset file secure
- Use appropriate Discord permissions for the bot
- Keep your Docker images updated
- Use environment variables for sensitive data
