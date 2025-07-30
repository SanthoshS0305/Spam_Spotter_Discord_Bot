# Discord Spam Detection Bot

A Discord bot that uses nearest neighbor classification to detect and filter spam messages based on content similarity. The bot trains on both internal flagged messages and an external dataset of known scam messages.

## Features

- **Content-based Spam Detection**: Uses sentence embeddings and nearest neighbor classification to identify spam messages
- **Hybrid Training**: Combines internal flagged messages with external scam dataset for better detection
- **Learning System**: Automatically learns from flagged/deleted messages to improve detection
- **Admin Approval**: Optional admin approval system for detected spam messages
- **Manual Flagging**: Administrators can manually flag messages as spam
- **Configurable**: Adjustable similarity thresholds and detection parameters

## How It Works

1. **Message Processing**: When a message is sent, the bot converts it to a vector embedding using the SentenceTransformer model
2. **Similarity Check**: The bot compares the message embedding with all known spam messages from both internal and external datasets
3. **Nearest Neighbor**: Uses k-nearest neighbors (k=3) with cosine similarity to find the most similar spam messages
4. **Threshold Decision**: If the similarity score exceeds the threshold (0.75), the message is flagged as potential spam
5. **Action**: Depending on server settings, the message is either deleted automatically or sent for admin approval

## Training Data

The bot uses two sources of training data:

1. **Internal Dataset**: Messages flagged as spam by administrators in your Discord server
2. **External Dataset**: Hugging Face dataset containing known phishing and scam messages
   - Source: `wangyuancheng/discord-phishing-scam-clean`
   - Contains messages labeled as scam (1) or not scam (0)
   - Automatically loaded on bot startup

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file with:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   DATASET_FILE=spam_dataset.csv
   COMMAND_PREFIX=!
   FLAG_COMMAND=!flag
   CONFIG_PATH=server_config.json
   ```

3. **Run the Bot**:
   ```bash
   python bot.py
   ```

## Commands

- `!setadmin #channel` - Set the admin channel for spam notifications
- `!retrain` - Retrain the spam detection model
- `!toggle_approval` - Toggle between automatic deletion and admin approval
- `!status` - Show bot status and configuration
- `!flag [reason]` - Reply to a message to flag it as spam (admin only)

## Configuration

### Similarity Threshold
- Default: 0.75 (75% similarity required)
- Lower values = more sensitive detection
- Higher values = less sensitive detection

### K Neighbors
- Default: 3
- Number of nearest neighbors to consider
- Higher values = more robust but slower

### Admin Approval
- Default: Enabled
- When enabled, detected spam is sent to admin channel for approval
- When disabled, spam is deleted automatically

## Datasets

### Internal Dataset (`spam_dataset.csv`)
Contains messages flagged as spam by administrators:
- Message content
- Timestamp
- Channel ID
- Server ID
- Deletion timestamp

### External Dataset
Automatically loaded from Hugging Face:
- Source: `wangyuancheng/discord-phishing-scam-clean`
- Contains thousands of known scam messages
- Used to improve initial detection before internal data is available

The bot combines both datasets for training, improving detection accuracy over time as more messages are flagged internally.

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (fast and effective for text similarity)
- **Classification**: Nearest Neighbors with cosine similarity
- **Training Data**: Internal flagged messages + external scam dataset
- **Storage**: CSV files for datasets and JSON for server configurations
- **Logging**: Comprehensive logging to `bot.log`

## Security Notes

- Only administrators can flag messages as spam
- Admin approval system prevents false positives
- All flagged messages are logged for audit purposes
- Bot permissions should be limited to message management only
- External dataset is from a trusted source (Hugging Face)