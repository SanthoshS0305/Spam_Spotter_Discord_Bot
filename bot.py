import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import discord
from discord.ext import commands
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Bot configuration
TOKEN = os.getenv('DISCORD_TOKEN')
DATASET_FILE = os.getenv('DATASET_FILE', 'global_dataset.csv')  # Global dataset file
COMMAND_PREFIX = os.getenv('COMMAND_PREFIX', '!')
FLAG_COMMAND = os.getenv('FLAG_COMMAND', '!flag')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'server_config.json')

# Time-based configuration
MIN_JOIN_TIME_DAYS = 30  # Messages from members who joined less than this many days ago get higher weight
MAX_JOIN_TIME_DAYS = 365  # Messages from members who joined more than this many days ago get lower weight

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents)

class ServerConfig:
    def __init__(self):
        self.config: Dict[str, Dict] = {}
        self.load_config()

    def load_config(self):
        """Load server configurations from JSON file."""
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configurations for {len(self.config)} servers")
            else:
                self.config = {}
                self.save_config()
                logger.info("Created new server configuration file")
        except Exception as e:
            logger.error(f"Error loading server config: {e}")
            self.config = {}

    def save_config(self):
        """Save server configurations to JSON file."""
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info("Server configurations saved successfully")
        except Exception as e:
            logger.error(f"Error saving server config: {e}")

    def get_admin_channel(self, guild_id: int) -> Optional[int]:
        """Get the admin channel ID for a specific server."""
        return self.config.get(str(guild_id), {}).get('admin_channel_id')

    def set_admin_channel(self, guild_id: int, channel_id: int):
        """Set the admin channel ID for a specific server."""
        if str(guild_id) not in self.config:
            self.config[str(guild_id)] = {}
        self.config[str(guild_id)]['admin_channel_id'] = channel_id
        self.save_config()

    def get_requires_approval(self, guild_id: int) -> bool:
        """Get whether a server requires admin approval."""
        return self.config.get(str(guild_id), {}).get('requires_approval', True)

    def set_requires_approval(self, guild_id: int, requires_approval: bool):
        """Set whether a server requires admin approval."""
        if str(guild_id) not in self.config:
            self.config[str(guild_id)] = {}
        self.config[str(guild_id)]['requires_approval'] = requires_approval
        self.save_config()

class MessageFilter:
    def __init__(self):
        self.dataset: Optional[pd.DataFrame] = None
        self.pending_deletions: Dict[str, Dict] = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.spam_patterns = []
        self.similarity_threshold = 0.8
        self.fuzzy_threshold = 0.85
        self.trained = False
        self.last_training_size = 0
        self.training_threshold = 10  # Retrain after this many new messages

    def get_dataset_path(self) -> str:
        """Get the path to the global dataset file."""
        return DATASET_FILE

    def load_dataset(self):
        """Load the global message dataset."""
        dataset_path = self.get_dataset_path()
        try:
            if os.path.exists(dataset_path):
                self.dataset = pd.read_csv(dataset_path)
                logger.info(f"Dataset loaded successfully with {len(self.dataset)} entries")
            else:
                # Create new dataset if it doesn't exist
                self.dataset = pd.DataFrame(columns=[
                    'author_id', 'content', 'timestamp', 'channel_id',
                    'server_id', 'status', 'deletion_timestamp', 'author_join_days',
                    'flagged_by', 'flag_reason'
                ])
                self.save_dataset()
                logger.info("Created new global dataset file")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.dataset = None

    def save_dataset(self):
        """Save the current dataset."""
        if self.dataset is not None:
            try:
                dataset_path = self.get_dataset_path()
                self.dataset.to_csv(dataset_path, index=False)
                logger.info("Dataset saved successfully")
            except Exception as e:
                logger.error(f"Error saving dataset: {e}")

    def calculate_join_time_weight(self, member: discord.Member) -> float:
        """Calculate a weight based on how long the member has been in the server."""
        join_time = member.joined_at
        if not join_time:
            return 1.0  # Default weight if join time is unknown
        
        days_in_server = (datetime.now(timezone.utc) - join_time).days
        
        if days_in_server < MIN_JOIN_TIME_DAYS:
            # Higher weight for very new members
            return 1.0
        elif days_in_server > MAX_JOIN_TIME_DAYS:
            # Lower weight for long-time members
            return 0.3
        else:
            # Linear interpolation between 1.0 and 0.3
            ratio = (days_in_server - MIN_JOIN_TIME_DAYS) / (MAX_JOIN_TIME_DAYS - MIN_JOIN_TIME_DAYS)
            return 1.0 - (0.7 * ratio)

    def add_message(self, message: discord.Message, status: str = "deleted", flagged_by: Optional[discord.Member] = None, flag_reason: Optional[str] = None):
        """Add a deleted message to the global dataset and trigger training if needed."""
        if self.dataset is None:
            self.load_dataset()
        
        if self.dataset is not None:
            member = message.guild.get_member(message.author.id)
            join_days = (datetime.now(timezone.utc) - member.joined_at).days if member and member.joined_at else None
            
            new_entry = {
                'author_id': message.author.id,
                'content': message.content,
                'timestamp': message.created_at.isoformat(),
                'channel_id': message.channel.id,
                'server_id': message.guild.id,
                'status': status,
                'deletion_timestamp': datetime.now(timezone.utc).isoformat(),
                'author_join_days': join_days,
                'flagged_by': flagged_by.id if flagged_by else None,
                'flag_reason': flag_reason
            }
            self.dataset = pd.concat([self.dataset, pd.DataFrame([new_entry])], ignore_index=True)
            self.save_dataset()
            
            # Trigger training if needed
            self.train_from_dataset()

    def train_from_dataset(self, force: bool = False):
        """Train the filter using the existing dataset."""
        if self.dataset is None or self.dataset.empty:
            logger.warning("No dataset available for training")
            return

        # Check if we need to retrain
        current_size = len(self.dataset)
        if not force and self.trained and (current_size - self.last_training_size) < self.training_threshold:
            return

        logger.info("Starting training from dataset...")
        
        # 1. Extract and learn spam patterns
        self._learn_spam_patterns()
        
        # 2. Calculate optimal thresholds
        self._calculate_thresholds()
        
        # 3. Fine-tune the embedding model
        self._fine_tune_embeddings()
        
        self.trained = True
        self.last_training_size = current_size
        logger.info(f"Training completed successfully. Dataset size: {current_size}")

    def _learn_spam_patterns(self):
        """Learn common spam patterns from the dataset."""
        # Get all deleted messages
        deleted_messages = self.dataset[self.dataset['status'] == 'deleted']
        
        # Common spam indicators
        spam_indicators = {
            'links': r'https?://[^\s]+',
            'discord_invites': r'discord\.gg/\w+',
            'short_urls': r'(bit\.ly|tinyurl\.com|goo\.gl|t\.co)/\w+',
            'common_spam_words': r'(free|giveaway|win|click|visit|join|subscribe|discord\.gg)',
            'repeated_chars': r'(.)\1{3,}',  # 4 or more repeated characters
            'excessive_caps': r'[A-Z]{5,}',  # 5 or more consecutive caps
            'excessive_punctuation': r'[!?]{3,}',  # 3 or more consecutive ! or ?
        }
        
        # Analyze patterns in deleted messages
        pattern_scores = {pattern: 0 for pattern in spam_indicators.values()}
        total_deleted = len(deleted_messages)
        
        if total_deleted == 0:
            return

        for _, row in deleted_messages.iterrows():
            content = str(row['content']).lower()
            for pattern in spam_indicators.values():
                if re.search(pattern, content, re.IGNORECASE):
                    pattern_scores[pattern] += 1
        
        # Select patterns that appear in more than 10% of deleted messages
        self.spam_patterns = [
            pattern for pattern, count in pattern_scores.items()
            if count / total_deleted > 0.1
        ]
        
        logger.info(f"Learned {len(self.spam_patterns)} spam patterns from dataset")

    def _calculate_thresholds(self):
        """Calculate optimal thresholds based on the dataset."""
        if self.dataset is None or self.dataset.empty:
            return

        # Get all deleted messages
        deleted_messages = self.dataset[self.dataset['status'] == 'deleted']
        
        if len(deleted_messages) < 2:
            return

        # Calculate similarity scores between deleted messages
        similarities = []
        from difflib import SequenceMatcher
        
        # Only compare recent messages for efficiency
        recent_messages = deleted_messages.tail(100)  # Last 100 messages
        
        for i, row1 in recent_messages.iterrows():
            for j, row2 in recent_messages.iterrows():
                if i != j:
                    ratio = SequenceMatcher(None, 
                                         str(row1['content']).lower(), 
                                         str(row2['content']).lower()).ratio()
                    similarities.append(ratio)
        
        if similarities:
            # Set threshold to 75th percentile of similarities
            self.fuzzy_threshold = np.percentile(similarities, 75)
            logger.info(f"Calculated fuzzy threshold: {self.fuzzy_threshold:.2f}")

    def _fine_tune_embeddings(self):
        """Fine-tune the embedding model on the dataset."""
        if self.dataset is None or self.dataset.empty:
            return

        try:
            # Get all deleted messages
            deleted_messages = self.dataset[self.dataset['status'] == 'deleted']
            
            if len(deleted_messages) < 2:
                return

            # Only use recent messages for efficiency
            recent_messages = deleted_messages.tail(100)  # Last 100 messages
            
            # Prepare training data
            texts = recent_messages['content'].astype(str).tolist()
            
            # Calculate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Calculate average similarity between deleted messages
            similarities = cosine_similarity(embeddings)
            np.fill_diagonal(similarities, 0)  # Remove self-similarities
            
            # Set threshold to 75th percentile of similarities
            self.similarity_threshold = np.percentile(similarities.flatten(), 75)
            logger.info(f"Calculated semantic similarity threshold: {self.similarity_threshold:.2f}")
            
        except Exception as e:
            logger.error(f"Error fine-tuning embeddings: {e}")

    def find_matches(self, message: discord.Message) -> List[Tuple[Dict, float]]:
        """Find matching messages using trained patterns and thresholds."""
        if self.dataset is None:
            self.load_dataset()
        if self.dataset is None or self.dataset.empty:
            return []

        # Train if not already trained or if dataset has grown significantly
        self.train_from_dataset()

        matches = []

        # 1. Regex pattern matching using learned patterns
        for pattern in self.spam_patterns:
            if re.search(pattern, message.content, re.IGNORECASE):
                for _, row in self.dataset.iterrows():
                    if re.search(pattern, str(row['content']), re.IGNORECASE):
                        matches.append((row.to_dict(), 1.0))

        # 2. Fuzzy string matching with learned threshold
        from difflib import SequenceMatcher
        for _, row in self.dataset.iterrows():
            ratio = SequenceMatcher(None, message.content.lower(), str(row['content']).lower()).ratio()
            if ratio > self.fuzzy_threshold and message.content.strip() and str(row['content']).strip():
                matches.append((row.to_dict(), ratio))

        # 3. Semantic similarity with learned threshold
        try:
            msg_emb = self.embedding_model.encode([message.content])
            dataset_contents = self.dataset['content'].astype(str).tolist()
            dataset_embs = self.embedding_model.encode(dataset_contents)
            sims = cosine_similarity(msg_emb, dataset_embs)[0]
            for idx, sim in enumerate(sims):
                if sim > self.similarity_threshold:
                    matches.append((self.dataset.iloc[idx].to_dict(), sim))
        except Exception as e:
            logger.error(f"Error in semantic similarity: {e}")

        # 4. Remove duplicates and sort by weight
        seen = set()
        unique_matches = []
        for match, weight in matches:
            key = (match['content'], match['server_id'])
            if key not in seen:
                unique_matches.append((match, weight))
                seen.add(key)
        unique_matches.sort(key=lambda x: x[1], reverse=True)
        return unique_matches

# Initialize message filter and server config
message_filter = MessageFilter()
server_config = ServerConfig()

@bot.event
async def on_ready():
    """Called when the bot is ready and connected to Discord."""
    logger.info(f'{bot.user} has connected to Discord!')
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s)")
    except Exception as e:
        logger.error(f"Failed to sync commands: {e}")

@bot.event
async def on_guild_join(guild: discord.Guild):
    """Called when the bot joins a new server."""
    logger.info(f"Joined new server: {guild.name}")
    # Initialize default configuration for the new server
    if str(guild.id) not in server_config.config:
        server_config.config[str(guild.id)] = {
            'requires_approval': True
        }
        server_config.save_config()
    # Initialize dataset for the new server
    message_filter.load_dataset()

@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages."""
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Process commands
    await bot.process_commands(message)

    # Handle message replies for flagging
    if message.reference and message.reference.resolved:
        if message.content.lower().startswith(FLAG_COMMAND.lower()):
            if message.author.guild_permissions.administrator:
                target_message = message.reference.resolved
                flag_reason = message.content[len(FLAG_COMMAND):].strip()
                
                # Create confirmation embed
                embed = discord.Embed(
                    title="Message Flagged for Deletion",
                    description="An administrator has flagged this message for deletion.",
                    color=discord.Color.red()
                )
                embed.add_field(name="Author", value=target_message.author.mention)
                embed.add_field(name="Content", value=target_message.content)
                embed.add_field(name="Channel", value=target_message.channel.mention)
                embed.add_field(name="Flagged By", value=message.author.mention)
                if flag_reason:
                    embed.add_field(name="Reason", value=flag_reason)
                
                # Add confirmation buttons
                view = discord.ui.View()
                confirm_button = discord.ui.Button(label="Confirm Delete", style=discord.ButtonStyle.danger)
                cancel_button = discord.ui.Button(label="Cancel", style=discord.ButtonStyle.secondary)

                async def confirm_callback(interaction: discord.Interaction):
                    if interaction.user.guild_permissions.administrator:
                        try:
                            await target_message.delete()
                            message_filter.add_message(
                                target_message,
                                status="deleted",
                                flagged_by=message.author,
                                flag_reason=flag_reason
                            )
                            await interaction.response.send_message("Message deleted successfully.", ephemeral=True)
                            await message.delete()  # Delete the flag command message
                        except Exception as e:
                            await interaction.response.send_message(f"Error deleting message: {e}", ephemeral=True)
                    else:
                        await interaction.response.send_message("You don't have permission to delete messages.", ephemeral=True)

                async def cancel_callback(interaction: discord.Interaction):
                    if interaction.user.guild_permissions.administrator:
                        await interaction.response.send_message("Deletion cancelled.", ephemeral=True)
                        await message.delete()  # Delete the flag command message
                    else:
                        await interaction.response.send_message("You don't have permission to cancel deletions.", ephemeral=True)

                confirm_button.callback = confirm_callback
                cancel_button.callback = cancel_callback
                view.add_item(confirm_button)
                view.add_item(cancel_button)

                # Send confirmation to admin channel or current channel
                admin_channel_id = server_config.get_admin_channel(message.guild.id)
                if admin_channel_id:
                    admin_channel = bot.get_channel(admin_channel_id)
                    if admin_channel:
                        await admin_channel.send(embed=embed, view=view)
                        return
                await message.channel.send(embed=embed, view=view)
                return

    # Check for matching messages
    matches = message_filter.find_matches(message)
    if matches:
        # Sort matches by weight (highest weight first)
        matches.sort(key=lambda x: x[1], reverse=True)
        match, weight = matches[0]  # Get the highest weighted match
        
        if server_config.get_requires_approval(message.guild.id):
            # Send to admin channel for approval
            admin_channel_id = server_config.get_admin_channel(message.guild.id)
            if admin_channel_id:
                admin_channel = bot.get_channel(admin_channel_id)
                if admin_channel:
                    member = message.guild.get_member(message.author.id)
                    join_days = (datetime.now(timezone.utc) - member.joined_at).days if member and member.joined_at else "Unknown"
                    
                    embed = discord.Embed(
                        title="Message Match Found",
                        description="A message matching the dataset was found.",
                        color=discord.Color.yellow()
                    )
                    embed.add_field(name="Author", value=message.author.mention)
                    embed.add_field(name="Channel", value=message.channel.mention)
                    embed.add_field(name="Content", value=message.content)
                    embed.add_field(name="Message Link", value=message.jump_url)
                    embed.add_field(name="Author Join Time", value=f"{join_days} days")
                    embed.add_field(name="Match Weight", value=f"{weight:.2f}")

                    # Add approval buttons
                    view = discord.ui.View()
                    approve_button = discord.ui.Button(label="Approve", style=discord.ButtonStyle.green)
                    reject_button = discord.ui.Button(label="Reject", style=discord.ButtonStyle.red)

                    async def approve_callback(interaction: discord.Interaction):
                        if interaction.user.guild_permissions.administrator:
                            try:
                                await message.delete()
                                message_filter.add_message(message)
                                await interaction.response.send_message("Message deleted successfully.", ephemeral=True)
                            except Exception as e:
                                await interaction.response.send_message(f"Error deleting message: {e}", ephemeral=True)
                        else:
                            await interaction.response.send_message("You don't have permission to approve deletions.", ephemeral=True)

                    async def reject_callback(interaction: discord.Interaction):
                        if interaction.user.guild_permissions.administrator:
                            await interaction.response.send_message("Deletion rejected.", ephemeral=True)
                        else:
                            await interaction.response.send_message("You don't have permission to reject deletions.", ephemeral=True)

                    approve_button.callback = approve_callback
                    reject_button.callback = reject_callback
                    view.add_item(approve_button)
                    view.add_item(reject_button)

                    await admin_channel.send(embed=embed, view=view)
        else:
            # Delete immediately if approval not required
            try:
                await message.delete()
                message_filter.add_message(message)
                logger.info(f"Deleted message from {message.author} in {message.channel} (weight: {weight:.2f})")
            except Exception as e:
                logger.error(f"Error deleting message: {e}")

@bot.command(name='setadmin')
@commands.has_permissions(administrator=True)
async def set_admin_channel(ctx, channel: discord.TextChannel):
    """Set the admin channel for the current server."""
    server_config.set_admin_channel(ctx.guild.id, channel.id)
    await ctx.send(f"Admin channel set to {channel.mention}")

@bot.command(name='load_dataset')
@commands.has_permissions(administrator=True)
async def load_dataset(ctx):
    """Reload the message dataset for the current server."""
    message_filter.load_dataset()
    await ctx.send("Dataset reloaded successfully.")

@bot.command(name='toggle_approval')
@commands.has_permissions(administrator=True)
async def toggle_approval(ctx):
    """Toggle the admin approval requirement for the current server."""
    current = server_config.get_requires_approval(ctx.guild.id)
    server_config.set_requires_approval(ctx.guild.id, not current)
    status = "enabled" if not current else "disabled"
    await ctx.send(f"Admin approval requirement {status}.")

@bot.command(name='status')
@commands.has_permissions(administrator=True)
async def status(ctx):
    """Display the current bot status for the server."""
    admin_channel_id = server_config.get_admin_channel(ctx.guild.id)
    admin_channel = f"<#{admin_channel_id}>" if admin_channel_id else "Not set"
    
    # Get dataset size for current server
    dataset_size = 0
    if message_filter.dataset is not None:
        dataset_size = len(message_filter.dataset)
    
    embed = discord.Embed(
        title="Bot Status",
        color=discord.Color.blue()
    )
    embed.add_field(name="Admin Approval", value="Enabled" if server_config.get_requires_approval(ctx.guild.id) else "Disabled")
    embed.add_field(name="Dataset Size", value=dataset_size)
    embed.add_field(name="Admin Channel", value=admin_channel)
    embed.add_field(name="Join Time Weighting", value=f"Enabled (Min: {MIN_JOIN_TIME_DAYS} days, Max: {MAX_JOIN_TIME_DAYS} days)")
    embed.add_field(name="Flag Command", value=f"Reply to a message with `{FLAG_COMMAND} [reason]` to flag it for deletion")
    await ctx.send(embed=embed)

# Run the bot
if __name__ == "__main__":
    if not TOKEN:
        logger.error("No Discord token found. Please set the DISCORD_TOKEN environment variable.")
        exit(1)
    
    bot.run(TOKEN) 