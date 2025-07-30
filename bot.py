import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import requests
from io import StringIO

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
DATASET_FILE = os.getenv('DATASET_FILE', 'global_dataset.csv')
REJECTED_DATASET_FILE = os.getenv('REJECTED_DATASET_FILE', 'rejected_dataset.csv')
COMMAND_PREFIX = os.getenv('COMMAND_PREFIX', '!')
FLAG_COMMAND = os.getenv('FLAG_COMMAND', '!flag')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'server_config.json')
EXTERNAL_DATASET_URL = "https://huggingface.co/datasets/wangyuancheng/discord-phishing-scam-clean/resolve/main/discord-phishing-scam-detection.csv"

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
        self.rejected_dataset: Optional[pd.DataFrame] = None
        self.external_dataset: Optional[pd.DataFrame] = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nearest_neighbor_model = None
        self.similarity_threshold = 0.8
        self.k_neighbors = 5
        self.trained = False
        self.last_training_size = 0
        self.training_threshold = 10
        self._training_texts = []
        self._training_embeddings = None

    def load_external_dataset(self):
        """Load the external phishing/scam dataset."""
        try:
            logger.info("Loading external dataset...")
            response = requests.get(EXTERNAL_DATASET_URL, timeout=30)
            response.raise_for_status()
            
            self.external_dataset = pd.read_csv(StringIO(response.text))
            scam_count = len(self.external_dataset[self.external_dataset['label'] == 1])
            logger.info(f"External dataset loaded: {scam_count} scam messages")
            
        except Exception as e:
            logger.error(f"Error loading external dataset: {e}")
            self.external_dataset = None

    def load_dataset(self):
        """Load all datasets."""
        # Load main dataset
        try:
            if os.path.exists(DATASET_FILE):
                self.dataset = pd.read_csv(DATASET_FILE)
                logger.info(f"Main dataset loaded: {len(self.dataset)} entries")
            else:
                self.dataset = pd.DataFrame(columns=[
                    'content', 'timestamp', 'channel_id', 'server_id', 'status', 'deletion_timestamp'
                ])
                self.save_dataset()
        except Exception as e:
            logger.error(f"Error loading main dataset: {e}")
            self.dataset = None

        # Load rejected dataset
        try:
            if os.path.exists(REJECTED_DATASET_FILE):
                self.rejected_dataset = pd.read_csv(REJECTED_DATASET_FILE)
                logger.info(f"Rejected dataset loaded: {len(self.rejected_dataset)} entries")
            else:
                self.rejected_dataset = pd.DataFrame(columns=[
                    'content', 'timestamp', 'channel_id', 'server_id', 'status', 'rejection_timestamp'
                ])
                self.save_rejected_dataset()
        except Exception as e:
            logger.error(f"Error loading rejected dataset: {e}")
            self.rejected_dataset = None

        # Load external dataset
        self.load_external_dataset()

    def save_dataset(self):
        """Save the main dataset."""
        if self.dataset is not None:
            try:
                self.dataset.to_csv(DATASET_FILE, index=False)
            except Exception as e:
                logger.error(f"Error saving main dataset: {e}")

    def save_rejected_dataset(self):
        """Save the rejected dataset."""
        if self.rejected_dataset is not None:
            try:
                self.rejected_dataset.to_csv(REJECTED_DATASET_FILE, index=False)
            except Exception as e:
                logger.error(f"Error saving rejected dataset: {e}")

    def add_message(self, message: discord.Message, status: str = "deleted"):
        """Add a deleted message to the dataset."""
        if self.dataset is None:
            self.load_dataset()
        
        if self.dataset is not None:
            new_entry = {
                'content': message.content,
                'timestamp': message.created_at.isoformat(),
                'channel_id': message.channel.id,
                'server_id': message.guild.id,
                'status': status,
                'deletion_timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.dataset = pd.concat([self.dataset, pd.DataFrame([new_entry])], ignore_index=True)
            self.save_dataset()
            self.train_from_dataset()

    def add_rejected_message(self, message: discord.Message):
        """Add a rejected message to the rejected dataset."""
        if self.rejected_dataset is None:
            self.load_dataset()
        
        if self.rejected_dataset is not None:
            new_entry = {
                'content': message.content,
                'timestamp': message.created_at.isoformat(),
                'channel_id': message.channel.id,
                'server_id': message.guild.id,
                'status': 'rejected',
                'rejection_timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.rejected_dataset = pd.concat([self.rejected_dataset, pd.DataFrame([new_entry])], ignore_index=True)
            self.save_rejected_dataset()

    def _prepare_training_data(self):
        """Prepare training data from both datasets."""
        # Get deleted messages from main dataset
        deleted_messages = self.dataset[self.dataset['status'] == 'deleted'] if self.dataset is not None else pd.DataFrame()
        internal_texts = deleted_messages['content'].astype(str).tolist() if len(deleted_messages) > 0 else []
        
        # Get scam messages from external dataset
        external_scam_texts = []
        if self.external_dataset is not None:
            scam_messages = self.external_dataset[self.external_dataset['label'] == 1]
            external_scam_texts = scam_messages['msg_content'].astype(str).tolist()
        
        return internal_texts + external_scam_texts

    def train_from_dataset(self, force: bool = False):
        """Train the nearest neighbor model."""
        if self.dataset is None or self.dataset.empty:
            return

        current_size = len(self.dataset)
        if not force and self.trained and (current_size - self.last_training_size) < self.training_threshold:
            return

        logger.info("Training nearest neighbor model...")
        
        try:
            training_texts = self._prepare_training_data()
            
            if len(training_texts) < 2:
                logger.warning("Not enough training data")
                return

            # Calculate embeddings
            embeddings = self.embedding_model.encode(training_texts)
            
            # Train model
            self.nearest_neighbor_model = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, len(embeddings)),
                metric='cosine',
                algorithm='auto'
            )
            self.nearest_neighbor_model.fit(embeddings)
            
            # Calculate threshold
            similarities = np.dot(embeddings, embeddings.T)
            np.fill_diagonal(similarities, 0)
            self.similarity_threshold = np.percentile(similarities.flatten(), 75)
            
            # Cache training data for faster inference
            self._training_texts = training_texts
            self._training_embeddings = embeddings
            
            self.trained = True
            self.last_training_size = current_size
            logger.info(f"Training completed: {len(training_texts)} samples")
            
        except Exception as e:
            logger.error(f"Training error: {e}")

    def find_matches(self, message: discord.Message) -> List[Tuple[Dict, float]]:
        """Find matching messages using nearest neighbor classification."""
        if not self.trained or self.nearest_neighbor_model is None:
            self.train_from_dataset()
            if not self.trained:
                return []

        try:
            # Encode message
            message_embedding = self.embedding_model.encode([message.content])
            
            # Find nearest neighbors
            distances, indices = self.nearest_neighbor_model.kneighbors(message_embedding)
            
            matches = []
            for distance, idx in zip(distances[0], indices[0]):
                similarity = 1 - distance
                
                if similarity > self.similarity_threshold:
                    # Determine source and create match object
                    if idx < len(self._training_texts) - (len(self.external_dataset[self.external_dataset['label'] == 1]) if self.external_dataset is not None else 0):
                        # Internal dataset match
                        deleted_messages = self.dataset[self.dataset['status'] == 'deleted']
                        matched_message = deleted_messages.iloc[idx].to_dict()
                        matched_message['source'] = 'internal'
                    else:
                        # External dataset match
                        matched_message = {
                            'content': self._training_texts[idx],
                            'source': 'external',
                            'label': 1
                        }
                    
                    matches.append((matched_message, similarity))
            
            return sorted(matches, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error in matching: {e}")
            return []

# Initialize components
message_filter = MessageFilter()
server_config = ServerConfig()

@bot.event
async def on_ready():
    """Called when the bot is ready."""
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
    if str(guild.id) not in server_config.config:
        server_config.config[str(guild.id)] = {'requires_approval': True}
        server_config.save_config()
    message_filter.load_dataset()

@bot.event
async def on_message(message: discord.Message):
    """Handle incoming messages."""
    if message.author == bot.user:
        return

    await bot.process_commands(message)

    # Handle flagging
    if message.reference and message.reference.resolved:
        if message.content.lower().startswith(FLAG_COMMAND.lower()):
            if message.author.guild_permissions.administrator:
                target_message = message.reference.resolved
                flag_reason = message.content[len(FLAG_COMMAND):].strip()
                
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
                
                view = discord.ui.View()
                confirm_button = discord.ui.Button(label="Confirm Delete", style=discord.ButtonStyle.danger)
                cancel_button = discord.ui.Button(label="Cancel", style=discord.ButtonStyle.secondary)

                async def confirm_callback(interaction: discord.Interaction):
                    if interaction.user.guild_permissions.administrator:
                        try:
                            await target_message.delete()
                            message_filter.add_message(target_message)
                            await interaction.response.send_message("Message deleted successfully.", ephemeral=True)
                            await message.delete()
                        except Exception as e:
                            await interaction.response.send_message(f"Error deleting message: {e}", ephemeral=True)
                    else:
                        await interaction.response.send_message("You don't have permission to delete messages.", ephemeral=True)

                async def cancel_callback(interaction: discord.Interaction):
                    if interaction.user.guild_permissions.administrator:
                        await interaction.response.send_message("Deletion cancelled.", ephemeral=True)
                        await message.delete()
                    else:
                        await interaction.response.send_message("You don't have permission to cancel deletions.", ephemeral=True)

                confirm_button.callback = confirm_callback
                cancel_button.callback = cancel_callback
                view.add_item(confirm_button)
                view.add_item(cancel_button)

                admin_channel_id = server_config.get_admin_channel(message.guild.id)
                if admin_channel_id:
                    admin_channel = bot.get_channel(admin_channel_id)
                    if admin_channel:
                        await admin_channel.send(embed=embed, view=view)
                        return
                await message.channel.send(embed=embed, view=view)
                return

    # Check for content-based matches
    matches = message_filter.find_matches(message)
    if matches:
        match, similarity = matches[0]
        
        if server_config.get_requires_approval(message.guild.id):
            admin_channel_id = server_config.get_admin_channel(message.guild.id)
            if admin_channel_id:
                admin_channel = bot.get_channel(admin_channel_id)
                if admin_channel:
                    match_source = match.get('source', 'internal') if isinstance(match, dict) else 'internal'
                    source_display = "External Dataset" if match_source == 'external' else "Internal Dataset"
                    
                    embed = discord.Embed(
                        title="Potential Spam Detected",
                        description="A message similar to previously deleted content was found.",
                        color=discord.Color.yellow()
                    )
                    embed.add_field(name="Author", value=message.author.mention)
                    embed.add_field(name="Channel", value=message.channel.mention)
                    embed.add_field(name="Content", value=message.content)
                    embed.add_field(name="Message Link", value=message.jump_url)
                    embed.add_field(name="Similarity Score", value=f"{similarity:.3f}")
                    embed.add_field(name="Match Source", value=source_display)

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
                            message_filter.add_rejected_message(message)
                            await interaction.response.send_message("Deletion rejected.", ephemeral=True)
                        else:
                            await interaction.response.send_message("You don't have permission to reject deletions.", ephemeral=True)

                    approve_button.callback = approve_callback
                    reject_button.callback = reject_callback
                    view.add_item(approve_button)
                    view.add_item(reject_button)

                    await admin_channel.send(embed=embed, view=view)
        else:
            try:
                await message.delete()
                message_filter.add_message(message)
                logger.info(f"Deleted message from {message.author} (similarity: {similarity:.3f})")
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
    """Reload the message dataset."""
    message_filter.load_dataset()
    await ctx.send("Dataset reloaded successfully.")

@bot.command(name='toggle_approval')
@commands.has_permissions(administrator=True)
async def toggle_approval(ctx):
    """Toggle the admin approval requirement."""
    current = server_config.get_requires_approval(ctx.guild.id)
    server_config.set_requires_approval(ctx.guild.id, not current)
    status = "enabled" if not current else "disabled"
    await ctx.send(f"Admin approval requirement {status}.")

@bot.command(name='status')
@commands.has_permissions(administrator=True)
async def status(ctx):
    """Display the current bot status."""
    admin_channel_id = server_config.get_admin_channel(ctx.guild.id)
    admin_channel = f"<#{admin_channel_id}>" if admin_channel_id else "Not set"
    
    main_size = len(message_filter.dataset) if message_filter.dataset is not None else 0
    rejected_size = len(message_filter.rejected_dataset) if message_filter.rejected_dataset is not None else 0
    external_size = len(message_filter.external_dataset) if message_filter.external_dataset is not None else 0
    
    embed = discord.Embed(title="Bot Status", color=discord.Color.blue())
    embed.add_field(name="Admin Approval", value="Enabled" if server_config.get_requires_approval(ctx.guild.id) else "Disabled")
    embed.add_field(name="Main Dataset Size", value=main_size)
    embed.add_field(name="Rejected Dataset Size", value=rejected_size)
    embed.add_field(name="External Dataset Size", value=external_size)
    embed.add_field(name="Admin Channel", value=admin_channel)
    embed.add_field(name="Detection Method", value="Content-based (Nearest Neighbor)")
    embed.add_field(name="Similarity Threshold", value=f"{message_filter.similarity_threshold:.3f}")
    embed.add_field(name="K Neighbors", value=message_filter.k_neighbors)
    embed.add_field(name="Flag Command", value=f"Reply to a message with `{FLAG_COMMAND} [reason]` to flag it for deletion")
    await ctx.send(embed=embed)

# Run the bot
if __name__ == "__main__":
    if not TOKEN:
        logger.error("No Discord token found. Please set the DISCORD_TOKEN environment variable.")
        exit(1)
    
    bot.run(TOKEN) 