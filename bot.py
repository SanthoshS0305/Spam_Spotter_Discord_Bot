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
DATASET_FILE = os.getenv('DATASET_FILE', 'spam_dataset.csv')
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
        self.external_dataset: Optional[pd.DataFrame] = None
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nearest_neighbor_model = None
        self.similarity_threshold = 0.75  # Lower threshold for more sensitive detection
        self.k_neighbors = 3
        self.trained = False
        self._training_texts = []
        self._training_embeddings = None

    def load_external_dataset(self):
        """Load the external phishing/scam dataset."""
        try:
            logger.info("Loading external dataset from Hugging Face...")
            response = requests.get(EXTERNAL_DATASET_URL, timeout=30)
            response.raise_for_status()
            
            self.external_dataset = pd.read_csv(StringIO(response.text))
            scam_count = len(self.external_dataset[self.external_dataset['label'] == 1])
            total_count = len(self.external_dataset)
            logger.info(f"External dataset loaded: {scam_count} scam messages out of {total_count} total messages")
            
        except Exception as e:
            logger.error(f"Error loading external dataset: {e}")
            self.external_dataset = None

    def load_dataset(self):
        """Load all datasets."""
        # Load main dataset
        try:
            if os.path.exists(DATASET_FILE):
                self.dataset = pd.read_csv(DATASET_FILE)
                logger.info(f"Spam dataset loaded: {len(self.dataset)} entries")
            else:
                self.dataset = pd.DataFrame(columns=[
                    'content', 'timestamp', 'channel_id', 'server_id', 'deletion_timestamp'
                ])
                self.save_dataset()
                logger.info("Created new spam dataset")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.dataset = None

        # Load external dataset
        self.load_external_dataset()

    def save_dataset(self):
        """Save the spam dataset."""
        if self.dataset is not None:
            try:
                self.dataset.to_csv(DATASET_FILE, index=False)
            except Exception as e:
                logger.error(f"Error saving dataset: {e}")

    def add_spam_message(self, message: discord.Message):
        """Add a spam message to the dataset."""
        if self.dataset is None:
            self.load_dataset()
        
        if self.dataset is not None:
            new_entry = {
                'content': message.content,
                'timestamp': message.created_at.isoformat(),
                'channel_id': message.channel.id,
                'server_id': message.guild.id,
                'deletion_timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.dataset = pd.concat([self.dataset, pd.DataFrame([new_entry])], ignore_index=True)
            self.save_dataset()
            self.train_model()

    def _prepare_training_data(self):
        """Prepare training data from both internal and external datasets."""
        training_texts = []
        
        # Get spam messages from internal dataset
        if self.dataset is not None and len(self.dataset) > 0:
            internal_texts = self.dataset['content'].astype(str).tolist()
            training_texts.extend(internal_texts)
            logger.info(f"Added {len(internal_texts)} internal spam messages to training data")
        
        # Get scam messages from external dataset (label == 1)
        if self.external_dataset is not None:
            scam_messages = self.external_dataset[self.external_dataset['label'] == 1]
            external_scam_texts = scam_messages['msg_content'].astype(str).tolist()
            training_texts.extend(external_scam_texts)
            logger.info(f"Added {len(external_scam_texts)} external scam messages to training data")
        
        return training_texts

    def train_model(self):
        """Train the nearest neighbor model on spam messages."""
        training_texts = self._prepare_training_data()
        
        if len(training_texts) < 2:
            logger.warning("Not enough spam messages to train model")
            return

        try:
            logger.info(f"Training nearest neighbor model on {len(training_texts)} spam messages...")
            
            # Calculate embeddings
            embeddings = self.embedding_model.encode(training_texts)
            
            # Train nearest neighbor model
            self.nearest_neighbor_model = NearestNeighbors(
                n_neighbors=min(self.k_neighbors, len(embeddings)),
                metric='cosine',
                algorithm='auto'
            )
            self.nearest_neighbor_model.fit(embeddings)
            
            # Cache training data
            self._training_texts = training_texts
            self._training_embeddings = embeddings
            
            self.trained = True
            logger.info(f"Model trained successfully on {len(training_texts)} samples")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            self.trained = False

    def is_spam(self, message: discord.Message) -> Tuple[bool, float, Optional[str]]:
        """Check if a message is spam using nearest neighbor classification."""
        if not self.trained or self.nearest_neighbor_model is None:
            self.train_model()
            if not self.trained:
                return False, 0.0, None

        try:
            # Encode the message
            message_embedding = self.embedding_model.encode([message.content])
            
            # Find nearest neighbors
            distances, indices = self.nearest_neighbor_model.kneighbors(message_embedding)
            
            # Calculate similarity scores
            similarities = 1 - distances[0]
            
            # Check if any neighbor is similar enough
            max_similarity = max(similarities)
            if max_similarity > self.similarity_threshold:
                # Find the most similar spam message
                best_match_idx = indices[0][np.argmax(similarities)]
                best_match_text = self._training_texts[best_match_idx]
                return True, max_similarity, best_match_text
            
            return False, max_similarity, None
            
        except Exception as e:
            logger.error(f"Error checking spam: {e}")
            return False, 0.0, None

# Initialize components
message_filter = MessageFilter()
server_config = ServerConfig()

@bot.event
async def on_ready():
    """Called when the bot is ready."""
    logger.info(f'{bot.user} has connected to Discord!')
    message_filter.load_dataset()
    message_filter.train_model()

@bot.event
async def on_guild_join(guild: discord.Guild):
    """Called when the bot joins a new server."""
    logger.info(f"Joined new server: {guild.name}")
    if str(guild.id) not in server_config.config:
        server_config.config[str(guild.id)] = {'requires_approval': True}
        server_config.save_config()

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
                            message_filter.add_spam_message(target_message)
                            await interaction.response.send_message("Message deleted and added to spam dataset.", ephemeral=True)
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

    # Check for spam using nearest neighbor
    is_spam, similarity, matched_text = message_filter.is_spam(message)
    
    if is_spam:
        if server_config.get_requires_approval(message.guild.id):
            admin_channel_id = server_config.get_admin_channel(message.guild.id)
            if admin_channel_id:
                admin_channel = bot.get_channel(admin_channel_id)
                if admin_channel:
                    embed = discord.Embed(
                        title="Potential Spam Detected",
                        description="A message similar to known spam was detected.",
                        color=discord.Color.yellow()
                    )
                    embed.add_field(name="Author", value=message.author.mention)
                    embed.add_field(name="Channel", value=message.channel.mention)
                    embed.add_field(name="Content", value=message.content)
                    embed.add_field(name="Message Link", value=message.jump_url)
                    embed.add_field(name="Similarity Score", value=f"{similarity:.3f}")
                    if matched_text:
                        embed.add_field(name="Matched Spam", value=matched_text[:100] + "..." if len(matched_text) > 100 else matched_text)

                    view = discord.ui.View()
                    approve_button = discord.ui.Button(label="Delete as Spam", style=discord.ButtonStyle.danger)
                    reject_button = discord.ui.Button(label="Allow Message", style=discord.ButtonStyle.green)

                    async def approve_callback(interaction: discord.Interaction):
                        if interaction.user.guild_permissions.administrator:
                            try:
                                await message.delete()
                                message_filter.add_spam_message(message)
                                await interaction.response.send_message("Message deleted and added to spam dataset.", ephemeral=True)
                            except Exception as e:
                                await interaction.response.send_message(f"Error deleting message: {e}", ephemeral=True)
                        else:
                            await interaction.response.send_message("You don't have permission to delete messages.", ephemeral=True)

                    async def reject_callback(interaction: discord.Interaction):
                        if interaction.user.guild_permissions.administrator:
                            await interaction.response.send_message("Message allowed to remain.", ephemeral=True)
                        else:
                            await interaction.response.send_message("You don't have permission to allow messages.", ephemeral=True)

                    approve_button.callback = approve_callback
                    reject_button.callback = reject_callback
                    view.add_item(approve_button)
                    view.add_item(reject_button)

                    await admin_channel.send(embed=embed, view=view)
        else:
            try:
                await message.delete()
                message_filter.add_spam_message(message)
                logger.info(f"Deleted spam message from {message.author} (similarity: {similarity:.3f})")
            except Exception as e:
                logger.error(f"Error deleting spam message: {e}")

@bot.command(name='setadmin')
@commands.has_permissions(administrator=True)
async def set_admin_channel(ctx, channel: discord.TextChannel):
    """Set the admin channel for the current server."""
    server_config.set_admin_channel(ctx.guild.id, channel.id)
    await ctx.send(f"Admin channel set to {channel.mention}")

@bot.command(name='retrain')
@commands.has_permissions(administrator=True)
async def retrain_model(ctx):
    """Retrain the spam detection model."""
    message_filter.train_model()
    await ctx.send("Spam detection model retrained successfully.")

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
    
    dataset_size = len(message_filter.dataset) if message_filter.dataset is not None else 0
    external_size = len(message_filter.external_dataset) if message_filter.external_dataset is not None else 0
    external_scam_count = len(message_filter.external_dataset[message_filter.external_dataset['label'] == 1]) if message_filter.external_dataset is not None else 0
    
    embed = discord.Embed(title="Spam Detection Bot Status", color=discord.Color.blue())
    embed.add_field(name="Admin Approval", value="Enabled" if server_config.get_requires_approval(ctx.guild.id) else "Disabled")
    embed.add_field(name="Internal Dataset Size", value=dataset_size)
    embed.add_field(name="External Dataset Size", value=external_size)
    embed.add_field(name="External Scam Messages", value=external_scam_count)
    embed.add_field(name="Admin Channel", value=admin_channel)
    embed.add_field(name="Detection Method", value="Nearest Neighbor (Content-based)")
    embed.add_field(name="Similarity Threshold", value=f"{message_filter.similarity_threshold:.3f}")
    embed.add_field(name="K Neighbors", value=message_filter.k_neighbors)
    embed.add_field(name="Model Trained", value="Yes" if message_filter.trained else "No")
    embed.add_field(name="Flag Command", value=f"Reply to a message with `{FLAG_COMMAND} [reason]` to flag it as spam")
    await ctx.send(embed=embed)

# Run the bot
if __name__ == "__main__":
    if not TOKEN:
        logger.error("No Discord token found. Please set the DISCORD_TOKEN environment variable.")
        exit(1)
    
    bot.run(TOKEN) 