#!/usr/bin/env python3

import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_similarity():
    """Test similarity scores for given messages."""
    
    # Load the dataset
    dataset_file = os.getenv('DATASET_FILE', 'global_dataset.csv')
    print(f"Loading dataset from: {dataset_file}")
    
    try:
        dataset = pd.read_csv(dataset_file)
        print(f"Loaded {len(dataset)} messages from dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Get spam messages from dataset
    if 'content' in dataset.columns:
        spam_texts = dataset['content'].astype(str).tolist()
    else:
        print("No 'content' column found in dataset")
        return
    
    print(f"Found {len(spam_texts)} spam messages for comparison")
    
    # Initialize the embedding model
    print("Loading sentence transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test messages
    test_messages = [
        "Hi guys Is there anyone who is interested in buying cold play tickets for next sunday? I bought some for my parents, but they decided to go to an arjit singh concert isntead. DM if interested",
        "Hello. @everyone I want to give out my HP Omen 16 Laptop for free, it's in perfect health and good as new, alongside a charger so it's perfect, I want to give it out because I just got a new model and I thought of giving out the old one to someone who can't afford one and is in need of it... Strictly First come first serve !"
    ]
    
    # Calculate embeddings for spam messages
    print("Calculating embeddings for spam messages...")
    spam_embeddings = embedding_model.encode(spam_texts)
    
    # Train nearest neighbor model
    print("Training nearest neighbor model...")
    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='auto')
    nn_model.fit(spam_embeddings)
    
    # Test each message
    for i, test_msg in enumerate(test_messages, 1):
        print(f"\n{'='*60}")
        print(f"TESTING MESSAGE {i}:")
        print(f"'{test_msg}'")
        print(f"{'='*60}")
        
        # Calculate embedding for test message
        test_embedding = embedding_model.encode([test_msg])
        
        # Find nearest neighbors
        distances, indices = nn_model.kneighbors(test_embedding)
        
        # Calculate similarity scores
        similarities = 1 - distances[0]
        
        print(f"Top 5 most similar spam messages:")
        print(f"Current threshold: 0.75 (75%)")
        print(f"{'Similarity':<10} {'Would Trigger':<12} {'Message'}")
        print(f"{'-'*60}")
        
        for j, (similarity, idx) in enumerate(zip(similarities, indices[0])):
            would_trigger = "YES" if similarity > 0.75 else "NO"
            spam_msg = spam_texts[idx][:100] + "..." if len(spam_texts[idx]) > 100 else spam_texts[idx]
            print(f"{similarity:.3f}      {would_trigger:<12} {spam_msg}")
        
        max_similarity = max(similarities)
        print(f"\nHighest similarity: {max_similarity:.3f}")
        print(f"Would be flagged as spam: {'YES' if max_similarity > 0.75 else 'NO'}")

if __name__ == "__main__":
    test_similarity() 