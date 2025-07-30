import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load dataset
print("Loading dataset...")
dataset = pd.read_csv('global_dataset.csv')
spam_texts = dataset['content'].astype(str).tolist()
print(f"Loaded {len(spam_texts)} spam messages")

# Initialize model
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test messages
test_messages = [
    "yo, is anyone interested in going to a coldplay concert? I've got spare tickets for 2. DM if interested"
]

# Calculate embeddings
print("Calculating embeddings...")
spam_embeddings = model.encode(spam_texts)

for i, test_msg in enumerate(test_messages, 1):
    print(f"\n{'='*50}")
    print(f"TESTING MESSAGE {i}:")
    print(f"'{test_msg}'")
    print(f"{'='*50}")
    
    # Get test embedding
    test_embedding = model.encode([test_msg])
    
    # Calculate similarities
    similarities = []
    for spam_emb in spam_embeddings:
        similarity = np.dot(test_embedding[0], spam_emb) / (np.linalg.norm(test_embedding[0]) * np.linalg.norm(spam_emb))
        similarities.append(similarity)
    
    # Get top 5 matches
    top_indices = np.argsort(similarities)[-5:][::-1]
    
    print(f"Threshold: 0.75")
    print(f"Top 5 similarities:")
    for idx in top_indices:
        similarity = similarities[idx]
        would_trigger = "YES" if similarity > 0.75 else "NO"
        spam_msg = spam_texts[idx][:80] + "..." if len(spam_texts[idx]) > 80 else spam_texts[idx]
        print(f"{similarity:.3f} ({would_trigger}) - {spam_msg}")
    
    max_sim = max(similarities)
    print(f"\nHighest similarity: {max_sim:.3f}")
    print(f"Would trigger: {'YES' if max_sim > 0.75 else 'NO'}") 