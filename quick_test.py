import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading dataset...")
dataset = pd.read_csv('global_dataset.csv')
spam_texts = dataset['content'].astype(str).tolist()

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

test_msg = "yo, is anyone interested in going to a coldplay concert? I've got spare tickets for 2. DM if interested"
print(f"\nTesting: '{test_msg}'")

# Get embeddings
test_embedding = model.encode([test_msg])
spam_embeddings = model.encode(spam_texts)

# Calculate similarities
similarities = []
for spam_emb in spam_embeddings:
    similarity = np.dot(test_embedding[0], spam_emb) / (np.linalg.norm(test_embedding[0]) * np.linalg.norm(spam_emb))
    similarities.append(similarity)

# Get top 3 matches
top_indices = np.argsort(similarities)[-3:][::-1]

print(f"\nThreshold: 0.70 (NEW)")
print(f"Top 3 similarities:")
for idx in top_indices:
    similarity = similarities[idx]
    would_trigger = "YES" if similarity > 0.70 else "NO"
    spam_msg = spam_texts[idx][:60] + "..." if len(spam_texts[idx]) > 60 else spam_texts[idx]
    print(f"{similarity:.3f} ({would_trigger}) - {spam_msg}")

max_sim = max(similarities)
print(f"\nHighest similarity: {max_sim:.3f}")
print(f"Would trigger: {'YES' if max_sim > 0.70 else 'NO'}")

# Test the other messages too
print(f"\n{'='*50}")
print("Testing other messages with new threshold:")
print(f"{'='*50}")

test_messages = [
    "Hi guys Is there anyone who is interested in buying cold play tickets for next sunday? I bought some for my parents, but they decided to go to an arjit singh concert isntead. DM if interested",
    "Hello. @everyone I want to give out my HP Omen 16 Laptop for free, it's in perfect health and good as new, alongside a charger so it's perfect, I want to give it out because I just got a new model and I thought of giving out the old one to someone who can't afford one and is in need of it... Strictly First come first serve !"
]

for i, msg in enumerate(test_messages, 1):
    test_emb = model.encode([msg])
    similarities = []
    for spam_emb in spam_embeddings:
        similarity = np.dot(test_emb[0], spam_emb) / (np.linalg.norm(test_emb[0]) * np.linalg.norm(spam_emb))
        similarities.append(similarity)
    
    max_sim = max(similarities)
    print(f"Message {i}: {max_sim:.3f} ({'YES' if max_sim > 0.70 else 'NO'})") 