"""
EMBEDDING GENERATION - Model 1: granite-embedding:30m
======================================================
Generates 384-dimensional embeddings using CRC Open WebUI API.
Includes checkpoint system for resumption after disconnection.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from tqdm import tqdm

# Configuration
PROJECT_DIR = "/groups/ecumming/rrazavi/twitter_project"
API_URL = "https://openwebui.crc.nd.edu/api/embed"
API_KEY = "YOUR_API_KEY_HERE"  # Replace with actual key
MODEL = "granite-embedding:30m"
EMBEDDING_DIM = 384
CHECKPOINT_EVERY = 5000

print("="*60)
print(f"EMBEDDING GENERATION - {MODEL}")
print("="*60)

# Load data
print("\nLoading preprocessed tweets...")
df = pd.read_csv(f'{PROJECT_DIR}/preprocessed_tweets.csv')
texts = df['clean_text'].tolist()
print(f"Total tweets: {len(texts)}")

# Initialize or load checkpoint
checkpoint_file = f'{PROJECT_DIR}/checkpoints/progress_granite_embedding_30m.json'
partial_file = f'{PROJECT_DIR}/checkpoints/partial_granite_embedding_30m.npy'

if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        progress = json.load(f)
    start_idx = progress['last_idx']
    failed_count = progress.get('failed', 0)
    embeddings = np.load(partial_file)
    print(f"Resuming from index {start_idx}")
else:
    start_idx = 0
    failed_count = 0
    embeddings = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
    print("Starting fresh")

# Generate embeddings
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

for i in tqdm(range(start_idx, len(texts)), desc="Generating embeddings"):
    text = texts[i]
    
    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"model": MODEL, "input": text},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            emb = data['embeddings'][0]
            if len(emb) == EMBEDDING_DIM:
                embeddings[i] = emb
            else:
                failed_count += 1
        else:
            failed_count += 1
            
    except Exception as e:
        failed_count += 1
    
    # Checkpoint
    if (i + 1) % CHECKPOINT_EVERY == 0:
        np.save(partial_file, embeddings)
        with open(checkpoint_file, 'w') as f:
            json.dump({'last_idx': i + 1, 'failed': failed_count}, f)
        print(f"\nCheckpoint saved at {i + 1}")

# Save final embeddings
output_file = f'{PROJECT_DIR}/checkpoints/embeddings_granite_embedding_30m.npy'
np.save(output_file, embeddings)
print(f"\nSaved embeddings to: {output_file}")
print(f"Shape: {embeddings.shape}")
print(f"Failed requests: {failed_count}")

# Clean up checkpoint files
if os.path.exists(partial_file):
    os.remove(partial_file)
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)

print("\n" + "="*60)
print("EMBEDDING GENERATION COMPLETE")
print("="*60)
