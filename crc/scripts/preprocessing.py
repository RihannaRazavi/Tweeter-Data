"""
PREPROCESSING SCRIPT
====================
Cleans and prepares Twitter data for embedding generation.
"""

import pandas as pd
import re
import os

print("="*60)
print("PREPROCESSING TWITTER DATA")
print("="*60)

# Load raw data
print("\nLoading raw data...")
df = pd.read_csv('/groups/ecumming/rrazavi/twitter_project/ALL_TWEETS_MASTER.csv')
print(f"Loaded {len(df)} tweets")

# Text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtag symbol but keep word
    text = re.sub(r'#', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase
    text = text.lower()
    return text

# Apply cleaning
print("Cleaning text...")
df['clean_text'] = df['text'].apply(clean_text)

# Remove empty tweets
df = df[df['clean_text'].str.len() > 10]
print(f"After cleaning: {len(df)} tweets")

# Save preprocessed data
output_path = '/groups/ecumming/rrazavi/twitter_project/preprocessed_tweets.csv'
df.to_csv(output_path, index=False)
print(f"\nSaved to: {output_path}")

print("\n" + "="*60)
print("PREPROCESSING COMPLETE")
print("="*60)
