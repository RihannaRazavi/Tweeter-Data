# Data Information

## Dataset Overview

**Note:** Actual data files are NOT included in this repository due to size and privacy considerations.

## Files Description

| File | Description | Size | Location |
|------|-------------|------|----------|
| `ALL_TWEETS_MASTER.csv` | Raw Twitter data | ~150 MB | Google Drive |
| `preprocessed_tweets.csv` | Cleaned tweets | ~120 MB | CRC |
| `embeddings_granite_embedding_30m.npy` | Model 1 embeddings | 465 MB | CRC |
| `embeddings_granite_embedding_278m.npy` | Model 2 embeddings | 930 MB | CRC |
| `embeddings_qwen3_embedding_8b.npy` | Model 3 embeddings | 4.9 GB | CRC |

## Data Statistics

- **Total Tweets:** 302,928
- **Collection Period:** 2015-2024
- **Language:** English
- **Search Queries:** 61 marriage-related terms

## Columns in Preprocessed Data

| Column | Description |
|--------|-------------|
| `text` | Original tweet text |
| `clean_text` | Preprocessed text (lowercase, no URLs, etc.) |
| `createdAt` | Tweet timestamp |
| `author_userName` | Twitter username |
| `likeCount` | Number of likes |
| `retweetCount` | Number of retweets |

## Data Access

Dataset available at: [Google Drive Link](https://drive.google.com/drive/folders/16XXPylcNEPI-ZffEvckNSaFuLzI0er9u)

## Embeddings Shape

| Model | Shape | Dimensions |
|-------|-------|------------|
| granite-30m | (302928, 384) | 384 |
| granite-278m | (302928, 768) | 768 |
| qwen3-8b | (302928, 4096) | 4096 |
