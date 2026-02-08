# Comparative Analysis of Embedding Models and Fine-Tuning Approaches for Social Media Text Clustering

## Project Overview

This project performs unsupervised topic discovery and sentiment analysis on a large-scale Twitter dataset (~303,000 marriage-related tweets) using two parallel approaches:

1. **Pre-trained Embedding Models** via Notre Dame CRC infrastructure
2. **Fine-tuned Language Models** using SimCSE (Simple Contrastive Learning of Sentence Embeddings)

The goal is to compare the effectiveness of off-the-shelf embedding models against self-supervised fine-tuning approaches for social media text clustering.

## Objectives

- Generate text embeddings using multiple pre-trained models of varying sizes
- Implement SimCSE to fine-tune BERT for domain-specific embeddings
- Apply clustering algorithms to discover latent topics
- Perform sentiment analysis on identified clusters
- Compare clustering quality across all embedding approaches
- Visualize results using dimensionality reduction techniques

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | Twitter API |
| Size | ~303,000 tweets |
| Focus | Marriage and relationship-related content |
| Language | English |
| Queries | 61 marriage-related search terms |

## Methods

### Approach 1: Pre-trained Embedding Models (via Notre Dame CRC)

Using CRC's Open WebUI API to generate embeddings without fine-tuning:

| Model | Parameters | Embedding Dimensions |
|-------|------------|---------------------|
| granite-embedding:30m | 30 million | 384 |
| granite-embedding:278m | 278 million | 768 |
| qwen3-embedding:8b | 8 billion | 4096 |

### Approach 2: Fine-tuned Language Models (SimCSE)

Self-supervised contrastive learning to create domain-specific embeddings:

| Model | Base Model | Approach |
|-------|------------|----------|
| BERT + SimCSE | bert-base-uncased (110M params) | Unsupervised contrastive learning |
| BERT + Cluster Fine-tuning | bert-base-uncased (110M params) | Supervised cluster-based training |

### Clustering Algorithms

- **K-means:** k = 8, 10, 15, 20
- **HDBSCAN:** Density-based clustering with automatic cluster detection

### Dimensionality Reduction

- **UMAP:** Uniform Manifold Approximation and Projection
- **t-SNE:** t-Distributed Stochastic Neighbor Embedding
- **PCA:** Principal Component Analysis

### Text Analysis

- TF-IDF word frequency analysis
- Word cloud generation
- Lexicon-based sentiment analysis

## Infrastructure

### CRC Resources (Pre-trained Models)

| Resource | Details |
|----------|---------|
| Computing | Notre Dame Center for Research Computing (CRC) |
| Job Scheduler | SGE (Sun Grid Engine) |
| API | CRC Open WebUI API |
| Environment | Python 3.11 with Conda |

### Fine-tuning Resources (SimCSE)

| Resource | Details |
|----------|---------|
| Platform | Google Colab Pro |
| GPU | Tesla T4 |
| Frameworks | PyTorch, HuggingFace Transformers |
