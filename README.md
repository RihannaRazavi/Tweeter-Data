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














----------------------------------------------------------------------------------------------


# Part 2: Dataset Acquisition and Description

## Dataset Overview

This project utilizes a large-scale Twitter dataset focused on marriage and relationship discourse. The dataset supports two distinct analytical approaches with different data handling strategies.

**Dataset Link:** [Google Drive - Marriage Tweets Dataset](https://drive.google.com/drive/folders/16XXPylcNEPI-ZffEvckNSaFuLzI0er9u)

## Data Source and Collection

| Attribute | Description |
|-----------|-------------|
| **Source** | Twitter API |
| **Collection Period** | 2015-2024 |
| **Total Samples** | 302,928 tweets |
| **Language** | English |
| **Search Queries** | 61 marriage-related terms (e.g., "husband", "wife", "married", "wedding", "spouse", "hubby", "wifey") |
| **File Format** | CSV |
| **Total Size** | ~150 MB (preprocessed) |

## Data Handling Strategy

This project employs **two different approaches** that require different data handling:

### Approach 1: CRC Pre-trained Embedding Models (No Split Required)

For the CRC embedding analysis using pre-trained models (granite-30m, granite-278m, qwen3-8b), **no train/validation/test split is required** because:

| Reason | Explanation |
|--------|-------------|
| **No Training** | We use pre-trained models via API calls - no model weights are updated |
| **Unsupervised Clustering** | K-means, HDBSCAN, and other clustering algorithms are unsupervised - they discover patterns without labeled "correct answers" |
| **Internal Evaluation Metrics** | Silhouette score, Calinski-Harabasz, and Davies-Bouldin indices evaluate cluster quality using the data itself, not held-out predictions |
| **Full Data Utilization** | Using all 302,928 tweets maximizes the representativeness of discovered clusters |

**CRC Pipeline:**
```
302,928 tweets (ALL DATA)
    ↓
Pre-trained embedding model (API call, no training)
    ↓
302,928 embeddings
    ↓
Clustering (unsupervised, uses all data)
    ↓
Evaluate with internal metrics (silhouette score)
```

### Approach 2: Fine-tuned Language Models (Split Required)

For the SimCSE fine-tuning and cluster-based fine-tuning approaches, **data splitting is necessary** because we are updating model weights and need to prevent overfitting:

| Subset | Percentage | Sample Count | Purpose |
|--------|------------|--------------|---------|
| **Training** | 60% | 181,757 | Fine-tuning model weights via SimCSE contrastive learning |
| **Validation** | 20% | 60,586 | Hyperparameter tuning, early stopping, monitoring generalization |
| **Test** | 20% | 60,585 | Final evaluation (untouched until final comparison) |

## Differences Between Subsets (For Fine-tuning Only)

### Training Set (60%)
- **Purpose:** Fine-tune BERT using SimCSE contrastive learning
- **Properties:** Largest portion to ensure robust learning of semantic relationships
- **Usage:** Model weight updates, contrastive pair generation

### Validation Set (20%)
- **Purpose:** Monitor generalization and prevent overfitting
- **Properties:** Stratified sampling to maintain topic distribution
- **Usage:** Early stopping, learning rate selection, epoch selection

### Test Set (20%)
- **Purpose:** Final unbiased evaluation comparing fine-tuned vs. pre-trained models
- **Properties:** Completely held out until final evaluation
- **Usage:** Compare SimCSE fine-tuned embeddings against CRC pre-trained embeddings

## Sample Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Text Length** | 10-280 characters (Twitter limit) |
| **Average Length** | ~120 characters |
| **Preprocessing** | Lowercased, URLs removed, special characters cleaned, emojis preserved |
| **Sentiment Distribution** | ~25% positive, ~65% neutral, ~10% negative |
| **Language** | English only |

## Sample Data Examples

```
"happy anniversary to my amazing husband of 10 years love you forever"
"my wife made the best dinner tonight so grateful for her"
"married life isnt always easy but its worth it"
"hubby surprised me with flowers today feeling blessed"
"celebrating 5 years of marriage with my best friend"
"wedding planning is stressful but excited for the big day"
```

## Models Used

### Approach 1: Pre-trained Embedding Models (Notre Dame CRC)

Using CRC's Open WebUI API - **no training required, uses full dataset:**

| Model | Parameters | Embedding Dimensions | Data Used |
|-------|------------|---------------------|-----------|
| granite-embedding:30m | 30 million | 384 | All 302,928 tweets |
| granite-embedding:278m | 278 million | 768 | All 302,928 tweets |
| qwen3-embedding:8b | 8 billion | 4096 | All 302,928 tweets |

### Approach 2: Fine-tuned Language Models

**Requires train/validation/test split:**

| Model | Parameters | Method | Data Split |
|-------|------------|--------|------------|
| BERT-base-uncased | 110 million | SimCSE contrastive learning | 60/20/20 |
| BERT-base-uncased | 110 million | Cluster-based fine-tuning | 60/20/20 |
| Sentence-BERT (all-MiniLM-L6-v2) | 23 million | Transfer learning | 60/20/20 |

### Sentiment and Topic Analysis Models

| Model | Type | Purpose |
|-------|------|---------|
| VADER | Rule-based | Fast sentiment scoring |
| BERT-sentiment | 110M transformer | Advanced sentiment classification |
| RoBERTa-emotion | 82M transformer | 7-class emotion detection |
| BART-MNLI | 406M transformer | Zero-shot topic categorization |
| LDA | Probabilistic | Unsupervised topic discovery |

### Clustering and Visualization

| Algorithm | Type | Purpose |
|-----------|------|---------|
| K-means | Partitioning | Primary clustering (k=3-15) |
| HDBSCAN | Density-based | Automatic cluster detection |
| Agglomerative | Hierarchical | Alternative clustering |
| Spectral | Graph-based | Non-linear cluster boundaries |
| PCA | Linear reduction | Fast visualization |
| t-SNE | Non-linear reduction | Local structure preservation |
| UMAP | Manifold learning | Best visualization quality |

**Total: 5 transformer models (~731M parameters) + 10 traditional ML algorithms**

## Evaluation Strategy

### For CRC Pre-trained Models (Unsupervised):
- **Silhouette Score:** Cluster separation quality (-1 to 1)
- **Calinski-Harabasz Index:** Between-cluster vs. within-cluster variance
- **Davies-Bouldin Index:** Average cluster similarity (lower is better)
- **Optimal K:** Testing k=3 to 15 clusters

### Cluster Stability via Subsampling

To evaluate clustering robustness, we employ a **subsampling stability approach** - a method proposed for assessing the reliability of clustering solutions. This technique addresses the critical question: *"Are the discovered clusters real patterns or artifacts of the specific data sample?"*

**Method:**
1. Randomly subsample the dataset multiple times (e.g., 10 iterations at 80% of data)
2. Apply the same clustering algorithm to each subsample
3. Measure cluster agreement across subsamples using Adjusted Rand Index (ARI)
4. Stable clustering = high ARI across subsamples (clusters persist regardless of which data points are included)

**Interpretation:**
| ARI Score | Stability Level | Interpretation |
|-----------|-----------------|----------------|
| 0.9 - 1.0 | Excellent | Clusters are highly stable and reproducible |
| 0.7 - 0.9 | Good | Clusters are reliable with minor variations |
| 0.5 - 0.7 | Moderate | Some cluster instability; interpret with caution |
| < 0.5 | Poor | Clusters are unstable; may not reflect true structure |

This subsampling approach is particularly important for our analysis because:
- It validates that clusters represent genuine discourse patterns, not sampling noise
- It helps select the optimal number of clusters (k) - stable k values are preferred
- It provides confidence in the interpretability of cluster themes

### For Fine-tuned Models (Supervised):
- **Training Loss:** Contrastive loss during SimCSE training
- **Validation Loss:** Monitor for overfitting
- **Test Evaluation:** Final silhouette score comparison on held-out data


## Infrastructure

| Resource | Specification |
|----------|---------------|
| **CRC Computing** | Notre Dame Center for Research Computing |
| **Job Scheduler** | SGE (Sun Grid Engine) |
| **Memory** | 64-128 GB per job |
| **Fine-tuning GPU** | Google Colab Pro (Tesla T4) |
| **Environment** | Python 3.11, PyTorch, HuggingFace Transformers |

## Data Attestation

I attest that I have physically downloaded and processed the dataset described in this report. The data is stored locally and on Notre Dame's CRC infrastructure for analysis.

---

*Report prepared for Data Science in Practice - Semester Project*  
*University of Notre Dame*
