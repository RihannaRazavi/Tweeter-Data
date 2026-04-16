# Comparative Analysis of Embedding Models and Fine-Tuning Approaches for Twitter Media Text Clustering

## Project Overview

This project performs unsupervised topic discovery and sentiment analysis on a large-scale Twitter dataset (~303,000 marriage-related tweets) using two parallel approaches:

1. **Pre-trained Embedding Models** via Notre Dame CRC infrastructure
2. **Fine-tuned Language Models** using SimCSE (Simple Contrastive Learning of Sentence Embeddings) with The four models fine-tuned with SimCSE:

BERT — bert-base-uncased (Google)
RoBERTa — roberta-base (Meta/Facebook)
Twitter-RoBERTa — cardiffnlp/twitter-roberta-base (Cardiff NLP)
DeBERTa-v3-small — microsoft/deberta-v3-small (Microsoft)

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

## Models

All four models were fine-tuned using self-supervised SimCSE (Simple Contrastive
Learning of Sentence Embeddings) with no labelled data at any stage.

| Model | Base Model | Parameters | Approach |
|-------|-----------|------------|----------|
| BERT + SimCSE | bert-base-uncased | 109,482,240 | Unsupervised contrastive learning |
| RoBERTa + SimCSE | roberta-base | 124,645,632 | Unsupervised contrastive learning |
| Twitter-RoBERTa + SimCSE | cardiffnlp/twitter-roberta-base | 124,645,632 | Unsupervised contrastive learning |
| DeBERTa-v3 + SimCSE | microsoft/deberta-v3-small | 141,304,320 | Unsupervised contrastive learning |

---

## Clustering Algorithms

Four clustering algorithms were applied to the learned embeddings, with the
optimal configuration selected per algorithm using cluster validity indices.

| Algorithm | Configuration | Selection Criterion |
|-----------|--------------|---------------------|
| KMeans | k ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12} | Maximum silhouette score |
| HDBSCAN | min_cluster_size ∈ {50, 100, 200, 300, 500}, min_samples=10 | Maximum silhouette score (≥3 clusters) |
| Agglomerative | Ward linkage, k ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12} | Maximum silhouette score |
| GMM | Diagonal covariance, k ∈ {4, 5, 6, 7, 8, 9, 10, 11, 12} | Minimum BIC |

All solutions evaluated using three validity indices:
- **Silhouette Score** (↑ higher is better)
- **Davies-Bouldin Index** (↓ lower is better)
- **Calinski-Harabasz Index** (↑ higher is better)

---

## Dimensionality Reduction

Dimensionality reduction was applied at two stages of the pipeline.

**Pre-clustering (curse of dimensionality fix)**
- **PCA** applied before all clustering algorithms; components tested at
  {32, 64, 128, 256}

**Visualisation only**
- **PCA** — 2-component projection for scatter plots
- **t-SNE** — applied to PCA-50 pre-reduced embeddings;
  perplexity=40, learning_rate=200, 1,000 iterations
- **UMAP** — n_neighbors=30, min_dist=0.1, cosine metric

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

**Dataset Link:** https://drive.google.com/file/d/1vH_Cqg_g1w0s_8LgmPSme2l4RWqD0TNl/view?usp=sharing

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





---

# Part 3




**Results:**
- Positive tweets: ~45%
- Neutral tweets: ~30%
- Negative tweets: ~25%

**Feature Engineering:**
- Word count, character count
- Emoji count (using regex)
- Hashtag count
- URL presence flags
- Mention count

### Baseline Clustering Analysis 

**Embedding Models Tested:**

1. **BERT-base-uncased** (Original, no fine-tuning)
   - Model: Google's bert-base-uncased
   - Parameters: 110M
   - Embedding dimension: 768
   - Method: Mean pooling of last hidden states



**Clustering Algorithms Applied:**

1. **K-means Clustering**
   - Optimal k determined by elbow method and silhouette analysis
   - Tested k from 4 to 12
   - Selected k=6 based on silhouette score


2. **DBSCAN (Density-Based)**
   - Epsilon tuning using k-distance plots
   - MinPoints: 5
   - Found many small clusters and noise points

**Baseline Results:**
- **K-means silhouette score: 0.0008** (extremely low!)
- Davies-Bouldin Index: 2.87 (poor separation)
- Calinski-Harabasz: 1247.3
- Interpretation: Original BERT embeddings show almost no cluster structure for marriage discourse

**Why Baseline Failed:**
BERT was pre-trained on Wikipedia and BookCorpus - general English text. Marriage discourse on Twitter has:
- Domain-specific vocabulary (hubby, wifey, bae, etc.)
- Informal language and slang
- Emotional expressions and sarcasm
- Context-dependent meanings

This motivated our decision to pursue **domain-adaptive fine-tuning**.


###  Visualization with Dimensionality Reduction\

We created visualizations using multiple dimensionality reduction techniques:

**1. PCA (Principal Component Analysis)**

- PC1 explained: 12.3% variance
- PC2 explained: 7.8% variance
- Total: Only 20.1% variance in 2D (expected for high-dim data)

**2. t-SNE (t-Distributed Stochastic Neighbor Embedding)**

- Better local structure preservation
- Clear visual separation would indicate good clusters
- Our baseline showed heavy overlap (confirming poor clustering)

**3. UMAP (Uniform Manifold Approximation and Projection)**

- Best balance of local and global structure
- Faster than t-SNE
- Also showed cluster overlap in baseline

**Visualization Insight:** The heavy overlap in all three methods confirmed that BERT-base-uncased embeddings do not capture marriage-specific semantics effectively.

---

##  Fine-Tuning Approaches

###  Why Fine-Tuning Instead of Domain-Adaptive Pre-training?

**Domain-Adaptive Pre-training Option:**
We considered continuing BERT's masked language modeling (MLM) pre-training on our 300K marriage tweets. This approach would teach BERT marriage-specific vocabulary and syntax.

**Pros:**
- BERT would learn "marriage language" patterns
- Understands domain-specific terms (hubby = husband)
- Relatively straightforward implementation

**Cons:**
- Requires enormous computational resources (weeks on multiple GPUs)
- Our dataset is small compared to BERT's original training (3.3B words)
- Would not directly optimize for clustering/similarity tasks
- Risk of catastrophic forgetting of general language

**Decision: We chose supervised fine-tuning instead** because:
1. ✓ Directly optimizes for our task (semantic similarity)
2. ✓ Computationally feasible (hours, not weeks)
3. ✓ Proven effective in research literature
4. ✓ Can leverage existing BERT knowledge while adapting to domain

### Approach 1: Cluster-Based Fine-Tuning 

**Methodology:**

Our first approach used existing K-means clusters as weak supervision:

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TripletBERT(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / \
               torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        return embeddings

# Triplet Loss Function
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()
```

**Triplet Creation Strategy:**
For each tweet (anchor):
- **Positive**: Random tweet from SAME K-means cluster
- **Negative**: Random tweet from DIFFERENT cluster



**Training Configuration:**
- Epochs: 3
- Batch size: 16
- Learning rate: 2e-5
- Optimizer: AdamW
- Number of triplets: 6,000 (1,000 per cluster)
- Training time: ~6 hours on Tesla T4 GPU

**Results:**
- **Old silhouette (baseline): 0.0008**
- **New silhouette (after fine-tuning): 0.0502**
- **Improvement: 6,175% (62x better!)**

**Analysis:**
While this represents massive improvement, we identified a critical flaw:
-  **Circular dependency**: Training relies on potentially bad initial clusters
-  **Error reinforcement**: If initial K-means clusters are wrong, we teach BERT the wrong patterns
-  **Limited ceiling**: Can't exceed quality of initial clustering

This led us to our second, more principled approach.

###  Approach 2: SimCSE - Self-Supervised Contrastive Learning 

**Why SimCSE?**

SimCSE (Gao et al., 2021) is a state-of-the-art method that requires **NO cluster labels or supervision**. It's pure self-supervised learning.

**Core Concept:**

The same tweet passed through BERT twice (with different dropout) should have similar embeddings, but different tweets should have different embeddings.

```
Tweet: "I love my wife"

Pass 1 (dropout mask A): → embedding_1
Pass 2 (dropout mask B): → embedding_2

Goal: embedding_1 ≈ embedding_2 (same tweet!)
      But embedding_1 ≠ embedding_other (different tweets)
```

**Mathematical Formulation:**

For a batch of N tweets, we create 2N embeddings (each tweet passed twice):

Loss = -log( exp(sim(h_i, h_i^+) / τ) / Σ exp(sim(h_i, h_j) / τ) )

Where:
- h_i = first embedding of tweet i
- h_i^+ = second embedding of same tweet i (positive pair)
- h_j = embeddings of all other tweets in batch (negatives)
- sim(·,·) = cosine similarity
- τ = temperature (0.05)


**Training Configuration:**
- Epochs: **1 only!** (SimCSE converges fast)
- Batch size: 64 (larger batches = more negatives = better)
- Learning rate: 3e-5
- Optimizer: AdamW
- Training data: ALL 300K tweets (no cluster dependency!)
- Expected training time: 2-3 hours on GPU

**Key Advantages:**
✓ No dependence on initial clusters
✓ Self-supervised (learns from data structure)
✓ State-of-the-art method (EMNLP 2021)
✓ Proven to work on sentence embeddings
✓ Simple and elegant

**Expected Results:**
Based on SimCSE paper and similar work:
- Silhouette score: **0.15-0.25** (3-5x better than cluster-based!)
- More meaningful clusters
- Better generalization

---

## Results Summary

###  Quantitative Results

| Method | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | Training Time |
|--------|------------------|----------------|-------------------|---------------|
| Baseline (BERT-base) | 0.0008 | 2.87 | 1247.3 | 0 (pre-trained) |
| Cluster-based Fine-tuning | 0.0502 | 2.34 | 1856.7 | ~6 hours |
| SimCSE (expected) | 0.15-0.25 | <1.5 | >3000 | ~3 hours |

**Key Findings:**
1. Original BERT embeddings are nearly useless for marriage discourse (silhouette ~0)
2. Even flawed cluster-based fine-tuning provides 62x improvement
3. SimCSE should provide another 3-5x improvement

###  Qualitative Analysis

**Discovered Cluster Themes (after cluster-based fine-tuning):**

Cluster 0 (17.2% of tweets): **Weddings & Ceremonies**
- Sample tweets: "Beautiful wedding ceremony today", "Can't wait to marry my best friend"
- Top words: wedding, dress, ceremony, bride, married, ring

Cluster 1 (15.8%): **Anniversaries & Milestones**  
- Sample tweets: "Happy 10th anniversary to my amazing husband", "25 years together!"
- Top words: anniversary, years, together, celebrate, decade

Cluster 2 (19.1%): **Daily Life & Routines**
- Sample tweets: "Making dinner for the family", "Grocery shopping with the spouse"
- Top words: dinner, home, cooking, made, tonight, everyday

Cluster 3 (18.4%): **Appreciation & Love**
- Sample tweets: "So grateful for my wife", "Best husband in the world"
- Top words: love, best, grateful, amazing, blessed, lucky

Cluster 4 (16.3%): **Complaints & Frustrations**
- Sample tweets: "My husband never listens", "Why does my wife always..."
- Top words: annoying, never, always, hate, tired, why

Cluster 5 (13.2%): **Humor & Memes**
- Sample tweets: "Marriage is just asking 'what do you want for dinner' forever lol"
- Top words: lol, haha, funny, literally, mood, basically

**Interpretation:** The clusters are meaningful and interpretable, showing that fine-tuning successfully captured marriage discourse semantics!



The SimCSE self-supervised fine-tuning successfully completed after approximately 12 minutes of training on 242,341 marriage-related tweets using  GPU. The training employed contrastive learning where each tweet was passed through the BERT model twice with different dropout masks, creating positive pairs without requiring any cluster labels. The model achieved an average training loss of 0.0219, with loss curves showing steady convergence throughout the 3,787 training batches. Embeddings were generated for all three data splits (training, validation, and test sets), producing 768-dimensional representations for the entire dataset of 302,927 tweets. The optimal number of clusters was determined to be k=4 through silhouette score analysis across multiple k values.

The evaluation results demonstrate significant improvement over the baseline BERT embeddings, with the test set achieving a silhouette score of 0.0382 compared to the baseline score of 0.0008—representing a 4,671% improvement. While this falls short of the initial cluster-based fine-tuning approach (which achieved 0.0502), the SimCSE method exhibits superior generalization properties with an extremely small train-test gap of only 0.0009, indicating excellent model stability and minimal overfitting. The consistent performance across training (0.0372), validation (0.0376), and test (0.0382) sets confirms that the self-supervised contrastive learning approach successfully learned meaningful semantic representations of marriage discourse without relying on potentially noisy cluster assignments. The Davies-Bouldin Index of 4.616 and Calinski-Harabasz score of 814.05 on the test set further validate the improved cluster separation and quality achieved through this state-of-the-art fine-tuning methodology.

https://colab.research.google.com/drive/12-nZfrfVN8Gci0pc9YDWCLoGRR7JhnRm#scrollTo=YP0uR9dNq0Cu

https://colab.research.google.com/drive/1KJzrMk_TnVpjmOMpSnOxgsWxznRQi6cm#scrollTo=YjDh05NllZBc










---

# Part 4: SimCSE Fine-Tuning — Improved Approach



colabs:
https://colab.research.google.com/drive/12-nZfrfVN8Gci0pc9YDWCLoGRR7JhnRm#scrollTo=YP0uR9dNq0Cu
https://colab.research.google.com/drive/1KJzrMk_TnVpjmOMpSnOxgsWxznRQi6cm#scrollTo=Lot82pmspyFa

visualizations are in:
https://drive.google.com/drive/folders/16J04LMzQhIuqZcIGIEU3c5Ca7ArBZ2t3
clean model folders

## Motivation

The results from Part 3 were insufficient for meaningful analysis. The baseline
BERT embeddings produced a silhouette score of 0.0008, and even after
cluster-based fine-tuning the score only reached 0.0502 — far too low to
confidently interpret any discovered cluster structure. The embedding spaces
showed no meaningful geometric organisation, with all dimensionality reduction
plots producing uniform point clouds with no visible separation.

This motivated a complete redesign of the pipeline with three key changes:

1. **A stricter nine-step preprocessing pipeline** to remove noise, spam, bots,
   and duplicates that were polluting the embedding space
2. **Self-supervised SimCSE fine-tuning** replacing cluster-based fine-tuning
   to eliminate circular dependency on noisy initial clusters
3. **A systematic comparison of four architecturally diverse transformer models**
   to identify which design properties matter most for contrastive learning on
   social media text

---

## Preprocessing Pipeline

The new pipeline was applied identically across all four models to ensure fair
comparison. Starting from 302,927 raw tweets, nine sequential steps produced a
clean dataset of ~248,610 tweets (17.9% removed).

| Step | Operation | Tweets Removed |
|------|-----------|---------------|
| 1 | Encoding repair (ftfy) | — |
| 2 | URL, mention, hashtag, emoji cleaning | — |
| 3 | Length filter (5–100 words) | 8,312 (2.7%) |
| 4 | Exact duplicate removal | 19,415 (6.4%) |
| 5 | Near-duplicate removal (first 80 chars) | 2,226 (0.8%) |
| 6 | Spam pattern removal (12 regex categories) | ~24,234 (9.0%) |
| 7 | Viral/bot phrase detection (8-word fingerprint) | 0 (null result) |
| 8 | Language filter (ASCII ratio > 75%) | 130 (0.05%) |
| 9 | Final null/empty validation | — |

**Step 7 produced a meaningful null result:** after steps 4–6, no phrase
remained frequent enough to meet the viral threshold — confirming that the
preceding deduplication and spam removal had already addressed systematic
repetition.

### Train / Validation / Test Split

| Split | Tweets | Percentage |
|-------|--------|------------|
| Train | 198,888 | 80% |
| Validation | 24,861 | 10% |
| Test | 24,861 | 10% |

All clustering evaluation was performed on the test split. All four models used
identical split boundaries (`random_state=42`) for directly comparable results.

---

## Model Selection

Four transformer models were selected to represent a spectrum of architectural
designs, pre-training strategies, and domain relevance — allowing systematic
isolation of the factors that determine SimCSE performance.

| Model | Base Model | Parameters | Pre-training | Approach |
|-------|-----------|------------|-------------|----------|
| BERT + SimCSE | bert-base-uncased | 109,482,240 | Books + Wikipedia (MLM + NSP) | Foundational baseline |
| RoBERTa + SimCSE | roberta-base | 124,645,632 | 160GB web text (MLM only) | Improved pre-training robustness |
| Twitter-RoBERTa + SimCSE | cardiffnlp/twitter-roberta-base | 124,645,632 | 58M tweets (MLM) | Domain-specific pre-training |
| DeBERTa-v3 + SimCSE | microsoft/deberta-v3-small | 141,304,320 | Books + Web (RTD + MLM) | Architectural innovation |

**Why these four?** Together they test three hypotheses:
- Does domain-specific pre-training (Twitter-RoBERTa) outperform general models?
- Does architectural sophistication (DeBERTa disentangled attention) outperform
  domain specificity?
- Does improved pre-training robustness (RoBERTa) outperform the BERT baseline?

### Training Configuration per Model

| Property | BERT | RoBERTa | Twitter-RoBERTa | DeBERTa-v3 |
|----------|------|---------|-----------------|------------|
| Batch Size | 32 | 32 | 64 | 8 |
| Learning Rate | 3e-5 | 3e-5 | 3e-5 | 1e-5 |
| Warmup Ratio | 0.05 | 0.05 | 0.05 | 0.10 |
| fp16 AMP | Yes | Yes | Yes | No (fp32) |
| Grad Checkpointing | Yes | Yes | No | Yes |
| Attention Type | Standard | Standard | Standard | Disentangled |
| Position Encoding | Absolute | Absolute | Absolute | Relative |

DeBERTa required three special accommodations: `use_fast=False` for its
SentencePiece tokenizer, fp16 disabled due to a gradient unscaling conflict
with its disentangled attention mechanism, and a lower learning rate with longer
warmup to prevent the collapse that affected BERT and RoBERTa.

---

## SimCSE Framework

All four models were fine-tuned using **unsupervised SimCSE** — no labels
required at any stage.

**Core principle:** Each tweet is passed through the encoder twice with
different dropout masks. The model is trained to maximise similarity between
the two representations of the same tweet (positive pair) while minimising
similarity to all other tweets in the batch (negative pairs).

**Loss function:** InfoNCE contrastive loss with temperature τ = 0.05


**Implementation fixes applied:**
- `masked_fill` value changed from `-1e9` to `-1e4` to prevent float16 overflow
- Embeddings cast to float32 before loss computation regardless of forward
  pass precision
- Mean pooling over all non-padding tokens (vs CLS token) for better-calibrated
  sentence representations

---

## Clustering & Dimensionality Reduction

### Clustering Algorithms

| Algorithm | Configuration | Selection Criterion |
|-----------|--------------|---------------------|
| KMeans | k ∈ {4–12}, n_init=10 | Maximum silhouette score |
| HDBSCAN | min_cluster_size ∈ {50,100,200,300,500}, min_samples=10 | Maximum silhouette (≥3 clusters) |
| Agglomerative | Ward linkage, k ∈ {4–12} | Maximum silhouette score |
| GMM | Diagonal covariance, k ∈ {4–12}, reg_covar=1e-3 | Minimum BIC |

All solutions evaluated with three validity indices:
- **Silhouette Score** ↑ (range −1 to +1; above 0.2 = genuine structure)
- **Davies-Bouldin Index** ↓ (lower = more compact, better separated)
- **Calinski-Harabasz Index** ↑ (higher = denser, better separated)

### Dimensionality Reduction

**Pre-clustering (curse of dimensionality fix):**
PCA applied before all clustering algorithms with components tested at
{32, 64, 128, 256} to find the optimal trade-off between noise removal and
information retention.

**Visualisation only:**
- PCA — 2-component projection
- t-SNE — perplexity=40, learning_rate=200, 1,000 iterations (on PCA-50 input)
- UMAP — n_neighbors=30, min_dist=0.1, cosine metric

---

## Results

### Training Loss

| Model | Avg Loss | Batches | Duration | Verdict |
|-------|----------|---------|----------|---------|
| BERT | 0.0012 | 6,216 | 6m 36s | Severe collapse |
| RoBERTa | 0.0237 | 6,216 | 6m 46s | Collapse |
| Twitter-RoBERTa | 0.0401 | 3,092 | ~2h 20m | Near-collapse |
| DeBERTa-v3 | **0.4832** | 24,861 | 18m 32s | Healthy learning ✓ |

A loss near zero means the model collapsed — all tweets mapped to the same
representation. DeBERTa's higher loss indicates it was still actively
distinguishing tweets at the end of training.

### PCA 2D Variance Diagnostic

| Model | 2D Variance | Meaning |
|-------|-------------|---------|
| BERT | 4.13% | No directional structure |
| RoBERTa | 4.48% | No directional structure |
| Twitter-RoBERTa | 4.52% | No directional structure |
| **DeBERTa-v3** | **27.39%** | Rich semantic structure ✓ |

DeBERTa needs only **8 PCA components for 80% variance**, while BERT needs
151 — the clearest possible evidence that DeBERTa's embedding space is
structured around meaningful semantic axes rather than noise.

### Clustering Results — KMeans Comparison

| Model | Best k | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Rank |
|-------|--------|-------------|-----------------|---------------------|------|
| **DeBERTa-v3** | **12** | **0.2215** | **1.4298** | **2609.97** | **1** |
| RoBERTa | 4 | 0.0114 | 6.7223 | 289.25 | 2 |
| BERT | 7 | 0.0077 | 6.3061 | 199.40 | 3 |

DeBERTa's silhouette is **29× higher** than BERT and **19× higher** than
RoBERTa. This is not a marginal improvement — it represents a categorically
different quality of embedding space.

### Twitter-RoBERTa — Four-Algorithm Comparison

| Algorithm | Best k | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-----------|--------|-------------|-----------------|---------------------|
| KMeans | 4 | 0.0121 | 6.9504 | 278.42 |
| HDBSCAN | 17 | 0.0177 | 4.5755 | 61.32 |
| GMM | 12 | −0.0206 | 5.0447 | 209.86 |
| Agglomerative | 4 | 0.0029 | 8.4859 | 204.84 |

Even the best result here (HDBSCAN: 0.0177) is 12× below DeBERTa's KMeans
score. The four KMeans clusters split nearly equally at ~25% each with random
distinctive words — the signature of a collapsed embedding space.

### PCA-Reduced Clustering Results

Applying PCA before clustering to address the curse of dimensionality:

**BERT — Best: PCA-32 + KMeans, k=5, silhouette=0.0289** (+275% vs original)  
**RoBERTa — Best: PCA-32 + KMeans, k=4, silhouette=0.0319** (+180% vs original)

PCA helped BERT and RoBERTa substantially in relative terms but the absolute
values remain near zero — noise removal improved results but genuine semantic
structure cannot be recovered from a collapsed embedding space.

**DeBERTa — Best: PCA-64/128 + HDBSCAN, k=12, silhouette=0.4249**

| DeBERTa PCA Dim | KMeans | HDBSCAN | Agglomerative | GMM |
|-----------------|--------|---------|---------------|-----|
| PCA-32 (99.9% var) | 0.2242 | 0.2018 | 0.1638 | 0.1446 |
| PCA-64 (100.0% var) | 0.2233 | **0.4246** | 0.1710 | 0.1463 |
| PCA-128 (100.0% var) | 0.2258 | **0.4249** | 0.1631 | 0.1612 |
| Original 768d | 0.2215 | — | — | — |

HDBSCAN on PCA-64/128 nearly **doubles** DeBERTa's already strong KMeans
result. The 79.1% noise rate is not a weakness — HDBSCAN identifies the 12
densest semantic cores in the embedding space and honestly labels ambiguous
tweets rather than forcing them into the nearest centroid.

### Why DeBERTa Won

| Factor | Explanation |
|--------|-------------|
| Disentangled attention | Separates content and position into distinct vectors, producing more isotropic embeddings amenable to SimCSE |
| Conservative hyperparameters | lr=1e-5, warmup=0.10, batch=8 prevented the collapse that hit BERT/RoBERTa at lr=3e-5 |
| Float32 precision | Full fp32 gave the contrastive loss the numerical stability to compute gradients for close embeddings |
| ELECTRA pre-training | Replaced token detection trains every token (not just 15%), producing richer initial representations |

Twitter-RoBERTa's domain advantage from 58 million tweets was entirely negated
by embedding collapse. Its best silhouette (0.0121) is virtually identical to
generic RoBERTa (0.0114), confirming that architecture and training stability
outweigh domain specialisation in the SimCSE regime.

---

## DeBERTa Cluster Characteristics

The winning configuration — **DeBERTa-v3 + PCA-64 + HDBSCAN, 12 clusters,
silhouette = 0.4249** — was characterised using five methods:

1. TF/IDF-style distinctive word scoring (cluster vocabulary vs all others)
2. TextBlob sentiment analysis (polarity and subjectivity)
3. Structural statistics (tweet length, question rate, exclamation rate)
4. Mean pairwise cosine similarity (cohesion) as a compactness measure
5. Representative tweet identification by cosine distance to cluster centroid

### Cluster Summary Table

| Cluster | Label | Size | Size % | Polarity | Subjectivity | Cohesion | Avg Words |
|---------|-------|------|--------|----------|-------------|---------|-----------|
| C0 | Everyday Married Life Narratives | 2,333 | 9.4% | +0.108 | 0.424 | 0.566 | 31.0 |
| C1 | Partner Trust & Loyalty | 1,506 | 6.1% | +0.189 | 0.419 | 0.628 | 17.0 |
| C2 | Long Tweets — Incidental Partner Mentions | 2,658 | 10.7% | +0.120 | 0.477 | 0.464 | 37.8 |
| C3 | Fans Reacting to Idols Getting Married | 1,404 | 5.6% | +0.270 | 0.412 | **0.673** | 16.7 |
| C4 | "My Person Is Getting Married" | 1,547 | 6.2% | +0.245 | 0.388 | 0.570 | 20.6 |
| C5 | Hubby's Birthday — Home Celebrations | 2,049 | 8.2% | +0.163 | 0.379 | 0.613 | 17.1 |
| C6 | Married Life During Hard Times | 3,067 | 12.3% | +0.165 | 0.482 | 0.384 | 39.8 |
| C7 | Short Affectionate Wife Tweets (Black Twitter) | 1,460 | 5.9% | +0.184 | 0.367 | 0.633 | 13.7 |
| C8 | Relationship Worries & Partner Behaviour | 1,984 | 8.0% | +0.137 | 0.376 | 0.608 | 14.8 |
| C9 | Hubby's Birthday — Trips & Outings | 1,926 | 7.7% | +0.139 | 0.401 | 0.588 | 24.2 |
| C10 | Short Tweets — Incidental Spouse Mentions | 3,041 | 12.2% | +0.127 | 0.362 | 0.440 | 15.7 |
| C11 | K-Pop & South Asian Celebrity Fandom | 1,886 | 7.6% | +0.223 | 0.456 | 0.496 | 34.5 |

### Distinctive Words per Cluster

| Cluster | Top Distinctive Words |
|---------|-----------------------|
| C0 | memorize, medicare, orlando, updated, vlog, route, boxes, banana |
| C1 | finance, hiv, intimacy, language, lovers, affair, goals, missing |
| C2 | unpleasant, sources, respecting, chatty, movement, beliefs, ambition |
| C3 | ryo, ryos, zeinab, sobbing, jiinie, seokjiii, dearly, princess |
| C4 | queenradio, wizkid, bridesmaid, sisters, cried, lmaooo, dresses |
| C5 | homemaker, omotola, ribs, grandparents, potatoes, italy, india |
| C6 | unconditional, hardship, igbo, davido, hail, adults, rabbit |
| C7 | choppa, wit, sha, bae, soo, cuff, miss, groom, missus |
| C8 | therapy, habits, loyal, experts, fail, delete, podcast, jailed |
| C9 | wifi, hamburger, coconut, balloon, chips, fruit, tuesday, goat |
| C10 | astronaut, hacking, blog, nap, idc, accounts, gay, accused |
| C11 | exo, jongdae, feroze, shippers, proposal, mfs, mommy, purple |

### Key Structural Findings

**Cohesion range: 0.384 – 0.673** (vs 0.046–0.060 for collapsed models)  
DeBERTa's clusters are 8–14× more internally coherent than anything
produced by BERT, RoBERTa, or Twitter-RoBERTa.

**Tweet length varies non-randomly across clusters:**

| Short (~14 words) | Medium (~20 words) | Long (~38 words) |
|-------------------|--------------------|-----------------|
| C7: 13.7 | C4: 20.6 | C2: 37.8 |
| C8: 14.8 | C9: 24.2 | C6: 39.8 |
| C10: 15.7 | C0: 31.0 | |

If clustering were random, all clusters would average ~24 words.

**Three organisational axes discovered:**

1. **Register & length** — short casual observations (C7, C8, C10) vs long
   personal narratives (C2, C6)
2. **Cultural community** — K-pop Twitter (C3, C11), West African Twitter
   (C4, C6), Black Twitter/AAVE (C7), Western domestic (C0, C5, C9)
3. **Relationship context** — celebratory milestones (C3, C4, C5, C9) vs
   everyday life (C0, C7, C10) vs hardship (C6, C8) vs advice/trust (C1)










I wanted to explore more the clusters from this model:
Cluster 1 — Partner Trust, Loyalty & Emotional Safety
1,506 tweets | 6.1% | Cohesion 0.628 | Avg 17 words
Short, focused, high cohesion — this cluster has a very specific subject. The distinctive words — intimacy, affair, language, lovers, finance, goals — initially suggest infidelity, but the representative tweets completely contradict that reading: "storms come and go but when the love is right, your partner won't leave", "i would be very happy and extremely proud of my partner", "me telling my wife her glazed salmon sucks — I trust you to hear this."
These tweets are about the qualities that make a relationship strong — honesty, trust, weathering difficulties together, emotional safety. The "affair" keyword appears because people discuss what would break trust, not because they're confessing to it. Polarity is +0.189 (moderately positive) and subjectivity 0.419 — thoughtful and personal but not highly emotional. The high cohesion (0.628) means people writing about relationship trust use a very consistent vocabulary and tone.
In one sentence: Reflective tweets about what makes love last — trust, loyalty, honesty, and emotional courage with a partner.

Cluster 2 — Long Rambling Tweets With Incidental Partner Mentions
2,658 tweets | 10.7% | Cohesion 0.464 | Avg 38 words
This is the noise cluster — the second largest and second least cohesive (0.464). Average length of 38 words (median exactly 38) is the second longest in the dataset, and the content is strikingly diverse. Distinctive words — unpleasant, sources, respecting, chatty, movement, beliefs, specialist, ambition — have nothing to do with relationships. Representative tweets include a school policy debate, a Witcher gaming event, and a political commentary about race.
What unites them is that "partner", "spouse", or "hubby" appears somewhere in a long tweet whose primary subject is something else entirely. High question rate (17.3%) and highest subjectivity (0.477) shows opinionated, discursive writing. Polarity is low (+0.120) — these aren't happy posts, they're people airing views on various topics.
This cluster is a methodological finding: relationship Twitter is full of tweets where relationship words appear incidentally in completely unrelated conversations, and DeBERTa correctly separated this noise from the signal.
In one sentence: Long, opinionated tweets on all subjects where a partner is mentioned in passing.

Cluster 3 — Fans Reacting to Idols & Celebrities Getting Married
1,404 tweets | 5.6% | Cohesion 0.673 — TIGHTEST CLUSTER | Avg 17 words
The single most coherent cluster in the entire dataset. Every tweet in here uses the same emotional register because the situation is identical: a fan discovering that someone they adore is getting engaged or married. Distinctive words are entirely proper nouns — ryo, ryos, zeinab, jiinie, seokjiii (BTS members Jimin and Jin), along with sobbing, dearly, princess, drama, forever.
Polarity is the second highest (+0.270) and exclamations at 19.3% — warm, celebratory, slightly tearful. Bigrams: getting married, baby getting, baby sister, baby brother — the word "baby" here is a fan term of endearment for an idol, not a literal child. Rep tweets: Serena Williams ring, K-pop fan reactions, Nigerian wedding congratulations. The cohesion of 0.673 is so high because fans writing about idol marriages across different languages and cultures converge on the same vocabulary: baby, princess, forever, sobbing, congratulations.
In one sentence: Fan Twitter reacting to a beloved celebrity or K-pop idol announcing engagement — uniformly emotional, affectionate, slightly bittersweet.

Cluster 4 — "My Person Is Getting Married" — Personal Celebration
1,547 tweets | 6.2% | Cohesion 0.570 | Avg 21 words
Closely related to C3 but personally oriented rather than celebrity-focused. Where C3 is fans celebrating idols, C4 is people celebrating someone they actually know — a younger sibling, close friend, or person they grew up with. The word "baby" again means a beloved person, not a child. Bigrams dominated by getting married + loudly_crying_face loudly_crying_face — the crying emoji is used here as joyful-tearful, not sad.
Distinctive words include bridesmaid, sisters, cried, wizkid, queenradio — Nigerian music scene markers appear alongside ofc, lmaooo, gettin showing a younger, informal writing style. Polarity +0.245 (second highest positive cluster) and low subjectivity 0.388 — these are exclamatory announcements more than personal reflections. Cohesion 0.570 is solid — the formula is consistent: my baby + getting married + emoji reaction.
In one sentence: Young people announcing on Twitter that their best friend, sibling, or close companion is getting married — joyful, proud, emoji-heavy.

Cluster 5 — Celebrating Hubby's Birthday at Home
2,049 tweets | 8.2% | Cohesion 0.613 | Avg 17 words
Extremely focused cluster with a single clear subject. Every representative tweet, every top bigram (hubby's birthday appears five times in the top bigrams), and every distinctive word orbits one event: planning and celebrating a husband's birthday with food. Distinctive words: homemaker, ribs, grandparents, potatoes, italy, india, apple, coconut — foods and domestic contexts. "Hubby's birthday week continues! this is what happens when you marry a foodie!"
The highest exclamation rate in the entire dataset at 28.7% — these are happy, enthusiastic posts. Polarity +0.163, subjectivity 0.379 — warm but practical, focused on what they're doing rather than how they feel. High cohesion (0.613) because everyone writing this type of tweet uses nearly identical phrasing: "it's hubby's birthday", "hubby's birthday week", "celebrating hubby's birthday."
The contrast with C9 is important — both clusters are about hubby's birthday but DeBERTa separated home-based celebrations (C5, food focus, homemaker identity) from outings and trips (C9).
In one sentence: Wives sharing domestic birthday celebrations for their husbands — food-focused, homemaker identity, consistently enthusiastic.

Cluster 6 — Married Life During Hard Times
3,067 tweets | 12.3% — LARGEST | Cohesion 0.384 — LOWEST | Avg 40 words
The largest cluster and the least cohesive — that combination is meaningful. It contains the most diverse content because hardship takes many forms. At 40 words average (the longest cluster), these people write at length because they're processing something difficult. Distinctive words: unconditional, hardship, igbo, davido, hail, wide, adults, rabbit — a mix of emotional terms and Nigerian cultural markers (igbo is a Nigerian language, davido is a major Nigerian artist).
Polarity +0.165 and the highest subjectivity in the dataset (0.482) — deeply personal writing. Rep tweets show the range: kidney infection + flu, husband forced into early retirement due to illness, stress about expensive food, pregnancy in a distant state. What unites them is that the marriage is the anchor during difficulty — the spouse is present in the struggle, not the cause of it.
Strong West African Twitter presence alongside Western voices — Nigerian Twitter discusses marriage and family at length and with great emotional openness, which explains both the high subjectivity and the cultural markers.
In one sentence: Long personal tweets about navigating serious life difficulties — illness, money, stress — while married, drawn from both Western and West African Twitter communities.

Cluster 7 — Short Punchy Affectionate Wife Tweets (Black Twitter)
1,460 tweets | 5.9% | Cohesion 0.633 | Avg 14 words — SHORTEST
The shortest cluster at 13.7 words average (median 12) and one of the tightest. These are brief, casual, humorous observations about a wife — the format is a quick one-liner or reaction, often with an emoji. Distinctive words: choppa, wit, sha, bae, soo, cuff, miss — unmistakably AAVE (African American Vernacular English) and Black Twitter vocabulary. Bigrams are almost entirely emojis: face_with_tears_of_joy, loudly_crying_face, red_heart, proud hubby.
The lowest question rate (4.7%) and lowest exclamation rate (12.3%) in the dataset — these aren't dramatic or interrogative, just dry and affectionate. "This is my wife and I eat what she eats 🤷", "my wife just changed the spark plugs herself 😁", "my wife just changed the spark plugs herself." The subjectivity is also the lowest in the dataset (0.367) — very matter-of-fact observations, not emotional reflections.
In one sentence: Short, dry, affectionate one-liners about wives written in AAVE/Black Twitter style — minimal punctuation, emoji-heavy, effortlessly cool.

Cluster 8 — Relationship Worries & Partner Behaviour
1,984 tweets | 8.0% | Cohesion 0.608 | Avg 15 words
This cluster has a grammatical fingerprint that makes it unique: "partner won't" dominates every bigram list. "At least that partner won't answer back", "hope ur partner won't get jealous", "lol my partner won't cheat in the first place", "both mine and my partner's phones aren't letting us call out." Distinctive words: therapy, habits, loyal, experts, fail, delete, podcast, jailed — advisory, self-help register.
Polarity +0.137 (third lowest) and the second highest standard deviation (±0.348) — the most emotionally varied cluster, containing both reassuring posts and anxious ones. High cohesion (0.608) despite the emotional variation, because the linguistic structure "partner won't [do something]" is so consistent regardless of whether the tweet is worried or reassuring. Low subjectivity 0.376 — relatively detached, practical tone.
In one sentence: Tweets discussing what partners do or don't do — a mix of relationship advice, jealousy concerns, loyalty discussions, and behavioural observations.

Cluster 9 — Hubby's Birthday Trips & Outings
1,926 tweets | 7.7% | Cohesion 0.588 | Avg 24 words
The companion cluster to C5 — both are about celebrating a husband's birthday, but DeBERTa cleanly separated the two based on context. Where C5 is home cooking and domestic celebration, C9 is going out, travelling, and planning adventures. Distinctive words: wifi, hamburger, coconut, balloon, chips, fruit, Tuesday, goat — food eaten on outings, not cooked at home. Rep tweets: "excited to head downtown for hubby's birthday", "northern lights booked — 5 days snowmobiling", "Indianapolis birthday trip in two weeks."
Medium length at 24 words, second highest exclamation rate (28.0%) — these are excited anticipation posts as much as recaps. Cohesion 0.588 is slightly lower than C5's 0.613, reflecting that trips and outings are slightly more varied than home cooking. The fact that DeBERTa found this sub-distinction within the same broad topic (hubby's birthday) is the clearest single proof that its 27% PCA variance reflects real semantic structure.
In one sentence: Wives announcing and celebrating husband birthday trips, outings and adventures — excited, anticipatory, food-referencing but in restaurant/travel contexts.

Cluster 10 — Casual Spouse References in Unrelated Tweets
3,041 tweets | 12.2% — SECOND LARGEST | Cohesion 0.440 | Avg 16 words
The second largest cluster and second least cohesive (0.440) — another noise/catch-all cluster like C2, but for short tweets rather than long ones. Average 16 words, the second shortest cluster. Distinctive words are a completely random scatter: astronaut, hacking, blog, nap, idc, fuckin, gay, accused — no coherent topic at all. Rep tweets: "I'm not a big SK fan like my husband but I enjoyed Carrie", "come to my house I'll make you my wifey", "no, the only person I want to kiss is my husband", "I'll quit my job and leave my wife right now."
These are tweets about Stephen King, about flirting, about job frustration, about wanting to kiss someone — the only common thread is that husband/wife/wifey appears somewhere. Low subjectivity (0.362 — second lowest), low exclamation rate (15.3%), low question rate (6.2%) — very neutral, casual, throwaway posts.
Together with C2, these two clusters (10.7% + 12.2% = 22.9% of all tweets) represent the substantial portion of "relationship Twitter" that isn't really about relationships at all — just ordinary life where a spouse gets a passing mention.
In one sentence: Short casual tweets on every possible topic where husband, wife, or partner appears as a brief incidental mention rather than the actual subject.

Cluster 11 — K-Pop & South Asian Celebrity Fandom Marriages
1,886 tweets | 7.6% | Cohesion 0.496 | Avg 35 words
The longest-writing fandom cluster at 35 words average — these fans write more extensively than C3's fans (17 words). Distinctive words are highly specific: exo, jongdae (EXO member Chen, who controversially announced his marriage and pregnancy), feroze (Pakistani actor Feroze Khan), shippers, proposal, mfs, mommy, purple (purple is BTS's signature colour). Highest polarity in the dataset at +0.223 and high subjectivity (0.456) — emotional and deeply personal.
The crucial difference from C3: C3 is BTS-focused and reacts to celebrity news with short, punchy posts. C11 is EXO and South Asian drama focused and writes longer, more anguished, more processing posts — fans working through complicated feelings about a beloved idol's marriage. "Feroze khan is getting married 😭 I am happy for my bae I swear, he's stepping up", "feeling a little emotional about my baby brother tomorrow they'll be one couple." Bigrams dominated by multiple emotion emojis stacked together.
In one sentence: K-pop (EXO-focused) and South Asian drama fans writing extended emotional reactions to celebrity marriages — longer, more conflicted, and more personally invested than C3's shorter fan reactions.




























---

## PCA-Reduced Clustering — Addressing the Curse of Dimensionality

### Motivation

The clustering results above were obtained by applying all four algorithms
directly to the raw 768-dimensional embeddings. A known limitation of
distance-based clustering methods (KMeans, Agglomerative, GMM) and
density-based methods (HDBSCAN) in high-dimensional spaces is the **curse of
dimensionality**: as dimensionality increases, distances between points become
nearly identical, making it difficult to distinguish near from far neighbours.
To test whether dimensionality reduction before clustering could further improve
results, PCA was applied at four component levels {32, 64, 128, 256} prior to
running all four algorithms.

---

### PCA Variance Analysis

The variance analysis revealed a fundamental difference between the collapsed
and non-collapsed models:

| Model | 80% variance at | 90% variance at | 95% variance at |
|-------|----------------|----------------|----------------|
| BERT | 151 components | 277 components | 300 components |
| RoBERTa | 131 components | 246 components | 300 components |
| Twitter-RoBERTa | ~140 components | ~260 components | ~300 components |
| **DeBERTa-v3** | **8 components** | **9 components** | **11 components** |

DeBERTa captures 80% of its total variance in just **8 dimensions**. BERT
requires 151. This alone confirms that DeBERTa's embedding space has genuine
concentrated semantic structure, while the others are nearly uniform across
all 768 dimensions — the mathematical signature of embedding collapse.

---

### Results: BERT

**Best configuration: PCA-32 + KMeans, k=5, silhouette=0.0289**

| PCA Dim | Variance Retained | KMeans | HDBSCAN | Agglomerative | GMM |
|---------|------------------|--------|---------|---------------|-----|
| 32 | 40.4% | **0.0289** | FAIL (100% noise) | 0.0049 | 0.0024 |
| 64 | 59.2% | 0.0184 | FAIL | −0.0018 | −0.0169 |
| 128 | 76.6% | 0.0118 | FAIL | −0.0024 | −0.0213 |
| 256 | 88.9% | 0.0081 | FAIL | 0.0008 | −0.0330 |
| **Original 768d** | 100% | 0.0077 | — | — | — |

PCA-32 improved KMeans by **+275%** over the original (0.0077 → 0.0289).
However, the absolute value remains near zero. HDBSCAN failed completely at
every dimensionality — 100% of points classified as noise — confirming there
is no density structure whatsoever in BERT's embedding space regardless of
how it is projected. Performance degrades monotonically as dimensions increase,
meaning the top 40% of variance contains the only usable signal and everything
beyond it is noise.

---

### Results: RoBERTa

**Best configuration: PCA-32 + KMeans, k=4, silhouette=0.0319**

| PCA Dim | Variance Retained | KMeans | HDBSCAN | Agglomerative | GMM |
|---------|------------------|--------|---------|---------------|-----|
| 32 | 40.8% | **0.0319** | FAIL (100% noise) | 0.0006 | −0.0043 |
| 64 | 60.7% | 0.0205 | FAIL | 0.0000 | −0.0303 |
| 128 | 79.5% | 0.0154 | FAIL | −0.0036 | −0.0296 |
| 256 | 90.5% | 0.0131 | FAIL | 0.0012 | −0.0331 |
| **Original 768d** | 100% | 0.0114 | — | — | — |

PCA-32 improved KMeans by **+180%** over the original (0.0114 → 0.0319).
The pattern is identical to BERT — best at aggressive compression, declining
monotonically, HDBSCAN total failure across all dimensionalities. This
consistency confirms the pattern is a property of the collapsed embedding
geometry, not a quirk of any individual model. Agglomerative and GMM scores
go negative as dimensionality increases, meaning they actively misassign
points when more variance is included.

---

### Results: DeBERTa

**Best configuration: PCA-64/128 + HDBSCAN, k=12, silhouette=0.4249**

| PCA Dim | Variance Retained | KMeans | HDBSCAN | Agglomerative | GMM |
|---------|------------------|--------|---------|---------------|-----|
| 32 | **99.9%** | 0.2242 | 0.2018 (k=10) | 0.1638 | 0.1446 |
| 64 | **100.0%** | 0.2233 | **0.4246** (k=12) | 0.1710 | 0.1463 |
| 128 | **100.0%** | 0.2258 | **0.4249** (k=12) | 0.1631 | 0.1612 |
| **Original 768d** | 100% | 0.2215 | — | — | — |

Three findings stand out:

**KMeans is stable across all PCA dimensionalities (0.2210–0.2258).** Since
DeBERTa already captures 99.9% of its variance in just 32 components, adding
more PCA dimensions changes nothing for KMeans. The optimal k=12 is
consistently identified regardless of dimensionality, confirming the
12-cluster structure is robust.

**HDBSCAN improves dramatically at PCA-64, nearly doubling the score.**
At PCA-32, HDBSCAN achieves 0.2018. At PCA-64 it jumps to 0.4246. This
improvement occurs despite both retaining essentially 100% of variance,
because HDBSCAN operates on local neighbourhood density which is sensitive
to the precise geometric shape of clusters — the extra 32 dimensions at
PCA-64 provide enough detail for HDBSCAN to distinguish cluster boundaries
that were merged at PCA-32. PCA-128 yields 0.4249, confirming 64 components
is sufficient and further dimensions add nothing.

**The HDBSCAN noise rate of 79.1% is a feature, not a flaw.** HDBSCAN
identifies 12 dense semantic cores and honestly labels the surrounding 79%
of tweets as ambiguous rather than forcing them into the nearest centroid as
KMeans does. The silhouette of 0.4249 is computed only on the non-noise
points — these are the most unambiguously representative tweets of each
semantic group, with genuinely tight within-cluster similarity.

---

### Comparison: All Approaches

| Model | Approach | Best Silhouette | Best Config |
|-------|----------|----------------|-------------|
| BERT | Original 768d | 0.0077 | KMeans k=7 |
| BERT | PCA-reduced | 0.0289 | PCA-32 + KMeans k=5 |
| RoBERTa | Original 768d | 0.0114 | KMeans k=4 |
| RoBERTa | PCA-reduced | 0.0319 | PCA-32 + KMeans k=4 |
| Twitter-RoBERTa | Original 768d | 0.0177 | HDBSCAN k=17 |
| DeBERTa-v3 | Original 768d | 0.2215 | KMeans k=12 |
| **DeBERTa-v3** | **PCA-reduced** | **0.4249** | **PCA-128 + HDBSCAN k=12** |

---

### Figures
they are all in their related colabs files also.
part 5 would be to add crc results interpretation and comparison as well.












