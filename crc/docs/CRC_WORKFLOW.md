# CRC Analysis - Complete Code Package

## Directory Structure for GitHub

```
crc_analysis/
├── README.md                          # This file
├── data/
│   └── DATA_INFO.md                   # Data description (don't upload actual data)
├── scripts/
│   ├── preprocessing.py               # Data preprocessing
│   ├── get_embeddings_model1.py       # Embeddings for granite-30m
│   ├── get_embeddings_model2.py       # Embeddings for granite-278m
│   ├── get_embeddings_model3.py       # Embeddings for qwen3-8b
│   ├── cluster_analysis_8.py          # 8-cluster analysis with wordclouds
│   ├── cluster_analysis_model2.py     # Cluster analysis for model 2
│   ├── cluster_analysis_model3.py     # Cluster analysis for model 3
│   ├── find_optimal_k.py              # Find optimal k (3-15)
│   └── clustering_comparison.py       # Compare clustering methods
├── jobs/
│   ├── job_embeddings_model1.sh       # SGE job for model 1
│   ├── job_embeddings_model2.sh       # SGE job for model 2
│   ├── job_embeddings_model3.sh       # SGE job for model 3
│   ├── job_cluster_analysis.sh        # SGE job for clustering
│   └── job_comparison.sh              # SGE job for comparison
├── results/
│   └── RESULTS_INFO.md                # Results description
└── docs/
    └── CRC_WORKFLOW.md                # Step-by-step workflow
```

---

# Step-by-Step CRC Workflow Documentation

## Overview

This document describes how the Twitter clustering analysis was performed using Notre Dame's Center for Research Computing (CRC) infrastructure.

## Step 1: Data Transfer to CRC

### 1.1 Connect to CRC
```bash
ssh rrazavi@crcfe01.crc.nd.edu
```

### 1.2 Create Project Directory
```bash
mkdir -p /groups/ecumming/rrazavi/twitter_project
cd /groups/ecumming/rrazavi/twitter_project
mkdir -p checkpoints results logs
```

### 1.3 Transfer Data from Local Machine
On your local Mac terminal:
```bash
scp ~/path/to/ALL_TWEETS_MASTER.csv rrazavi@crcfe01.crc.nd.edu:/groups/ecumming/rrazavi/twitter_project/
```

## Step 2: Environment Setup

### 2.1 Create Conda Environment
```bash
module load conda/25.9.1
conda create -p /groups/ecumming/rrazavi/twitter_project/myenv2 python=3.11 -y
conda activate /groups/ecumming/rrazavi/twitter_project/myenv2
```

### 2.2 Install Required Packages
```bash
pip install pandas numpy matplotlib seaborn scikit-learn umap-learn hdbscan wordcloud requests tqdm
```

## Step 3: Data Preprocessing

### 3.1 Create Preprocessing Script
See `scripts/preprocessing.py`

### 3.2 Run Preprocessing
```bash
python preprocessing.py
```

Output: `preprocessed_tweets.csv` (302,928 cleaned tweets)

## Step 4: Embedding Generation

### 4.1 CRC Open WebUI API Configuration
- **API Endpoint:** https://openwebui.crc.nd.edu/api/embed
- **API Key:** Stored securely (not in code)

### 4.2 Embedding Models Used

| Model | Parameters | Dimensions | File Size |
|-------|------------|------------|-----------|
| granite-embedding:30m | 30M | 384 | 465 MB |
| granite-embedding:278m | 278M | 768 | 930 MB |
| qwen3-embedding:8b | 8B | 4096 | 4.9 GB |

### 4.3 Submit Embedding Jobs
```bash
qsub jobs/job_embeddings_model1.sh
qsub jobs/job_embeddings_model2.sh
qsub jobs/job_embeddings_model3.sh
```

### 4.4 Monitor Progress
```bash
qstat -u rrazavi
tail -20 /groups/ecumming/rrazavi/twitter_project/logs/model1_*.out
```

### 4.5 Checkpoint System
- Saves progress every 5,000 tweets
- Allows resumption after disconnection
- Files: `partial_*.npy`, `progress_*.json`

## Step 5: Clustering Analysis

### 5.1 8-Cluster Analysis (K-means)
```bash
qsub jobs/job_cluster_analysis.sh
```

Generates:
- Word clouds for each cluster
- Sentiment analysis per cluster
- Top words heatmap
- Cluster size distribution

### 5.2 Optimal K Search (k=3-15)
```bash
qsub jobs/job_optimal_k.sh
```

Tests silhouette scores for k=3 to k=15.

## Step 6: Results Download

### 6.1 Download Results to Local Machine
On Mac terminal:
```bash
mkdir -p ~/Downloads/twitter_results
scp rrazavi@crcfe01.crc.nd.edu:"/groups/ecumming/rrazavi/twitter_project/results/8_cluster_analysis/*.png" ~/Downloads/twitter_results/
scp rrazavi@crcfe01.crc.nd.edu:"/groups/ecumming/rrazavi/twitter_project/results/8_cluster_analysis/*.txt" ~/Downloads/twitter_results/
```

## Step 7: Results Summary

### Clustering Metrics (K-means, k=8)

| Model | Silhouette Score |
|-------|------------------|
| granite-30m | 0.041 |
| granite-278m | **0.048** ✅ |
| qwen3-8b | 0.044 |

### Key Finding
The medium-sized model (278m) achieved the best clustering quality, outperforming both smaller and larger models.

---

## SGE Job Submission Reference

### Common SGE Commands
```bash
qsub script.sh      # Submit job
qstat -u rrazavi    # Check job status
qdel JOB_ID         # Cancel job
qstat -j JOB_ID     # Job details
```

### SGE Script Template
```bash
#!/bin/bash
#$ -N job_name
#$ -o /path/to/logs/job_$JOB_ID.out
#$ -e /path/to/logs/job_$JOB_ID.err
#$ -q long
#$ -l h_vmem=64G
#$ -M rrazavi@nd.edu
#$ -m bea

source /software/c/conda/25.9.1/etc/profile.d/conda.sh
conda activate /groups/ecumming/rrazavi/twitter_project/myenv2
cd /groups/ecumming/rrazavi/twitter_project
python script.py
```

---

## File Locations on CRC

| File Type | Location |
|-----------|----------|
| Raw Data | `/groups/ecumming/rrazavi/twitter_project/ALL_TWEETS_MASTER.csv` |
| Preprocessed | `/groups/ecumming/rrazavi/twitter_project/preprocessed_tweets.csv` |
| Embeddings | `/groups/ecumming/rrazavi/twitter_project/checkpoints/` |
| Results | `/groups/ecumming/rrazavi/twitter_project/results/` |
| Logs | `/groups/ecumming/rrazavi/twitter_project/logs/` |
| Scripts | `/groups/ecumming/rrazavi/twitter_project/*.py` |
| Jobs | `/groups/ecumming/rrazavi/twitter_project/*.sh` |
