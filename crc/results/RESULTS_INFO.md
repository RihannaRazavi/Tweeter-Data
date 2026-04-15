# Results Information

## Clustering Results Summary

### Models Compared

| Model | Parameters | Dimensions | Silhouette Score (k=8) |
|-------|------------|------------|------------------------|
| granite-embedding:30m | 30M | 384 | 0.041 |
| granite-embedding:278m | 278M | 768 | **0.048** ✅ Best |
| qwen3-embedding:8b | 8B | 4096 | 0.044 |

### Key Finding

The medium-sized model (278m) achieved the best clustering quality, outperforming both smaller (30m) and larger (8b) models.

## Output Files

### 8-Cluster Analysis (per model)

| File | Description |
|------|-------------|
| `cluster_sizes.png` | Bar chart of cluster size distribution |
| `all_wordclouds.png` | Word clouds for all 8 clusters |
| `wordcloud_cluster_0.png` to `_7.png` | Individual word clouds |
| `sentiment_analysis.png` | Sentiment distribution by cluster |
| `top_words_heatmap.png` | TF-IDF heatmap of top words |
| `cluster_themes_report.txt` | Detailed text report |
| `cluster_statistics.csv` | Cluster sizes and percentages |
| `sentiment_by_cluster.csv` | Sentiment scores per cluster |
| `tweets_with_clusters_and_sentiment.csv` | Full data with labels |

### Cluster Themes Discovered (Model 1: granite-30m)

| Cluster | Size | Theme | Top Words |
|---------|------|-------|-----------|
| 0 | 1.6% | Horoscopes/Astrology | Aquarius, disagreement |
| 1 | 2.3% | Cheating/Trust Issues | Flirting, kissing, text |
| 2 | 14.5% | Hubby - Daily Life | Hubby, proud, love |
| 3 | 17.0% | Partner/Relationship | Love, partner, relationship |
| 4 | 7.6% | Birthday Celebrations | Birthday, cake, celebrate |
| 5 | 20.7% | Marriage/Spouse | Husband, wife, married |
| 6 | 14.6% | Baby/Family | Baby, sister, wedding |
| 7 | 21.4% | Wife - General | Wife, miss, love |

## Visualization Outputs

Results are stored in:
- `/groups/ecumming/rrazavi/twitter_project/results/8_cluster_analysis/`
- `/groups/ecumming/rrazavi/twitter_project/results/granite_embedding_278m_8clusters/`
- `/groups/ecumming/rrazavi/twitter_project/results/qwen3_embedding_8b_8clusters/`
