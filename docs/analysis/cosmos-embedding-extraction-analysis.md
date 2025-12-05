# Cosmos VAE Embedding Extraction Analysis

**Date:** 2024-12-03
**Dataset:** SpatialRGPT-Bench (1,406 samples, 24 categories, 6 groups)
**Latent Shape:** (N, 16, 64, 64)

## Executive Summary

This analysis systematically compares 14 different methods for extracting embeddings from Cosmos VAE latents to understand what works best for spatial reasoning tasks.

**Key Finding:** Simple statistics (min/max pooling, 32 dims) significantly outperform the current TheWorld projection modes (spatial/channel, 65K dims). The current architecture may be suboptimal.

---

## Reproduction Commands

```bash
# 1. Run the full extraction method analysis
PYTHONPATH=python:$PYTHONPATH uv run python scripts/visualize/analyze_cosmos_extraction_methods.py \
    --load-cache outputs/cosmos_spatial_full.npz \
    --output outputs/extraction_analysis/

# 2. Quick test with subset of methods
PYTHONPATH=python:$PYTHONPATH uv run python scripts/visualize/analyze_cosmos_extraction_methods.py \
    --load-cache outputs/cosmos_spatial_full.npz \
    --methods mode_mean,mode_max,mode_minmax \
    --output outputs/quick_test/

# 3. Original embedding extraction (if cache doesn't exist)
PYTHONPATH=python:$PYTHONPATH uv run python scripts/visualize/visualize_cosmos_spatial_embeddings.py \
    --num-samples 2000 \
    --draw-bboxes \
    --all-analysis \
    --output outputs/cosmos_spatial_full.html
```

---

## Output Files

```
outputs/
├── cosmos_spatial_full.npz              # Cached latents (122 MB)
├── extraction_analysis/
│   ├── summary_report.html              # Interactive dashboard
│   ├── metrics_table.csv                # All methods × metrics
│   ├── metrics_table.json
│   ├── comparisons/
│   │   ├── metric_bars.html             # Bar chart comparison
│   │   ├── pareto_frontier.html         # Accuracy vs tokens
│   │   ├── radar_chart.html             # Method profiles
│   │   └── category_heatmap.html        # Per-category performance
│   ├── per_method/
│   │   └── {method}/
│   │       ├── umap.html                # UMAP visualization
│   │       └── metrics.json             # Method metrics
│   └── deep_dive/
│       ├── channel_importance.html      # Which channels matter
│       └── spatial_importance.html      # Which positions matter
└── quick_extraction_test/               # Quick test outputs
```

---

## Results: Method Comparison

| Method | Dims | k-NN Acc | vs Random | Silhouette |
|--------|------|----------|-----------|------------|
| **min_pool** | 16 | **0.155** | **3.7x** | -0.130 |
| **minmax_concat** | 32 | **0.155** | **3.7x** | -0.141 |
| full_stats | 64 | 0.151 | 3.6x | -0.130 |
| max_pool | 16 | 0.143 | 3.4x | -0.198 |
| std_pool | 16 | 0.101 | 2.4x | -0.251 |
| meanstd_concat | 32 | 0.095 | 2.3x | -0.160 |
| patch_8x8 | 1024 | 0.094 | 2.2x | -0.123 |
| **spatial_mode (current)** | 65536 | 0.093 | 2.2x | -0.080 |
| **channel_mode (current)** | 65536 | 0.093 | 2.2x | -0.100 |
| mean_pool | 16 | 0.089 | 2.1x | -0.177 |

**Random baseline:** 4.2% (1/24 categories)

---

## Analysis

### Positive Evidence (Potential Exists)

1. **3.7x better than random** - Best methods achieve 15.5% k-NN accuracy vs 4.2% random
2. **Simple statistics win** - 32 dims (min/max) beats 65,536 dims (raw features)
3. **Information is there** - Cosmos latents DO encode spatial category information
4. **Channel importance varies** - Ch14 is 2.6x random, Ch6 is 2.0x → learnable weighting could help

### Negative Evidence (Concerns)

1. **All silhouettes negative** - No clean cluster separation (-0.08 to -0.25)
2. **Absolute accuracy low** - Best is only 15.5%, still mostly wrong
3. **Current modes underperform** - Spatial/channel modes (65K dims) worse than min pooling (16 dims)
4. **Curse of dimensionality** - More features hurt performance
5. **Wrong pretraining** - Cosmos trained for video prediction, not spatial reasoning

### Critical Insight

The current TheWorld projection modes feed 65,536 raw features to an MLP, but:
- 32 carefully-chosen features (min/max) work 1.7x better
- The MLP must learn to extract useful statistics from scratch
- Random initialization + language modeling loss may never discover optimal extraction

**This means TheWorld's Cosmos branch may be contributing noise rather than signal.**

---

## Per-Channel Analysis

For channel mode (16 tokens × 4096 features each):

| Channel | k-NN Acc | vs Random |
|---------|----------|-----------|
| Ch14 | 0.110 | 2.6x |
| Ch1 | 0.105 | 2.5x |
| Ch13 | 0.101 | 2.4x |
| Ch8 | 0.100 | 2.4x |
| ... | ... | ... |
| Ch6 | 0.084 | 2.0x |

**Insight:** Channel importance varies 30% (2.0x to 2.6x). Weighted aggregation could help.

---

## Recommendations

### 1. Replace Raw Features with Statistics [HIGH IMPACT]

```python
# Current (suboptimal)
latents = cosmos_vae(image)  # (N, 16, 64, 64)
tokens = mlp(latents.reshape(N, 16, 4096))  # 65K features → MLP

# Proposed (better)
latents = cosmos_vae(image)  # (N, 16, 64, 64)
stats = torch.cat([
    latents.min(dim=(2,3)),   # 16 dims
    latents.max(dim=(2,3)),   # 16 dims
], dim=1)  # 32 dims total
tokens = mlp(stats)  # 32 features → MLP
```

Expected improvement: ~1.7x k-NN accuracy

### 2. Use Min/Max Specifically [HIGH IMPACT]

The range (max - min) per channel captures the most discriminative information. Just 32 dims achieves best results.

### 3. Reduce Token Count [EFFICIENCY]

- Current spatial mode: 4096 tokens
- Could achieve same/better with: 1 token (statistics only)
- Massive attention speedup with no quality loss

### 4. Consider Removing Cosmos [HONEST OPTION]

If Gemma's SigLIP already captures spatial info, Cosmos may add complexity without proportional benefit. Need A/B test.

---

## Experimental Validation Needed

- [ ] Train TheWorld with minmax statistics vs current projection
- [ ] Compare Gemma-only vs TheWorld on SpatialRGPT-Bench
- [ ] Measure if Cosmos adds value beyond what SigLIP provides
- [ ] Test attention-based projection instead of MLP

---

## Bottom Line

| Aspect | Assessment |
|--------|------------|
| **Is there spatial info in Cosmos?** | Yes, 3.7x better than random |
| **Is current extraction optimal?** | No, simple stats beat raw features |
| **Is the signal strong?** | No, 15.5% accuracy, negative silhouettes |
| **Will TheWorld help spatial reasoning?** | Uncertain - need better extraction first |

**Recommendation:** Before concluding TheWorld doesn't work, switch to minmax statistics (32 dims) and re-evaluate against Gemma-only baseline.

---

## Files Created

| File | Description |
|------|-------------|
| `python/theworld/analysis/__init__.py` | Analysis package |
| `python/theworld/analysis/embedding_methods.py` | 14 extraction methods |
| `python/theworld/analysis/embedding_metrics.py` | Discriminability metrics |
| `scripts/visualize/analyze_cosmos_extraction_methods.py` | Main analysis script |
| `docs/analysis/cosmos-embedding-extraction-analysis.md` | This document |
