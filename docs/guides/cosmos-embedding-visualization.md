# Cosmos Embedding Visualization

Visualize Cosmos VAE embeddings to explore whether similar spatial relation patterns produce similar encodings.

## Quick Start

```bash
# Install new dependencies
uv sync

# Run on 50 samples (quick test, ~2 min)
python scripts/visualize/visualize_cosmos_spatial_embeddings.py \
    --num-samples 50 \
    --draw-bboxes \
    --output outputs/cosmos_spatial_test.html

# Run on full SpatialRGPT-Bench (~2000 samples, ~20 min)
python scripts/visualize/visualize_cosmos_spatial_embeddings.py \
    --num-samples 2000 \
    --draw-bboxes \
    --output outputs/cosmos_spatial_full.html
```

## What It Does

1. **Loads SpatialRGPT-Bench** - 2000 spatial reasoning samples with 19 relation types
2. **Optionally draws bboxes** - Visualize which regions are being compared
3. **Encodes through Cosmos VAE** - Extracts (N, 16, 64, 64) latent representations
4. **Mean-pools to 16-dim** - One vector per image
5. **Reduces to 2D** - UMAP/t-SNE/PCA for visualization
6. **Creates interactive HTML** - Hover to see images, colored by category group

## Output Files

| File | Description |
|------|-------------|
| `cosmos_spatial.html` | Interactive scatter plot with image hover |
| `cosmos_spatial.npz` | Cached embeddings (reload without GPU) |
| `channel_means_heatmap.html` | Per-channel analysis (if `--channel-analysis`) |
| `channel_stds_heatmap.html` | Per-channel variance (if `--channel-analysis`) |

## Re-visualize from Cache

Once you've extracted embeddings, you can quickly try different visualizations without re-encoding:

```bash
# Try t-SNE instead of UMAP
python scripts/visualize/visualize_cosmos_spatial_embeddings.py \
    --load-cache outputs/cosmos_spatial.npz \
    --reduction tsne \
    --output outputs/cosmos_spatial_tsne.html

# Try PCA
python scripts/visualize/visualize_cosmos_spatial_embeddings.py \
    --load-cache outputs/cosmos_spatial.npz \
    --reduction pca \
    --output outputs/cosmos_spatial_pca.html

# All methods side-by-side
python scripts/visualize/visualize_cosmos_spatial_embeddings.py \
    --load-cache outputs/cosmos_spatial.npz \
    --reduction all \
    --output outputs/cosmos_spatial_all.html
```

## Spatial Category Groups

The 19 spatial categories are grouped into 5 semantic clusters for visualization:

| Group | Categories | Color |
|-------|-----------|-------|
| **Size** | small, big, tall, short, thin predicates | Orange |
| **Depth** | behind predicate | Blue |
| **Horizontal** | left, right predicates/choices | Teal |
| **Vertical** | above, tall, short choices | Yellow |
| **Measurements** | height, width, distance data | Pink |

## CLI Options

```
--num-samples N       Number of samples (default: 500)
--batch-size N        Batch size for encoding (default: 4)
--draw-bboxes         Draw bounding boxes on images before encoding
--reduction METHOD    pca, tsne, umap, or all (default: umap)
--thumbnail-size N    Hover image size in pixels (default: 150)
--channel-analysis    Generate per-channel heatmaps
--load-cache PATH     Load from .npz cache (skip encoding)
--save-cache PATH     Custom cache path (default: alongside output)
--device DEVICE       cuda or cpu (default: auto)
```

## Understanding the Results

- **Tight clusters by color** = Cosmos learns similar representations for similar spatial relations
- **Mixed colors** = Spatial relation type doesn't strongly influence Cosmos encoding
- **Channel heatmaps** = Show which of the 16 latent channels vary most per category

## Cache Format

The `.npz` cache contains:
- `latents`: (N, 16, 64, 64) float32 - Full spatial latents
- `metadata`: JSON string with id, qa_category, qa_type per sample
- `thumbnails`: JSON string with base64 images (for reload without re-encoding)

This lets you:
1. Run expensive GPU encoding once
2. Experiment with visualizations on CPU
3. Try different pooling strategies (max, percentile, etc.)
