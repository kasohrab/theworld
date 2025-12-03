"""
Visualize Cosmos VAE embeddings for SpatialRGPT-Bench samples.

This script explores whether similar spatial relation categories produce similar
Cosmos VAE encodings. It extracts 16-dimensional latent embeddings and visualizes
them using dimensionality reduction (UMAP/t-SNE/PCA) with interactive Plotly plots.

Key features:
- Caches full latents to .npz for fast re-visualization
- Interactive HTML with image thumbnails on hover
- Groups 19 spatial categories into 5 semantic groups
- Optional channel-wise heatmap analysis

Usage:
    # Extract embeddings and visualize (500 samples, UMAP, with bboxes)
    python scripts/visualize/visualize_cosmos_spatial_embeddings.py \
        --num-samples 500 \
        --draw-bboxes \
        --output outputs/cosmos_spatial.html

    # Re-visualize from cached embeddings (no GPU needed)
    python scripts/visualize/visualize_cosmos_spatial_embeddings.py \
        --load-cache outputs/cosmos_spatial_cache.npz \
        --reduction tsne \
        --output outputs/cosmos_spatial_tsne.html

    # Full benchmark with all visualizations
    python scripts/visualize/visualize_cosmos_spatial_embeddings.py \
        --num-samples 2000 \
        --draw-bboxes \
        --reduction all \
        --channel-analysis \
        --output outputs/cosmos_spatial_full.html
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "python"))


# =============================================================================
# Spatial Category Grouping
# =============================================================================

SPATIAL_CATEGORY_GROUPS = {
    "Size": ["small_predicate", "big_predicate", "tall_predicate", "short_predicate", "thin_predicate"],
    "Depth": ["behind_predicate"],
    "Horizontal": ["left_predicate", "right_predicate", "left_choice", "right_choice"],
    "Vertical": ["above_choice", "tall_choice", "short_choice"],
    "Measurements": [
        "height_data",
        "width_data",
        "distance_data",
        "horizontal_distance_data",
        "vertical_distance_data",
        "direction",
    ],
}

# Colorblind-friendly palette
GROUP_COLORS = {
    "Size": "#E69F00",  # Orange
    "Depth": "#56B4E9",  # Sky blue
    "Horizontal": "#009E73",  # Teal
    "Vertical": "#F0E442",  # Yellow
    "Measurements": "#CC79A7",  # Pink
    "Unknown": "#999999",  # Gray
}


def get_category_group(category: str) -> str:
    """Get the semantic group for a spatial category."""
    for group, categories in SPATIAL_CATEGORY_GROUPS.items():
        if category in categories:
            return group
    return "Unknown"


# =============================================================================
# Argument Parsing
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Cosmos VAE embeddings for spatial relation categories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to analyze (default: 500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for Cosmos encoding (default: 4)",
    )
    parser.add_argument(
        "--draw-bboxes",
        action="store_true",
        help="Draw bounding boxes on images before encoding",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/cosmos_spatial_embeddings.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        default="umap",
        choices=["pca", "tsne", "umap", "all"],
        help="Dimensionality reduction method (default: umap)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for Cosmos VAE (default: auto-detect)",
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=150,
        help="Thumbnail size for hover images (default: 150)",
    )
    parser.add_argument(
        "--channel-analysis",
        action="store_true",
        help="Include per-channel variance analysis",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Create 4x4 grid visualization of each channel's spatial patterns",
    )
    parser.add_argument(
        "--discriminability",
        action="store_true",
        help="Analyze which channels best discriminate spatial categories (silhouette + F-ratio)",
    )
    parser.add_argument(
        "--spatial-discriminability",
        action="store_true",
        help="Analyze which spatial positions (64x64) best discriminate categories",
    )
    parser.add_argument(
        "--all-analysis",
        action="store_true",
        help="Run all analysis types (channel, per-channel, discriminability, spatial, neighbors, correlations)",
    )
    parser.add_argument(
        "--neighbors",
        action="store_true",
        help="Analyze nearest neighbors: do similar embeddings have same category?",
    )
    parser.add_argument(
        "--correlations",
        action="store_true",
        help="Analyze channel correlations: are channels redundant?",
    )
    parser.add_argument(
        "--load-cache",
        type=str,
        default=None,
        help="Load embeddings from existing .npz file (skip encoding)",
    )
    parser.add_argument(
        "--save-cache",
        type=str,
        default=None,
        help="Path to save embeddings cache (default: alongside output)",
    )
    return parser.parse_args()


# =============================================================================
# Cosmos VAE Loading (Standalone)
# =============================================================================


def load_cosmos_vae_standalone(device: Optional[str] = None):
    """Load Cosmos VAE encoder standalone (without full TheWorld model).

    This is more memory-efficient than loading the full model (~8GB saved).
    """
    from cosmos_guardrail import CosmosSafetyChecker
    from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline

    from theworld.modeling.cosmos_vae_encoder import CosmosVAEEncoder

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Cosmos VAE encoder (device={device})...")

    # Load the full pipeline to access the VAE
    safety_checker = CosmosSafetyChecker()
    cosmos_pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        "nvidia/Cosmos-Predict2-2B-Video2World",
        torch_dtype=torch.bfloat16,
        safety_checker=safety_checker,
        low_cpu_mem_usage=True,
    )

    # Extract VAE and move to device
    cosmos_vae = cosmos_pipe.vae.to(device)

    # Create encoder wrapper
    encoder = CosmosVAEEncoder(
        cosmos_vae=cosmos_vae,
        device=device,
        freeze_vae=True,
    )

    # Workaround for RetinaFace bug (disables gradients globally)
    torch.set_grad_enabled(True)

    print(f"  Cosmos VAE loaded (z_dim={encoder.z_dim})")
    return encoder


# =============================================================================
# Data Loading
# =============================================================================


def load_spatial_rgpt_bench(num_samples: int, draw_bboxes: bool) -> List[Dict[str, Any]]:
    """Load SpatialRGPT-Bench dataset with optional bbox drawing.

    Returns list of dicts with:
        - image: PIL.Image (with or without bboxes)
        - qa_category: str (one of 19 spatial categories)
        - qa_type: str (qualitative or quantitative)
        - id: str
        - metadata: dict (original sample)
    """
    from datasets import load_dataset

    from theworld.datasets.spatial_rgpt import SpatialRGPTDataset

    print(f"Loading SpatialRGPT-Bench (num_samples={num_samples}, draw_bboxes={draw_bboxes})...")

    # Load from HuggingFace
    hf_dataset = load_dataset("a8cheng/SpatialRGPT-Bench", split="val")
    print(f"  HuggingFace dataset has {len(hf_dataset)} samples")

    # Wrap with SpatialRGPTDataset (handles bbox drawing internally)
    dataset = SpatialRGPTDataset(
        data_source=hf_dataset,
        image_folder=None,  # Images included in HF dataset
        draw_bboxes=draw_bboxes,
        num_samples=num_samples,
    )

    # Extract relevant fields
    samples = []
    for i in tqdm(range(len(dataset)), desc="Loading samples"):
        try:
            sample = dataset[i]
            samples.append(
                {
                    "image": sample["image"],
                    "qa_category": sample["qa_category"] or "Unknown",
                    "qa_type": sample["qa_type"] or "Unknown",
                    "id": sample["id"],
                    "metadata": sample["metadata"],
                }
            )
        except Exception as e:
            print(f"  Warning: Failed to load sample {i}: {e}")
            continue

    print(f"  Loaded {len(samples)} samples successfully")

    # Print category distribution
    categories: Dict[str, int] = {}
    for s in samples:
        cat = s["qa_category"]
        categories[cat] = categories.get(cat, 0) + 1
    print("Category distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        group = get_category_group(cat)
        print(f"  {cat} ({group}): {count}")

    return samples


# =============================================================================
# Embedding Extraction
# =============================================================================


def extract_embeddings(
    encoder,
    samples: List[Dict[str, Any]],
    batch_size: int = 4,
) -> np.ndarray:
    """Extract Cosmos VAE embeddings from images.

    Args:
        encoder: CosmosVAEEncoder instance
        samples: List of sample dicts with "image" field
        batch_size: Batch size for encoding

    Returns:
        full_latents: (N, 16, 64, 64) array - full spatial latents
    """
    print(f"Extracting embeddings (batch_size={batch_size})...")

    full_latents_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size), desc="Encoding"):
            batch_end = min(i + batch_size, len(samples))
            batch_images = [samples[j]["image"] for j in range(i, batch_end)]

            # Encode through Cosmos VAE
            latents = encoder(batch_images)  # (B, 16, H, W)
            full_latents_list.append(latents.cpu().float().numpy())

    full_latents = np.concatenate(full_latents_list, axis=0)  # (N, 16, H, W)
    print(f"  Full latents shape: {full_latents.shape}")

    return full_latents


# =============================================================================
# Cache Management
# =============================================================================


def save_cache(
    cache_path: Path,
    latents: np.ndarray,
    samples: List[Dict[str, Any]],
    thumbnails: Optional[List[str]] = None,
) -> None:
    """Save embeddings and metadata to .npz cache file."""
    print(f"Saving cache to {cache_path}...")

    # Extract metadata (without PIL images)
    metadata = []
    for s in samples:
        metadata.append(
            {
                "id": s["id"],
                "qa_category": s["qa_category"],
                "qa_type": s["qa_type"],
            }
        )

    # Save to npz
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        latents=latents,
        metadata=json.dumps(metadata),
        thumbnails=json.dumps(thumbnails) if thumbnails else "",
    )
    print(f"  Cache saved ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")


def load_cache(cache_path: Path) -> Tuple[np.ndarray, List[Dict[str, Any]], Optional[List[str]]]:
    """Load embeddings and metadata from .npz cache file."""
    print(f"Loading cache from {cache_path}...")

    data = np.load(cache_path, allow_pickle=True)
    latents = data["latents"]
    metadata = json.loads(str(data["metadata"]))

    thumbnails = None
    if "thumbnails" in data and str(data["thumbnails"]):
        thumbnails = json.loads(str(data["thumbnails"]))

    # Convert metadata to samples format
    samples = []
    for m in metadata:
        samples.append(
            {
                "id": m["id"],
                "qa_category": m["qa_category"],
                "qa_type": m["qa_type"],
                "image": None,  # Not available from cache
                "metadata": m,
            }
        )

    print(f"  Loaded {len(samples)} samples, latents shape: {latents.shape}")
    return latents, samples, thumbnails


# =============================================================================
# Dimensionality Reduction
# =============================================================================


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = "umap",
) -> Dict[str, np.ndarray]:
    """Reduce embeddings to 2D for visualization.

    Args:
        embeddings: (N, 16) array (mean-pooled latents)
        method: "pca", "tsne", "umap", or "all"

    Returns:
        Dict mapping method name to (N, 2) coordinates
    """
    results = {}

    if method in ["pca", "all"]:
        print("Computing PCA...")
        pca = PCA(n_components=2)
        results["pca"] = pca.fit_transform(embeddings)
        print(f"  PCA explained variance: {pca.explained_variance_ratio_}")

    if method in ["tsne", "all"]:
        print("Computing t-SNE (this may take a minute)...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), max_iter=1000, random_state=42)
        results["tsne"] = tsne.fit_transform(embeddings)

    if method in ["umap", "all"]:
        print("Computing UMAP...")
        import umap

        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
        results["umap"] = reducer.fit_transform(embeddings)

    return results


# =============================================================================
# Image Encoding for Hover
# =============================================================================


def image_to_base64(image: Image.Image, size: int = 150) -> str:
    """Convert PIL image to base64 string for Plotly hover."""
    # Create a copy and resize to thumbnail
    img_copy = image.copy()
    img_copy.thumbnail((size, size), Image.Resampling.LANCZOS)

    # Convert to JPEG bytes
    buffer = io.BytesIO()
    img_copy.save(buffer, format="JPEG", quality=80)
    buffer.seek(0)

    # Encode to base64
    img_bytes = buffer.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    return f"data:image/jpeg;base64,{img_b64}"


def encode_thumbnails(samples: List[Dict[str, Any]], size: int = 150) -> List[str]:
    """Pre-encode all sample images as base64 thumbnails."""
    print(f"Encoding {len(samples)} thumbnails...")
    thumbnails = []
    for s in tqdm(samples, desc="Thumbnails"):
        if s["image"] is not None:
            thumbnails.append(image_to_base64(s["image"], size))
        else:
            # Placeholder for cached data without images
            thumbnails.append("")
    return thumbnails


# =============================================================================
# Interactive Visualization
# =============================================================================


def create_interactive_visualization(
    coords_dict: Dict[str, np.ndarray],
    samples: List[Dict[str, Any]],
    thumbnails: List[str],
    thumbnail_size: int,
    output_path: Path,
) -> None:
    """Create interactive Plotly visualization with hover images."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Creating interactive visualization...")

    # Prepare data
    categories = [s["qa_category"] for s in samples]
    groups = [get_category_group(cat) for cat in categories]
    qa_types = [s["qa_type"] for s in samples]
    ids = [s["id"] for s in samples]

    # Create subplot for each reduction method
    n_methods = len(coords_dict)
    fig = make_subplots(
        rows=1,
        cols=n_methods,
        subplot_titles=[f"{method.upper()}" for method in coords_dict.keys()],
        horizontal_spacing=0.05,
    )

    # Unique groups for legend
    unique_groups = sorted(set(groups))

    for col, (method, coords) in enumerate(coords_dict.items(), start=1):
        for group in unique_groups:
            # Filter points for this group
            mask = [g == group for g in groups]
            x = coords[np.array(mask), 0]
            y = coords[np.array(mask), 1]

            # Filter corresponding data
            group_cats = [c for c, m in zip(categories, mask) if m]
            group_types = [t for t, m in zip(qa_types, mask) if m]
            group_ids = [i for i, m in zip(ids, mask) if m]
            group_thumbs = [t for t, m in zip(thumbnails, mask) if m]

            # Custom hover text with image
            hover_texts = []
            for cat, qtype, sid, thumb in zip(group_cats, group_types, group_ids, group_thumbs):
                if thumb:
                    hover_texts.append(
                        f"<b>Category:</b> {cat}<br>"
                        f"<b>Type:</b> {qtype}<br>"
                        f"<b>ID:</b> {sid}<br>"
                        f"<img src='{thumb}' width='{thumbnail_size}'>"
                    )
                else:
                    hover_texts.append(f"<b>Category:</b> {cat}<br>" f"<b>Type:</b> {qtype}<br>" f"<b>ID:</b> {sid}")

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name=group,
                    marker=dict(
                        color=GROUP_COLORS.get(group, "#999999"),
                        size=8,
                        opacity=0.7,
                        line=dict(width=1, color="white"),
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                    legendgroup=group,
                    showlegend=(col == 1),  # Only show legend for first subplot
                ),
                row=1,
                col=col,
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Cosmos VAE Embeddings by Spatial Relation Category",
            x=0.5,
            font=dict(size=20),
        ),
        height=700,
        width=500 * n_methods,
        template="plotly_white",
        legend=dict(
            title="Category Group",
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
    )

    # Save to HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    print(f"  Saved interactive visualization to: {output_path}")


# =============================================================================
# Channel Analysis
# =============================================================================


def analyze_channels(
    full_latents: np.ndarray,
    samples: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Analyze what each of the 16 latent channels encodes.

    Creates:
    - Per-channel mean heatmap by category
    - Per-channel variance heatmap by category
    """
    import pandas as pd
    import plotly.express as px

    print("Analyzing channels...")

    # Mean-pool spatial dimensions: (N, 16, H, W) -> (N, 16)
    channel_means = full_latents.mean(axis=(2, 3))  # (N, 16)

    # Get categories
    categories = [s["qa_category"] for s in samples]
    unique_cats = sorted(set(categories))

    # Compute per-category channel statistics
    cat_channel_means: Dict[str, np.ndarray] = {}
    cat_channel_stds: Dict[str, np.ndarray] = {}

    for cat in unique_cats:
        mask = [c == cat for c in categories]
        cat_latents = channel_means[np.array(mask)]  # (n_cat, 16)
        cat_channel_means[cat] = cat_latents.mean(axis=0)
        cat_channel_stds[cat] = cat_latents.std(axis=0)

    # Create dataframe for heatmap
    df_means = pd.DataFrame(cat_channel_means).T
    df_means.columns = [f"Ch{i}" for i in range(16)]

    # Heatmap of channel means by category
    fig = px.imshow(
        df_means.values,
        x=df_means.columns.tolist(),
        y=df_means.index.tolist(),
        color_continuous_scale="RdBu_r",
        title="Channel Mean Values by Spatial Category",
        labels=dict(x="Latent Channel", y="Spatial Category", color="Mean Value"),
    )
    fig.update_layout(height=600, width=900)
    fig.write_html(str(output_dir / "channel_means_heatmap.html"))

    # Variance heatmap
    df_stds = pd.DataFrame(cat_channel_stds).T
    df_stds.columns = [f"Ch{i}" for i in range(16)]

    fig2 = px.imshow(
        df_stds.values,
        x=df_stds.columns.tolist(),
        y=df_stds.index.tolist(),
        color_continuous_scale="Viridis",
        title="Channel Standard Deviation by Spatial Category",
        labels=dict(x="Latent Channel", y="Spatial Category", color="Std Dev"),
    )
    fig2.update_layout(height=600, width=900)
    fig2.write_html(str(output_dir / "channel_stds_heatmap.html"))

    print(f"  Saved channel analysis to: {output_dir}")


def visualize_per_channel(
    full_latents: np.ndarray,
    samples: List[Dict[str, Any]],
    thumbnails: List[str],
    thumbnail_size: int,
    output_path: Path,
    reduction: str = "umap",
) -> None:
    """Create per-channel 2D visualizations (16 subplots).

    Each channel's mean-pooled spatial value is used, then all 16 channels
    are visualized in a 4x4 grid with the same dimensionality reduction.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Creating per-channel visualization...")

    # Get metadata
    categories = [s["qa_category"] for s in samples]
    groups = [get_category_group(cat) for cat in categories]
    unique_groups = sorted(set(groups))

    # Mean-pool spatial: (N, 16, 64, 64) -> (N, 16)
    channel_values = full_latents.mean(axis=(2, 3))  # (N, 16)

    # For each channel, we'll do a 2D embedding using the spatial pattern
    # Option 1: Use the 64x64 spatial values per channel (4096-dim per sample)
    # Option 2: Use pairs of channels
    # Let's do Option 1 - flatten spatial dims per channel

    # Create 4x4 subplot grid
    fig = make_subplots(
        rows=4,
        cols=4,
        subplot_titles=[f"Channel {i}" for i in range(16)],
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
    )

    # For each channel, compute 2D projection of its spatial pattern
    for ch in tqdm(range(16), desc="Per-channel reduction"):
        # Get this channel's spatial data: (N, 64, 64) -> (N, 4096)
        ch_spatial = full_latents[:, ch, :, :].reshape(len(samples), -1)

        # Reduce to 2D
        if reduction == "pca":
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(ch_spatial)
        elif reduction == "tsne":
            reducer = TSNE(n_components=2, perplexity=min(30, len(samples) - 1), max_iter=500, random_state=42)
            coords = reducer.fit_transform(ch_spatial)
        else:  # umap
            import umap

            reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, verbose=False)
            coords = reducer.fit_transform(ch_spatial)

        # Plot each group
        row = ch // 4 + 1
        col = ch % 4 + 1

        for group in unique_groups:
            mask = np.array([g == group for g in groups])
            x = coords[mask, 0]
            y = coords[mask, 1]

            # Hover text
            group_cats = [c for c, m in zip(categories, mask) if m]
            group_thumbs = [t for t, m in zip(thumbnails, mask) if m]
            hover_texts = []
            for cat, thumb in zip(group_cats, group_thumbs):
                if thumb:
                    hover_texts.append(f"<b>{cat}</b><br><img src='{thumb}' width='{thumbnail_size}'>")
                else:
                    hover_texts.append(f"<b>{cat}</b>")

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    name=group,
                    marker=dict(
                        color=GROUP_COLORS.get(group, "#999999"),
                        size=5,
                        opacity=0.6,
                    ),
                    text=hover_texts,
                    hoverinfo="text",
                    legendgroup=group,
                    showlegend=(ch == 0),  # Only show legend once
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=dict(
            text=f"Per-Channel Spatial Patterns ({reduction.upper()})",
            x=0.5,
            font=dict(size=20),
        ),
        height=1200,
        width=1400,
        template="plotly_white",
        legend=dict(
            title="Category Group",
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
        ),
        hoverlabel=dict(bgcolor="white", font_size=10),
    )

    # Hide axis labels for cleaner look
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    print(f"  Saved per-channel visualization to: {output_path}")


def analyze_channel_discriminability(
    full_latents: np.ndarray,
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Analyze which channels best discriminate between spatial categories.

    Computes:
    - Silhouette score per channel (higher = better separation by category)
    - Between-class / within-class variance ratio (F-ratio)
    - Creates bar chart showing discriminative power per channel
    """
    import plotly.graph_objects as go
    from sklearn.metrics import silhouette_score

    print("Analyzing channel discriminability...")

    categories = [s["qa_category"] for s in samples]
    groups = [get_category_group(cat) for cat in categories]

    # Convert to numeric labels
    unique_groups = sorted(set(groups))
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    labels = np.array([group_to_idx[g] for g in groups])

    # Per-channel analysis
    n_channels = full_latents.shape[1]
    silhouette_scores = []
    f_ratios = []

    for ch in range(n_channels):
        # Flatten spatial: (N, 64, 64) -> (N, 4096)
        ch_data = full_latents[:, ch, :, :].reshape(len(samples), -1)

        # Silhouette score (how well clusters separate)
        try:
            sil = silhouette_score(ch_data, labels, sample_size=min(1000, len(labels)))
        except Exception:
            sil = 0.0
        silhouette_scores.append(sil)

        # F-ratio (between-class variance / within-class variance)
        # Higher = better separation
        overall_mean = ch_data.mean(axis=0)
        between_var = 0.0
        within_var = 0.0
        for grp_idx in range(len(unique_groups)):
            grp_mask = labels == grp_idx
            grp_data = ch_data[grp_mask]
            grp_mean = grp_data.mean(axis=0)
            n_grp = grp_mask.sum()
            # Between-class: n_k * ||mean_k - overall_mean||^2
            between_var += n_grp * np.sum((grp_mean - overall_mean) ** 2)
            # Within-class: sum of ||x - mean_k||^2
            within_var += np.sum((grp_data - grp_mean) ** 2)

        f_ratio = between_var / (within_var + 1e-8)
        f_ratios.append(f_ratio)

    # Create visualization
    fig = go.Figure()

    # Silhouette scores
    fig.add_trace(
        go.Bar(
            x=[f"Ch{i}" for i in range(n_channels)],
            y=silhouette_scores,
            name="Silhouette Score",
            marker_color="#56B4E9",
            yaxis="y",
        )
    )

    # F-ratios on secondary axis (normalized for visualization)
    f_ratios_norm = np.array(f_ratios) / max(f_ratios) if max(f_ratios) > 0 else f_ratios
    fig.add_trace(
        go.Scatter(
            x=[f"Ch{i}" for i in range(n_channels)],
            y=f_ratios_norm,
            name="F-ratio (normalized)",
            mode="lines+markers",
            marker_color="#E69F00",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title=dict(
            text="Channel Discriminability: Which Channels Best Separate Spatial Categories?",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(title="Latent Channel"),
        yaxis=dict(title="Silhouette Score", side="left", showgrid=True),
        yaxis2=dict(title="F-ratio (normalized)", side="right", overlaying="y", showgrid=False),
        height=500,
        width=900,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    # Add annotation for best channels
    best_sil_ch = np.argmax(silhouette_scores)
    best_f_ch = np.argmax(f_ratios)
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=f"Best by Silhouette: Channel {best_sil_ch} ({silhouette_scores[best_sil_ch]:.3f}) | "
        f"Best by F-ratio: Channel {best_f_ch}",
        showarrow=False,
        font=dict(size=12),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    print(f"  Saved discriminability analysis to: {output_path}")
    print(f"  Best discriminative channel (silhouette): {best_sil_ch} (score={silhouette_scores[best_sil_ch]:.3f})")
    print(f"  Best discriminative channel (F-ratio): {best_f_ch}")


def analyze_spatial_discriminability(
    full_latents: np.ndarray,
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Analyze which spatial positions best discriminate between spatial categories.

    For each position (i,j) in the 64x64 grid, computes F-ratio across the 16 channels.
    Creates heatmap showing discriminative power at each spatial location.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Analyzing spatial discriminability (this may take a moment)...")

    categories = [s["qa_category"] for s in samples]
    groups = [get_category_group(cat) for cat in categories]

    # Convert to numeric labels
    unique_groups = sorted(set(groups))
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    labels = np.array([group_to_idx[g] for g in groups])

    # Get spatial dimensions
    N, C, H, W = full_latents.shape  # (N, 16, 64, 64)

    # Compute F-ratio for each spatial position
    # For position (i,j), we have N samples with C=16 features
    f_ratio_map = np.zeros((H, W))

    for i in tqdm(range(H), desc="Spatial analysis"):
        for j in range(W):
            # Get all channels at this position: (N, 16)
            pos_data = full_latents[:, :, i, j]

            # Compute F-ratio
            overall_mean = pos_data.mean(axis=0)
            between_var = 0.0
            within_var = 0.0

            for grp_idx in range(len(unique_groups)):
                grp_mask = labels == grp_idx
                grp_data = pos_data[grp_mask]
                if len(grp_data) == 0:
                    continue
                grp_mean = grp_data.mean(axis=0)
                n_grp = grp_mask.sum()
                between_var += n_grp * np.sum((grp_mean - overall_mean) ** 2)
                within_var += np.sum((grp_data - grp_mean) ** 2)

            f_ratio_map[i, j] = between_var / (within_var + 1e-8)

    # Also compute per-channel spatial maps
    channel_spatial_maps = np.zeros((C, H, W))
    for ch in range(C):
        for i in range(H):
            for j in range(W):
                # Single value per sample at this (ch, i, j)
                pos_data = full_latents[:, ch, i, j].reshape(-1, 1)

                overall_mean = pos_data.mean()
                between_var = 0.0
                within_var = 0.0

                for grp_idx in range(len(unique_groups)):
                    grp_mask = labels == grp_idx
                    grp_data = pos_data[grp_mask]
                    if len(grp_data) == 0:
                        continue
                    grp_mean = grp_data.mean()
                    n_grp = grp_mask.sum()
                    between_var += n_grp * (grp_mean - overall_mean) ** 2
                    within_var += np.sum((grp_data - grp_mean) ** 2)

                channel_spatial_maps[ch, i, j] = between_var / (within_var + 1e-8)

    # Create visualization with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Overall Spatial Discriminability (all 16 channels)", "Best spatial position per channel"],
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4],
    )

    # Overall spatial heatmap
    fig.add_trace(
        go.Heatmap(
            z=f_ratio_map,
            colorscale="Viridis",
            colorbar=dict(title="F-ratio", x=1.02, len=0.4, y=0.8),
        ),
        row=1,
        col=1,
    )

    # Find best position for each channel
    best_positions = []
    for ch in range(C):
        best_idx = np.unravel_index(np.argmax(channel_spatial_maps[ch]), (H, W))
        best_positions.append((ch, best_idx[0], best_idx[1], channel_spatial_maps[ch, best_idx[0], best_idx[1]]))

    # Bar chart of max F-ratio per channel
    fig.add_trace(
        go.Bar(
            x=[f"Ch{i}" for i in range(C)],
            y=[bp[3] for bp in best_positions],
            marker_color="#56B4E9",
            text=[f"({bp[1]},{bp[2]})" for bp in best_positions],
            textposition="outside",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title=dict(
            text="Spatial Discriminability: Which Positions Best Separate Categories?",
            x=0.5,
            font=dict(size=16),
        ),
        height=900,
        width=800,
        template="plotly_white",
        showlegend=False,
    )

    fig.update_xaxes(title_text="Spatial X", row=1, col=1)
    fig.update_yaxes(title_text="Spatial Y", row=1, col=1)
    fig.update_xaxes(title_text="Channel", row=2, col=1)
    fig.update_yaxes(title_text="Max F-ratio at best position", row=2, col=1)

    # Find overall best position
    best_overall = np.unravel_index(np.argmax(f_ratio_map), (H, W))
    fig.add_annotation(
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        text=f"Best overall position: ({best_overall[0]}, {best_overall[1]}) with F-ratio={f_ratio_map[best_overall]:.4f}",
        showarrow=False,
        font=dict(size=12),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)

    # Also save per-channel spatial maps as separate file
    perchannel_path = output_path.parent / "spatial_discriminability_per_channel.html"
    fig2 = make_subplots(
        rows=4,
        cols=4,
        subplot_titles=[f"Channel {i}" for i in range(16)],
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
    )

    for ch in range(C):
        row = ch // 4 + 1
        col = ch % 4 + 1
        fig2.add_trace(
            go.Heatmap(
                z=channel_spatial_maps[ch],
                colorscale="Viridis",
                showscale=(ch == 15),  # Only show colorbar for last
            ),
            row=row,
            col=col,
        )

    fig2.update_layout(
        title=dict(
            text="Per-Channel Spatial Discriminability Maps",
            x=0.5,
            font=dict(size=16),
        ),
        height=1000,
        width=1000,
        template="plotly_white",
    )
    fig2.update_xaxes(showticklabels=False)
    fig2.update_yaxes(showticklabels=False)
    fig2.write_html(str(perchannel_path), include_plotlyjs=True, full_html=True)

    print(f"  Saved spatial discriminability to: {output_path}")
    print(f"  Saved per-channel spatial maps to: {perchannel_path}")
    print(f"  Best discriminative position: ({best_overall[0]}, {best_overall[1]})")


def analyze_nearest_neighbors(
    full_latents: np.ndarray,
    samples: List[Dict[str, Any]],
    output_path: Path,
    k: int = 10,
) -> None:
    """Analyze if nearest neighbors in embedding space share the same category.

    For each sample, finds k nearest neighbors and computes:
    - % of neighbors with same category
    - % of neighbors with same category group
    - Confusion matrix showing which categories are "close" in embedding space
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.neighbors import NearestNeighbors

    print(f"Analyzing nearest neighbors (k={k})...")

    # Mean-pool to get per-image embeddings
    embeddings = full_latents.mean(axis=(2, 3))  # (N, 16)

    categories = [s["qa_category"] for s in samples]
    groups = [get_category_group(cat) for cat in categories]
    unique_cats = sorted(set(categories))
    unique_groups = sorted(set(groups))

    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")  # +1 because query point is included
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    # Compute per-sample neighbor accuracy
    same_cat_counts = []
    same_group_counts = []

    for i in range(len(samples)):
        neighbor_indices = indices[i, 1:]  # Exclude self
        neighbor_cats = [categories[j] for j in neighbor_indices]
        neighbor_groups = [groups[j] for j in neighbor_indices]

        same_cat = sum(1 for c in neighbor_cats if c == categories[i])
        same_group = sum(1 for g in neighbor_groups if g == groups[i])

        same_cat_counts.append(same_cat / k)
        same_group_counts.append(same_group / k)

    # Compute confusion matrix (which categories have neighbors of other categories)
    confusion = np.zeros((len(unique_cats), len(unique_cats)))
    cat_to_idx = {c: i for i, c in enumerate(unique_cats)}

    for i in range(len(samples)):
        src_idx = cat_to_idx[categories[i]]
        neighbor_indices = indices[i, 1:]
        for j in neighbor_indices:
            dst_idx = cat_to_idx[categories[j]]
            confusion[src_idx, dst_idx] += 1

    # Normalize rows
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)

    # Create visualization
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Same Category in Top-{k} Neighbors",
            f"Same Group in Top-{k} Neighbors",
            "Category Confusion (Normalized)",
            "Summary Statistics",
        ],
        specs=[[{"type": "histogram"}, {"type": "histogram"}], [{"type": "heatmap"}, {"type": "table"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Histogram of same-category accuracy
    fig.add_trace(
        go.Histogram(x=same_cat_counts, nbinsx=20, marker_color="#56B4E9", name="Same Category"),
        row=1,
        col=1,
    )

    # Histogram of same-group accuracy
    fig.add_trace(
        go.Histogram(x=same_group_counts, nbinsx=20, marker_color="#E69F00", name="Same Group"),
        row=1,
        col=2,
    )

    # Confusion heatmap
    fig.add_trace(
        go.Heatmap(
            z=confusion_norm,
            x=unique_cats,
            y=unique_cats,
            colorscale="Blues",
            showscale=True,
        ),
        row=2,
        col=1,
    )

    # Summary statistics table
    stats_data = [
        ["Mean same-category rate", f"{np.mean(same_cat_counts):.3f}"],
        ["Mean same-group rate", f"{np.mean(same_group_counts):.3f}"],
        ["Median same-category rate", f"{np.median(same_cat_counts):.3f}"],
        ["Median same-group rate", f"{np.median(same_group_counts):.3f}"],
        ["Random baseline (category)", f"{1/len(unique_cats):.3f}"],
        ["Random baseline (group)", f"{1/len(unique_groups):.3f}"],
    ]
    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], fill_color="#56B4E9", font=dict(color="white")),
            cells=dict(values=list(zip(*stats_data)), fill_color="white"),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"Nearest Neighbor Analysis: Do Similar Embeddings Have Same Category?",
            x=0.5,
            font=dict(size=16),
        ),
        height=900,
        width=1100,
        template="plotly_white",
        showlegend=False,
    )

    fig.update_xaxes(title_text="Fraction of neighbors with same category", row=1, col=1)
    fig.update_xaxes(title_text="Fraction of neighbors with same group", row=1, col=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)

    print(f"  Saved nearest neighbor analysis to: {output_path}")
    print(f"  Mean same-category rate: {np.mean(same_cat_counts):.3f} (random: {1/len(unique_cats):.3f})")
    print(f"  Mean same-group rate: {np.mean(same_group_counts):.3f} (random: {1/len(unique_groups):.3f})")


def analyze_channel_correlations(
    full_latents: np.ndarray,
    samples: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Analyze correlations between the 16 channels.

    Shows which channels are redundant (highly correlated) vs independent.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    print("Analyzing channel correlations...")

    # Mean-pool spatial: (N, 16, 64, 64) -> (N, 16)
    embeddings = full_latents.mean(axis=(2, 3))

    # Compute correlation matrix
    corr_matrix = np.corrcoef(embeddings.T)  # (16, 16)

    # Compute covariance matrix
    cov_matrix = np.cov(embeddings.T)

    # Eigenvalue analysis (how much variance each PC captures)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = eigenvalues[::-1]  # Descending order
    explained_var = eigenvalues / eigenvalues.sum()
    cumulative_var = np.cumsum(explained_var)

    # Create visualization
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Channel Correlation Matrix",
            "Channel Covariance Matrix",
            "Eigenvalue Spectrum (PCA)",
            "Cumulative Variance Explained",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Correlation heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix,
            x=[f"Ch{i}" for i in range(16)],
            y=[f"Ch{i}" for i in range(16)],
            colorscale="RdBu_r",
            zmid=0,
            showscale=True,
            colorbar=dict(x=0.45, len=0.4, y=0.8),
        ),
        row=1,
        col=1,
    )

    # Covariance heatmap
    fig.add_trace(
        go.Heatmap(
            z=cov_matrix,
            x=[f"Ch{i}" for i in range(16)],
            y=[f"Ch{i}" for i in range(16)],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(x=1.0, len=0.4, y=0.8),
        ),
        row=1,
        col=2,
    )

    # Eigenvalue spectrum
    fig.add_trace(
        go.Bar(
            x=[f"PC{i}" for i in range(16)],
            y=explained_var,
            marker_color="#56B4E9",
        ),
        row=2,
        col=1,
    )

    # Cumulative variance
    fig.add_trace(
        go.Scatter(
            x=[f"PC{i}" for i in range(16)],
            y=cumulative_var,
            mode="lines+markers",
            marker_color="#E69F00",
        ),
        row=2,
        col=2,
    )

    # Add 90% threshold line
    fig.add_hline(y=0.9, line_dash="dash", line_color="red", row=2, col=2)

    fig.update_layout(
        title=dict(
            text="Channel Correlation Analysis: Are Channels Redundant?",
            x=0.5,
            font=dict(size=16),
        ),
        height=900,
        width=1000,
        template="plotly_white",
        showlegend=False,
    )

    fig.update_yaxes(title_text="Explained Variance", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Variance", row=2, col=2)

    # Find effective dimensionality (PCs needed for 90% variance)
    effective_dim = np.argmax(cumulative_var >= 0.9) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)

    print(f"  Saved channel correlation analysis to: {output_path}")
    print(f"  Effective dimensionality (90% variance): {effective_dim} out of 16 channels")
    print(f"  Top 2 PCs explain: {cumulative_var[1]:.1%} of variance")


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    # Handle --all-analysis flag
    if args.all_analysis:
        args.channel_analysis = True
        args.per_channel = True
        args.discriminability = True
        args.spatial_discriminability = True
        args.neighbors = True
        args.correlations = True

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine cache path
    if args.save_cache:
        cache_path = Path(args.save_cache)
    else:
        cache_path = output_path.with_suffix(".npz")

    # Load or extract embeddings
    if args.load_cache:
        # Load from cache
        full_latents, samples, thumbnails = load_cache(Path(args.load_cache))
        if thumbnails is None:
            thumbnails = [""] * len(samples)  # No thumbnails available
    else:
        # Extract embeddings
        # 1. Load Cosmos VAE
        encoder = load_cosmos_vae_standalone(device=args.device)

        # 2. Load SpatialRGPT-Bench
        samples = load_spatial_rgpt_bench(
            num_samples=args.num_samples,
            draw_bboxes=args.draw_bboxes,
        )

        # 3. Encode thumbnails (before encoding, while images are available)
        thumbnails = encode_thumbnails(samples, args.thumbnail_size)

        # 4. Extract embeddings
        full_latents = extract_embeddings(
            encoder=encoder,
            samples=samples,
            batch_size=args.batch_size,
        )

        # 5. Save cache
        save_cache(cache_path, full_latents, samples, thumbnails)

    # Mean-pool for dimensionality reduction
    embeddings = full_latents.mean(axis=(2, 3))  # (N, 16)
    print(f"Mean-pooled embeddings shape: {embeddings.shape}")

    # Dimensionality reduction
    coords_dict = reduce_dimensions(embeddings, method=args.reduction)

    # Create visualization
    create_interactive_visualization(
        coords_dict=coords_dict,
        samples=samples,
        thumbnails=thumbnails,
        thumbnail_size=args.thumbnail_size,
        output_path=output_path,
    )

    # Optional channel analysis
    if args.channel_analysis:
        analyze_channels(
            full_latents=full_latents,
            samples=samples,
            output_dir=output_path.parent,
        )

    # Optional per-channel visualization
    if args.per_channel:
        per_channel_path = output_path.parent / "per_channel_visualization.html"
        visualize_per_channel(
            full_latents=full_latents,
            samples=samples,
            thumbnails=thumbnails,
            thumbnail_size=args.thumbnail_size,
            output_path=per_channel_path,
            reduction=args.reduction if args.reduction != "all" else "umap",
        )

    # Optional discriminability analysis
    if args.discriminability:
        discrim_path = output_path.parent / "channel_discriminability.html"
        analyze_channel_discriminability(
            full_latents=full_latents,
            samples=samples,
            output_path=discrim_path,
        )

    # Optional spatial discriminability analysis
    if args.spatial_discriminability:
        spatial_path = output_path.parent / "spatial_discriminability.html"
        analyze_spatial_discriminability(
            full_latents=full_latents,
            samples=samples,
            output_path=spatial_path,
        )

    # Optional nearest neighbors analysis
    if args.neighbors:
        neighbors_path = output_path.parent / "nearest_neighbors.html"
        analyze_nearest_neighbors(
            full_latents=full_latents,
            samples=samples,
            output_path=neighbors_path,
            k=10,
        )

    # Optional channel correlations analysis
    if args.correlations:
        correlations_path = output_path.parent / "channel_correlations.html"
        analyze_channel_correlations(
            full_latents=full_latents,
            samples=samples,
            output_path=correlations_path,
        )

    print("\nDone!")
    print(f"  Main visualization: {output_path}")
    print(f"  Cache file: {cache_path}")
    if args.channel_analysis:
        print(f"  Channel analysis: {output_path.parent}/channel_*.html")
    if args.per_channel:
        print(f"  Per-channel: {output_path.parent}/per_channel_visualization.html")
    if args.discriminability:
        print(f"  Discriminability: {output_path.parent}/channel_discriminability.html")
    if args.spatial_discriminability:
        print(f"  Spatial discriminability: {output_path.parent}/spatial_discriminability*.html")
    if args.neighbors:
        print(f"  Nearest neighbors: {output_path.parent}/nearest_neighbors.html")
    if args.correlations:
        print(f"  Channel correlations: {output_path.parent}/channel_correlations.html")


if __name__ == "__main__":
    main()
