"""Comprehensive analysis of Cosmos VAE embedding extraction methods.

This script systematically compares different methods for extracting embeddings
from Cosmos VAE latents and evaluates their discriminability for spatial
reasoning tasks using SpatialRGPT-Bench categories.

Methods compared:
- Spatial pooling: mean, max, std, patches
- Channel reduction: PCA
- Projection modes: spatial vs channel

Metrics computed:
- Discriminability: silhouette, F-ratio, Davies-Bouldin, Calinski-Harabasz
- Information: effective dimensionality, entropy
- Nearest neighbor: k-NN accuracy

Usage:
    # Full analysis from cached latents
    python scripts/visualize/analyze_cosmos_extraction_methods.py \
        --load-cache outputs/cosmos_spatial_full.npz \
        --output outputs/extraction_analysis/

    # Quick test with subset of methods
    python scripts/visualize/analyze_cosmos_extraction_methods.py \
        --load-cache outputs/cosmos_spatial_full.npz \
        --methods mode_mean,mode_max,mode_std \
        --output outputs/quick_analysis/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "python"))

from theworld.analysis.embedding_methods import (
    EXTRACTION_METHODS,
    EmbeddingExtractor,
    get_method_info,
)
from theworld.analysis.embedding_metrics import (
    compute_all_metrics,
    compute_channel_importance,
    compute_pairwise_distances,
    compute_per_class_metrics,
    compute_spatial_importance,
)


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

GROUP_COLORS = {
    "Size": "#E69F00",
    "Depth": "#56B4E9",
    "Horizontal": "#009E73",
    "Vertical": "#F0E442",
    "Measurements": "#CC79A7",
    "Unknown": "#999999",
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
        description="Analyze Cosmos VAE embedding extraction methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--load-cache",
        type=str,
        required=True,
        help="Path to cached latents .npz file (required)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of methods to test (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/extraction_analysis",
        help="Output directory path",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip generating visualizations (metrics only)",
    )
    parser.add_argument(
        "--skip-deep-dive",
        action="store_true",
        help="Skip deep dive analysis (channel/spatial importance)",
    )
    return parser.parse_args()


# =============================================================================
# Data Loading
# =============================================================================


def load_cache(cache_path: Path) -> tuple:
    """Load cached latents and metadata."""
    print(f"Loading cache from {cache_path}...")
    data = np.load(cache_path, allow_pickle=True)

    latents = data["latents"]  # (N, 16, 64, 64)
    metadata = json.loads(str(data["metadata"]))

    # Extract categories and groups
    categories = [m["qa_category"] for m in metadata]
    groups = [get_category_group(c) for c in categories]

    print(f"  Loaded {len(metadata)} samples")
    print(f"  Latents shape: {latents.shape}")
    print(f"  Categories: {len(set(categories))} unique")
    print(f"  Groups: {len(set(groups))} unique")

    return latents, np.array(categories), np.array(groups), metadata


# =============================================================================
# Analysis Functions
# =============================================================================


def analyze_method(
    extractor: EmbeddingExtractor,
    latents: np.ndarray,
    categories: np.ndarray,
    groups: np.ndarray,
    method: str,
) -> Dict[str, Any]:
    """Analyze a single extraction method.

    Returns dict with:
    - embeddings: extracted embeddings
    - metrics: computed metrics
    - info: method info
    """
    print(f"\n  Analyzing method: {method}")

    # Extract embeddings
    embeddings = extractor.extract(latents, method)
    print(f"    Extracted shape: {embeddings.shape}")

    # Compute metrics
    metrics = compute_all_metrics(
        embeddings=embeddings,
        labels=categories,
        groups=groups,
        k=10,
    )

    # Add pairwise distance metrics
    dist_metrics = compute_pairwise_distances(embeddings, categories)
    metrics.update(dist_metrics)

    # Add method info
    info = get_method_info(method)
    metrics["method"] = method
    metrics["description"] = info["description"]
    metrics["output_shape"] = info["output_shape"]
    metrics["n_tokens"] = info["n_tokens"]

    return {
        "embeddings": embeddings,
        "metrics": metrics,
        "info": info,
    }


def create_metrics_table(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison table of all methods."""
    rows = []
    for method, result in all_results.items():
        row = result["metrics"].copy()
        rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns
    priority_cols = [
        "method",
        "n_dimensions",
        "n_tokens",
        "silhouette",
        "f_ratio",
        "knn_category_acc",
        "knn_group_acc",
        "effective_dim_90",
        "distance_ratio",
    ]
    other_cols = [c for c in df.columns if c not in priority_cols]
    df = df[[c for c in priority_cols if c in df.columns] + other_cols]

    return df


# =============================================================================
# Visualization Functions
# =============================================================================


def create_comparison_bar_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Create bar chart comparing key metrics across methods."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metrics_to_plot = [
        ("silhouette", "Silhouette Score", "higher is better"),
        ("f_ratio", "F-Ratio", "higher is better"),
        ("knn_category_acc", "k-NN Category Accuracy", "higher is better"),
        ("effective_dim_90", "Effective Dimensionality (90%)", "lower means compressed"),
        ("distance_ratio", "Inter/Intra Distance Ratio", "higher is better"),
    ]

    # Filter to metrics that exist
    metrics_to_plot = [(m, t, d) for m, t, d in metrics_to_plot if m in df.columns]

    fig = make_subplots(
        rows=len(metrics_to_plot),
        cols=1,
        subplot_titles=[f"{title} ({desc})" for _, title, desc in metrics_to_plot],
        vertical_spacing=0.08,
    )

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
    ]

    for i, (metric, title, _) in enumerate(metrics_to_plot, 1):
        values = df[metric].values
        methods = df["method"].values

        # Sort by value for better visualization
        sorted_idx = np.argsort(values)[::-1]
        sorted_values = values[sorted_idx]
        sorted_methods = methods[sorted_idx]

        fig.add_trace(
            go.Bar(
                x=sorted_methods,
                y=sorted_values,
                marker_color=[colors[j % len(colors)] for j in range(len(sorted_methods))],
                showlegend=False,
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        title=dict(
            text="Embedding Extraction Method Comparison",
            x=0.5,
            font=dict(size=20),
        ),
        height=300 * len(metrics_to_plot),
        width=1000,
        template="plotly_white",
    )

    fig.write_html(str(output_path))
    print(f"  Saved comparison bar chart: {output_path}")


def create_pareto_frontier(df: pd.DataFrame, output_path: Path) -> None:
    """Create Pareto frontier plot: discriminability vs token count."""
    import plotly.express as px

    # Use silhouette as discriminability metric
    df_plot = df.copy()
    df_plot["log_tokens"] = np.log10(df_plot["n_tokens"].clip(lower=1))

    fig = px.scatter(
        df_plot,
        x="n_tokens",
        y="silhouette",
        text="method",
        color="f_ratio",
        size="knn_category_acc",
        hover_data=["n_dimensions", "effective_dim_90"],
        title="Pareto Frontier: Discriminability vs Token Count",
        labels={
            "n_tokens": "Number of Tokens (log scale)",
            "silhouette": "Silhouette Score",
            "f_ratio": "F-Ratio",
        },
    )

    fig.update_traces(textposition="top center")
    fig.update_xaxes(type="log")
    fig.update_layout(
        height=600,
        width=900,
        template="plotly_white",
    )

    fig.write_html(str(output_path))
    print(f"  Saved Pareto frontier: {output_path}")


def create_radar_chart(df: pd.DataFrame, output_path: Path) -> None:
    """Create radar chart comparing method profiles."""
    import plotly.graph_objects as go

    # Select metrics for radar (normalize to 0-1 range)
    radar_metrics = ["silhouette", "knn_category_acc", "distance_ratio"]

    # Normalize metrics
    df_norm = df.copy()
    for metric in radar_metrics:
        if metric in df_norm.columns:
            min_val = df_norm[metric].min()
            max_val = df_norm[metric].max()
            if max_val > min_val:
                df_norm[f"{metric}_norm"] = (df_norm[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[f"{metric}_norm"] = 0.5

    # Add inverted metrics (lower is better)
    if "effective_dim_90" in df_norm.columns:
        max_dim = df_norm["effective_dim_90"].max()
        df_norm["compression_norm"] = 1 - (df_norm["effective_dim_90"] / max_dim)
        radar_metrics.append("compression")

    if "n_tokens" in df_norm.columns:
        max_tokens = df_norm["n_tokens"].max()
        df_norm["efficiency_norm"] = 1 - (np.log10(df_norm["n_tokens"].clip(lower=1)) / np.log10(max_tokens + 1))
        radar_metrics.append("efficiency")

    fig = go.Figure()

    # Add trace for each method
    for idx, row in df_norm.iterrows():
        values = []
        for metric in radar_metrics:
            norm_col = f"{metric}_norm"
            if norm_col in df_norm.columns:
                values.append(row[norm_col])
            elif metric == "compression":
                values.append(row.get("compression_norm", 0.5))
            elif metric == "efficiency":
                values.append(row.get("efficiency_norm", 0.5))

        # Close the polygon
        values.append(values[0])
        categories = radar_metrics + [radar_metrics[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                name=row["method"],
                fill="toself",
                opacity=0.5,
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=dict(
            text="Method Profiles (Normalized Metrics)",
            x=0.5,
            font=dict(size=16),
        ),
        height=600,
        width=800,
    )

    fig.write_html(str(output_path))
    print(f"  Saved radar chart: {output_path}")


def create_per_method_umap(
    embeddings: np.ndarray,
    categories: np.ndarray,
    groups: np.ndarray,
    method: str,
    output_path: Path,
) -> None:
    """Create UMAP visualization for a single method."""
    import plotly.express as px

    # Skip for very high-dimensional embeddings
    if embeddings.shape[1] > 10000:
        print(f"    Skipping UMAP for {method} (too high dimensional: {embeddings.shape[1]})")
        return

    try:
        import umap

        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, verbose=False)
        coords = reducer.fit_transform(embeddings)

        df_plot = pd.DataFrame(
            {
                "UMAP1": coords[:, 0],
                "UMAP2": coords[:, 1],
                "category": categories,
                "group": groups,
            }
        )

        fig = px.scatter(
            df_plot,
            x="UMAP1",
            y="UMAP2",
            color="group",
            hover_data=["category"],
            title=f"UMAP: {method}",
            color_discrete_map=GROUP_COLORS,
        )

        fig.update_layout(
            height=500,
            width=600,
            template="plotly_white",
        )

        fig.write_html(str(output_path))
        print(f"    Saved UMAP: {output_path}")

    except Exception as e:
        print(f"    Failed to create UMAP for {method}: {e}")


def create_heatmap_methods_vs_categories(
    all_results: Dict[str, Dict],
    categories: np.ndarray,
    output_path: Path,
) -> None:
    """Create heatmap showing per-category performance for each method."""
    import plotly.graph_objects as go

    unique_cats = sorted(set(categories))
    methods = list(all_results.keys())

    # Compute per-category k-NN accuracy for each method
    heatmap_data = np.zeros((len(methods), len(unique_cats)))

    for i, method in enumerate(methods):
        embeddings = all_results[method]["embeddings"]

        for j, cat in enumerate(unique_cats):
            mask = categories == cat
            cat_embeddings = embeddings[mask]

            if len(cat_embeddings) > 1:
                # Compute same-category neighbor rate for this category
                from sklearn.neighbors import NearestNeighbors

                k = min(10, len(cat_embeddings) - 1)
                nn = NearestNeighbors(n_neighbors=k + 1)
                nn.fit(embeddings)
                _, indices = nn.kneighbors(cat_embeddings)

                rates = []
                for idx in range(len(cat_embeddings)):
                    neighbor_cats = categories[indices[idx, 1:]]
                    same_cat = np.sum(neighbor_cats == cat)
                    rates.append(same_cat / k)

                heatmap_data[i, j] = np.mean(rates)

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data,
            x=unique_cats,
            y=methods,
            colorscale="Viridis",
            colorbar=dict(title="k-NN Same-Cat Rate"),
        )
    )

    fig.update_layout(
        title=dict(
            text="Per-Category Performance by Method",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(title="Category", tickangle=45),
        yaxis=dict(title="Method"),
        height=max(400, len(methods) * 30),
        width=max(800, len(unique_cats) * 50),
        template="plotly_white",
    )

    fig.write_html(str(output_path))
    print(f"  Saved heatmap: {output_path}")


def create_channel_importance_chart(
    latents: np.ndarray,
    categories: np.ndarray,
    output_path: Path,
) -> None:
    """Create channel importance visualization."""
    import plotly.graph_objects as go

    print("  Computing channel importance...")
    importance = compute_channel_importance(latents, categories)

    fig = go.Figure(
        data=go.Bar(
            x=[f"Ch{i}" for i in range(len(importance))],
            y=importance,
            marker_color="#56B4E9",
        )
    )

    # Add annotations for top channels
    sorted_idx = np.argsort(importance)[::-1]
    top_3 = sorted_idx[:3]

    fig.update_layout(
        title=dict(
            text=f"Channel Importance (F-ratio per channel)<br>Top 3: {', '.join([f'Ch{i}' for i in top_3])}",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(title="Latent Channel"),
        yaxis=dict(title="F-Ratio (discriminability)"),
        height=400,
        width=700,
        template="plotly_white",
    )

    fig.write_html(str(output_path))
    print(f"  Saved channel importance: {output_path}")


def create_spatial_importance_heatmap(
    latents: np.ndarray,
    categories: np.ndarray,
    output_path: Path,
) -> None:
    """Create spatial importance heatmap."""
    import plotly.graph_objects as go

    print("  Computing spatial importance (this may take a moment)...")
    importance = compute_spatial_importance(latents, categories)

    fig = go.Figure(
        data=go.Heatmap(
            z=importance,
            colorscale="Viridis",
            colorbar=dict(title="F-Ratio"),
        )
    )

    # Find best position
    best_pos = np.unravel_index(np.argmax(importance), importance.shape)

    fig.update_layout(
        title=dict(
            text=f"Spatial Importance (F-ratio per position)<br>Best position: ({best_pos[0]}, {best_pos[1]})",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(title="Spatial X"),
        yaxis=dict(title="Spatial Y"),
        height=500,
        width=550,
        template="plotly_white",
    )

    fig.write_html(str(output_path))
    print(f"  Saved spatial importance: {output_path}")


def create_summary_report(
    df: pd.DataFrame,
    all_results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Create interactive HTML summary report."""
    # Find best methods for each metric
    best_silhouette = df.loc[df["silhouette"].idxmax(), "method"]
    best_f_ratio = df.loc[df["f_ratio"].idxmax(), "method"]
    best_knn = df.loc[df["knn_category_acc"].idxmax(), "method"]
    most_compressed = df.loc[df["effective_dim_90"].idxmin(), "method"]
    fewest_tokens = df.loc[df["n_tokens"].idxmin(), "method"]

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cosmos VAE Embedding Extraction Analysis</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 10px 0; }}
        .best {{ color: #28a745; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4a90d9; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .iframe-container {{ margin: 20px 0; }}
        iframe {{ border: 1px solid #ddd; border-radius: 8px; }}
        .findings {{ background: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .key-finding {{ margin: 10px 0; padding: 10px; background: white; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Cosmos VAE Embedding Extraction Analysis</h1>

    <div class="findings">
        <h2>Key Findings</h2>
        <div class="key-finding">
            <strong>Best Silhouette Score:</strong> <span class="best">{best_silhouette}</span>
            ({df.loc[df['method']==best_silhouette, 'silhouette'].values[0]:.4f})
        </div>
        <div class="key-finding">
            <strong>Best F-Ratio:</strong> <span class="best">{best_f_ratio}</span>
            ({df.loc[df['method']==best_f_ratio, 'f_ratio'].values[0]:.4f})
        </div>
        <div class="key-finding">
            <strong>Best k-NN Accuracy:</strong> <span class="best">{best_knn}</span>
            ({df.loc[df['method']==best_knn, 'knn_category_acc'].values[0]:.4f})
        </div>
        <div class="key-finding">
            <strong>Most Compressed:</strong> <span class="best">{most_compressed}</span>
            (effective dim: {df.loc[df['method']==most_compressed, 'effective_dim_90'].values[0]})
        </div>
        <div class="key-finding">
            <strong>Fewest Tokens:</strong> <span class="best">{fewest_tokens}</span>
            ({df.loc[df['method']==fewest_tokens, 'n_tokens'].values[0]} tokens)
        </div>
    </div>

    <h2>Methods Tested: {len(df)}</h2>
    <p>Total samples analyzed: {df['n_samples'].iloc[0]}</p>

    <h2>Metrics Comparison Table</h2>
    {df.to_html(index=False, float_format='%.4f')}

    <h2>Visualizations</h2>
    <div class="iframe-container">
        <h3>Metric Comparison</h3>
        <iframe src="comparisons/metric_bars.html" width="100%" height="1200px"></iframe>
    </div>

    <div class="iframe-container">
        <h3>Pareto Frontier</h3>
        <iframe src="comparisons/pareto_frontier.html" width="100%" height="650px"></iframe>
    </div>

    <div class="iframe-container">
        <h3>Method Profiles</h3>
        <iframe src="comparisons/radar_chart.html" width="100%" height="650px"></iframe>
    </div>

    <h2>Research Questions Answered</h2>
    <div class="findings">
        <div class="key-finding">
            <strong>Q: Is mean pooling sufficient, or is spatial structure needed?</strong><br>
            Compare mode_mean vs mode_flatten vs mode_patch* metrics above.
        </div>
        <div class="key-finding">
            <strong>Q: Are all 16 channels necessary?</strong><br>
            Compare mode_mean vs mode_mean_pca8 vs mode_mean_pca4 metrics above.
        </div>
        <div class="key-finding">
            <strong>Q: What's the optimal token count?</strong><br>
            See Pareto frontier for discriminability vs tokens tradeoff.
        </div>
    </div>

</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"  Saved summary report: {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    # Create output directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_method").mkdir(exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)
    (output_dir / "deep_dive").mkdir(exist_ok=True)

    # Load cached latents
    latents, categories, groups, metadata = load_cache(Path(args.load_cache))

    # Determine methods to test
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
        # Validate methods
        invalid = [m for m in methods if m not in EXTRACTION_METHODS]
        if invalid:
            print(f"Warning: Unknown methods ignored: {invalid}")
            methods = [m for m in methods if m in EXTRACTION_METHODS]
    else:
        methods = list(EXTRACTION_METHODS.keys())

    print(f"\nAnalyzing {len(methods)} extraction methods...")

    # Initialize extractor
    extractor = EmbeddingExtractor()

    # Analyze each method
    all_results = {}
    for method in tqdm(methods, desc="Methods"):
        result = analyze_method(extractor, latents, categories, groups, method)
        all_results[method] = result

    # Create metrics table
    print("\nCreating metrics table...")
    df = create_metrics_table(all_results)

    # Save metrics as CSV
    csv_path = output_dir / "metrics_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved metrics CSV: {csv_path}")

    # Save metrics as JSON
    json_path = output_dir / "metrics_table.json"
    df.to_json(json_path, orient="records", indent=2)
    print(f"  Saved metrics JSON: {json_path}")

    # Create visualizations
    if not args.skip_viz:
        print("\nCreating visualizations...")

        # Comparison charts
        create_comparison_bar_chart(df, output_dir / "comparisons" / "metric_bars.html")
        create_pareto_frontier(df, output_dir / "comparisons" / "pareto_frontier.html")
        create_radar_chart(df, output_dir / "comparisons" / "radar_chart.html")
        create_heatmap_methods_vs_categories(
            all_results, categories, output_dir / "comparisons" / "category_heatmap.html"
        )

        # Per-method UMAP visualizations
        print("\nCreating per-method visualizations...")
        for method, result in tqdm(all_results.items(), desc="Per-method viz"):
            method_dir = output_dir / "per_method" / method
            method_dir.mkdir(exist_ok=True)

            # Save method metrics
            with open(method_dir / "metrics.json", "w") as f:
                json.dump(result["metrics"], f, indent=2, default=str)

            # Create UMAP
            create_per_method_umap(
                result["embeddings"],
                categories,
                groups,
                method,
                method_dir / "umap.html",
            )

    # Deep dive analysis
    if not args.skip_deep_dive:
        print("\nCreating deep dive analysis...")
        create_channel_importance_chart(latents, categories, output_dir / "deep_dive" / "channel_importance.html")
        create_spatial_importance_heatmap(latents, categories, output_dir / "deep_dive" / "spatial_importance.html")

    # Create summary report
    print("\nCreating summary report...")
    create_summary_report(df, all_results, output_dir / "summary_report.html")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Methods analyzed: {len(methods)}")
    print(f"Samples analyzed: {len(categories)}")

    print("\n--- TOP METHODS BY METRIC ---")
    for metric in ["silhouette", "f_ratio", "knn_category_acc"]:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_method = df.loc[best_idx, "method"]
            best_value = df.loc[best_idx, metric]
            print(f"  {metric}: {best_method} ({best_value:.4f})")

    print(f"\nSummary report: {output_dir / 'summary_report.html'}")


if __name__ == "__main__":
    main()
