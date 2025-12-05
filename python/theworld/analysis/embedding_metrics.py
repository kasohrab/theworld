"""Metrics for evaluating embedding quality.

This module provides comprehensive metrics for analyzing how well
embeddings separate spatial reasoning categories.

Metrics include:
- Discriminability: silhouette, F-ratio, Davies-Bouldin, Calinski-Harabasz
- Information: effective dimensionality, entropy
- Nearest neighbor: k-NN accuracy, retrieval mAP
- Practical: dimensions, extraction time
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


def compute_all_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    k: int = 10,
    sample_size: Optional[int] = None,
) -> Dict[str, float]:
    """Compute all embedding quality metrics.

    Args:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of category labels
        groups: (N,) array of group labels (optional, for group-level metrics)
        k: Number of neighbors for k-NN analysis
        sample_size: Subsample for expensive metrics (None = use all)

    Returns:
        Dict of metric name -> value
    """
    N, D = embeddings.shape

    # Encode labels to integers
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    n_classes = len(le.classes_)

    # Optionally subsample for expensive metrics
    if sample_size is not None and sample_size < N:
        idx = np.random.choice(N, sample_size, replace=False)
        emb_sample = embeddings[idx]
        lab_sample = labels_encoded[idx]
    else:
        emb_sample = embeddings
        lab_sample = labels_encoded

    metrics = {}

    # ==========================================================================
    # Discriminability Metrics
    # ==========================================================================

    # Silhouette score: measures cluster cohesion and separation
    try:
        metrics["silhouette"] = silhouette_score(emb_sample, lab_sample, sample_size=min(1000, len(lab_sample)))
    except Exception:
        metrics["silhouette"] = 0.0

    # F-ratio: between-class variance / within-class variance
    metrics["f_ratio"] = compute_f_ratio(embeddings, labels_encoded)

    # Davies-Bouldin index: lower is better (cluster similarity)
    try:
        metrics["davies_bouldin"] = davies_bouldin_score(emb_sample, lab_sample)
    except Exception:
        metrics["davies_bouldin"] = float("inf")

    # Calinski-Harabasz index: higher is better (variance ratio)
    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(emb_sample, lab_sample)
    except Exception:
        metrics["calinski_harabasz"] = 0.0

    # ==========================================================================
    # Information Metrics
    # ==========================================================================

    # Effective dimensionality (PCs for 90% variance)
    metrics["effective_dim_90"] = compute_effective_dim(embeddings, threshold=0.90)
    metrics["effective_dim_95"] = compute_effective_dim(embeddings, threshold=0.95)
    metrics["effective_dim_99"] = compute_effective_dim(embeddings, threshold=0.99)

    # Channel/feature entropy
    metrics["feature_entropy"] = compute_feature_entropy(embeddings)

    # ==========================================================================
    # Nearest Neighbor Metrics
    # ==========================================================================

    # k-NN category accuracy
    knn_cat_acc, knn_cat_rates = compute_knn_accuracy(embeddings, labels_encoded, k=k)
    metrics["knn_category_acc"] = knn_cat_acc
    metrics["knn_category_rate_mean"] = np.mean(knn_cat_rates)
    metrics["knn_category_rate_median"] = np.median(knn_cat_rates)

    # k-NN group accuracy (if groups provided)
    if groups is not None:
        le_group = LabelEncoder()
        groups_encoded = le_group.fit_transform(groups)
        knn_group_acc, knn_group_rates = compute_knn_accuracy(embeddings, groups_encoded, k=k)
        metrics["knn_group_acc"] = knn_group_acc
        metrics["knn_group_rate_mean"] = np.mean(knn_group_rates)

    # Random baseline
    metrics["random_baseline_category"] = 1.0 / n_classes
    if groups is not None:
        n_groups = len(np.unique(groups))
        metrics["random_baseline_group"] = 1.0 / n_groups

    # ==========================================================================
    # Practical Metrics
    # ==========================================================================

    metrics["n_samples"] = N
    metrics["n_dimensions"] = D
    metrics["n_classes"] = n_classes

    return metrics


def compute_f_ratio(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute F-ratio (between-class / within-class variance).

    Higher F-ratio indicates better class separation.
    """
    unique_labels = np.unique(labels)
    overall_mean = embeddings.mean(axis=0)

    between_var = 0.0
    within_var = 0.0

    for label in unique_labels:
        mask = labels == label
        class_data = embeddings[mask]
        class_mean = class_data.mean(axis=0)
        n_class = mask.sum()

        # Between-class variance: n_k * ||mean_k - overall_mean||^2
        between_var += n_class * np.sum((class_mean - overall_mean) ** 2)

        # Within-class variance: sum of ||x - mean_k||^2
        within_var += np.sum((class_data - class_mean) ** 2)

    return between_var / (within_var + 1e-8)


def compute_effective_dim(embeddings: np.ndarray, threshold: float = 0.90) -> int:
    """Compute effective dimensionality (PCs needed for threshold variance).

    Args:
        embeddings: (N, D) array
        threshold: Cumulative variance threshold (default 0.90)

    Returns:
        Number of principal components needed
    """
    n_components = min(embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    effective_dim = np.argmax(cumulative_var >= threshold) + 1

    return int(effective_dim)


def compute_feature_entropy(embeddings: np.ndarray, n_bins: int = 50) -> float:
    """Compute average entropy across features.

    Higher entropy indicates more diverse/spread activations.
    """
    entropies = []
    for i in range(embeddings.shape[1]):
        feature = embeddings[:, i]
        # Discretize into bins
        hist, _ = np.histogram(feature, bins=n_bins, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist))
        entropies.append(entropy)

    return float(np.mean(entropies))


def compute_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> Tuple[float, np.ndarray]:
    """Compute k-NN same-class accuracy.

    Args:
        embeddings: (N, D) array
        labels: (N,) array of class labels
        k: Number of neighbors

    Returns:
        (overall_accuracy, per_sample_rates) tuple
    """
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)

    per_sample_rates = []
    for i in range(len(embeddings)):
        neighbor_indices = indices[i, 1:]  # Exclude self
        neighbor_labels = labels[neighbor_indices]
        same_class = np.sum(neighbor_labels == labels[i])
        per_sample_rates.append(same_class / k)

    per_sample_rates = np.array(per_sample_rates)
    return float(np.mean(per_sample_rates)), per_sample_rates


def compute_per_class_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for each class separately.

    Returns dict mapping class label to metrics dict.
    """
    unique_labels = np.unique(labels)
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    per_class = {}
    for i, label in enumerate(unique_labels):
        mask = labels == label
        class_embeddings = embeddings[mask]

        per_class[str(label)] = {
            "count": int(mask.sum()),
            "mean_norm": float(np.linalg.norm(class_embeddings.mean(axis=0))),
            "intra_class_std": float(class_embeddings.std()),
            "intra_class_variance": float(np.var(class_embeddings)),
        }

    return per_class


def compute_pairwise_distances(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Compute inter-class and intra-class distances.

    Returns:
        Dict with mean_intra_distance, mean_inter_distance, distance_ratio
    """
    unique_labels = np.unique(labels)

    # Compute class centroids
    centroids = {}
    for label in unique_labels:
        mask = labels == label
        centroids[label] = embeddings[mask].mean(axis=0)

    # Intra-class distances (average distance to centroid)
    intra_distances = []
    for label in unique_labels:
        mask = labels == label
        class_embeddings = embeddings[mask]
        centroid = centroids[label]
        distances = np.linalg.norm(class_embeddings - centroid, axis=1)
        intra_distances.extend(distances.tolist())

    # Inter-class distances (between centroids)
    inter_distances = []
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i < j:
                dist = np.linalg.norm(centroids[label_i] - centroids[label_j])
                inter_distances.append(dist)

    mean_intra = np.mean(intra_distances) if intra_distances else 0.0
    mean_inter = np.mean(inter_distances) if inter_distances else 0.0

    return {
        "mean_intra_distance": float(mean_intra),
        "mean_inter_distance": float(mean_inter),
        "distance_ratio": float(mean_inter / (mean_intra + 1e-8)),
    }


def compute_channel_importance(
    latents: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute importance score for each latent channel.

    Uses F-ratio per channel to rank importance.

    Args:
        latents: (N, 16, H, W) raw latents
        labels: (N,) class labels

    Returns:
        (16,) array of importance scores (higher = more discriminative)
    """
    N, C, H, W = latents.shape

    # Mean-pool spatial dimensions
    channel_values = latents.mean(axis=(2, 3))  # (N, C)

    # Compute F-ratio per channel
    importance = []
    for c in range(C):
        f_ratio = compute_f_ratio(channel_values[:, c : c + 1], labels)
        importance.append(f_ratio)

    return np.array(importance)


def compute_spatial_importance(
    latents: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute importance score for each spatial position.

    Uses F-ratio per position to create discriminability heatmap.

    Args:
        latents: (N, 16, H, W) raw latents
        labels: (N,) class labels

    Returns:
        (H, W) array of importance scores
    """
    N, C, H, W = latents.shape
    importance = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            # Get all channels at this position: (N, C)
            pos_data = latents[:, :, i, j]
            f_ratio = compute_f_ratio(pos_data, labels)
            importance[i, j] = f_ratio

    return importance
