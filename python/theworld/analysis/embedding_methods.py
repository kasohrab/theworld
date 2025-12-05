"""Embedding extraction methods for Cosmos VAE latents.

This module provides different methods for extracting and aggregating
embeddings from Cosmos VAE latent tensors (N, 16, 64, 64).

Methods include:
- VAE output extraction: mode, sample, raw
- Spatial aggregation: mean, max, std, patches, flatten
- Channel reduction: PCA, top-k discriminative
"""

from typing import Dict, Literal, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


# =============================================================================
# Method Registry
# =============================================================================

EXTRACTION_METHODS = {
    # Spatial pooling methods (apply to mode latents)
    "mode_mean": "Global average pooling → (N, 16)",
    "mode_max": "Global max pooling → (N, 16)",
    "mode_std": "Global std dev → (N, 16)",
    "mode_meanstd": "Concat mean and std → (N, 32)",
    "mode_minmax": "Concat min and max → (N, 32)",
    # Patch pooling
    "mode_patch4": "4×4 spatial patches → (N, 256)",
    "mode_patch8": "8×8 spatial patches → (N, 1024)",
    "mode_patch16": "16×16 spatial patches → (N, 4096)",
    # Flatten
    "mode_flatten": "Full flatten → (N, 65536)",
    # Channel reduction (applied after mean pooling)
    "mode_mean_pca8": "Mean pool → PCA(8) → (N, 8)",
    "mode_mean_pca4": "Mean pool → PCA(4) → (N, 4)",
    "mode_mean_pca2": "Mean pool → PCA(2) → (N, 2)",
    # Spatial projection modes (for TheWorld integration)
    "spatial_flat": "Spatial mode: (N, 4096, 16) flattened",
    "channel_flat": "Channel mode: (N, 16, 4096) flattened",
}


class EmbeddingExtractor:
    """Extract embeddings from Cosmos VAE latents using various methods."""

    def __init__(self):
        """Initialize the extractor."""
        self._pca_models: Dict[int, PCA] = {}

    def get_available_methods(self) -> Dict[str, str]:
        """Return available extraction methods with descriptions."""
        return EXTRACTION_METHODS.copy()

    def extract(
        self,
        latents: np.ndarray,
        method: str,
        fit_pca: bool = True,
    ) -> np.ndarray:
        """Extract embeddings using the specified method.

        Args:
            latents: Raw latents from Cosmos VAE, shape (N, 16, 64, 64)
            method: Extraction method name (see EXTRACTION_METHODS)
            fit_pca: Whether to fit PCA models (True for training, False for inference)

        Returns:
            Extracted embeddings with method-specific shape
        """
        if method not in EXTRACTION_METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(EXTRACTION_METHODS.keys())}")

        # Validate input shape
        if latents.ndim != 4:
            raise ValueError(f"Expected 4D latents (N, C, H, W), got shape {latents.shape}")

        N, C, H, W = latents.shape

        # Spatial pooling methods
        if method == "mode_mean":
            return self._mean_pool(latents)
        elif method == "mode_max":
            return self._max_pool(latents)
        elif method == "mode_std":
            return self._std_pool(latents)
        elif method == "mode_meanstd":
            return self._meanstd_pool(latents)
        elif method == "mode_minmax":
            return self._minmax_pool(latents)

        # Patch pooling methods
        elif method == "mode_patch4":
            return self._patch_pool(latents, patch_size=16)  # 64/16 = 4
        elif method == "mode_patch8":
            return self._patch_pool(latents, patch_size=8)  # 64/8 = 8
        elif method == "mode_patch16":
            return self._patch_pool(latents, patch_size=4)  # 64/4 = 16

        # Flatten
        elif method == "mode_flatten":
            return latents.reshape(N, -1)

        # PCA reduction (applied after mean pooling)
        elif method == "mode_mean_pca8":
            mean_pooled = self._mean_pool(latents)
            return self._pca_reduce(mean_pooled, n_components=8, fit=fit_pca)
        elif method == "mode_mean_pca4":
            mean_pooled = self._mean_pool(latents)
            return self._pca_reduce(mean_pooled, n_components=4, fit=fit_pca)
        elif method == "mode_mean_pca2":
            mean_pooled = self._mean_pool(latents)
            return self._pca_reduce(mean_pooled, n_components=2, fit=fit_pca)

        # Spatial projection modes
        elif method == "spatial_flat":
            # (N, 16, 64, 64) -> (N, 64, 64, 16) -> (N, 4096, 16) -> (N, 65536)
            spatial = np.transpose(latents, (0, 2, 3, 1))  # (N, H, W, C)
            spatial = spatial.reshape(N, H * W, C)  # (N, 4096, 16)
            return spatial.reshape(N, -1)  # (N, 65536)
        elif method == "channel_flat":
            # (N, 16, 64, 64) -> (N, 16, 4096) -> (N, 65536)
            channel = latents.reshape(N, C, H * W)  # (N, 16, 4096)
            return channel.reshape(N, -1)  # (N, 65536)

        else:
            raise ValueError(f"Method {method} not implemented")

    def extract_all(
        self,
        latents: np.ndarray,
        methods: Optional[list] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract embeddings using all specified methods.

        Args:
            latents: Raw latents from Cosmos VAE, shape (N, 16, 64, 64)
            methods: List of method names, or None for all methods

        Returns:
            Dict mapping method name to extracted embeddings
        """
        if methods is None:
            methods = list(EXTRACTION_METHODS.keys())

        results = {}
        for method in methods:
            try:
                results[method] = self.extract(latents, method)
            except Exception as e:
                print(f"Warning: Failed to extract with method {method}: {e}")
                continue

        return results

    # =========================================================================
    # Pooling Methods
    # =========================================================================

    def _mean_pool(self, latents: np.ndarray) -> np.ndarray:
        """Global average pooling: (N, C, H, W) -> (N, C)."""
        return latents.mean(axis=(2, 3))

    def _max_pool(self, latents: np.ndarray) -> np.ndarray:
        """Global max pooling: (N, C, H, W) -> (N, C)."""
        return latents.max(axis=(2, 3))

    def _std_pool(self, latents: np.ndarray) -> np.ndarray:
        """Global std pooling: (N, C, H, W) -> (N, C)."""
        return latents.std(axis=(2, 3))

    def _meanstd_pool(self, latents: np.ndarray) -> np.ndarray:
        """Concatenate mean and std: (N, C, H, W) -> (N, 2C)."""
        mean = self._mean_pool(latents)
        std = self._std_pool(latents)
        return np.concatenate([mean, std], axis=1)

    def _minmax_pool(self, latents: np.ndarray) -> np.ndarray:
        """Concatenate min and max: (N, C, H, W) -> (N, 2C)."""
        min_val = latents.min(axis=(2, 3))
        max_val = latents.max(axis=(2, 3))
        return np.concatenate([min_val, max_val], axis=1)

    def _patch_pool(self, latents: np.ndarray, patch_size: int) -> np.ndarray:
        """Patch pooling: divide into patches and average.

        Args:
            latents: (N, C, H, W)
            patch_size: Size of each patch (e.g., 16 for 4×4 output)

        Returns:
            (N, C * n_patches) where n_patches = (H/patch_size) * (W/patch_size)
        """
        N, C, H, W = latents.shape
        n_h = H // patch_size
        n_w = W // patch_size

        # Reshape to patches
        patches = latents.reshape(N, C, n_h, patch_size, n_w, patch_size)
        # Average over patch dimensions
        patches = patches.mean(axis=(3, 5))  # (N, C, n_h, n_w)
        # Flatten spatial dimensions
        return patches.reshape(N, -1)  # (N, C * n_h * n_w)

    # =========================================================================
    # Dimensionality Reduction
    # =========================================================================

    def _pca_reduce(
        self,
        embeddings: np.ndarray,
        n_components: int,
        fit: bool = True,
    ) -> np.ndarray:
        """Apply PCA dimensionality reduction.

        Args:
            embeddings: (N, D) input embeddings
            n_components: Number of PCA components
            fit: Whether to fit a new PCA model

        Returns:
            (N, n_components) reduced embeddings
        """
        if fit or n_components not in self._pca_models:
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(embeddings)
            self._pca_models[n_components] = pca
            return reduced
        else:
            return self._pca_models[n_components].transform(embeddings)

    def get_pca_explained_variance(self, n_components: int) -> Optional[np.ndarray]:
        """Get explained variance ratio for fitted PCA model."""
        if n_components in self._pca_models:
            return self._pca_models[n_components].explained_variance_ratio_
        return None


def get_method_info(method: str) -> Dict:
    """Get detailed information about an extraction method.

    Returns dict with:
    - description: Human-readable description
    - output_shape: Expected output shape given (N, 16, 64, 64) input
    - n_dims: Number of output dimensions
    - n_tokens: Number of tokens (for TheWorld integration)
    """
    N = "N"
    info = {
        "mode_mean": {"output_shape": f"({N}, 16)", "n_dims": 16, "n_tokens": 1},
        "mode_max": {"output_shape": f"({N}, 16)", "n_dims": 16, "n_tokens": 1},
        "mode_std": {"output_shape": f"({N}, 16)", "n_dims": 16, "n_tokens": 1},
        "mode_meanstd": {"output_shape": f"({N}, 32)", "n_dims": 32, "n_tokens": 1},
        "mode_minmax": {"output_shape": f"({N}, 32)", "n_dims": 32, "n_tokens": 1},
        "mode_patch4": {"output_shape": f"({N}, 256)", "n_dims": 256, "n_tokens": 16},
        "mode_patch8": {"output_shape": f"({N}, 1024)", "n_dims": 1024, "n_tokens": 64},
        "mode_patch16": {"output_shape": f"({N}, 4096)", "n_dims": 4096, "n_tokens": 256},
        "mode_flatten": {"output_shape": f"({N}, 65536)", "n_dims": 65536, "n_tokens": 4096},
        "mode_mean_pca8": {"output_shape": f"({N}, 8)", "n_dims": 8, "n_tokens": 1},
        "mode_mean_pca4": {"output_shape": f"({N}, 4)", "n_dims": 4, "n_tokens": 1},
        "mode_mean_pca2": {"output_shape": f"({N}, 2)", "n_dims": 2, "n_tokens": 1},
        "spatial_flat": {"output_shape": f"({N}, 65536)", "n_dims": 65536, "n_tokens": 4096},
        "channel_flat": {"output_shape": f"({N}, 65536)", "n_dims": 65536, "n_tokens": 16},
    }

    if method not in info:
        return {"output_shape": "unknown", "n_dims": -1, "n_tokens": -1}

    result = info[method]
    result["description"] = EXTRACTION_METHODS.get(method, "Unknown method")
    return result
