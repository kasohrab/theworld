"""Analysis utilities for Cosmos VAE embeddings.

This package provides tools for analyzing and comparing different embedding
extraction methods from the Cosmos VAE encoder.
"""

from theworld.analysis.embedding_methods import EmbeddingExtractor
from theworld.analysis.embedding_metrics import compute_all_metrics

__all__ = ["EmbeddingExtractor", "compute_all_metrics"]
