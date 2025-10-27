"""Output dataclasses for TheWorld model components."""

from dataclasses import dataclass
from torch import Tensor


@dataclass
class FusionOutput:
    """Output from EmbeddingFusion module.

    Attributes:
        combined_embeds: Fused embeddings with world tokens inserted (B, combined_len, dim)
        combined_attention_mask: Updated attention mask (B, combined_len)
    """

    combined_embeds: Tensor  # (B, combined_len, dim)
    combined_attention_mask: Tensor  # (B, combined_len)