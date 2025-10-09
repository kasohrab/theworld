"""Output dataclasses for TheWorld model components."""

from dataclasses import dataclass
from torch import Tensor


@dataclass
class GemmaVisionOutput:
    """Output from GemmaVisionEncoder module.

    Attributes:
        embeddings: Combined vision+text embeddings (B, seq_len, 2304)
        input_ids: Token IDs for reference (B, seq_len)
        attention_mask: Attention mask (B, seq_len)
    """

    embeddings: Tensor  # (B, seq_len, 2304)
    input_ids: Tensor  # (B, seq_len)
    attention_mask: Tensor  # (B, seq_len)


@dataclass
class FusionOutput:
    """Output from EmbeddingFusion module.

    Attributes:
        combined_embeds: Fused embeddings with world tokens inserted (B, combined_len, 2304)
        combined_attention_mask: Updated attention mask (B, combined_len)
    """

    combined_embeds: Tensor  # (B, combined_len, 2304)
    combined_attention_mask: Tensor  # (B, combined_len)
