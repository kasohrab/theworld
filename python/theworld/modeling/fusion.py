"""Embedding fusion module for TheWorld model."""

import torch
import torch.nn as nn
from torch import Tensor
from .outputs import FusionOutput


class EmbeddingFusion(nn.Module):
    """Fuse Gemma vision embeddings with Cosmos world embeddings.

    Inserts world tokens between special bracket tokens <start_of_world> and <end_of_world>.

    Args:
        sow_token_id: Token ID for <start_of_world> (SOW)
        eow_token_id: Token ID for <end_of_world> (EOW)

    Input shapes:
        gemma_embeds: (B, seq_len, gemma_dim) - From Gemma model (Gemma 3 4B: 2560)
        world_embeds: (B, num_world_tokens, gemma_dim) - From CosmosEncoder (Gemma 3 4B: 2560)
        input_ids: (B, seq_len) - For finding bracket positions
        attention_mask: (B, seq_len) - To update for world tokens

    Output:
        FusionOutput with:
            combined_embeds: (B, combined_len, gemma_dim) - Gemma 3 4B: 2560
            combined_attention_mask: (B, combined_len)
        where combined_len = seq_len - 2 + num_world_tokens
    """

    def __init__(self, sow_token_id: int, eow_token_id: int):
        super().__init__()
        self.sow_token_id = sow_token_id
        self.eow_token_id = eow_token_id

    def forward(
        self,
        gemma_embeds: Tensor,
        world_embeds: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> FusionOutput:
        """Fuse Gemma and world embeddings by inserting world tokens.

        Args:
            gemma_embeds: Gemma vision+text embeddings (B, seq_len, gemma_dim)
            world_embeds: Cosmos world embeddings (B, num_world_tokens, gemma_dim)
                        (Gemma 3 4B uses gemma_dim=2560)
            input_ids: Token IDs for finding brackets (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            FusionOutput with combined embeddings and mask
        """
        # Input validation
        assert gemma_embeds.dim() == 3, f"Expected 3D gemma_embeds, got {gemma_embeds.dim()}D"
        assert world_embeds.dim() == 3, f"Expected 3D world_embeds, got {world_embeds.dim()}D"
        assert input_ids.dim() == 2, f"Expected 2D input_ids, got {input_ids.dim()}D"
        assert attention_mask.dim() == 2, f"Expected 2D attention_mask, got {attention_mask.dim()}D"

        batch_size, seq_len, embed_dim = gemma_embeds.shape
        num_world_tokens = world_embeds.size(1)

        assert world_embeds.size(0) == batch_size, "Batch size mismatch between gemma and world embeds"
        assert world_embeds.size(2) == embed_dim, "Embedding dimension mismatch"
        assert input_ids.shape == (batch_size, seq_len), "input_ids shape mismatch"
        assert attention_mask.shape == (batch_size, seq_len), "attention_mask shape mismatch"

        # Find bracket token positions
        start_positions = (input_ids == self.sow_token_id).nonzero(as_tuple=True)[1]
        end_positions = (input_ids == self.eow_token_id).nonzero(as_tuple=True)[1]

        assert len(start_positions) > 0, "No <start_of_world> token found in input_ids"
        assert len(end_positions) > 0, "No <end_of_world> token found in input_ids"

        # Use tensor-based indexing to avoid .item() CPU-GPU sync
        start_pos = start_positions[0]
        end_pos = end_positions[0]

        # Slice embeddings: [before_start] + [<start>] + [WORLD] + [<end>] + [after_end]
        # We keep the bracket tokens for consistency
        # PyTorch supports slicing with scalar tensors (no .item() needed)
        embeddings_before = gemma_embeds[:, : start_pos + 1, :]  # Up to and including <start>
        embeddings_after = gemma_embeds[:, end_pos:, :]  # From <end> onwards

        # Move world embeddings to target device if needed
        device = gemma_embeds.device
        if world_embeds.device != device:
            world_embeds = world_embeds.to(device, non_blocking=True)

        # Concatenate to insert world tokens between brackets
        combined_embeds = torch.cat([embeddings_before, world_embeds, embeddings_after], dim=1)

        # Update attention mask similarly
        attention_mask_before = attention_mask[:, : start_pos + 1]
        attention_mask_after = attention_mask[:, end_pos:]
        world_attention_mask = torch.ones((batch_size, num_world_tokens), dtype=torch.long, device=device)
        combined_attention_mask = torch.cat([attention_mask_before, world_attention_mask, attention_mask_after], dim=1)

        # Output validation (compute expected length using tensor operations to avoid sync)
        expected_len = (start_pos + 1) + num_world_tokens + (seq_len - end_pos)
        assert (
            combined_embeds.size(1) == expected_len
        ), f"Combined length mismatch: got {combined_embeds.size(1)}, expected {expected_len}"
        assert combined_attention_mask.size(1) == expected_len, "Combined mask length mismatch"

        return FusionOutput(
            combined_embeds=combined_embeds,
            combined_attention_mask=combined_attention_mask,
        )
