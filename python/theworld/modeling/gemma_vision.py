"""Gemma vision encoder module for TheWorld model."""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import Gemma3ForConditionalGeneration
from .outputs import GemmaVisionOutput


class GemmaVisionEncoder(nn.Module):
    """Process vision through SigLIP and combine with text token embeddings.

    This module encapsulates Gemma's vision processing:
    1. Get text token embeddings from input_ids
    2. Process images through SigLIP encoder
    3. Replace image placeholder tokens with real vision features

    Args:
        gemma_model: Reference to parent Gemma3ForConditionalGeneration model

    Input shapes:
        input_ids: (B, seq_len) - Token IDs with image placeholders
        pixel_values: (B, C, H, W) - Preprocessed images for SigLIP
        attention_mask: (B, seq_len) - Attention mask

    Output:
        GemmaVisionOutput with:
            embeddings: (B, seq_len, 2304) - Combined vision+text embeddings
            input_ids: (B, seq_len) - Pass-through for reference
            attention_mask: (B, seq_len) - Pass-through
    """

    def __init__(self, gemma_model: Gemma3ForConditionalGeneration):
        super().__init__()
        self.gemma = gemma_model

    def forward(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
        attention_mask: Tensor,
    ) -> GemmaVisionOutput:
        """Process vision and text into combined embeddings.

        Args:
            input_ids: Token IDs (B, seq_len)
            pixel_values: Preprocessed images (B, C, H, W)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            GemmaVisionOutput with combined embeddings
        """
        # Input validation
        assert input_ids.dim() == 2, f"Expected 2D input_ids, got {input_ids.dim()}D"
        assert pixel_values.dim() == 4, f"Expected 4D pixel_values, got {pixel_values.dim()}D"
        assert attention_mask.dim() == 2, f"Expected 2D attention_mask, got {attention_mask.dim()}D"
        batch_size, seq_len = input_ids.shape
        assert attention_mask.shape == (batch_size, seq_len), "Attention mask shape mismatch"

        # Step 1: Get text token embeddings (image tokens are placeholders at this point)
        # Reference: Gemma3Model.forward() line 887-888
        inputs_embeds = self.gemma.model.language_model.embed_tokens(input_ids)  # (B, seq_len, 2304)

        # Step 2: Process vision through SigLIP + multi-modal projector
        # Reference: Gemma3Model.forward() line 897-903
        # IMPORTANT: Don't use torch.no_grad() here! Even though SigLIP is frozen,
        # we need gradients to flow through these features back to the projection layer
        # Get image features from SigLIP encoder
        image_features = self.gemma.model.get_image_features(pixel_values)  # (B, num_image_tokens, 2304)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

        # Step 3: Replace image token placeholders with real SigLIP vision features
        # Reference: Gemma3Model.forward() line 900-903
        # Cast to specific tensor types for get_placeholder_mask
        # torch.long() and torch.float() return Tensor, but Gemma expects LongTensor/FloatTensor
        # These are the same type at runtime (LongTensor is just Tensor with dtype=long)
        input_ids_casted = input_ids.long()
        image_features_casted = image_features.float()
        special_image_mask = self.gemma.model.get_placeholder_mask(
            input_ids_casted,  # pyright: ignore[reportArgumentType]
            inputs_embeds=inputs_embeds,
            image_features=image_features_casted,  # pyright: ignore[reportArgumentType]
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Output validation
        assert inputs_embeds.dim() == 3, f"Expected 3D embeddings, got {inputs_embeds.dim()}D"
        assert inputs_embeds.shape[:2] == (batch_size, seq_len), "Embeddings shape mismatch"
        gemma_dim = self.gemma.config.text_config.hidden_size
        assert inputs_embeds.size(2) == gemma_dim, f"Expected dim {gemma_dim}, got {inputs_embeds.size(2)}"

        return GemmaVisionOutput(
            embeddings=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
