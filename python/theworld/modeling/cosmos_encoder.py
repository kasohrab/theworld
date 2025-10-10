"""Cosmos VAE encoder module for TheWorld model."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline
from torch import Tensor


class CosmosEncoder(nn.Module):
    """Encode images using Cosmos VAE → projection to Gemma space.

    This module encapsulates all Cosmos-related processing:
    1. VAE encoding (single frame only)
    2. Projection from 16-dim latent space to 2304-dim Gemma space

    Args:
        cosmos_pipe: Cosmos2VideoToWorldPipeline instance (uses only the VAE component)
        cosmos_dim: Dimension of Cosmos latent space (default: 16)
        gemma_dim: Dimension of Gemma embedding space (default: 2304)
        device: Device to place parameters on
        freeze_vae: Whether to freeze VAE weights (default: True)

    Input shapes:
        images: List[PIL.Image] - Raw PIL images (B images)

    Output shape:
        world_embeds: (B, num_tokens, 2304) where num_tokens = H × W (spatial dimensions from VAE)
    """

    def __init__(
        self,
        cosmos_pipe: Cosmos2VideoToWorldPipeline,
        cosmos_dim: int = 16,
        gemma_dim: int = 2304,
        device: str = "cuda",
        freeze_vae: bool = True,
    ):
        super().__init__()
        self.cosmos_pipe = cosmos_pipe
        self.device = device
        self.cosmos_dim = cosmos_dim
        self.freeze_vae = freeze_vae

        # Projection: 16-dim latent → 2304-dim Gemma embedding space
        self.world_projection = nn.Linear(cosmos_dim, gemma_dim, dtype=torch.bfloat16).to(device)

    def forward(self, images: List[Image.Image]) -> Tensor:
        """Encode images into world embeddings.

        Args:
            images: List of PIL images (length B)

        Returns:
            world_embeds: Tensor of shape (B, num_tokens, 2304)
                         where num_tokens = H × W (spatial tokens from VAE)
        """
        # Input validation
        assert isinstance(images, list), f"images must be List[PIL.Image], got {type(images)}"
        batch_size = len(images)

        # Convert PIL images to tensors for VAE encoding
        tensor_images = []
        for img in images:
            if isinstance(img, Image.Image):
                img_np = np.array(img.convert("RGB"))
                tensor_img = torch.from_numpy(img_np).permute(2, 0, 1)  # (C, H, W)
            elif isinstance(img, np.ndarray):
                tensor_img = torch.from_numpy(img).permute(2, 0, 1)
            else:
                # Already a tensor
                tensor_img = img if img.ndim == 3 else img[0]
            tensor_images.append(tensor_img)

        # Stack into batch: (B, C, H, W)
        tensor_batch = torch.stack(tensor_images, dim=0).to(self.device, dtype=torch.bfloat16)

        # Add time dimension for VAE: (B, C, 1, H, W) where T=1 for single frame
        cosmos_input_5d = tensor_batch.unsqueeze(2)

        # Encode through Cosmos VAE (not the full pipeline)
        # Use vae.encode().latent_dist.mode() for deterministic latents
        # Move VAE to device on first use (avoid repeated .to() calls)
        if not hasattr(self, '_vae_device_set'):
            self.cosmos_pipe.vae = self.cosmos_pipe.vae.to(self.device)
            self._vae_device_set = True

        if self.freeze_vae:
            with torch.no_grad():
                latent_dist = self.cosmos_pipe.vae.encode(cosmos_input_5d).latent_dist
                latents = latent_dist.mode()  # Deterministic: use mode, not mean or sample
        else:
            latent_dist = self.cosmos_pipe.vae.encode(cosmos_input_5d).latent_dist
            latents = latent_dist.mode()

        # Shape: (B, 16, 1, H, W) - single frame latent
        b, c, t, h, w = latents.shape
        assert b == batch_size, f"Batch size mismatch: expected {batch_size}, got {b}"
        assert c == self.cosmos_dim, f"Latent dim mismatch: expected {self.cosmos_dim}, got {c}"
        assert t == 1, f"Time dimension should be 1, got {t}"

        # Remove time dimension and permute to (B, H, W, C)
        latents = latents.squeeze(2).permute(0, 2, 3, 1)

        # Reshape to 2D for projection: (B, H*W, 16)
        num_tokens = h * w
        reshaped_latents = latents.reshape(b, num_tokens, c)

        # Ensure correct dtype
        reshaped_latents = reshaped_latents.to(dtype=torch.bfloat16)

        # Project to Gemma dimension: (B, H*W, 16) → (B, H*W, 2304)
        projected_embeds = self.world_projection(reshaped_latents)

        # Output validation
        assert projected_embeds.dim() == 3, f"Expected 3D tensor, got {projected_embeds.dim()}D"
        assert projected_embeds.size(0) == batch_size, f"Batch size mismatch in output"
        assert projected_embeds.size(1) == num_tokens, f"Token count mismatch"
        assert projected_embeds.size(2) == self.world_projection.out_features, f"Projection dim mismatch"

        return projected_embeds
