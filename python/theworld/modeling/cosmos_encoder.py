"""Cosmos VAE encoder module for TheWorld model."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, cast
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torch import Tensor
import torchvision.transforms.functional as TF


class CosmosEncoder(nn.Module):
    """Encode images using Cosmos VAE → projection to Gemma space.

    This module encapsulates all Cosmos-related processing:
    1. VAE encoding (single frame only)
    2. Projection from 16-dim latent space to 2304-dim Gemma space

    Args:
        cosmos_vae: AutoencoderKL instance (Cosmos VAE model)
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
        cosmos_vae: AutoencoderKL,
        cosmos_dim: int = 16,
        gemma_dim: int = 2304,
        device: str = "cuda",
        freeze_vae: bool = True,
    ):
        super().__init__()
        self.cosmos_vae = cosmos_vae
        self.device: str = device
        self.cosmos_dim = cosmos_dim
        self.freeze_vae = freeze_vae

        # Projection: 16-dim latent → 2304-dim Gemma embedding space
        self.world_projection = nn.Linear(cosmos_dim, gemma_dim, dtype=torch.bfloat16)
        self.world_projection.to(device)

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

        # Convert PIL images to tensors with consistent size for VAE encoding
        # Cosmos VAE works best with power-of-2 sizes
        target_size = (512, 512)  # (H, W)
        tensor_images = []

        for img in images:
            # Convert to PIL Image if needed
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                # Already a tensor - convert to PIL
                if img.ndim == 4:
                    img = img[0]  # Remove batch dim
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img_np)

            # Ensure RGB
            img = img.convert("RGB")

            # Resize to target size (bilinear interpolation)
            img = TF.resize(img, list(target_size), interpolation=TF.InterpolationMode.BILINEAR)

            # Convert to tensor: (C, H, W), range [0, 255]
            tensor_img = TF.to_tensor(img) * 255.0  # torchvision normalizes to [0,1], scale back
            tensor_images.append(tensor_img)

        # Stack into batch: (B, C, H, W)
        tensor_batch = torch.stack(tensor_images, dim=0).to(self.device, dtype=torch.bfloat16)

        # Add time dimension for VAE: (B, C, 1, H, W) where T=1 for single frame
        cosmos_input_5d = tensor_batch.unsqueeze(2)

        # Encode through Cosmos VAE (not the full pipeline)
        # Use vae.encode().latent_dist.mode() for deterministic latents
        # Move VAE to device on first use (avoid repeated .to() calls)
        if not hasattr(self, '_vae_device_set'):
            target_device: torch.device = torch.device(self.device)
            _ = self.cosmos_vae.to(target_device)
            self._vae_device_set = True

        # IMPORTANT: Don't use torch.no_grad() here! Even though VAE is frozen,
        # we need gradients to flow through these latents back to the projection layer
        encoder_output = cast(AutoencoderKLOutput, self.cosmos_vae.encode(cosmos_input_5d))
        latent_dist = encoder_output.latent_dist
        latents = latent_dist.mode()  # Deterministic: use mode, not mean or sample

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
