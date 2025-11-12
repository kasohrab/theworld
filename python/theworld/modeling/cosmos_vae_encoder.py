"""Cosmos VAE encoder for latent space encoding."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Optional, cast
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from torch import Tensor
import torchvision.transforms.functional as TF


class CosmosVAEEncoder(nn.Module):
    """Encodes images using Cosmos VAE into latent space.

    This module handles:
    1. Image preprocessing (PIL/numpy → tensor, resize to 512×512)
    2. VAE encoding via Cosmos
    3. Returns deterministic latents (using .mode())

    Args:
        cosmos_vae: Cosmos AutoencoderKL instance
        device: Device to place computations on
        freeze_vae: Whether to freeze VAE weights

    Input: List[PIL.Image]
    Output: (B, z_dim, H, W) latents where H=64, W=64 for 512×512 input
    """

    def __init__(
        self,
        cosmos_vae: AutoencoderKL,
        device: Optional[str] = None,
        freeze_vae: bool = True,
    ):
        super().__init__()

        # Auto-detect device if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cosmos_vae = cosmos_vae
        self.device: str = device
        self.freeze_vae = freeze_vae

        # Get latent dimension from config
        self.z_dim = getattr(self.cosmos_vae.config, "z_dim", 16)

    def forward(self, images: List[Image.Image]) -> Tensor:
        """Encode images to latent space.

        Args:
            images: List of PIL images (length B)

        Returns:
            Latents of shape (B, z_dim, H, W) where:
                - z_dim is from cosmos_vae.config (typically 16)
                - H, W are spatial dimensions (64, 64 for 512×512 input)
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
        # Use non_blocking=True for async transfer (works best with pin_memory=True in DataLoader)
        tensor_batch = torch.stack(tensor_images, dim=0).to(self.device, dtype=torch.bfloat16, non_blocking=True)

        # Add time dimension for VAE: (B, C, 1, H, W) where T=1 for single frame
        cosmos_input_5d = tensor_batch.unsqueeze(2)

        # Encode through Cosmos VAE (not the full pipeline)
        # Use vae.encode().latent_dist.mode() for deterministic latents
        # Move VAE to device on first use (avoid repeated .to() calls)
        if not hasattr(self, '_vae_device_set'):
            target_device: torch.device = torch.device(self.device)
            _ = self.cosmos_vae.to(target_device)
            self._vae_device_set = True

        encoder_output = cast(AutoencoderKLOutput, self.cosmos_vae.encode(cosmos_input_5d))
        latent_dist = encoder_output.latent_dist
        latents = latent_dist.mode()  # Deterministic: use mode, not mean or sample

        # Shape: (B, z_dim, 1, H, W) - single frame latent
        b, c, t, h, w = latents.shape
        assert b == batch_size, f"Batch size mismatch: expected {batch_size}, got {b}"
        assert c == self.z_dim, f"Latent dim mismatch: expected {self.z_dim}, got {c}"
        assert t == 1, f"Time dimension should be 1, got {t}"

        # Remove time dimension: (B, z_dim, H, W)
        latents = latents.squeeze(2)

        return latents
