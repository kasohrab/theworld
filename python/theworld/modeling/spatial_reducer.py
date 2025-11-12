"""Token projection modes for Cosmos VAE latents."""

from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn


@dataclass
class WorldProjectionConfig:
    """Configuration for world token projection.

    Controls how Cosmos VAE latents are reshaped into tokens
    before projection to Gemma's embedding space.

    Attributes:
        mode: Projection mode
            - "spatial": Each spatial position → token (current default)
                (B, z_dim, H, W) → (B, H*W, z_dim)
                Pros: Preserves spatial locality
                Cons: Many tokens (e.g., 4096 for 64×64)

            - "channel": Each channel → token (experimental)
                (B, z_dim, H, W) → (B, z_dim, H*W)
                Pros: Few tokens (e.g., 16), global spatial context
                Cons: Loses spatial structure
    """
    mode: Literal["spatial", "channel"] = "spatial"

    def validate(self):
        """Validate configuration."""
        if self.mode not in ["spatial", "channel"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Choose 'spatial' or 'channel'.")


class SpatialReducer(nn.Module):
    """Reshapes Cosmos VAE latents to tokens for projection.

    Handles two projection modes:
    - Spatial: Preserves spatial structure (each position is a token)
    - Channel: Preserves channel structure (each channel is a token)

    Example - Spatial mode (current):
        >>> config = WorldProjectionConfig(mode="spatial")
        >>> reducer = SpatialReducer(config, z_dim=16)
        >>> latents = torch.randn(2, 16, 64, 64)  # (B, z_dim, H, W)
        >>> tokens = reducer(latents)             # (B, 4096, 16)

    Example - Channel mode (new):
        >>> config = WorldProjectionConfig(mode="channel")
        >>> reducer = SpatialReducer(config, z_dim=16)
        >>> latents = torch.randn(2, 16, 64, 64)  # (B, z_dim, H, W)
        >>> tokens = reducer(latents)             # (B, 16, 4096)
    """

    def __init__(self, config: WorldProjectionConfig, z_dim: int):
        """Initialize spatial reducer.

        Args:
            config: World projection configuration
            z_dim: Cosmos latent dimension (from cosmos_vae.config.z_dim)
        """
        super().__init__()
        self.config = config
        self.z_dim = z_dim
        config.validate()

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert VAE latents to tokens.

        Args:
            latents: VAE latent embeddings of shape (B, z_dim, H, W)
                Example: (B, 16, 64, 64) for 512×512 input

        Returns:
            Tokens for projection layer:
                - Spatial mode: (B, H*W, z_dim)
                - Channel mode: (B, z_dim, H*W)
        """
        B, C, H, W = latents.shape
        assert C == self.z_dim, f"Expected {self.z_dim} channels, got {C}"

        if self.config.mode == "spatial":
            # Spatial positions as tokens
            # (B, z_dim, H, W) → (B, H, W, z_dim) → (B, H*W, z_dim)
            tokens = latents.permute(0, 2, 3, 1)  # (B, H, W, z_dim)
            tokens = tokens.reshape(B, H * W, C)   # (B, H*W, z_dim)

        elif self.config.mode == "channel":
            # Channels as tokens
            # (B, z_dim, H, W) → (B, z_dim, H*W)
            tokens = latents.reshape(B, C, H * W)  # (B, z_dim, H*W)

        return tokens

    def get_output_shape(self, spatial_h: int, spatial_w: int) -> tuple[int, int]:
        """Get output shape (num_tokens, token_dim).

        Args:
            spatial_h: Spatial height after VAE (e.g., 64)
            spatial_w: Spatial width after VAE (e.g., 64)

        Returns:
            (num_tokens, token_dim) tuple where:
                - Spatial mode: (H*W, z_dim)
                - Channel mode: (z_dim, H*W)
        """
        if self.config.mode == "spatial":
            return (spatial_h * spatial_w, self.z_dim)
        else:  # channel
            return (self.z_dim, spatial_h * spatial_w)
