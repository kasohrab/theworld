"""World token projection module."""

import torch
import torch.nn as nn
from torch import Tensor

from .spatial_reducer import SpatialReducer, WorldProjectionConfig


class WorldProjector(nn.Module):
    """Projects world latents to Gemma embeddings.

    This module orchestrates the full pipeline from Cosmos latents to
    Gemma-compatible embeddings:
    1. Reshape latents to tokens (via SpatialReducer)
    2. Project tokens to Gemma dimension (via MLP)

    The projection input dimension is automatically derived from the
    reduction mode and latent dimension.

    Args:
        config: World projection configuration (spatial vs channel mode)
        z_dim: Cosmos latent dimension (from cosmos_vae.config.z_dim)
        gemma_dim: Gemma embedding dimension (Gemma 3 4B: 2560, default: 2304 for compatibility)

    Input: (B, z_dim, H, W) latents from CosmosVAEEncoder
    Output: (B, num_tokens, gemma_dim) projected embeddings where:
        - Spatial mode: num_tokens = H*W (e.g., 4096)
        - Channel mode: num_tokens = z_dim (e.g., 16)
    """

    def __init__(
        self,
        config: WorldProjectionConfig,
        z_dim: int,
        gemma_dim: int = 2304,
        device: str = "cuda",
    ):
        super().__init__()

        self.config = config
        self.z_dim = z_dim
        self.gemma_dim = gemma_dim
        self.device = device

        # Create reducer (handles latent → token reshaping)
        self.reducer = SpatialReducer(config=config, z_dim=z_dim)

        # Derive projection input dimension based on mode
        # - Spatial: tokens are (num_spatial, z_dim) → project z_dim
        # - Channel: tokens are (z_dim, num_spatial) → project num_spatial
        if config.mode == "spatial":
            proj_input_dim = z_dim
        else:  # channel
            # For 512×512 input: H*W = 64*64 = 4096
            # Cosmos VAE has 8× spatial compression
            proj_input_dim = 64 * 64

        # Create projection layer(s) based on architecture config
        self.projection = self._build_projection(
            input_dim=proj_input_dim,
            output_dim=gemma_dim,
            architecture=config.architecture,
            device=device,
        )

    def _build_projection(
        self,
        input_dim: int,
        output_dim: int,
        architecture: str,
        device: str,
    ) -> nn.Module:
        """Build projection layer(s) based on architecture config.

        Args:
            input_dim: Input dimension (z_dim for spatial, H*W for channel)
            output_dim: Output dimension (Gemma embedding dimension, Gemma 3 4B: 2560)
            architecture: Architecture type ("mlp", "mlp_no_final_gelu", "linear")
            device: Device to place layers on

        Returns:
            Projection module (nn.Sequential or nn.Linear)
        """
        if architecture == "mlp":
            # Default: 2-layer MLP with GELU after both layers
            # Backward compatible with existing checkpoints
            return nn.Sequential(
                nn.Linear(input_dim, output_dim, dtype=torch.bfloat16, device=device),
                nn.GELU(),
                nn.Linear(output_dim, output_dim, dtype=torch.bfloat16, device=device),
                nn.GELU(),
            )
        elif architecture == "mlp_no_final_gelu":
            # 2-layer MLP without final GELU activation
            return nn.Sequential(
                nn.Linear(input_dim, output_dim, dtype=torch.bfloat16, device=device),
                nn.GELU(),
                nn.Linear(output_dim, output_dim, dtype=torch.bfloat16, device=device),
            )
        elif architecture == "linear":
            # Single linear layer (simplest projection)
            return nn.Linear(input_dim, output_dim, dtype=torch.bfloat16, device=device)
        else:
            raise ValueError(
                f"Invalid architecture '{architecture}'. "
                "Choose 'mlp', 'mlp_no_final_gelu', or 'linear'."
            )

    def forward(self, latents: Tensor) -> Tensor:
        """Project latents to Gemma embeddings.

        Args:
            latents: Cosmos VAE latents of shape (B, z_dim, H, W)

        Returns:
            Projected embeddings of shape (B, num_tokens, gemma_dim)
        """
        # Step 1: Reshape latents to tokens
        # Spatial: (B, z_dim, H, W) → (B, H*W, z_dim)
        # Channel: (B, z_dim, H, W) → (B, z_dim, H*W)
        tokens = self.reducer(latents)

        # Ensure bfloat16 dtype
        tokens = tokens.to(dtype=torch.bfloat16)

        # Step 2: Project to Gemma dimension
        # (B, num_tokens, token_dim) → (B, num_tokens, gemma_dim)
        embeddings = self.projection(tokens)

        return embeddings
