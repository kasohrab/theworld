"""Cosmos VAE encoder module for TheWorld model."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Union, Any
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline
from diffusers.pipelines.cosmos.pipeline_output import CosmosPipelineOutput
from torch import Tensor


class CosmosEncoder(nn.Module):
    """Encode images using Cosmos VAE → temporal embeddings → projection to Gemma space.

    This module encapsulates all Cosmos-related processing:
    1. VAE encoding (single-step) or autoregressive rollout (multi-step)
    2. Temporal embedding addition
    3. Projection from 16-dim latent space to 2304-dim Gemma space

    Args:
        cosmos_pipe: Cosmos2VideoToWorldPipeline instance
        cosmos_dim: Dimension of Cosmos latent space (default: 16)
        gemma_dim: Dimension of Gemma embedding space (default: 2304)
        max_world_steps: Maximum number of future steps for temporal embeddings (default: 16)
        device: Device to place parameters on

    Input shapes:
        images: List[PIL.Image] - Raw PIL images (B images)
        texts: List[str] - Text prompts for conditioning (B prompts)
        num_world_steps: int - Number of future frames (0 = current only)

    Output shape:
        world_embeds: (B, num_world_tokens, 2304) where num_world_tokens = 784 * (1 + num_world_steps)
    """

    def __init__(
        self,
        cosmos_pipe: Cosmos2VideoToWorldPipeline,
        cosmos_dim: int = 16,
        gemma_dim: int = 2304,
        max_world_steps: int = 16,
        device: str = "cuda",
        freeze_vae: bool = True,
    ):
        super().__init__()
        self.cosmos_pipe = cosmos_pipe
        self.device = device
        self.cosmos_dim = cosmos_dim
        self.freeze_vae = freeze_vae

        # Temporal embeddings: Helps distinguish between t=0 (current), t=1, t=2, ... (future)
        self.temporal_embedding = nn.Embedding(max_world_steps + 1, cosmos_dim, dtype=torch.bfloat16).to(device)

        # Projection: 16-dim latent → 2304-dim Gemma embedding space
        self.world_projection = nn.Linear(cosmos_dim, gemma_dim, dtype=torch.bfloat16).to(device)

    def forward(
        self,
        images: List[Image.Image],
        texts: List[str],
        num_world_steps: int = 0,
    ) -> Tensor:
        """Encode images into world embeddings.

        Args:
            images: List of PIL images (length B)
            texts: List of text prompts for conditioning (length B)
            num_world_steps: Number of future frames to predict (0 = current frame only)

        Returns:
            world_embeds: Tensor of shape (B, num_world_tokens, 2304)
                         where num_world_tokens = 784 * (1 + num_world_steps)
        """
        # Input validation
        assert isinstance(images, list), f"images must be List[PIL.Image], got {type(images)}"
        assert isinstance(texts, list), f"texts must be List[str], got {type(texts)}"
        assert len(images) == len(texts), f"Batch size mismatch: {len(images)} images vs {len(texts)} texts"
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

        # Add time dimension for VAE: (B, C, T, H, W) where T=1 for single frame
        cosmos_input_5d = tensor_batch.unsqueeze(2)

        # Encode through Cosmos pipeline
        # Use pipeline for all cases (single-step and multi-step) for consistency
        # Only use no_grad if VAE is frozen, otherwise allow gradients to flow

        # Ensure pipeline is on correct device
        self.cosmos_pipe = self.cosmos_pipe.to(self.device)

        # Process each image individually (pipeline expects single image)
        all_latents = []

        if self.freeze_vae:
            with torch.no_grad():
                for pil_img, text_prompt in zip(images, texts):
                    output: Union[CosmosPipelineOutput, Any] = self.cosmos_pipe(
                        prompt=text_prompt,
                        image=pil_img,  # PIL Image required for pipeline
                        num_frames=1 + num_world_steps,  # Current + future (1 for single-step)
                        num_inference_steps=10,
                        output_type="latent",  # Don't decode to pixels
                        return_dict=True,
                    )
                    # Access frames attribute (guaranteed to exist when return_dict=True)
                    assert isinstance(
                        output, CosmosPipelineOutput
                    ), "Expected CosmosPipelineOutput with return_dict=True"
                    all_latents.append(output.frames)
        else:
            # VAE unfrozen - allow gradients to flow
            for pil_img, text_prompt in zip(images, texts):
                output: Union[CosmosPipelineOutput, Any] = self.cosmos_pipe(
                    prompt=text_prompt,
                    image=pil_img,  # PIL Image required for pipeline
                    num_frames=1 + num_world_steps,  # Current + future (1 for single-step)
                    num_inference_steps=10,
                    output_type="latent",  # Don't decode to pixels
                    return_dict=True,
                )
                # Access frames attribute (guaranteed to exist when return_dict=True)
                assert isinstance(
                    output, CosmosPipelineOutput
                ), "Expected CosmosPipelineOutput with return_dict=True"
                all_latents.append(output.frames)

        # Stack latents: (B, 16, T, H, W) where T=1+num_world_steps
        latent_img_embeds = torch.cat(all_latents, dim=0)

        # Shape validation
        b, c, t, h, w = latent_img_embeds.shape
        assert b == batch_size, f"Batch size mismatch: expected {batch_size}, got {b}"
        assert c == self.cosmos_dim, f"Latent dim mismatch: expected {self.cosmos_dim}, got {c}"
        assert t == 1 + num_world_steps, f"Time steps mismatch: expected {1 + num_world_steps}, got {t}"

        # Permute to (B, T, H, W, C) for easier processing
        latent_img_embeds = latent_img_embeds.permute(0, 2, 3, 4, 1)

        # Add temporal embeddings to distinguish timesteps
        temporal_ids = torch.arange(t, device=self.device)  # [0, 1, 2, ...]
        temporal_embeds = self.temporal_embedding(temporal_ids)  # (T, 16)

        # Broadcast temporal embeddings: (T, 16) → (1, T, 1, 1, 16) → add to (B, T, H, W, 16)
        latent_img_embeds = latent_img_embeds + temporal_embeds.view(1, t, 1, 1, c)

        # Reshape to 2D for projection: (B, T*H*W, 16)
        num_world_tokens = t * h * w
        reshaped_world_embeds = latent_img_embeds.reshape(b, num_world_tokens, c)

        # Ensure correct dtype
        reshaped_world_embeds = reshaped_world_embeds.to(dtype=torch.bfloat16)

        # Project to Gemma dimension: (B, T*H*W, 16) → (B, T*H*W, 2304)
        projected_world_embeds = self.world_projection(reshaped_world_embeds)

        # Output validation
        assert projected_world_embeds.dim() == 3, f"Expected 3D tensor, got {projected_world_embeds.dim()}D"
        assert projected_world_embeds.size(0) == batch_size, f"Batch size mismatch in output"
        assert projected_world_embeds.size(1) == num_world_tokens, f"Token count mismatch"
        assert projected_world_embeds.size(2) == self.world_projection.out_features, f"Projection dim mismatch"

        return projected_world_embeds
