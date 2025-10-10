"""Integration tests for CosmosEncoder module.

These tests load the actual Cosmos model and verify VAE encoding works correctly.
"""

import torch
import pytest
from PIL import Image
import numpy as np
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline
from theworld.modeling.cosmos_encoder import CosmosEncoder


@pytest.fixture(scope="module")
def cosmos_encoder():
    """Load Cosmos pipeline and create encoder (shared across tests to save time)."""
    cosmos_pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        "nvidia/Cosmos-Predict2-2B-Video2World",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    encoder = CosmosEncoder(
        cosmos_pipe=cosmos_pipe,
        cosmos_dim=16,
        gemma_dim=2304,
        device="cuda" if torch.cuda.is_available() else "cpu",
        freeze_vae=True,
    )
    return encoder


@pytest.fixture
def test_image():
    """Create a simple test image."""
    # Create 512x512 RGB image with a gradient pattern
    img_array = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        img_array[i, :, 0] = i // 2  # Red gradient
        img_array[:, i, 1] = i // 2  # Green gradient
    img_array[:, :, 2] = 128  # Blue constant
    return Image.fromarray(img_array)


def test_single_image_encoding(cosmos_encoder, test_image):
    """Test encoding a single image."""
    images = [test_image]

    # Encode
    world_embeds = cosmos_encoder(images)

    # Verify output shape: (B=1, num_tokens, dim=2304)
    assert world_embeds.dim() == 3, f"Expected 3D tensor, got {world_embeds.dim()}D"
    assert world_embeds.shape[0] == 1, f"Expected batch size 1, got {world_embeds.shape[0]}"
    assert world_embeds.shape[2] == 2304, f"Expected embedding dim 2304, got {world_embeds.shape[2]}"

    # Verify dtype
    assert world_embeds.dtype == torch.bfloat16, f"Expected bfloat16, got {world_embeds.dtype}"

    # Verify no NaN or Inf values
    assert not torch.isnan(world_embeds).any(), "Output contains NaN values"
    assert not torch.isinf(world_embeds).any(), "Output contains Inf values"

    # Verify tensor is on correct device
    expected_device = cosmos_encoder.device
    assert str(world_embeds.device).startswith(expected_device.split(":")[0]), \
        f"Expected device {expected_device}, got {world_embeds.device}"


def test_batch_processing(cosmos_encoder, test_image):
    """Test encoding multiple images in a batch."""
    # Create two different test images
    img1 = test_image
    img2_array = np.array(test_image)
    img2_array = 255 - img2_array  # Invert colors
    img2 = Image.fromarray(img2_array.astype(np.uint8))

    images = [img1, img2]

    # Batch encoding
    world_embeds = cosmos_encoder(images)

    # Verify batch dimension
    assert world_embeds.shape[0] == 2, f"Expected batch size 2, got {world_embeds.shape[0]}"
    assert world_embeds.shape[2] == 2304, f"Expected embedding dim 2304, got {world_embeds.shape[2]}"

    # Verify the two embeddings are different (not identical)
    assert not torch.allclose(world_embeds[0], world_embeds[1], atol=1e-3), \
        "Embeddings for different images should not be identical"


def test_output_consistency(cosmos_encoder, test_image):
    """Test that encoding the same image twice produces consistent results."""
    images = [test_image]

    # Encode twice
    world_embeds1 = cosmos_encoder(images)
    world_embeds2 = cosmos_encoder(images)

    # With freeze_vae=True and using .mode(), results should be identical
    # (deterministic, no dropout, no sampling)
    assert torch.allclose(world_embeds1, world_embeds2, atol=1e-5), \
        "Deterministic encoding should produce identical results"


def test_spatial_dimensions(cosmos_encoder, test_image):
    """Test that spatial dimensions are correct."""
    images = [test_image]
    world_embeds = cosmos_encoder(images)

    # Get number of spatial tokens
    num_tokens = world_embeds.shape[1]

    # Verify it's a square number (H × W)
    sqrt_tokens = int(num_tokens ** 0.5)
    assert sqrt_tokens * sqrt_tokens == num_tokens, \
        f"Token count {num_tokens} should be a square number (H × W)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
