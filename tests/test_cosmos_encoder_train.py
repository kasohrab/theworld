"""Test 1: Can CosmosEncoder train by itself?

Tests if Cosmos VAE → projection layer can create and backpropagate gradients.
"""

import torch
from PIL import Image
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("TEST 1: CosmosEncoder Training Test")
print("=" * 80)

# Load Cosmos components
print("\n[1/4] Loading Cosmos pipeline...")
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline
from cosmos_guardrail import CosmosSafetyChecker

# CRITICAL FIX: Cosmos pipeline import disables gradients globally
torch.set_grad_enabled(True)

safety_checker = CosmosSafetyChecker()
cosmos_pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
    "nvidia/Cosmos-Predict2-2B-Video2World",
    torch_dtype=torch.bfloat16,
    safety_checker=safety_checker,
    low_cpu_mem_usage=True,
    local_files_only=False,
)
cosmos_pipe = cosmos_pipe.to("cuda")
print("✓ Cosmos pipeline loaded")

# Create CosmosEncoder
print("\n[2/4] Creating CosmosEncoder...")
from theworld.modeling.cosmos_encoder import CosmosEncoder

cosmos_encoder = CosmosEncoder(
    cosmos_pipe=cosmos_pipe,
    cosmos_dim=16,
    gemma_dim=2304,
    device="cuda",
    freeze_vae=False,  # Make everything trainable
)
cosmos_encoder.train()

# Make sure all params are trainable
for param in cosmos_encoder.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in cosmos_encoder.parameters() if p.requires_grad)
total = sum(p.numel() for p in cosmos_encoder.parameters())
print(f"✓ CosmosEncoder created: {trainable:,} / {total:,} params trainable")

# Create test data
print("\n[3/4] Creating test data...")
dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
images = [dummy_image]
print("✓ Test image created")

# Forward pass
print("\n[4/4] Testing forward + backward pass...")
cosmos_encoder.zero_grad()
world_embeds = cosmos_encoder(images=images)

print(f"  Output shape: {world_embeds.shape}")
print(f"  Output requires_grad: {world_embeds.requires_grad}")
print(f"  Output grad_fn: {world_embeds.grad_fn}")

# Backward pass
if world_embeds.requires_grad:
    loss = world_embeds.sum()
    print(f"  Loss value: {loss.item():.4f}")
    loss.backward()

    # Check gradients
    proj_grad = cosmos_encoder.world_projection.weight.grad
    if proj_grad is not None:
        print(f"  ✓ Projection grad norm: {proj_grad.norm().item():.4f}")

        vae_grads = sum(1 for p in cosmos_pipe.vae.parameters() if p.grad is not None and p.requires_grad)
        print(f"  ✓ VAE params with grad: {vae_grads}")

        print("\n" + "=" * 80)
        print("RESULT: ✓ CosmosEncoder CAN TRAIN!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("RESULT: ✗ CosmosEncoder forward works but backward FAILED")
        print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("RESULT: ✗ CosmosEncoder output has NO GRADIENTS")
    print("=" * 80)
