"""Test 3: Can EmbeddingFusion train by itself?

Tests if concatenating gemma + world embeddings preserves gradients.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("TEST 3: EmbeddingFusion Training Test")
print("=" * 80)

# Create EmbeddingFusion
print("\n[1/3] Creating EmbeddingFusion...")
from theworld.modeling.fusion import EmbeddingFusion

# CRITICAL FIX: Cosmos pipeline import disables gradients globally
# Re-enable them for training
torch.set_grad_enabled(True)
print(f"  [FIX] Re-enabled gradients after import (was disabled by Cosmos pipeline)")

fusion = EmbeddingFusion(sow_token_id=12345, eow_token_id=12346)
fusion.train()
print(f"✓ EmbeddingFusion created (no trainable params, pure ops)")

# Create test data
print("\n[2/3] Creating test data...")
batch_size = 1
seq_len = 100
embed_dim = 2304
num_world_tokens = 784

# Create dummy inputs WITH requires_grad
dummy_gemma = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
dummy_world = torch.randn(batch_size, num_world_tokens, embed_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# Create input_ids with bracket tokens at positions 10 and 11
input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device="cuda")
input_ids[0, 10] = 12345  # sow_token_id
input_ids[0, 11] = 12346  # eow_token_id

attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")

print(f"✓ Test data created")
print(f"  Gemma embeds shape: {dummy_gemma.shape}, requires_grad: {dummy_gemma.requires_grad}")
print(f"  World embeds shape: {dummy_world.shape}, requires_grad: {dummy_world.requires_grad}")

# Forward pass
print("\n[3/3] Testing forward + backward pass...")

# Debug: Check inputs before fusion
print(f"  [DEBUG] Before fusion:")
print(f"    dummy_gemma device: {dummy_gemma.device}, requires_grad: {dummy_gemma.requires_grad}")
print(f"    dummy_world device: {dummy_world.device}, requires_grad: {dummy_world.requires_grad}")

output = fusion(gemma_embeds=dummy_gemma, world_embeds=dummy_world, input_ids=input_ids, attention_mask=attention_mask)

print(f"  [DEBUG] After fusion:")
print(f"    Output shape: {output.combined_embeds.shape}")
print(f"    Output requires_grad: {output.combined_embeds.requires_grad}")
print(f"    Output grad_fn: {output.combined_embeds.grad_fn}")

# Backward pass
if output.combined_embeds.requires_grad:
    loss = output.combined_embeds.sum()
    print(f"  Loss value: {loss.item():.4f}")
    loss.backward()

    # Check gradients
    print(f"  ✓ Dummy gemma grad is not None: {dummy_gemma.grad is not None}")
    print(f"  ✓ Dummy world grad is not None: {dummy_world.grad is not None}")

    if dummy_gemma.grad is not None and dummy_world.grad is not None:
        print(f"  Gemma grad norm: {dummy_gemma.grad.norm().item():.4f}")
        print(f"  World grad norm: {dummy_world.grad.norm().item():.4f}")

    print("\n" + "=" * 80)
    print("RESULT: ✓ EmbeddingFusion CAN TRAIN!")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("RESULT: ✗ EmbeddingFusion output has NO GRADIENTS")
    print("=" * 80)
