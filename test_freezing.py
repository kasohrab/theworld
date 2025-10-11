"""Quick test: Verify selective freezing works correctly."""
import torch
from theworld import TheWorld

print("=" * 80)
print("TEST: Selective Freezing (Projection Only)")
print("=" * 80)

# Create model with default freezing (all frozen except projection)
print("\n[1/2] Creating TheWorld with default freezing...")
model = TheWorld(
    "google/gemma-3-4b-it",
    freeze_gemma_vision=True,
    freeze_gemma_language=True,
    freeze_cosmos_vae=True,
)

print("\n[2/2] Verifying trainable parameters...")

# Check each component
gemma_vision_trainable = sum(p.numel() for p in model.gemma.model.vision_tower.parameters() if p.requires_grad)
gemma_lm_trainable = sum(p.numel() for p in model.gemma.model.language_model.parameters() if p.requires_grad)
gemma_head_trainable = sum(p.numel() for p in model.gemma.lm_head.parameters() if p.requires_grad)
cosmos_vae_trainable = sum(p.numel() for p in model.cosmos_vae_encoder.parameters() if p.requires_grad)
projection_trainable = sum(p.numel() for p in model.cosmos_encoder.world_projection.parameters() if p.requires_grad)

print(f"\n  Component-level breakdown:")
print(f"  - Gemma vision tower: {gemma_vision_trainable:,} trainable")
print(f"  - Gemma language model: {gemma_lm_trainable:,} trainable")
print(f"  - Gemma lm_head: {gemma_head_trainable:,} trainable")
print(f"  - Cosmos VAE encoder: {cosmos_vae_trainable:,} trainable")
print(f"  - World projection: {projection_trainable:,} trainable")

# Overall check
trainable, total, pct = model.get_trainable_parameters()

print(f"\n  Overall: {trainable:,} / {total:,} ({pct:.4f}%)")

# Validate expectations
if gemma_vision_trainable == 0 and gemma_lm_trainable == 0 and cosmos_vae_trainable == 0:
    if projection_trainable > 0:
        print("\n" + "=" * 80)
        print("RESULT: ✓ Selective freezing WORKS! Only projection layer trainable.")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("RESULT: ✗ Projection layer is NOT trainable!")
        print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("RESULT: ✗ Some frozen components are still trainable!")
    print("=" * 80)
