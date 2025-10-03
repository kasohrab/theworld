import torch
from model import TheWorld
import os

def main():
    # Set device
    device = "cuda"
    print(f"Using device: {device}")

    # Create a single model instance
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)

    model = TheWorld("google/gemma-3-4b-it", device=device, num_world_steps=4)

    input_pixels = torch.randn(1, 3, 896, 896)  # Gemma 3 uses 896x896
    text = "What is in this image?"

    # Example 1: Single-step (no rollout) - Fast inference
    print("\n" + "="*60)
    print("Example 1: Single-step world model (no future prediction)")
    print("="*60)

    print("Running single-step forward pass (override to 0 steps)...")
    outputs_single = model.forward(input_pixels, text, num_world_steps=0)

    print(f"✓ Output shape: {outputs_single.logits.shape}")
    print(f"  - Includes: Gemma vision tokens + Cosmos world tokens + text")
    print(f"  - Total context: {outputs_single.logits.shape[1]} tokens")

    # Example 2: Multi-step rollout (4 future frames)
    print("\n" + "="*60)
    print("Example 2: Multi-step rollout (predict 4 future frames)")
    print("="*60)

    print("Running multi-step forward pass (predicting 4 future frames)...")
    print("This will take longer as Cosmos generates future states...")
    outputs_multi = model.forward(input_pixels, text)  # Uses default num_world_steps=4

    print(f"✓ Output shape: {outputs_multi.logits.shape}")
    print(f"  - Gemma vision: 256 tokens (896x896 at 14px patches)")
    print(f"  - Cosmos world: ~3920 tokens (5 frames × 28×28)")
    print(f"  - Total context: {outputs_multi.logits.shape[1]} tokens")
    print(f"  - Frames: 1 (input) + 4 (predicted future)")

    # Example 3: Override rollout at inference time
    print("\n" + "="*60)
    print("Example 3: Override rollout length at inference time")
    print("="*60)

    print("Model initialized with num_world_steps=4, but overriding to 2...")
    outputs_override = model.forward(
        input_pixels,
        text,
        num_world_steps=2  # Override to predict only 2 future frames
    )

    print(f"✓ Output shape: {outputs_override.logits.shape}")
    print(f"  - Cosmos world: ~2352 tokens (3 frames × 28×28)")
    print(f"  - Frames: 1 (input) + 2 (predicted future)")

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
    print("\nKey features:")
    print("  - Dual vision: Gemma's SigLIP + Cosmos world model")
    print("  - Gemma provides: static visual understanding (objects, scenes)")
    print("  - Cosmos provides: temporal dynamics (motion, physics, future states)")
    print("\nSee docs/ for details:")
    print("  - autoregressive_world_rollout.md: How rollout works")
    print("  - world_model_latent_space.md: Cosmos latent extraction")


if __name__ == "__main__":
    main()
