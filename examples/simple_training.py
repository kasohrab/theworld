import torch
from theworld import TheWorld
from PIL import Image
import numpy as np


def main():
    print("=" * 60)
    print("TheWorld Model - Training Example")
    print("=" * 60)

    # Initialize model with trainable projection layers only
    print("\n1. Loading model...")
    model = TheWorld(
        "google/gemma-3-4b-it",
        device="cuda",
        num_world_steps=0,  # Start with single-step for faster training
        freeze_gemma_vision=True,  # Freeze vision encoder
        freeze_gemma_language=True,  # Freeze language model
        freeze_cosmos_vae=True,  # Freeze Cosmos VAE
    )

    # Check trainable parameters
    trainable, total, percentage = model.get_trainable_parameters()
    print(f"\n2. Model configuration:")
    print(f"   Total parameters: {total:,}")
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Trainable percentage: {percentage:.4f}%")
    print(f"\n   Trainable components:")
    print(f"   - temporal_embedding: {sum(p.numel() for p in model.temporal_embedding.parameters()):,}")
    print(f"   - world_projection: {sum(p.numel() for p in model.world_projection.parameters()):,}")

    # Create dummy training data
    print("\n3. Creating dummy training batch...")
    # Random image (896x896 for Gemma 3)
    dummy_image = torch.randn(1, 3, 896, 896)

    # Text prompt
    text_prompt = "What is in this image?"

    # Dummy labels (in reality, these would be the expected output tokens)
    # For now, just use input_ids as labels for demonstration
    print("\n4. Running forward pass...")

    # Forward pass
    outputs = model.forward(
        input_pixels=dummy_image, text=text_prompt, labels=None  # Will be auto-generated from input_ids in forward()
    )

    print(f"   ✓ Forward pass successful!")
    # Handle both dict and object outputs
    if isinstance(outputs, dict):
        logits = outputs["logits"]
        loss = outputs.get("loss", None)
    else:
        logits = outputs.logits
        loss = outputs.loss if hasattr(outputs, "loss") else None

    print(f"   Output logits shape: {logits.shape}")
    if loss is not None:
        print(f"   Loss: {loss.item()}")
    else:
        print(f"   Loss: N/A (no labels provided)")

    # Backward pass
    if loss is not None:
        print("\n5. Running backward pass...")
        loss.backward()
        print("   ✓ Backward pass successful!")

        # Check gradients
        print("\n6. Checking gradients...")
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                print(f"   ✓ {name}: grad_norm={param.grad.norm().item():.6f}")
                has_grad = True

        if not has_grad:
            print("   ⚠ No gradients found! This might be expected if labels=None")
    else:
        print("\n5. No loss computed (labels not provided)")
        print("   To enable training, pass labels to forward()")

    print("\n" + "=" * 60)
    print("Training setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Load a real dataset with PIL images and labels")
    print("  2. Create a DataLoader")
    print("  3. Set up optimizer (e.g., AdamW)")
    print("  4. Training loop with forward/backward/optimizer steps")
    print("\nTo unfreeze components:")
    print("  model = TheWorld(..., freeze_gemma_vision=False)  # Train vision")
    print("  model = TheWorld(..., freeze_gemma_language=False)  # Train language")


if __name__ == "__main__":
    main()
