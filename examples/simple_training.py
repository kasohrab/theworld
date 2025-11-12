from theworld import TheWorld
from theworld.constants import DEFAULT_GEMMA_MODEL
from theworld.data import create_theworld_collator
from PIL import Image
import sys


def test_projection_mode(mode: str):
    """Test forward/backward pass with given projection mode."""
    print("=" * 80)
    print(f"TheWorld Model - Training Example ({mode.upper()} mode)")
    print("=" * 80)

    # Initialize model with trainable projection layers only
    print(f"\n1. Loading model with {mode} projection mode...")
    model = TheWorld.from_pretrained(
        DEFAULT_GEMMA_MODEL,
        enable_world=True,
        world_projection_mode=mode,
        device_map="auto",
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
    print(f"   - world_projector: {sum(p.numel() for p in model.world_projector.parameters()):,}")

    # Create dummy training data
    print("\n3. Creating dummy training batch...")
    # Random image (896x896 for Gemma 3)
    dummy_image = Image.new("RGB", (896, 896), color=(100, 150, 200))

    # Text prompt
    text_prompt = "What is in this image?"

    # Expected answer (label)
    expected_answer = "This is a test image."

    # Create collator for preprocessing
    collate_fn = create_theworld_collator(model)

    # Prepare batch
    batch = [{"image": dummy_image, "text": text_prompt, "label": expected_answer}]
    inputs = collate_fn(batch)

    # Get main compute device for device_map='auto'
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        # Use first available device from device map
        device_id = list(model.hf_device_map.values())[0]
        device = device_id if isinstance(device_id, str) else f"cuda:{device_id}"
    else:
        device = next(model.parameters()).device

    # Move tensors to device (preserve TypedDict type)
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    if inputs["labels"] is not None:
        inputs["labels"] = inputs["labels"].to(device)

    print("\n4. Running forward pass...")

    # Forward pass
    outputs = model.forward(**inputs)

    assert outputs.logits is not None, "Expected logits in output"
    print(f"   ✓ Forward pass successful!")
    print(f"   Output logits shape: {outputs.logits.shape}")
    if outputs.loss is not None:
        print(f"   Loss: {outputs.loss.item():.4f}")
    else:
        print(f"   Loss: N/A (no labels provided)")

    # Backward pass
    if outputs.loss is not None:
        print("\n5. Running backward pass...")
        outputs.loss.backward()
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

    print("\n" + "=" * 80)
    print(f"✓ {mode.upper()} mode test complete!")
    print("=" * 80)

    return model


def main():
    """Test both projection modes."""
    import torch

    # Parse command line args
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ["spatial", "channel"]:
            print(f"Error: Invalid mode '{mode}'. Use 'spatial' or 'channel'.")
            sys.exit(1)
        test_projection_mode(mode)
    else:
        # Test both modes
        print("\nTesting SPATIAL mode (default: 4096 tokens)...")
        test_projection_mode("spatial")

        # Clean up GPU memory
        torch.cuda.empty_cache()

        print("\n\nTesting CHANNEL mode (experimental: 16 tokens)...")
        test_projection_mode("channel")

    print("\n" + "=" * 80)
    print("All tests complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Load a real dataset with PIL images and labels")
    print("  2. Create a DataLoader with create_theworld_collator(model)")
    print("  3. Set up optimizer (e.g., AdamW)")
    print("  4. Training loop with forward/backward/optimizer steps")
    print("\nTo unfreeze components:")
    print("  model = TheWorld.from_pretrained(..., freeze_gemma_vision=False)  # Train vision")
    print("  model = TheWorld.from_pretrained(..., freeze_gemma_language=False)  # Train language")


if __name__ == "__main__":
    main()
