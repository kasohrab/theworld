from theworld import TheWorld
from theworld.constants import DEFAULT_GEMMA_MODEL
from theworld.data import create_theworld_collator
from theworld.generation import greedy_decode
from PIL import Image


def main():
    # Set device
    device = "cuda"
    print(f"Using device: {device}")

    # Create a single model instance
    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)

    model = TheWorld(DEFAULT_GEMMA_MODEL, device=device, num_world_steps=4)

    # Create dummy image (random for testing)
    dummy_image = Image.new("RGB", (896, 896), color=(100, 150, 200))
    text_prompt = "What is in this image?"

    # Create collator for preprocessing
    collate_fn = create_theworld_collator(model)

    # Example 1: Single-step (no rollout) - Fast inference
    print("\n" + "=" * 60)
    print("Example 1: Single-step world model (no future prediction)")
    print("=" * 60)

    print("Running single-step forward pass (override to 0 steps)...")

    # Prepare batch
    batch = [{"image": dummy_image, "text": text_prompt, "label": None}]
    inputs = collate_fn(batch)

    # Move tensors to device (preserve TypedDict type)
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    if inputs["labels"] is not None:
        inputs["labels"] = inputs["labels"].to(device)

    # Forward pass with override
    outputs_single = model.forward(**inputs, num_world_steps=0)

    assert outputs_single.logits is not None, "Expected logits in output"
    print(f"✓ Output shape: {outputs_single.logits.shape}")
    print(f"  - Includes: Gemma vision tokens + Cosmos world tokens + text")
    print(f"  - Total context: {outputs_single.logits.shape[1]} tokens")

    # Decode next token prediction
    next_token_id, next_token_text = greedy_decode(outputs_single, model.processor.tokenizer)
    print(f"\n  Next token prediction:")
    print(f"    Token ID: {next_token_id}")
    print(f"    Decoded: '{next_token_text}'")
    print(f"    (Greedy decoding - highest probability token)")

    # Example 2: Multi-step rollout (4 future frames)
    print("\n" + "=" * 60)
    print("Example 2: Multi-step rollout (predict 4 future frames)")
    print("=" * 60)

    print("Running multi-step forward pass (predicting 4 future frames)...")
    print("This will take longer as Cosmos generates future states...")

    # Prepare batch (same as before)
    batch = [{"image": dummy_image, "text": text_prompt, "label": None}]
    inputs = collate_fn(batch)

    # Move tensors to device (preserve TypedDict type)
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    if inputs["labels"] is not None:
        inputs["labels"] = inputs["labels"].to(device)

    # Forward pass with default num_world_steps=4
    outputs_multi = model.forward(**inputs)

    assert outputs_multi.logits is not None, "Expected logits in output"
    print(f"✓ Output shape: {outputs_multi.logits.shape}")
    print(f"  - Gemma vision: ~256 tokens (896x896 at 14px patches)")
    print(f"  - Cosmos world: ~3920 tokens (5 frames × 28×28)")
    print(f"  - Total context: {outputs_multi.logits.shape[1]} tokens")
    print(f"  - Frames: 1 (input) + 4 (predicted future)")

    # Decode next token prediction
    next_token_id, next_token_text = greedy_decode(outputs_multi, model.processor.tokenizer)
    print(f"\n  Next token prediction (with temporal context):")
    print(f"    Token ID: {next_token_id}")
    print(f"    Decoded: '{next_token_text}'")

    # Example 3: Override rollout at inference time
    print("\n" + "=" * 60)
    print("Example 3: Override rollout length at inference time")
    print("=" * 60)

    print("Model initialized with num_world_steps=4, but overriding to 2...")

    batch = [{"image": dummy_image, "text": text_prompt, "label": None}]
    inputs = collate_fn(batch)

    # Move tensors to device (preserve TypedDict type)
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    if inputs["labels"] is not None:
        inputs["labels"] = inputs["labels"].to(device)

    # Override to predict only 2 future frames
    outputs_override = model.forward(**inputs, num_world_steps=2)

    assert outputs_override.logits is not None, "Expected logits in output"
    print(f"✓ Output shape: {outputs_override.logits.shape}")
    print(f"  - Cosmos world: ~2352 tokens (3 frames × 28×28)")
    print(f"  - Frames: 1 (input) + 2 (predicted future)")

    # Decode next token prediction
    next_token_id, next_token_text = greedy_decode(outputs_override, model.processor.tokenizer)
    print(f"\n  Next token prediction:")
    print(f"    Token ID: {next_token_id}")
    print(f"    Decoded: '{next_token_text}'")

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nKey features:")
    print("  - Dual vision: Gemma's SigLIP + Cosmos world model")
    print("  - Gemma provides: static visual understanding (objects, scenes)")
    print("  - Cosmos provides: temporal dynamics (motion, physics, future states)")
    print("\nSee docs/ for details:")
    print("  - autoregressive_world_rollout.md: How rollout works")
    print("  - world_model_latent_space.md: Cosmos latent extraction")


if __name__ == "__main__":
    main()
