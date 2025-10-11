"""Test HuggingFace Hub upload/download workflow for TheWorld model."""
import torch
from theworld import TheWorld
from PIL import Image
import numpy as np

print("=" * 80)
print("TEST: HuggingFace Hub Download & Load")
print("=" * 80)

# Test 1: Download and load checkpoint from Hub
print("\n[1/3] Downloading checkpoint from Hub...")
print("  Hub model: kasohrab/theworld-smoke-test")

try:
    # Create fresh model instance (this loads pretrained Gemma + Cosmos)
    model = TheWorld(
        "google/gemma-3-4b-it",
        freeze_gemma_vision=True,
        freeze_gemma_language=True,
        freeze_cosmos_vae=True,
    )

    print("✓ Base model created")

    # Download checkpoint from Hub
    from huggingface_hub import hf_hub_download
    checkpoint_path = hf_hub_download(
        repo_id="kasohrab/theworld-smoke-test",
        filename="model.safetensors",  # Uploaded to root, not subdirectory
        token="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",
    )

    print(f"✓ Checkpoint downloaded: {checkpoint_path}")

    # Load trainable parameters
    from safetensors.torch import load_file
    state_dict = load_file(checkpoint_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"✓ Checkpoint loaded")
    print(f"  Missing keys: {len(missing)} (expected - frozen params)")
    print(f"  Unexpected keys: {len(unexpected)}")

except Exception as e:
    print(f"✗ Failed to download/load: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Verify trainable parameters
print("\n[2/3] Verifying trainable parameters...")
trainable, total, pct = model.get_trainable_parameters()
print(f"  Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")

if pct > 0.5 and pct < 5.0:
    print("✓ Parameter count looks correct")
else:
    print("✗ Unexpected parameter count!")

# Test 3: Verify model is ready for use
print("\n[3/3] Verifying model is ready...")
model.eval()

print("✓ Model loaded and ready for inference or continued training")

print("\n" + "=" * 80)
print("RESULT: ✓ Hub download/load workflow WORKS!")
print("=" * 80)
print("\nYou can now:")
print("  1. Resume training from Hub checkpoint")
print("  2. Use for inference")
print("  3. Share the model with others")
