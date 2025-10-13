"""
Quick validation test for refactored TheWorld model.

Tests:
1. Model instantiation
2. HuggingFace methods (state_dict, save/load_pretrained, checkpointing)
3. Gradient checkpointing
4. Basic forward pass
"""

import sys
from pathlib import Path
import torch
import tempfile
import shutil
from PIL import Image
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld


def create_dummy_image():
    """Create a simple test image."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_instantiation():
    """Test that model can be instantiated with refactored version."""
    print("=" * 60)
    print("TEST 1: Model Instantiation")
    print("=" * 60)

    model = TheWorld.from_pretrained(
        "google/gemma-3-4b-it",
        enable_world=True,
        device="cuda",
        freeze_gemma_vision=True,
        freeze_gemma_language=True,
        freeze_cosmos_vae=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(f"✓ Model instantiated successfully")
    print(f"✓ Model type: {type(model).__name__}")
    print(f"✓ Is TheWorld: {isinstance(model, TheWorld)}")

    # Check trainable parameters
    trainable, total, pct = model.get_trainable_parameters()
    print(f"✓ Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)")

    return model


def test_hf_methods(model):
    """Test HuggingFace methods."""
    print("\n" + "=" * 60)
    print("TEST 2: HuggingFace Methods")
    print("=" * 60)

    # Test state_dict (trainable only)
    state = model.state_dict()
    print(f"✓ state_dict() works: {len(state)} trainable parameters")

    # Test save_pretrained
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model"
        model.save_pretrained(str(save_path))
        print(f"✓ save_pretrained() works")

        # Check checkpoint file exists
        checkpoint_file = save_path / "pytorch_model.bin"
        assert checkpoint_file.exists(), "Checkpoint file not created"
        print(f"✓ Checkpoint file created: {checkpoint_file.stat().st_size / 1e6:.1f} MB")

    # Test save_checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        # Create dummy optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Save checkpoint
        model.save_checkpoint(
            str(checkpoint_path),
            optimizer=optimizer,
            epoch=5,
            step=1000,
        )
        print(f"✓ save_checkpoint() works")

        # Load checkpoint
        info = model.load_checkpoint(str(checkpoint_path), optimizer=optimizer)
        print(f"✓ load_checkpoint() works: epoch={info['epoch']}, step={info['step']}")
        assert info['epoch'] == 5, "Epoch not restored correctly"
        assert info['step'] == 1000, "Step not restored correctly"


def test_gradient_checkpointing(model):
    """Test gradient checkpointing method."""
    print("\n" + "=" * 60)
    print("TEST 3: Gradient Checkpointing")
    print("=" * 60)

    # This should work without errors
    model.enable_gradient_checkpointing()
    print("✓ enable_gradient_checkpointing() works")


def test_forward_and_backward(model):
    """Test both forward pass and backward pass (gradient flow)."""
    print("\n" + "=" * 60)
    print("TEST 4: Forward & Backward Pass")
    print("=" * 60)

    # Create dummy inputs
    image = create_dummy_image()
    text = "What is in this image?"

    # Process inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text}
            ]
        }
    ]

    inputs = model.processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Detect the model's device (with device_map="auto", different parts can be on different devices)
    embed_device = model.get_input_embeddings().weight.device
    print(f"✓ Detected model embedding device: {embed_device}")

    # Move inputs to the correct device
    inputs = {k: v.to(embed_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Create dummy labels for loss computation (needed for backward pass)
    labels = inputs["input_ids"].clone()

    # Forward pass WITH gradients (for backward test)
    outputs = model.forward(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        attention_mask=inputs["attention_mask"],
        labels=labels,
    )

    print(f"✓ Forward pass works")
    print(f"✓ Output logits shape: {outputs.logits.shape}")
    print(f"✓ Loss computed: {outputs.loss.item():.4f}")

    # Verify output
    assert outputs.logits is not None, "No logits in output"
    assert len(outputs.logits.shape) == 3, "Wrong logits shape"
    assert outputs.loss is not None, "No loss in output"
    print(f"✓ Output format correct")

    # Test backward pass (gradient flow)
    print("\nTesting backward pass...")
    outputs.loss.backward()

    # Check that trainable parameters have gradients
    trainable_params_with_grad = 0
    trainable_params_total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params_total += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                trainable_params_with_grad += 1

    print(f"✓ Backward pass works")
    print(f"✓ Trainable parameters with gradients: {trainable_params_with_grad}/{trainable_params_total}")

    # Verify gradients flow to trainable parameters
    assert trainable_params_with_grad > 0, "No gradients computed for trainable parameters"
    print(f"✓ Gradients flow correctly to trainable parameters")


def test_world_augmented_forward(model):
    """Test forward pass WITH world tokens (using Cosmos world model)."""
    print("\n" + "=" * 60)
    print("TEST 5: World-Augmented Forward & Backward")
    print("=" * 60)

    # Create dummy inputs
    image = create_dummy_image()
    text = "Describe what happens next in this image."

    # Detect device
    embed_device = model.get_input_embeddings().weight.device
    print(f"✓ Detected model embedding device: {embed_device}")

    # Manually create input with SOW/EOW tokens
    # Format: [BOS, SOW, EOW, text_tokens, image_placeholder, EOS]
    processor = model.processor

    # Tokenize text
    text_tokens = processor.tokenizer.encode(text, add_special_tokens=False)

    # Build sequence: [BOS, SOW, EOW, text, image_placeholders (256), EOS]
    BOS = 2
    EOS = 1
    IMAGE_TOKEN = 262144
    NUM_IMAGE_TOKENS = 256  # SigLIP produces 256 image features
    SOW = model.sow_token_id
    EOW = model.eow_token_id

    input_ids = torch.tensor(
        [[BOS, SOW, EOW] + text_tokens + [IMAGE_TOKEN] * NUM_IMAGE_TOKENS + [EOS]],
        device=embed_device
    )
    attention_mask = torch.ones_like(input_ids)

    # Process image for pixel_values (for Gemma SigLIP)
    from PIL import Image as PILImage
    pixel_values = processor.image_processor(image, return_tensors="pt")["pixel_values"].to(embed_device)

    # Create dummy labels
    labels = input_ids.clone()

    print(f"✓ Created input with world tokens: SOW={SOW}, EOW={EOW}")
    print(f"✓ Input shape: {input_ids.shape}")

    # Forward pass WITH world tokens and raw images
    outputs = model.forward(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        labels=labels,
        images=[image],  # Raw PIL image for Cosmos
    )

    print(f"✓ World-augmented forward pass works")
    print(f"✓ Output logits shape: {outputs.logits.shape}")
    print(f"✓ Loss computed: {outputs.loss.item():.4f}")

    # Verify output
    assert outputs.logits is not None, "No logits in output"
    assert outputs.loss is not None, "No loss in output"
    print(f"✓ World model integration works")

    # Test backward pass with world model
    print("\nTesting backward pass with world model...")
    outputs.loss.backward()

    # Check gradients flow to projection layer
    projection_grad_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad and "cosmos_encoder" in name:
            if param.grad is not None and param.grad.abs().sum() > 0:
                projection_grad_count += 1
                print(f"✓ Gradient flows to: {name}")

    print(f"✓ Backward pass works with world model")
    assert projection_grad_count > 0, "No gradients in Cosmos projection layer"
    print(f"✓ Gradients flow correctly to world projection layer")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("REFACTORED THEWORLD MODEL VALIDATION")
    print("=" * 60)

    try:
        # Test 1: Instantiation
        model = test_instantiation()

        # Test 2: HF methods
        test_hf_methods(model)

        # Test 3: Gradient checkpointing
        test_gradient_checkpointing(model)

        # Test 4: Forward & Backward pass
        test_forward_and_backward(model)

        # Test 5: World-augmented forward (WITH world tokens)
        test_world_augmented_forward(model)

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe refactored TheWorld model works correctly!")
        print("All HuggingFace integration methods are functional.")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
