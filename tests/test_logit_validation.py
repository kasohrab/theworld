"""
Logit Validation Test: TheWorld vs Pure Gemma3

This test validates that TheWorld(enable_world=False) produces identical logits
to pure Gemma3ForConditionalGeneration.

This is the gold standard test that proves our refactoring preserves exact
behavior when world features are disabled.
"""

import pytest
import torch
import numpy as np
import sys
from PIL import Image

sys.path.insert(0, "/storage/ice1/7/7/ksohrab3/theworld/python")

from theworld.modeling.theworld_refactored import TheWorld
from transformers import Gemma3ForConditionalGeneration, AutoProcessor


@pytest.mark.skip(reason="Requires model download (~16GB), run manually")
def test_logit_validation_forward_pass():
    """
    Test that TheWorld(enable_world=False) produces identical logits to Gemma3.

    This validates that our refactoring is correct by comparing numerical outputs.
    """
    print("\n" + "=" * 80)
    print("LOGIT VALIDATION TEST: TheWorld vs Gemma3")
    print("=" * 80)

    model_name = "google/gemma-3-4b-it"
    device = "cpu"  # Use CPU for deterministic results

    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("\n[1/5] Loading TheWorld with enable_world=False...")
    theworld = TheWorld.from_pretrained(
        model_name,
        enable_world=False,  # No world features, should be identical to Gemma3
        dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    theworld.eval()  # CRITICAL: Put in eval mode for deterministic behavior
    print(f"   ✓ TheWorld loaded")
    print(f"   - Vocabulary size: {len(theworld.processor.tokenizer)}")
    print(f"   - Config: enable_world=False")
    print(f"   - Dtype: {next(theworld.parameters()).dtype}")
    print(f"   - Training mode: {theworld.training}")

    print("\n[2/5] Loading pure Gemma3ForConditionalGeneration...")
    gemma3 = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    gemma3.eval()  # Ensure eval mode
    print(f"   ✓ Gemma3 loaded")
    print(f"   - Dtype: {next(gemma3.parameters()).dtype}")
    print(f"   - Training mode: {gemma3.training}")

    # IMPORTANT: Use TheWorld's processor for both models to ensure identical preprocessing
    processor = theworld.processor
    print(f"   - Using TheWorld's processor for both models (ensures identical preprocessing)")

    print("\n[3/5] Creating test inputs...")
    # Create a deterministic test image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    test_prompt = "What is in this image?"

    # Prepare inputs using SAME processor for both models
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": test_image},
            {"type": "text", "text": test_prompt}
        ]
    }]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    print(f"   ✓ Test inputs created")
    print(f"   - Input shape: {inputs['input_ids'].shape}")
    print(f"   - Sequence length: {inputs['input_ids'].shape[1]}")
    print(f"   - First 10 token IDs: {inputs['input_ids'][0, :10].tolist()}")

    # IMPORTANT: Warmup pass - Gemma3 is non-deterministic on first run!
    print(f"\n[3.5/5] Running warmup passes (Gemma3 needs cache warmup)...")
    with torch.no_grad():
        _ = theworld(**inputs)  # Warmup TheWorld
        _ = gemma3(**inputs)    # Warmup Gemma3
    print(f"   ✓ Warmup completed")

    print("\n[4/5] Running actual forward pass on both models...")

    # Forward pass on TheWorld
    with torch.no_grad():
        theworld_outputs = theworld(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
        )

    # Forward pass on Gemma3
    with torch.no_grad():
        gemma3_outputs = gemma3(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
        )

    print(f"   ✓ Forward passes completed")
    print(f"   - TheWorld logits shape: {theworld_outputs.logits.shape}")
    print(f"   - Gemma3 logits shape: {gemma3_outputs.logits.shape}")

    print("\n[5/5] Comparing logits...")

    # Compare shapes
    assert theworld_outputs.logits.shape == gemma3_outputs.logits.shape, \
        f"Logit shapes don't match: {theworld_outputs.logits.shape} vs {gemma3_outputs.logits.shape}"
    print(f"   ✓ Shapes match: {theworld_outputs.logits.shape}")

    # Compare values (convert to same dtype first)
    theworld_logits = theworld_outputs.logits.float()
    gemma3_logits = gemma3_outputs.logits.float()

    # Calculate differences
    abs_diff = torch.abs(theworld_logits - gemma3_logits)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    rel_diff = abs_diff / (torch.abs(gemma3_logits) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"\n   Difference Statistics:")
    print(f"   - Max absolute diff:  {max_abs_diff:.2e}")
    print(f"   - Mean absolute diff: {mean_abs_diff:.2e}")
    print(f"   - Max relative diff:  {max_rel_diff:.2e}")
    print(f"   - Mean relative diff: {mean_rel_diff:.2e}")

    # Check if values are close (using reasonable tolerance for bfloat16)
    atol = 1e-5
    rtol = 1e-5

    are_close = torch.allclose(
        theworld_logits,
        gemma3_logits,
        atol=atol,
        rtol=rtol
    )

    if are_close:
        print(f"\n   ✅ PASS: Logits are identical within tolerance!")
        print(f"   - Absolute tolerance: {atol}")
        print(f"   - Relative tolerance: {rtol}")
    else:
        print(f"\n   ❌ FAIL: Logits differ beyond tolerance!")
        print(f"   - Absolute tolerance: {atol} (max diff: {max_abs_diff:.2e})")
        print(f"   - Relative tolerance: {rtol} (max rel diff: {max_rel_diff:.2e})")

        # Find where the differences are largest
        diff_flat = abs_diff.flatten()
        top_diffs = torch.topk(diff_flat, k=5)
        print(f"\n   Top 5 largest differences:")
        for i, (diff_val, idx) in enumerate(zip(top_diffs.values, top_diffs.indices)):
            # Convert flat index to position
            batch_idx = idx // (theworld_logits.shape[1] * theworld_logits.shape[2])
            seq_idx = (idx // theworld_logits.shape[2]) % theworld_logits.shape[1]
            vocab_idx = idx % theworld_logits.shape[2]

            theworld_val = theworld_logits[batch_idx, seq_idx, vocab_idx].item()
            gemma3_val = gemma3_logits[batch_idx, seq_idx, vocab_idx].item()

            print(f"   {i+1}. Position [batch={batch_idx}, seq={seq_idx}, vocab={vocab_idx}]")
            print(f"      TheWorld: {theworld_val:.6f}")
            print(f"      Gemma3:   {gemma3_val:.6f}")
            print(f"      Diff:     {diff_val:.6e}")

        pytest.fail(f"Logits differ: max_abs_diff={max_abs_diff:.2e}, max_rel_diff={max_rel_diff:.2e}")

    print("\n" + "=" * 80)
    print("✅ LOGIT VALIDATION TEST PASSED")
    print("=" * 80)
    print("\nConclusion: TheWorld(enable_world=False) is numerically identical to Gemma3")
    print("The refactoring successfully preserves exact behavior!")

    assert are_close, "Logits should be identical"


@pytest.mark.skip(reason="Requires model download (~16GB), run manually")
def test_logit_validation_with_labels():
    """
    Test that loss computation is also identical.

    This validates that training behavior will be identical when enable_world=False.
    """
    print("\n" + "=" * 80)
    print("LOSS VALIDATION TEST: TheWorld vs Gemma3")
    print("=" * 80)

    model_name = "google/gemma-3-4b-it"
    device = "cpu"

    torch.manual_seed(42)
    np.random.seed(42)

    print("\n[1/4] Loading models...")
    theworld = TheWorld.from_pretrained(
        model_name,
        enable_world=False,
        dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    gemma3 = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    gemma3_processor = AutoProcessor.from_pretrained(model_name)
    print("   ✓ Models loaded")
    print(f"   - TheWorld dtype: {next(theworld.parameters()).dtype}")
    print(f"   - Gemma3 dtype: {next(gemma3.parameters()).dtype}")

    print("\n[2/4] Creating test inputs with labels...")
    test_image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": test_image},
            {"type": "text", "text": "Describe this image."}
        ]
    }]

    theworld_inputs = theworld.processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)

    gemma3_inputs = gemma3_processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(device)

    # Create labels (use input_ids as labels for simplicity)
    theworld_inputs["labels"] = theworld_inputs["input_ids"].clone()
    gemma3_inputs["labels"] = gemma3_inputs["input_ids"].clone()

    print("   ✓ Inputs with labels created")

    print("\n[3/4] Running forward pass with loss computation...")

    with torch.no_grad():
        theworld_outputs = theworld(**theworld_inputs)
        gemma3_outputs = gemma3(**gemma3_inputs)

    print("   ✓ Forward passes completed")

    print("\n[4/4] Comparing losses...")

    theworld_loss = theworld_outputs.loss.item()
    gemma3_loss = gemma3_outputs.loss.item()
    loss_diff = abs(theworld_loss - gemma3_loss)

    print(f"   - TheWorld loss: {theworld_loss:.6f}")
    print(f"   - Gemma3 loss:   {gemma3_loss:.6f}")
    print(f"   - Difference:    {loss_diff:.2e}")

    # Loss should be identical (within floating point precision)
    assert loss_diff < 1e-5, f"Loss differs: {loss_diff:.2e}"

    print(f"\n   ✅ PASS: Losses are identical!")
    print("\n" + "=" * 80)
    print("✅ LOSS VALIDATION TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    """Run tests manually without pytest."""
    print("\n🔬 Running Logit Validation Tests\n")

    try:
        # Test 1: Logit validation
        print("=" * 80)
        print("TEST 1: Forward Pass Logit Validation")
        print("=" * 80)
        test_logit_validation_forward_pass()

        # Test 2: Loss validation
        print("\n\n")
        print("=" * 80)
        print("TEST 2: Loss Computation Validation")
        print("=" * 80)
        test_logit_validation_with_labels()

        print("\n\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  ✅ ALL VALIDATION TESTS PASSED".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print("\n🎉 TheWorld refactoring is validated!")
        print("   TheWorld(enable_world=False) is numerically identical to Gemma3")

    except Exception as e:
        print("\n\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  ❌ VALIDATION FAILED".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        raise
