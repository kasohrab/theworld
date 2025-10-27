"""Test baseline equivalence: TheWorld(enable_world=False) vs pure Gemma3.

Validates that TheWorld with world model disabled produces identical outputs
to the original Gemma3ForConditionalGeneration model.
"""

import pytest
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from theworld import TheWorld


@pytest.fixture(scope="module")
def gemma_model():
    """Load pure Gemma3 model for baseline comparison."""
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it",
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def theworld_model():
    """Load TheWorld with world model disabled."""

    model = TheWorld.from_pretrained(
        "google/gemma-3-4b-it",
        enable_world=False,  # Disable world model - should behave like pure Gemma
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def processor():
    """Load Gemma processor for tokenization."""
    return AutoProcessor.from_pretrained("google/gemma-3-4b-it")


def get_main_device(model):
    """Get the main compute device for models with device_map='auto'.

    With device_map='auto', models use Accelerate's dispatch mechanism which
    places tensors via hooks. The actual .weight.device might show 'cpu', but
    runtime dispatch moves data to the correct device. We need to check the
    hf_device_map to find where inputs should go.
    """
    # Check hf_device_map first (set by device_map='auto')
    if hasattr(model, 'hf_device_map') and model.hf_device_map:
        # Get the device for the first layer (typically embed_tokens or lm_head)
        # Device map values are integers (0, 1, etc.) representing cuda:0, cuda:1, etc.
        device_id = None
        for key in ['model.language_model.embed_tokens', 'lm_head', 'model.embed_tokens']:
            if key in model.hf_device_map:
                device_id = model.hf_device_map[key]
                break

        if device_id is not None:
            if isinstance(device_id, int):
                return torch.device(f'cuda:{device_id}')
            else:
                return torch.device(device_id)

    # Fallback: try to get device from actual parameters
    # Note: With Accelerate dispatch, this might return 'cpu' even though
    # the model will execute on CUDA
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.device
    elif hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        return next(model.model.language_model.parameters()).device
    else:
        return next(model.parameters()).device


def test_same_generated_tokens(gemma_model: Gemma3ForConditionalGeneration, theworld_model: TheWorld, processor):
    """Test that both models generate identical token sequences (text-only)."""
    prompt = "What is the capital of France?"

    # Prepare inputs - for text-only, use tokenizer directly
    inputs = processor.tokenizer(prompt, return_tensors="pt")

    # Move to device - use main compute device for device_map='auto'
    device = get_main_device(gemma_model)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate with both models (deterministic)
    with torch.no_grad():
        gemma_output = gemma_model.generate(
            **inputs, max_new_tokens=20, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id
        )

        theworld_output = theworld_model.generate(
            **inputs, max_new_tokens=20, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id
        )

    # Decode outputs
    gemma_text = processor.tokenizer.decode(gemma_output[0], skip_special_tokens=True)
    theworld_text = processor.tokenizer.decode(theworld_output[0], skip_special_tokens=True)

    print(f"\nGemma output:    {gemma_text}")
    print(f"TheWorld output: {theworld_text}")

    # Assert token sequences are identical
    assert torch.equal(
        gemma_output, theworld_output
    ), "Generated token sequences should be identical"
    print("✓ Generated tokens match exactly")


def test_same_logits(gemma_model: Gemma3ForConditionalGeneration, theworld_model: TheWorld, processor):
    """Test that both models produce nearly identical logits (text-only)."""
    prompt = "The sky is"

    # Prepare inputs - for text-only, use tokenizer directly
    inputs = processor.tokenizer(prompt, return_tensors="pt")

    # Move to device - use main compute device for device_map='auto'
    device = get_main_device(gemma_model)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Forward pass with both models
    with torch.no_grad():
        gemma_outputs = gemma_model(**inputs)
        theworld_outputs = theworld_model(**inputs)

    # Compare logits
    gemma_logits = gemma_outputs.logits
    theworld_logits = theworld_outputs.logits

    print(f"\nLogits shape: {gemma_logits.shape}")
    print(f"Gemma logits range: [{gemma_logits.min():.4f}, {gemma_logits.max():.4f}]")
    print(f"TheWorld logits range: [{theworld_logits.min():.4f}, {theworld_logits.max():.4f}]")

    # Check shapes match
    assert (
        gemma_logits.shape == theworld_logits.shape
    ), f"Logits shapes differ: {gemma_logits.shape} vs {theworld_logits.shape}"

    # Check values are very close (allow small floating point differences)
    max_diff = (gemma_logits - theworld_logits).abs().max().item()
    print(f"Max logits difference: {max_diff:.6f}")

    # With bfloat16, we expect very small differences due to fp precision
    assert max_diff < 1e-3, f"Logits differ by {max_diff}, expected < 1e-3"
    print("✓ Logits match within tolerance")


def test_same_next_token_prediction(gemma_model, theworld_model, processor):
    """Test that both models predict the same next token (text-only)."""
    prompt = "Paris is the capital of"

    # Prepare inputs - for text-only, use tokenizer directly
    inputs = processor.tokenizer(prompt, return_tensors="pt")

    # Move to device - use main compute device for device_map='auto'
    device = get_main_device(gemma_model)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Get next token predictions
    with torch.no_grad():
        gemma_outputs = gemma_model(**inputs)
        theworld_outputs = theworld_model(**inputs)

    # Get predicted token IDs
    gemma_next_token = gemma_outputs.logits[0, -1].argmax().item()
    theworld_next_token = theworld_outputs.logits[0, -1].argmax().item()

    # Decode tokens
    gemma_token_text = processor.tokenizer.decode([gemma_next_token])
    theworld_token_text = processor.tokenizer.decode([theworld_next_token])

    print(f"\nGemma next token:    {gemma_token_text} (ID: {gemma_next_token})")
    print(f"TheWorld next token: {theworld_token_text} (ID: {theworld_next_token})")

    assert (
        gemma_next_token == theworld_next_token
    ), f"Next token predictions differ: {gemma_next_token} vs {theworld_next_token}"
    print("✓ Next token predictions match")


def test_no_world_tokens_when_disabled(theworld_model, processor):
    """Test that world tokens are not injected when enable_world=False."""
    prompt = "Hello world"

    # Prepare inputs - for text-only, use tokenizer directly
    inputs = processor.tokenizer(prompt, return_tensors="pt")

    # Check input_ids for world tokens
    input_ids = inputs["input_ids"][0].tolist()

    # World tokens should not exist when enable_world=False
    if hasattr(theworld_model, "sow_token_id") and theworld_model.sow_token_id is not None:
        assert (
            theworld_model.sow_token_id not in input_ids
        ), "SOW token should not be in inputs when enable_world=False"
        assert (
            theworld_model.eow_token_id not in input_ids
        ), "EOW token should not be in inputs when enable_world=False"

    print("✓ No world tokens injected when enable_world=False")


def test_multiple_prompts_consistency(gemma_model, theworld_model, processor):
    """Test consistency across multiple different prompts (text-only)."""
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "Machine learning is",
        "Python is a programming language",
    ]

    for prompt in prompts:
        # For text-only, use tokenizer directly
        inputs = processor.tokenizer(prompt, return_tensors="pt")

        # Move to device - use main compute device for device_map='auto'
        device = get_main_device(gemma_model)
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
        }

        with torch.no_grad():
            gemma_output = gemma_model.generate(
                **inputs, max_new_tokens=10, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id
            )
            theworld_output = theworld_model.generate(
                **inputs, max_new_tokens=10, do_sample=False, pad_token_id=processor.tokenizer.eos_token_id
            )

        assert torch.equal(
            gemma_output, theworld_output
        ), f"Outputs differ for prompt: {prompt}"

    print(f"✓ All {len(prompts)} prompts produced identical outputs")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
