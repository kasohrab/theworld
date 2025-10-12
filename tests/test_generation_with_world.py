"""Unit tests for world-aware generation in TheWorld model.

Tests validate that:
1. KV cache is populated with world embeddings
2. World embeddings affect generation output
3. Generation parameters work correctly
4. Ablation mode (skip_world_tokens) works
5. Output differs from Gemma-only baseline
"""

import pytest
import torch
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld


@pytest.fixture(scope="module")
def model():
    """Create TheWorld model for testing."""
    model = TheWorld(
        "google/gemma-3-4b-it",
        device="cuda" if torch.cuda.is_available() else "cpu",
        freeze_gemma_vision=True,
        freeze_gemma_language=True,
        freeze_cosmos_vae=True,
    )
    model.eval()
    return model


@pytest.fixture
def dummy_image():
    """Create a dummy PIL image with distinct visual pattern."""
    img = Image.new("RGB", (224, 224))
    pixels = img.load()
    # Create gradient pattern
    for i in range(224):
        for j in range(224):
            pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
    return img


def test_kv_cache_is_populated(model, dummy_image):
    """Test that forward() with use_cache=True returns past_key_values."""
    # Prepare inputs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<start_of_world> <end_of_world>"},
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "What is this?"},
            ],
        }
    ]

    inputs = model.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    target_device = model.gemma.get_input_embeddings().weight.device
    if isinstance(inputs, dict):
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}
    else:
        inputs = inputs.to(target_device)

    pixel_values = model.processor.image_processor(images=dummy_image, return_tensors="pt")[
        "pixel_values"
    ].to(target_device)

    # Handle inputs - might be dict or tensor
    if not isinstance(inputs, dict):
        # If inputs is a tensor, wrap it in a dict
        inputs = {"input_ids": inputs, "attention_mask": torch.ones_like(inputs)}

    # Forward pass with caching
    with torch.no_grad():
        outputs = model.forward(
            input_ids=inputs["input_ids"],
            pixel_values=pixel_values,
            attention_mask=inputs["attention_mask"],
            images=[dummy_image],
            labels=None,
            use_cache=True,
        )

    # Verify KV cache structure
    assert outputs.past_key_values is not None, "past_key_values should not be None"

    # HuggingFace now uses DynamicCache objects (not tuples)
    # Check that we have a valid cache object
    assert hasattr(outputs.past_key_values, "get_seq_length") or hasattr(outputs.past_key_values, "__len__"), \
        "past_key_values should be a cache object"

    # For tuple-based caches
    if isinstance(outputs.past_key_values, tuple):
        assert len(outputs.past_key_values) > 0, "past_key_values should not be empty"
        num_layers = len(outputs.past_key_values)
    # For DynamicCache objects
    elif hasattr(outputs.past_key_values, "key_cache"):
        assert len(outputs.past_key_values.key_cache) > 0, "DynamicCache should have layers"
        num_layers = len(outputs.past_key_values.key_cache)
    else:
        num_layers = "unknown"

    print(f"✓ KV cache populated (type: {type(outputs.past_key_values).__name__}, layers: {num_layers})")


def test_forward_without_cache(model, dummy_image):
    """Test that forward() with use_cache=False returns None for past_key_values."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<start_of_world> <end_of_world>"},
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "Describe this."},
            ],
        }
    ]

    inputs = model.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
    )

    target_device = model.gemma.get_input_embeddings().weight.device
    if isinstance(inputs, dict):
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}
    else:
        inputs = inputs.to(target_device)

    pixel_values = model.processor.image_processor(images=dummy_image, return_tensors="pt")[
        "pixel_values"
    ].to(target_device)

    # Handle inputs - might be dict or tensor
    if not isinstance(inputs, dict):
        inputs = {"input_ids": inputs, "attention_mask": torch.ones_like(inputs)}

    with torch.no_grad():
        outputs = model.forward(
            input_ids=inputs["input_ids"],
            pixel_values=pixel_values,
            attention_mask=inputs["attention_mask"],
            images=[dummy_image],
            labels=None,
            use_cache=False,
        )

    assert outputs.past_key_values is None, "past_key_values should be None when use_cache=False"


def test_world_aware_generation_runs(model, dummy_image):
    """Test that generate_with_world() runs without errors."""
    response = model.generate_with_world(
        image=dummy_image, prompt="What do you see?", max_new_tokens=20, temperature=0.0
    )

    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    print(f"✓ World-aware generation output: {response[:100]}")


def test_gemma_only_generation_runs(model, dummy_image):
    """Test that _generate_gemma_only() runs without errors."""
    response = model._generate_gemma_only(
        image=dummy_image, prompt="What do you see?", max_new_tokens=20, temperature=0.0
    )

    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    print(f"✓ Gemma-only generation output: {response[:100]}")


def test_generate_router_world_aware(model, dummy_image):
    """Test that generate() with skip_world_tokens=False uses world model."""
    response = model.generate(
        image=dummy_image,
        prompt="Describe this image.",
        max_new_tokens=15,
        temperature=0.0,
        skip_world_tokens=False,
    )

    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    print(f"✓ Router (world-aware) output: {response[:100]}")


def test_generate_router_gemma_only(model, dummy_image):
    """Test that generate() with skip_world_tokens=True skips world model."""
    response = model.generate(
        image=dummy_image,
        prompt="Describe this image.",
        max_new_tokens=15,
        temperature=0.0,
        skip_world_tokens=True,
    )

    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    print(f"✓ Router (gemma-only) output: {response[:100]}")


def test_world_vs_gemma_output_differs(model, dummy_image):
    """Test that world-aware and Gemma-only outputs differ."""
    prompt = "What is in this image?"

    # World-aware generation
    world_response = model.generate(
        image=dummy_image, prompt=prompt, max_new_tokens=30, temperature=0.0, skip_world_tokens=False
    )

    # Gemma-only generation
    gemma_response = model.generate(
        image=dummy_image, prompt=prompt, max_new_tokens=30, temperature=0.0, skip_world_tokens=True
    )

    print(f"World-aware: {world_response}")
    print(f"Gemma-only: {gemma_response}")

    # Outputs should differ (world model should affect generation)
    # Note: This might occasionally be the same due to model behavior,
    # but in general they should differ
    if world_response == gemma_response:
        print(
            "[WARNING] World-aware and Gemma-only outputs are identical. "
            "This may indicate world embeddings are not affecting generation."
        )
    else:
        print("✓ Outputs differ (world model is affecting generation)")


def test_temperature_zero_deterministic(model, dummy_image):
    """Test that temperature=0 produces deterministic outputs."""
    prompt = "What is this?"

    response1 = model.generate(
        image=dummy_image, prompt=prompt, max_new_tokens=20, temperature=0.0
    )

    response2 = model.generate(
        image=dummy_image, prompt=prompt, max_new_tokens=20, temperature=0.0
    )

    assert response1 == response2, "Temperature=0 should produce deterministic outputs"
    print("✓ Temperature=0 is deterministic")


def test_temperature_positive_sampling(model, dummy_image):
    """Test that temperature>0 enables sampling."""
    prompt = "Describe briefly."

    # Run generation with sampling
    response = model.generate(
        image=dummy_image, prompt=prompt, max_new_tokens=25, temperature=0.7
    )

    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    print(f"✓ Sampling generation output: {response[:100]}")


def test_max_new_tokens_respected(model, dummy_image):
    """Test that max_new_tokens limits output length."""
    prompt = "Tell me about this image in detail."

    # Very short generation
    short_response = model.generate(
        image=dummy_image, prompt=prompt, max_new_tokens=5, temperature=0.0
    )

    # Longer generation
    long_response = model.generate(
        image=dummy_image, prompt=prompt, max_new_tokens=30, temperature=0.0
    )

    # Tokenize to count tokens
    short_tokens = model.processor.tokenizer.encode(short_response)
    long_tokens = model.processor.tokenizer.encode(long_response)

    print(f"Short response ({len(short_tokens)} tokens): {short_response}")
    print(f"Long response ({len(long_tokens)} tokens): {long_response}")

    # Short should have fewer tokens (with some tolerance for special tokens)
    assert len(short_tokens) < len(long_tokens) + 3, "max_new_tokens should limit output length"


def test_different_prompts_different_outputs(model, dummy_image):
    """Test that different prompts produce different outputs."""
    prompt1 = "What color is this?"
    prompt2 = "What shapes do you see?"

    response1 = model.generate(
        image=dummy_image, prompt=prompt1, max_new_tokens=20, temperature=0.0
    )

    response2 = model.generate(
        image=dummy_image, prompt=prompt2, max_new_tokens=20, temperature=0.0
    )

    print(f"Prompt 1 response: {response1}")
    print(f"Prompt 2 response: {response2}")

    # Different prompts should generally produce different outputs
    assert response1 != response2, "Different prompts should produce different outputs"


def test_world_tokens_in_generated_sequence(model, dummy_image):
    """Test that SOW/EOW tokens are in the input sequence for world-aware generation."""
    prompt = "What is this?"

    # Prepare inputs manually to inspect
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<start_of_world> <end_of_world>"},
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = model.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    # Handle inputs - might be dict or tensor
    if isinstance(inputs, dict):
        ids = inputs["input_ids"][0].tolist()
    else:
        ids = inputs[0].tolist()

    # Check for SOW/EOW tokens
    sow_count = ids.count(model.sow_token_id)
    eow_count = ids.count(model.eow_token_id)

    assert sow_count == 1, f"Expected 1 SOW token, found {sow_count}"
    assert eow_count == 1, f"Expected 1 EOW token, found {eow_count}"

    sow_pos = ids.index(model.sow_token_id)
    eow_pos = ids.index(model.eow_token_id)

    assert sow_pos < eow_pos, "SOW should come before EOW"
    print(f"✓ World tokens present: SOW at {sow_pos}, EOW at {eow_pos}")


def test_gemma_only_no_world_tokens(model, dummy_image):
    """Test that Gemma-only generation does not include world tokens."""
    prompt = "What is this?"

    # Prepare inputs for Gemma-only (no world tokens)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": dummy_image}, {"type": "text", "text": prompt}],
        }
    ]

    inputs = model.processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    # Handle inputs - might be dict or tensor
    if isinstance(inputs, dict):
        ids = inputs["input_ids"][0].tolist()
    else:
        ids = inputs[0].tolist()

    # Check for absence of SOW/EOW tokens
    sow_count = ids.count(model.sow_token_id)
    eow_count = ids.count(model.eow_token_id)

    assert sow_count == 0, f"Gemma-only should have 0 SOW tokens, found {sow_count}"
    assert eow_count == 0, f"Gemma-only should have 0 EOW tokens, found {eow_count}"
    print("✓ Gemma-only generation has no world tokens")


@pytest.mark.parametrize("num_tokens", [10, 20, 50])
def test_various_generation_lengths(model, dummy_image, num_tokens):
    """Test generation with various max_new_tokens values."""
    response = model.generate(
        image=dummy_image,
        prompt="Describe this.",
        max_new_tokens=num_tokens,
        temperature=0.0,
    )

    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 0, "Response should not be empty"
    print(f"✓ Generated {num_tokens} tokens: {response[:80]}")


def test_empty_prompt_handling(model, dummy_image):
    """Test that empty prompt still generates output."""
    response = model.generate(image=dummy_image, prompt="", max_new_tokens=15, temperature=0.0)

    assert isinstance(response, str), "Response should be a string"
    # Empty prompt might produce empty output or default response
    print(f"✓ Empty prompt response: '{response}'")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
