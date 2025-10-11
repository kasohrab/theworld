"""Unit tests for tokenization in TheWorld model.

Tests validate that:
1. BOS token is at position 0
2. Image tokens are present
3. SOW/EOW custom tokens are correctly registered
4. Chat template produces valid sequences
5. Label construction preserves BOS
"""

import pytest
import torch
from PIL import Image
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld, create_theworld_collator
from theworld.constants import BOS_TOKEN_ID, EOS_TOKEN_ID, IMAGE_SOFT_TOKEN_ID


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
    return model


@pytest.fixture
def dummy_image():
    """Create a dummy PIL image."""
    return Image.new("RGB", (224, 224), color=(100, 150, 200))


def test_bos_token_constant():
    """Test that BOS token ID constant is correct."""
    assert BOS_TOKEN_ID == 2
    assert EOS_TOKEN_ID == 1
    assert IMAGE_SOFT_TOKEN_ID == 262144


def test_custom_tokens_registered(model):
    """Test that custom tokens are registered in vocabulary."""
    vocab = model.processor.tokenizer.get_vocab()

    # Check tokens exist
    assert "<start_of_world>" in vocab, "SOW token not in vocabulary"
    assert "<end_of_world>" in vocab, "EOW token not in vocabulary"

    # Check IDs are accessible
    assert hasattr(model, "sow_token_id")
    assert hasattr(model, "eow_token_id")
    assert isinstance(model.sow_token_id, int)
    assert isinstance(model.eow_token_id, int)

    # Check IDs are different
    assert model.sow_token_id != model.eow_token_id

    print(f"✓ SOW token ID: {model.sow_token_id}")
    print(f"✓ EOW token ID: {model.eow_token_id}")


def test_embedding_layer_resized(model):
    """Test that embedding layer was resized to include custom tokens."""
    vocab_size = len(model.processor.tokenizer)
    embedding_size = model.gemma.get_input_embeddings().num_embeddings

    assert embedding_size == vocab_size, (
        f"Embedding size ({embedding_size}) doesn't match vocab size ({vocab_size})"
    )

    # Should be base vocab (256000) + 2 custom tokens
    assert vocab_size >= 256002


def test_chat_template_adds_bos(model, dummy_image):
    """Test that chat template adds BOS token at position 0."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "What is this?"}
            ]
        }
    ]

    inputs = model.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )

    ids = inputs["input_ids"][0].tolist()

    # BOS should be at position 0
    assert ids[0] == BOS_TOKEN_ID, f"Expected BOS (ID 2) at position 0, got {ids[0]}"


def test_chat_template_adds_image_tokens(model, dummy_image):
    """Test that chat template processes images correctly."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "Describe this image."}
            ]
        }
    ]

    inputs = model.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )

    ids = inputs["input_ids"][0].tolist()

    # Image soft tokens should be present
    image_token_count = ids.count(IMAGE_SOFT_TOKEN_ID)
    assert image_token_count > 0, "No image tokens found after applying chat template"

    print(f"✓ Image token count: {image_token_count}")


def test_collator_adds_world_tokens(model, dummy_image):
    """Test that collator adds SOW/EOW tokens."""
    collate_fn = create_theworld_collator(model)

    batch = [
        {
            "image": dummy_image,
            "text": "What is in this image?",
            "label": "A test image."
        }
    ]

    inputs = collate_fn(batch)

    # Check SOW/EOW tokens present
    ids = inputs["input_ids"][0].tolist()
    sow_count = ids.count(model.sow_token_id)
    eow_count = ids.count(model.eow_token_id)

    assert sow_count == 1, f"Expected 1 SOW token, found {sow_count}"
    assert eow_count == 1, f"Expected 1 EOW token, found {eow_count}"


def test_world_tokens_ordered_correctly(model, dummy_image):
    """Test that SOW comes before EOW in sequence."""
    collate_fn = create_theworld_collator(model)

    batch = [
        {
            "image": dummy_image,
            "text": "Test question?",
            "label": "Test answer."
        }
    ]

    inputs = collate_fn(batch)
    ids = inputs["input_ids"][0].tolist()

    sow_pos = ids.index(model.sow_token_id)
    eow_pos = ids.index(model.eow_token_id)

    assert sow_pos < eow_pos, f"SOW position ({sow_pos}) must be before EOW ({eow_pos})"


def test_token_sequence_structure(model, dummy_image):
    """Test complete token sequence structure."""
    collate_fn = create_theworld_collator(model)

    batch = [
        {
            "image": dummy_image,
            "text": "What do you see?",
            "label": "I see a test image."
        }
    ]

    inputs = collate_fn(batch)
    ids = inputs["input_ids"][0].tolist()

    # 1. BOS at position 0
    assert ids[0] == BOS_TOKEN_ID, "BOS not at position 0"

    # 2. Image tokens present
    assert ids.count(IMAGE_SOFT_TOKEN_ID) > 0, "No image tokens"

    # 3. SOW/EOW present and ordered
    assert model.sow_token_id in ids, "SOW not in sequence"
    assert model.eow_token_id in ids, "EOW not in sequence"
    assert ids.index(model.sow_token_id) < ids.index(model.eow_token_id)

    print(f"✓ Token sequence structure valid")
    print(f"  BOS at position 0: {ids[0] == BOS_TOKEN_ID}")
    print(f"  Image tokens: {ids.count(IMAGE_SOFT_TOKEN_ID)}")
    print(f"  SOW position: {ids.index(model.sow_token_id)}")
    print(f"  EOW position: {ids.index(model.eow_token_id)}")


def test_decode_skips_special_tokens(model):
    """Test that decode with skip_special_tokens works correctly."""
    # Create token sequence with special tokens
    token_ids = torch.tensor([
        [BOS_TOKEN_ID, 1234, 5678, EOS_TOKEN_ID]
    ])

    # Decode with skipping
    text_skip = model.processor.decode(token_ids[0], skip_special_tokens=True)

    # Decode without skipping
    text_no_skip = model.processor.decode(token_ids[0], skip_special_tokens=False)

    # With skipping should not contain <bos> or <eos>
    assert "<bos>" not in text_skip.lower()
    assert "<eos>" not in text_skip.lower()

    # Without skipping might contain special token representations
    # (This is processor-dependent, so we just verify it's different)
    assert text_skip != text_no_skip or len(text_skip) < len(text_no_skip)


def test_add_generation_prompt_difference(model, dummy_image):
    """Test difference between training and inference prompt formats."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "Hello"}
            ]
        }
    ]

    # Training format (no generation prompt)
    train_inputs = model.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )

    # Inference format (with generation prompt)
    infer_inputs = model.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Inference should have more tokens (includes model turn prefix)
    assert infer_inputs["input_ids"].shape[1] > train_inputs["input_ids"].shape[1]


@pytest.mark.parametrize("num_samples", [1, 2, 4])
def test_batch_tokenization(model, dummy_image, num_samples):
    """Test that collator handles batches correctly."""
    collate_fn = create_theworld_collator(model)

    batch = [
        {
            "image": dummy_image,
            "text": f"Question {i}?",
            "label": f"Answer {i}."
        }
        for i in range(num_samples)
    ]

    inputs = collate_fn(batch)

    # Check batch size
    assert inputs["input_ids"].shape[0] == num_samples
    assert inputs["attention_mask"].shape[0] == num_samples
    assert inputs["pixel_values"].shape[0] == num_samples
    assert len(inputs["images"]) == num_samples

    # All samples should have BOS at position 0
    for i in range(num_samples):
        ids = inputs["input_ids"][i].tolist()
        non_pad = [id for id in ids if id != 0]  # Remove padding
        if len(non_pad) > 0:
            assert non_pad[0] == BOS_TOKEN_ID, f"Sample {i} missing BOS"


def test_validate_tokenization_function(model, dummy_image):
    """Test the validation function from documentation."""
    collate_fn = create_theworld_collator(model)

    batch = [
        {
            "image": dummy_image,
            "text": "Test",
            "label": "Response"
        }
    ]

    inputs = collate_fn(batch)

    # Run validation (from docs/tokenization_and_special_tokens.md)
    ids = inputs["input_ids"][0].tolist()

    # 1. BOS token
    assert ids[0] == BOS_TOKEN_ID

    # 2. Image tokens
    img_count = ids.count(IMAGE_SOFT_TOKEN_ID)
    assert img_count > 0

    # 3. SOW/EOW tokens
    sow_count = ids.count(model.sow_token_id)
    eow_count = ids.count(model.eow_token_id)
    assert sow_count == 1
    assert eow_count == 1

    sow_pos = ids.index(model.sow_token_id)
    eow_pos = ids.index(model.eow_token_id)
    assert sow_pos < eow_pos

    print("✓ All validation checks passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
