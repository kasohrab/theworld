"""
Tests for multi-turn conversation support in TheWorld.

Verifies:
1. SpatialRGPTDataset returns messages field
2. Collator handles multi-turn correctly
3. Label masking is correct (user turns masked, assistant turns kept)
"""

import pytest
import torch
from PIL import Image
import numpy as np

from theworld.datasets.spatial_rgpt import SpatialRGPTDataset
from theworld.data import theworld_collate_fn, create_multi_turn_labels


@pytest.fixture
def mock_spatial_data():
    """Create mock spatial reasoning data with multi-turn conversations."""
    # Create a simple test image
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    return {
        "filename": "test_image",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nDoes <mask> <depth> have a greater width compared to <mask> <depth>?",
            },
            {"from": "gpt", "value": "In fact, Region [0] might be narrower than Region [1]."},
            {"from": "human", "value": "Which of these two, <mask> <depth> or <mask> <depth>, stands taller?"},
            {"from": "gpt", "value": "Standing taller between the two is Region [0]."},
            {
                "from": "human",
                "value": "How far is <mask> <depth> from <mask> <depth> vertically?",
            },
            {"from": "gpt", "value": "The vertical distance of Region [0] from Region [1] is 1.49 inches."},
        ],
        "bbox": [[10, 10, 50, 50], [60, 60, 90, 90]],
        "rle": [],
        "image": img,
    }


def test_dataset_returns_messages(mock_spatial_data, tmp_path):
    """Test that SpatialRGPTDataset returns messages field."""
    # Save mock data to temp file
    import json

    data_file = tmp_path / "test_data.json"
    with open(data_file, "w") as f:
        # Remove PIL image for JSON serialization
        data_dict = {k: v for k, v in mock_spatial_data.items() if k != "image"}
        json.dump([data_dict], f)

    # Create temp image file
    img_file = tmp_path / "test_image.jpg"
    mock_spatial_data["image"].save(img_file)

    # Load dataset
    dataset = SpatialRGPTDataset(
        data_source=str(data_file), image_folder=str(tmp_path), draw_bboxes=False, num_samples=1
    )

    # Get first sample
    sample = dataset[0]

    # Verify messages field exists
    assert "messages" in sample, "Dataset should return 'messages' field"
    assert isinstance(sample["messages"], list), "Messages should be a list"

    # Verify all conversation turns are present
    assert len(sample["messages"]) == 6, f"Should have 6 messages (3 Q&A pairs), got {len(sample['messages'])}"

    # Verify role alternation
    assert sample["messages"][0]["role"] == "user"
    assert sample["messages"][1]["role"] == "assistant"
    assert sample["messages"][2]["role"] == "user"
    assert sample["messages"][3]["role"] == "assistant"

    # Verify <mask> <depth> tokens are replaced with Region [N]
    assert "Region [0]" in sample["messages"][0]["content"]
    assert "Region [1]" in sample["messages"][0]["content"]
    assert "<mask>" not in sample["messages"][0]["content"]
    assert "<depth>" not in sample["messages"][0]["content"]


def test_create_multi_turn_labels():
    """Test create_multi_turn_labels masks user turns correctly."""
    from transformers import AutoProcessor

    # Load processor (needed for tokenizer)
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True)

    # Create mock input_ids representing a conversation
    # Format: [BOS] <start_of_turn>user\nQ1<end_of_turn><start_of_turn>model\nA1<end_of_turn>
    tokenizer = processor.tokenizer

    # Tokenize a simple conversation
    messages = [
        {"role": "user", "content": "What is this?"},
        {"role": "assistant", "content": "A cat."},
        {"role": "user", "content": "What color?"},
        {"role": "assistant", "content": "Orange."},
    ]

    # Build full conversation (text only for this test)
    messages_for_template = []
    for msg in messages:
        messages_for_template.append({"role": msg["role"], "content": msg["content"]})

    # Tokenize with chat template
    input_ids = tokenizer.apply_chat_template(messages_for_template, tokenize=True, return_tensors="pt")[0]

    # Create labels
    labels = create_multi_turn_labels(input_ids, messages, processor)

    # Verify labels shape matches input_ids
    assert labels.shape == input_ids.shape, "Labels should have same shape as input_ids"

    # Verify some tokens are masked
    masked_count = (labels == -100).sum().item()
    assert masked_count > 0, "Some tokens should be masked with -100"

    # Verify not all tokens are masked (assistant responses should be kept)
    non_masked_count = (labels != -100).sum().item()
    assert non_masked_count > 0, "Some tokens should be kept (assistant responses)"

    # Verify non-masked tokens match input_ids
    non_masked_mask = labels != -100
    assert torch.all(labels[non_masked_mask] == input_ids[non_masked_mask]), "Non-masked labels should match input_ids"

    print(f"\n✓ Created labels: {masked_count} masked, {non_masked_count} kept")


def test_collator_multi_turn(mock_spatial_data, tmp_path):
    """Test that collator handles multi-turn messages correctly."""
    from transformers import AutoProcessor

    # Load processor
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True)
    tokenizer = processor.tokenizer

    # Save mock data
    import json

    data_file = tmp_path / "test_data.json"
    with open(data_file, "w") as f:
        data_dict = {k: v for k, v in mock_spatial_data.items() if k != "image"}
        json.dump([data_dict], f)

    img_file = tmp_path / "test_image.jpg"
    mock_spatial_data["image"].save(img_file)

    # Load dataset
    dataset = SpatialRGPTDataset(
        data_source=str(data_file), image_folder=str(tmp_path), draw_bboxes=False, num_samples=1
    )

    # Get sample
    sample = dataset[0]

    # Collate into batch
    batch = theworld_collate_fn([sample], processor, tokenizer, max_length=2048)

    # Verify batch structure
    assert "input_ids" in batch
    assert "labels" in batch
    assert "pixel_values" in batch
    assert "attention_mask" in batch

    # Verify input_ids and labels have same shape
    assert batch["input_ids"].shape == batch["labels"].shape

    # Verify some labels are masked
    masked_count = (batch["labels"] == -100).sum().item()
    non_masked_count = (batch["labels"] != -100).sum().item()

    assert masked_count > 0, "Some tokens should be masked (user questions)"
    assert non_masked_count > 0, "Some tokens should be kept (assistant answers)"

    # Verify sequence is not empty
    seq_len = batch["input_ids"].shape[1]
    assert seq_len > 0, "Sequence should not be empty"

    print(f"\n✓ Collated batch: seq_len={seq_len}, masked={masked_count}, kept={non_masked_count}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
