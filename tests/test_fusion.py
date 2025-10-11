"""Unit tests for EmbeddingFusion module."""

import torch
import pytest
from theworld.modeling import EmbeddingFusion


def test_embedding_fusion_basic():
    """Test basic fusion of Gemma and world embeddings."""
    # Setup
    sow_token_id = 1000
    eow_token_id = 1001
    fusion = EmbeddingFusion(sow_token_id=sow_token_id, eow_token_id=eow_token_id)

    # Create test inputs with brackets at positions 10 and 11
    batch_size = 2
    seq_len = 50
    embed_dim = 2304
    num_world_tokens = 784

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 10] = sow_token_id  # <start_of_world>
    input_ids[:, 11] = eow_token_id  # <end_of_world>

    gemma_embeds = torch.randn(batch_size, seq_len, embed_dim)
    world_embeds = torch.randn(batch_size, num_world_tokens, embed_dim)
    attention_mask = torch.ones(batch_size, seq_len)

    # Execute
    output = fusion(gemma_embeds, world_embeds, input_ids, attention_mask)

    # Verify
    # Expected length: seq_len - 2 (remove brackets) + num_world_tokens + 2 (keep brackets) = seq_len + num_world_tokens
    # Actually: [0:11] (11 tokens including start) + world_tokens + [11:] (seq_len - 11 tokens from end)
    # = 11 + 784 + (50-11) = 11 + 784 + 39 = 834
    expected_len = 11 + num_world_tokens + (seq_len - 11)
    assert output.combined_embeds.shape == (batch_size, expected_len, embed_dim)
    assert output.combined_attention_mask.shape == (batch_size, expected_len)


def test_embedding_fusion_preserves_bracket_tokens():
    """Test that fusion keeps the bracket tokens in the sequence."""
    sow_token_id = 1000
    eow_token_id = 1001
    fusion = EmbeddingFusion(sow_token_id=sow_token_id, eow_token_id=eow_token_id)

    batch_size = 1
    seq_len = 30
    embed_dim = 2304
    num_world_tokens = 100

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 5] = sow_token_id
    input_ids[:, 6] = eow_token_id

    gemma_embeds = torch.randn(batch_size, seq_len, embed_dim)
    world_embeds = torch.randn(batch_size, num_world_tokens, embed_dim)
    attention_mask = torch.ones(batch_size, seq_len)

    output = fusion(gemma_embeds, world_embeds, input_ids, attention_mask)

    # Expected: [0:6] (6 tokens up to and including start) + world + [6:] (24 tokens from end onwards)
    # = 6 + 100 + 24 = 130
    expected_len = 6 + num_world_tokens + (seq_len - 6)
    assert output.combined_embeds.shape[1] == expected_len


def test_embedding_fusion_device_consistency():
    """Test that fusion handles device transfers correctly."""
    sow_token_id = 1000
    eow_token_id = 1001
    fusion = EmbeddingFusion(sow_token_id=sow_token_id, eow_token_id=eow_token_id)

    batch_size = 1
    seq_len = 20
    embed_dim = 2304
    num_world_tokens = 50

    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    input_ids[:, 3] = sow_token_id
    input_ids[:, 4] = eow_token_id

    # Gemma embeds on CPU
    gemma_embeds = torch.randn(batch_size, seq_len, embed_dim)
    # World embeds on CPU (will be moved to match gemma_embeds device)
    world_embeds = torch.randn(batch_size, num_world_tokens, embed_dim)
    attention_mask = torch.ones(batch_size, seq_len)

    output = fusion(gemma_embeds, world_embeds, input_ids, attention_mask)

    # Verify all outputs are on same device as gemma_embeds
    assert output.combined_embeds.device == gemma_embeds.device
    assert output.combined_attention_mask.device == gemma_embeds.device


def test_embedding_fusion_missing_brackets():
    """Test that fusion raises error when bracket tokens are missing."""
    sow_token_id = 1000
    eow_token_id = 1001
    fusion = EmbeddingFusion(sow_token_id=sow_token_id, eow_token_id=eow_token_id)

    batch_size = 1
    seq_len = 20
    embed_dim = 2304
    num_world_tokens = 50

    # No bracket tokens
    input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    gemma_embeds = torch.randn(batch_size, seq_len, embed_dim)
    world_embeds = torch.randn(batch_size, num_world_tokens, embed_dim)
    attention_mask = torch.ones(batch_size, seq_len)

    # Should raise assertion error
    with pytest.raises(AssertionError, match="No <start_of_world> token found"):
        fusion(gemma_embeds, world_embeds, input_ids, attention_mask)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
