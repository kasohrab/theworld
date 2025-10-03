"""
Text generation utilities for TheWorld model.
Handles sampling and decoding of model outputs.
"""

import torch


def get_next_token_prediction(outputs, tokenizer, temperature=1.0, top_k=None, top_p=None):
    """
    Get and decode the predicted next token from model outputs.

    Args:
        outputs: Model outputs with .logits attribute
        tokenizer: HuggingFace tokenizer for decoding
        temperature: Sampling temperature (default: 1.0, use 0.0 for greedy)
        top_k: If set, only sample from top k tokens
        top_p: If set, use nucleus sampling with probability mass p

    Returns:
        tuple: (token_id, decoded_text)
    """
    # Get logits for last position (where we predict next token)
    next_token_logits = outputs.logits[0, -1, :]  # Shape: (vocab_size,)

    # Apply temperature
    if temperature > 0:
        next_token_logits = next_token_logits / temperature
    else:
        # Greedy decoding (temperature = 0)
        next_token_id = torch.argmax(next_token_logits).item()
        next_token_text = tokenizer.decode([next_token_id])
        return next_token_id, next_token_text

    # Apply top-k filtering
    if top_k is not None:
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        next_token_logits[indices_to_remove] = float("-inf")

    # Apply top-p (nucleus) filtering
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_token_logits[indices_to_remove] = float("-inf")

    # Sample from the filtered distribution
    probs = torch.softmax(next_token_logits, dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1).item()

    # Decode to text
    next_token_text = tokenizer.decode([next_token_id])

    return next_token_id, next_token_text


def greedy_decode(outputs, tokenizer):
    """
    Simple greedy decoding - always pick the most likely token.

    Args:
        outputs: Model outputs with .logits attribute
        tokenizer: HuggingFace tokenizer for decoding

    Returns:
        tuple: (token_id, decoded_text)
    """
    return get_next_token_prediction(outputs, tokenizer, temperature=0.0)


def sample_with_temperature(outputs, tokenizer, temperature=0.7):
    """
    Sample next token with temperature scaling.

    Args:
        outputs: Model outputs with .logits attribute
        tokenizer: HuggingFace tokenizer for decoding
        temperature: Controls randomness (lower = more deterministic)

    Returns:
        tuple: (token_id, decoded_text)
    """
    return get_next_token_prediction(outputs, tokenizer, temperature=temperature)


def nucleus_sample(outputs, tokenizer, top_p=0.9, temperature=0.7):
    """
    Nucleus (top-p) sampling for diverse but coherent generation.

    Args:
        outputs: Model outputs with .logits attribute
        tokenizer: HuggingFace tokenizer for decoding
        top_p: Cumulative probability threshold
        temperature: Sampling temperature

    Returns:
        tuple: (token_id, decoded_text)
    """
    return get_next_token_prediction(outputs, tokenizer, temperature=temperature, top_p=top_p)
