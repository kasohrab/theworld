"""Constants for TheWorld model.

This module defines shared constants used across the codebase, including
special token IDs and configuration values.
"""

# Gemma 3 Special Token IDs
# Reference: https://huggingface.co/google/gemma-3-4b-it/blob/main/tokenizer_config.json
BOS_TOKEN_ID = 2  # <bos> - Beginning of sequence
EOS_TOKEN_ID = 1  # <eos> - End of sequence
PAD_TOKEN_ID = 0  # <pad> - Padding token
IMAGE_SOFT_TOKEN_ID = 262144  # <image_soft_token> - Vision encoder output placeholder

# Custom World Tokens
# These tokens are added to Gemma's vocabulary using HuggingFace's add_special_tokens() API.
# They mark the boundaries where world model embeddings are inserted into the sequence.
#
# Implementation: We use processor.tokenizer.add_special_tokens({"additional_special_tokens": [...]})
# This appends tokens to the vocabulary rather than using Gemma's native custom token slots.
#
# Token IDs (after adding to vocab_size=262145):
#   <start_of_world>: 262145 (original_vocab_size + 0)
#   <end_of_world>:   262146 (original_vocab_size + 1)
#
# Note: Gemma 3 provides 99 reserved "unused" token slots via Google's gemma package
# (accessible via gm.text.Gemma3Tokenizer with custom_tokens parameter), but we use
# HuggingFace's standard API for better compatibility with the transformers ecosystem.
CUSTOM_TOKEN_SOW = "<start_of_world>"  # SOW - marks beginning of world embeddings
CUSTOM_TOKEN_EOW = "<end_of_world>"  # EOW - marks end of world embeddings

# Default Model Names
DEFAULT_GEMMA_MODEL = "google/gemma-3-4b-it"
DEFAULT_COSMOS_MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"
