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

# Custom Token Slot Assignments
# Gemma reserves 99 custom token slots (tokenizer.special_tokens.CUSTOM + 0 to 98)
# We use the first two slots for world model bracket tokens
CUSTOM_TOKEN_SLOT_SOW = 0  # <start_of_world>
CUSTOM_TOKEN_SLOT_EOW = 1  # <end_of_world>

# Custom Token Names
CUSTOM_TOKEN_SOW = "<start_of_world>"  # SOW - marks beginning of world embeddings
CUSTOM_TOKEN_EOW = "<end_of_world>"  # EOW - marks end of world embeddings

# Default Model Names
DEFAULT_GEMMA_MODEL = "google/gemma-3-4b-it"
DEFAULT_COSMOS_MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"
