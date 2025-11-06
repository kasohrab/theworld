"""
Data utilities for TheWorld model training.
Includes dataset classes and collate functions for HuggingFace Trainer.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, TYPE_CHECKING, Optional, TypedDict
from PIL import Image
import numpy as np
from pathlib import Path
import logging

from .constants import BOS_TOKEN_ID, IMAGE_SOFT_TOKEN_ID

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import AutoProcessor, AutoTokenizer


class TheWorldBatch(TypedDict):
    """Type definition for TheWorld collator output."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    images: List[Image.Image]
    labels: Optional[torch.Tensor]  # Optional field


class TheWorldDataset(Dataset):
    """Dataset for TheWorld model training.

    Expected data format:
        Each item should be a dictionary with:
        - image: PIL Image, path to image, or numpy array
        - text: Text prompt/question (string)
        - label: Expected response text (string)

    Example:
        >>> data = [
        ...     {
        ...         "image": Image.open("path/to/image.jpg"),
        ...         "text": "What is in this image?",
        ...         "label": "A cat sitting on a couch."
        ...     },
        ... ]
        >>> dataset = TheWorldDataset(data)
    """

    def __init__(
        self, data: List[Dict[str, Any]], image_key: str = "image", text_key: str = "text", label_key: str = "label"
    ):
        """Initialize TheWorldDataset.

        Args:
            data: List of dictionaries with image, text, and label
            image_key: Key for image in data dict (default: "image")
            text_key: Key for text in data dict (default: "text")
            label_key: Key for label in data dict (default: "label")
        """
        self.data = data
        self.image_key = image_key
        self.text_key = text_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Load image if path provided
        image = item[self.image_key]
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        # Otherwise assume it's already a PIL Image

        return {
            "image": image,
            "text": item[self.text_key],
            "label": item.get(self.label_key, None),
        }


class HFDatasetWrapper(Dataset):
    """Wrapper for HuggingFace datasets to work with TheWorld.

    Example:
        >>> from datasets import load_dataset
        >>> hf_dataset = load_dataset("some_dataset", split="train")
        >>> dataset = HFDatasetWrapper(hf_dataset, image_key="image", text_key="question", label_key="answer")
    """

    def __init__(
        self,
        hf_dataset,
        image_key: str = "image",
        text_key: str = "text",
        label_key: str = "label",
    ):
        """Initialize HFDatasetWrapper.

        Args:
            hf_dataset: HuggingFace Dataset object
            image_key: Key for image in dataset
            text_key: Key for text/question in dataset
            label_key: Key for label/answer in dataset
        """
        self.hf_dataset = hf_dataset
        self.image_key = image_key
        self.text_key = text_key
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.hf_dataset[idx]

        # Extract fields
        image = item[self.image_key]
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        return {
            "image": image,
            "text": item[self.text_key],
            "label": item.get(self.label_key, None),
        }


def create_multi_turn_labels(
    input_ids: torch.Tensor, messages: List[Dict[str, str]], processor: "AutoProcessor"
) -> torch.Tensor:
    """
    Create labels for multi-turn conversation by masking user turns.

    Strategy: Find turn boundaries using Gemma's special tokens, then mask user turns with -100
    while keeping assistant turns as actual token IDs for loss calculation.

    Args:
        input_ids: Full tokenized conversation sequence
        messages: List of messages with role and content
        processor: Gemma processor for tokenizer access

    Returns:
        labels: Same as input_ids but with user portions masked with -100
    """
    labels = input_ids.clone()
    tokenizer = processor.tokenizer

    # Get special token IDs
    start_turn_id = tokenizer.convert_tokens_to_ids("<start_of_turn>")
    end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")

    # Tokenize "user" and "model" to get role indicators
    user_tokens = tokenizer.encode("user", add_special_tokens=False)
    model_tokens = tokenizer.encode("model", add_special_tokens=False)

    # Handle case where role is single token or multiple tokens
    user_id = user_tokens[0] if user_tokens else None
    model_id = model_tokens[0] if model_tokens else None

    # Scan through tokens and mask user turns
    i = 0
    while i < len(input_ids):
        # Find <start_of_turn>
        if input_ids[i] == start_turn_id and i + 1 < len(input_ids):
            role_token = input_ids[i + 1]

            # Find matching <end_of_turn>
            j = i + 2
            while j < len(input_ids) and input_ids[j] != end_turn_id:
                j += 1

            # If this is a user turn, mask it
            if role_token == user_id:
                labels[i : j + 1] = -100  # Mask entire turn including markers
            # If model turn, keep it (already copied from input_ids)
            # else: labels remain as input_ids (model turn gets trained)

            i = j + 1
        else:
            i += 1

    return labels


def theworld_collate_fn(
    batch: List[Dict[str, Any]],
    processor: "AutoProcessor",
    tokenizer: "AutoTokenizer",
    max_length: int = 2048,
) -> TheWorldBatch:
    """
    Collate function for batching TheWorld inputs.

    This function correctly prepares data for Causal LM fine-tuning by:
    1. Creating the full conversation sequence (prompt + label).
    2. Tokenizing the full sequence to get input_ids.
    3. Creating a copy for the labels.
    4. Masking out the prompt portion of the labels with -100.
    5. Filtering out samples that exceed max_length.
    6. Padding all tensors to the maximum length in the batch.

    Args:
        batch: List of samples with image, text, and label
        processor: Gemma processor for tokenization and image processing
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length (samples exceeding this are skipped)
    """
    images_filtered = []
    texts_filtered = []
    labels_filtered = []

    # First pass: filter out samples that exceed max_length
    images = [item["image"].convert("RGB") if item["image"].mode != "RGB" else item["image"] for item in batch]

    # We need to process each item to find the length of the prompt part,
    # then the length of the full part, so we can mask the labels correctly.
    input_ids_list = []
    labels_list = []
    pixel_values_list = []

    for item, image in zip(batch, images):
        # Check if item has 'messages' field (multi-turn) or text/label (single-turn)
        if "messages" in item and item["messages"] is not None:
            # Multi-turn format
            messages = item["messages"]

            # Build messages_full for apply_chat_template
            # First message includes image
            messages_full = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image}, {"type": "text", "text": messages[0]["content"]}],
                }
            ]

            # Add remaining turns (text only)
            for msg in messages[1:]:
                messages_full.append({"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]})
        else:
            # Single-turn format (backward compatible)
            text = item["text"]
            label = item.get("label")
            messages = None  # Mark as single-turn

            messages_full = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
            ]
            if label is not None:
                messages_full.append({"role": "assistant", "content": [{"type": "text", "text": label}]})

        # Tokenize the full conversation to get the combined input_ids
        full_tokenized = processor.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False,  # The template adds assistant markers
            return_dict=True,
            return_tensors="pt",
        )
        full_ids = full_tokenized["input_ids"][0]
        seq_len = len(full_ids)

        # Skip samples that exceed max_length
        if seq_len > max_length:
            logger.warning(f"Skipping sample with sequence length {seq_len} > max_length {max_length}")
            continue

        # Create labels based on format
        if messages is not None:
            # Multi-turn: Use sophisticated turn masking
            current_labels = create_multi_turn_labels(full_ids, messages, processor)
        else:
            # Single-turn: Use simple prompt masking
            # Calculate prompt length using tokenizer only (avoid re-processing image)
            text = item["text"]
            messages_prompt_text = [
                {"role": "user", "content": text},
            ]
            # Use tokenizer directly (no image processing) with chat template
            prompt_ids = processor.tokenizer.apply_chat_template(
                messages_prompt_text,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            prompt_len = prompt_ids.shape[1]

            # Create labels: a copy of the full IDs, but with the prompt masked out
            current_labels = full_ids.clone()
            current_labels[:prompt_len] = -100

        input_ids_list.append(full_ids)
        labels_list.append(current_labels)

        # Gemma's apply_chat_template doesn't return pixel_values, so preprocess manually
        if "pixel_values" in full_tokenized:
            pixel_values_list.append(full_tokenized["pixel_values"])
        else:
            # Fallback: use processor's image_processor to preprocess
            processed = processor.image_processor(images=image, return_tensors="pt")
            pixel_values_list.append(processed["pixel_values"])

        images_filtered.append(image)

    # Handle empty batch (all samples filtered out)
    if len(input_ids_list) == 0:
        logger.error(
            f"All {len(batch)} samples in batch exceeded max_length {max_length}. "
            f"This should be rare - consider increasing max_length or checking your data. "
            f"Batch will be skipped."
        )
        # Return None to signal the trainer to skip this batch
        # Trainer will handle this gracefully
        raise ValueError(
            f"All {len(batch)} samples exceeded max_length {max_length}. " f"Increase max_length or check dataset."
        )

    # 4. Pad everything to the max length of the batch
    max_len = max(len(ids) for ids in input_ids_list)

    # Use the tokenizer's pad token ID
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id  # Fallback if no pad token is set

    # Pad input_ids and attention_mask
    # Use len(input_ids_list) not len(batch) - some samples may have been filtered for max_length
    input_ids_padded = torch.full((len(input_ids_list), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        input_ids_padded[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1

    # Pad labels
    labels_padded = torch.full((len(input_ids_list), max_len), -100, dtype=torch.long)
    for i, label_ids in enumerate(labels_list):
        labels_padded[i, : len(label_ids)] = label_ids

    # Concatenate pixel values (should all have the same shape)
    pixel_values = torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels_padded,
        "images": images_filtered,
    }


def create_theworld_collator(
    model,
    max_length: int = 2048,
):
    """Create a collate function bound to a specific model instance.

    This is a convenience function that creates a collator with the model's
    processor, tokenizer, and special tokens already configured.

    Args:
        model: TheWorld model instance
        max_length: Maximum sequence length (samples exceeding this are skipped)

    Returns:
        Collate function that can be passed to DataLoader or Trainer

    Example:
        >>> from theworld import TheWorld
        >>> model = TheWorld("google/gemma-3-4b-it")
        >>> collate_fn = create_theworld_collator(model, max_length=2048)
        >>> trainer = Trainer(..., data_collator=collate_fn)
    """

    def collate_fn(batch):
        return theworld_collate_fn(
            batch,
            processor=model.processor,
            tokenizer=model.processor.tokenizer,
            max_length=max_length,
        )

    return collate_fn
