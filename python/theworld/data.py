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

from .constants import BOS_TOKEN_ID, IMAGE_SOFT_TOKEN_ID

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


def theworld_collate_fn(
    batch: List[Dict[str, Any]],
    processor: "AutoProcessor",
    tokenizer: "AutoTokenizer",
    max_length: int = 2048,
    sow_token_id: Optional[int] = None,  # Unused but kept for compatibility
    eow_token_id: Optional[int] = None,  # Unused but kept for compatibility
    num_world_tokens: int = 784,  # Default: 28x28 for single frame
) -> TheWorldBatch:
    """Collate function for batching TheWorld inputs.

    This function:
    1. Formats chat template as text only (NO vision encoding - that happens in model)
    2. Tokenizes text with image placeholders
    3. Preprocesses images for SigLIP (resize/normalize only, no encoding)
    4. Keeps raw PIL images for Cosmos

    Args:
        batch: List of dictionaries with 'image', 'text', 'label'
        processor: Gemma processor (for image preprocessing and tokenization)
        tokenizer: Gemma tokenizer (for text)
        max_length: Maximum sequence length
        sow_token_id: Token ID for <start_of_world> (unused, for compatibility)
        eow_token_id: Token ID for <end_of_world> (unused, for compatibility)
        num_world_tokens: Number of world tokens (unused, for compatibility)

    Returns:
        Dictionary with:
            - input_ids: Token IDs for input
            - attention_mask: Attention mask
            - pixel_values: Preprocessed image tensors (for SigLIP, not encoded)
            - images: Raw PIL images (for Cosmos)
            - labels: Labels for loss computation
    """
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    labels_raw = [item.get("label", None) for item in batch]

    # Use processor's chat template to properly format messages with image placeholders
    # This ensures <start_of_image> and <end_of_image> tokens are correctly inserted
    messages_batch = []
    for i, (image, text) in enumerate(zip(images, texts)):
        # DEBUG: Check image type
        print(f"[DEBUG] Sample {i}: image type = {type(image)}, text = '{text[:50]}...'")

        # Ensure image is PIL and RGB (handle grayscale, RGBA, palette, etc.)
        if isinstance(image, Image.Image):
            # Convert any image mode to RGB (handles L, LA, P, RGBA, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<start_of_world> <end_of_world>"},  # World token brackets
                    {"type": "image", "image": image},  # Image placeholder (now guaranteed RGB)
                    {"type": "text", "text": text},  # Question/prompt
                ],
            }
        ]
        messages_batch.append(messages)

    # Apply chat template to get properly formatted input_ids with image tokens
    # We process each item individually because apply_chat_template doesn't support batching
    input_ids_list = []
    attention_mask_list = []
    pixel_values_list = []  # Extract pixel_values from apply_chat_template
    for i, messages in enumerate(messages_batch):
        formatted = processor.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
            messages,
            tokenize=True,
            add_generation_prompt=False,  # We don't want assistant prefix yet
            return_dict=True,
            return_tensors="pt",
        )
        # Validation: Check token structure
        ids = formatted["input_ids"][0].tolist()
        image_token_count = ids.count(IMAGE_SOFT_TOKEN_ID)

        # Validate BOS token at position 0
        if ids[0] != BOS_TOKEN_ID:
            print(f"[WARNING] Sample {i}: BOS token missing at position 0! Found token ID {ids[0]} instead.")

        # Validate image tokens present
        if image_token_count == 0:
            print(f"[WARNING] Sample {i}: No image tokens found after applying chat template!")

        print(
            f"[DEBUG] Sample {i}: input_ids shape = {formatted['input_ids'].shape}, <image_soft_token> count = {image_token_count}"
        )
        print(f"[DEBUG] Sample {i}: first 20 token IDs = {ids[:20]}")

        input_ids_list.append(formatted["input_ids"])
        attention_mask_list.append(formatted["attention_mask"])

        # Extract pixel_values from apply_chat_template (already preprocessed!)
        if "pixel_values" in formatted:
            pixel_values_list.append(formatted["pixel_values"])
        else:
            # Fallback: manual preprocessing if not present (shouldn't happen)
            processed = processor.image_processor(images=images[i], return_tensors="pt")
            pixel_values_list.append(processed["pixel_values"])

    # Pad to same length
    max_len = max(ids.size(1) for ids in input_ids_list)
    input_ids = torch.zeros(len(input_ids_list), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(attention_mask_list), max_len, dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        input_ids[i, : ids.size(1)] = ids
        attention_mask[i, : mask.size(1)] = mask

    # Validate SOW/EOW tokens present (if sow_token_id and eow_token_id are provided)
    if sow_token_id is not None and eow_token_id is not None:
        for i in range(len(input_ids)):
            ids_list = input_ids[i].tolist()
            sow_count = ids_list.count(sow_token_id)
            eow_count = ids_list.count(eow_token_id)
            if sow_count != 1 or eow_count != 1:
                print(
                    f"[WARNING] Sample {i}: Expected 1 SOW and 1 EOW token, "
                    f"found {sow_count} SOW and {eow_count} EOW"
                )

    # Concatenate pixel_values (already extracted from apply_chat_template above)
    pixel_values = torch.cat(pixel_values_list, dim=0)

    # Process labels if provided
    labels = None
    if any(label is not None for label in labels_raw):
        # Tokenize labels
        label_encodings = tokenizer(  # pyright: ignore[reportCallIssue] - AutoTokenizer is callable at runtime
            [label if label is not None else "" for label in labels_raw],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = label_encodings["input_ids"]

        # Replace padding token id with -100 (ignore index)
        labels[labels == tokenizer.pad_token_id] = -100  # pyright: ignore[reportAttributeAccessIssue]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
        "images": images,  # Raw PIL images for Cosmos processing
    }


def create_theworld_collator(model):
    """Create a collate function bound to a specific model instance.

    This is a convenience function that creates a collator with the model's
    processor, tokenizer, and special tokens already configured.

    Args:
        model: TheWorld model instance

    Returns:
        Collate function that can be passed to DataLoader or Trainer

    Example:
        >>> from theworld import TheWorld
        >>> model = TheWorld("google/gemma-3-4b-it")
        >>> collate_fn = create_theworld_collator(model)
        >>> trainer = Trainer(..., data_collator=collate_fn)
    """

    def collate_fn(batch):
        # Estimate num_world_tokens for bracket placement
        # Cosmos VAE produces variable spatial dimensions based on input size
        # For 512x512 input: approximately 80x176 = 14,080 tokens
        # This is an estimate - actual count determined at runtime during forward pass
        # The brackets just reserve space; actual world tokens inserted during forward()
        estimated_world_tokens = 14080  # Approximate for typical input sizes

        return theworld_collate_fn(
            batch,
            processor=model.processor,
            tokenizer=model.processor.tokenizer,
            max_length=2048,
            sow_token_id=model.sow_token_id,
            eow_token_id=model.eow_token_id,
            num_world_tokens=estimated_world_tokens,
        )

    return collate_fn
