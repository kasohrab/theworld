"""
Data utilities for TheWorld model training.
Includes dataset classes and collate functions for HuggingFace Trainer.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np
from pathlib import Path


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
    processor,
    tokenizer,
    max_length: int = 2048,
    world_start_id: int = None,
    world_end_id: int = None,
    num_world_tokens: int = 784,  # Default: 28x28 for single frame
) -> Dict[str, torch.Tensor]:
    """Collate function for batching TheWorld inputs.

    This function:
    1. Processes images through Gemma processor (handles chat template)
    2. Tokenizes labels
    3. Creates combined labels with -100 for vision/world tokens

    Args:
        batch: List of dictionaries with 'image', 'text', 'label'
        processor: Gemma processor (handles images + text)
        tokenizer: Gemma tokenizer (for labels)
        max_length: Maximum sequence length
        world_start_id: Token ID for <the_world_start>
        world_end_id: Token ID for <the_world_end>
        num_world_tokens: Number of world tokens (depends on num_world_steps)

    Returns:
        Dictionary with:
            - input_ids: Token IDs for input
            - attention_mask: Attention mask
            - pixel_values: Image tensors
            - labels: Labels for loss computation
    """
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    labels = [item.get("label", None) for item in batch]

    # Create messages format for Gemma processor
    # Gemma processor expects: [{"role": "user", "content": [{"type": "text"}, {"type": "image"}]}]
    messages_batch = []
    for image, text in zip(images, texts):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<the_world_start> <the_world_end>"},  # Placeholder for world tokens
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]
        messages_batch.append(messages)

    # Process through Gemma processor
    # This handles chat template, image preprocessing, etc.
    processed = []
    for messages in messages_batch:
        proc_item = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            max_length=max_length,
            truncation=True,
        )
        processed.append(proc_item)

    # Stack batch
    input_ids = torch.cat([p["input_ids"] for p in processed], dim=0)
    attention_mask = torch.cat([p["attention_mask"] for p in processed], dim=0)
    pixel_values = torch.cat([p["pixel_values"] for p in processed], dim=0)

    # Process labels if provided
    if any(label is not None for label in labels):
        # Tokenize labels
        label_encodings = tokenizer(
            [label if label is not None else "" for label in labels],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        label_ids = label_encodings["input_ids"]

        # Replace padding token id with -100 (ignore index)
        label_ids[label_ids == tokenizer.pad_token_id] = -100

        # Note: The actual label alignment with world tokens happens in model.forward()
        # We just provide the text labels here
        combined_labels = label_ids
    else:
        combined_labels = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": combined_labels,
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
        # Calculate num_world_tokens based on model configuration
        # Default: 28x28 spatial tokens per frame
        spatial_tokens = 28 * 28  # 784
        num_frames = 1 + model.num_world_steps  # Current + predicted future
        num_world_tokens = spatial_tokens * num_frames

        return theworld_collate_fn(
            batch,
            processor=model.processor,
            tokenizer=model.processor.tokenizer,
            max_length=2048,
            world_start_id=model.world_start_id,
            world_end_id=model.world_end_id,
            num_world_tokens=num_world_tokens,
        )

    return collate_fn
