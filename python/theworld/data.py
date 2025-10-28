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
) -> TheWorldBatch:
    """
    Collate function for batching TheWorld inputs.

    This function correctly prepares data for Causal LM fine-tuning by:
    1. Creating the full conversation sequence (prompt + label).
    2. Tokenizing the full sequence to get input_ids.
    3. Creating a copy for the labels.
    4. Masking out the prompt portion of the labels with -100.
    5. Padding all tensors to the maximum length in the batch.
    """
    images = [item["image"].convert("RGB") if item["image"].mode != "RGB" else item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    labels_raw = [item.get("label") for item in batch]

    # We need to process each item to find the length of the prompt part,
    # then the length of the full part, so we can mask the labels correctly.
    input_ids_list = []
    labels_list = []
    pixel_values_list = []

    for image, text, label in zip(images, texts, labels_raw):
        # 1. Create the full conversation with correctly formatted content
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

        # 2. Create and tokenize ONLY the prompt part to find its length
        messages_prompt = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
        ]
        # We add a generation prompt to ensure the template includes the assistant markers
        # (e.g., <start_of_turn>model\n) so we know where the prompt ends.
        prompt_tokenized = processor.apply_chat_template(
            messages_prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_len = prompt_tokenized["input_ids"].shape[1]

        # 3. Create labels: a copy of the full IDs, but with the prompt masked out
        current_labels = full_ids.clone()
        current_labels[:prompt_len] = -100

        input_ids_list.append(full_ids)
        labels_list.append(current_labels)
        pixel_values_list.append(full_tokenized.get("pixel_values"))

    # 4. Pad everything to the max length of the batch
    max_len = max(len(ids) for ids in input_ids_list)

    # Use the tokenizer's pad token ID
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id  # Fallback if no pad token is set

    # Pad input_ids and attention_mask
    input_ids_padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, ids in enumerate(input_ids_list):
        input_ids_padded[i, : len(ids)] = ids
        attention_mask[i, : len(ids)] = 1

    # Pad labels
    labels_padded = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, label_ids in enumerate(labels_list):
        labels_padded[i, : len(label_ids)] = label_ids

    # Concatenate pixel values (should all have the same shape)
    pixel_values = torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels_padded,
        "images": images,
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
        return theworld_collate_fn(
            batch,
            processor=model.processor,
            tokenizer=model.processor.tokenizer,
            max_length=8192,
        )

    return collate_fn
