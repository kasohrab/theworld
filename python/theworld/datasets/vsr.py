"""
Visual Spatial Reasoning (VSR) dataset loader for TheWorld model.

VSR is a binary visual entailment benchmark that evaluates spatial understanding.
Each sample contains an image and a caption describing spatial relations between objects.
The model must predict whether the caption correctly describes the image (True/False).

Dataset: https://huggingface.co/datasets/cambridgeltl/vsr_random
Paper: https://arxiv.org/abs/2205.00363
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


class VSRDataset(TorchDataset):
    """PyTorch Dataset wrapper for Visual Spatial Reasoning (VSR).

    Handles:
    - Loading from HuggingFace datasets (vsr_random or vsr_zeroshot)
    - Mapping image_link to local image files
    - Converting binary labels to text responses ("True"/"False")
    - Formatting as QA task for TheWorld training

    Example:
        >>> from datasets import load_dataset
        >>> hf_dataset = load_dataset("cambridgeltl/vsr_random", split="train")
        >>> dataset = VSRDataset(
        ...     hf_dataset,
        ...     image_folder="/path/to/vsr_images",
        ... )
    """

    def __init__(
        self,
        hf_dataset,
        image_folder: str,
        num_samples: Optional[int] = None,
        question_template: str = "Statement: {caption}\nAnswer (only '0' or '1'):",
        skip_on_error: bool = True,
    ):
        """Initialize VSR dataset.

        Args:
            hf_dataset: HuggingFace dataset object
            image_folder: Path to folder containing VSR images
            num_samples: Limit to N samples (None = use all)
            question_template: Template for formatting captions as questions
            skip_on_error: If True, skip samples where image loading fails
        """
        self.hf_dataset = hf_dataset
        self.image_folder = Path(image_folder)
        self.num_samples = num_samples
        self.question_template = question_template
        self.skip_on_error = skip_on_error

        # Take subset if specified
        if num_samples and hasattr(hf_dataset, "select"):
            self.hf_dataset = hf_dataset.select(range(min(num_samples, len(hf_dataset))))

        # Validate image folder exists
        if not self.image_folder.exists():
            raise ValueError(f"Image folder does not exist: {self.image_folder}")

        # Track statistics
        self.num_failed_loads = 0

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index.

        Returns:
            Dictionary with 'image', 'text', 'label'

        Raises:
            RuntimeError: If image loading fails and skip_on_error is False
        """
        max_retries = 10  # Try up to 10 samples if loading fails

        for retry in range(max_retries):
            try_idx = (idx + retry) % len(self.hf_dataset)
            item = self.hf_dataset[try_idx]
            result = self._process_item(item)

            if result is not None:
                return result

            if not self.skip_on_error:
                raise RuntimeError(f"Failed to load image for sample {try_idx}")

        # All retries failed
        raise RuntimeError(
            f"Failed to load valid item after trying {max_retries} samples starting from index {idx}. "
            f"Check that image files exist in {self.image_folder}"
        )

    def _process_item(self, item: Dict) -> Optional[Dict[str, Any]]:
        """Process a single dataset item.

        Args:
            item: Raw item from HuggingFace dataset

        Returns:
            Processed item or None if image loading fails
        """
        # Extract fields
        image_link = item.get("image_link")
        caption = item.get("caption", "")
        label = item.get("label")  # 0 or 1

        if image_link is None or label is None:
            return None

        # Load image from local file
        # image_link is typically just a filename like "000000000142.jpg"
        image_filename = os.path.basename(image_link)
        image_path = self.image_folder / image_filename

        try:
            image = Image.open(image_path).convert("RGB")

            # Validate image dimensions
            width, height = image.size
            if width < 16 or height < 16:
                self.num_failed_loads += 1
                return None

        except Exception:
            self.num_failed_loads += 1
            return None

        # Convert label to binary string (0 or 1)
        # This matches the evaluation format for single-token prediction
        if isinstance(label, int):
            label_text = "1" if label == 1 else "0"
        else:
            # Handle ClassLabel objects from HuggingFace datasets
            label_text = str(label)
            if label_text == "1" or label_text.lower() == "true":
                label_text = "1"
            else:
                label_text = "0"

        # Format question
        question = self.question_template.format(caption=caption)

        return {
            "image": image,
            "text": question,
            "label": label_text,
        }


def load_vsr(
    split: str = "train",
    variant: str = "random",
    image_folder: str = "/home/hice1/ksohrab3/scratch/theworld/data/images",
    num_samples: Optional[int] = None,
    question_template: str = "Statement: {caption}\nAnswer (only '0' or '1'):",
    hf_token: Optional[str] = None,
) -> VSRDataset:
    """Load Visual Spatial Reasoning (VSR) dataset for TheWorld training.

    Args:
        split: Dataset split ("train", "validation", or "test")
        variant: Dataset variant ("random" or "zeroshot")
        image_folder: Path to folder containing VSR images
        num_samples: Limit to N samples (None = use all)
        question_template: Template for formatting captions as questions
        hf_token: HuggingFace API token (optional, dataset is public)

    Returns:
        VSRDataset instance

    Example:
        >>> # Load training set
        >>> dataset = load_vsr(split="train", variant="random")
        >>>
        >>> # Load test set with custom image folder
        >>> dataset = load_vsr(
        ...     split="test",
        ...     variant="zeroshot",
        ...     image_folder="/path/to/images"
        ... )
    """
    from datasets import load_dataset as hf_load_dataset

    # Construct dataset name
    dataset_name = f"cambridgeltl/vsr_{variant}"

    print(f"Loading VSR dataset ({dataset_name}, split={split})...")

    # Load HuggingFace dataset
    hf_dataset = hf_load_dataset(
        dataset_name,
        split=split,
        token=hf_token,
    )

    print(f"  Loaded {len(hf_dataset)} samples")

    # Wrap in VSRDataset
    dataset = VSRDataset(
        hf_dataset,
        image_folder=image_folder,
        num_samples=num_samples,
        question_template=question_template,
    )

    print(f"✓ VSR dataset ready (image_folder={image_folder})")
    return dataset
