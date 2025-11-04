"""
LLaVA-CC3M-Pretrain-595K dataset loader for TheWorld model.

Dataset: https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K

This dataset contains 595K image-caption pairs from CC-3M with synthesized
conversational instructions for vision-language pretraining. Images are
included in the dataset as images.zip (6.46GB).

Structure:
- chat.json: Conversations with random instructions (e.g., "Describe this image")
- images.zip: Raw images from filtered CC-3M subset
- Each sample has: {"conversations": [...], "image": "filename.jpg", "id": "..."}
"""

import os
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from huggingface_hub import hf_hub_download


class LLaVAPretrainDataset(TorchDataset):
    """PyTorch Dataset wrapper for LLaVA-CC3M-Pretrain-595K.

    Handles:
    - Loading chat.json from HuggingFace dataset
    - Auto-downloading and extracting images.zip
    - Parsing conversations (human question + gpt response)
    - Converting to TheWorld format (image, text, label)

    Example:
        >>> dataset = LLaVAPretrainDataset(
        ...     image_folder="data/llava-cc3m/images",
        ...     num_samples=100,
        ...     hf_token=None
        ... )
    """

    def __init__(
        self,
        image_folder: str = "data/llava-cc3m/images",
        num_samples: Optional[int] = None,
        hf_token: Optional[str] = None,
        auto_download: bool = True,
    ):
        """Initialize LLaVA pretrain dataset.

        Args:
            image_folder: Path to extracted images folder
            num_samples: Limit to N samples (None = use all 595K)
            hf_token: HuggingFace API token (optional, dataset is public)
            auto_download: If True, auto-download images.zip if not found
        """
        self.image_folder = Path(image_folder)
        self.num_samples = num_samples
        self.hf_token = hf_token

        # Load chat.json from HuggingFace
        print("Loading LLaVA-CC3M-Pretrain dataset...")
        from datasets import load_dataset as hf_load_dataset

        # Use split parameter to load only the requested number of samples
        # This avoids loading all 595K examples when we only need a subset
        if num_samples is not None:
            split_str = f"train[:{num_samples}]"
            print(f"  Loading first {num_samples} samples...")
        else:
            split_str = "train"
            print(f"  Loading full dataset (595K samples)...")

        self.hf_dataset = hf_load_dataset(
            "liuhaotian/LLaVA-CC3M-Pretrain-595K",
            data_files="chat.json",  # Only load conversation data, not images.zip
            split=split_str,
            token=hf_token,
        )

        print(f"✓ Loaded {len(self.hf_dataset)} samples")

        # Download and extract images if needed
        if auto_download:
            self._ensure_images_downloaded()

        # Filter out samples with missing images
        # (CC3M dataset has some missing/broken images)
        self._filter_missing_images()

    def _ensure_images_downloaded(self):
        """Download and extract images.zip if not already present.

        This method:
        1. Checks if images are already extracted (skip everything)
        2. Downloads zip only if not in HF cache (reuses cached file)
        3. Extracts zip only if needed
        4. Keeps zip file cached for future use (don't delete)
        """
        # Check if images are already extracted
        if self.image_folder.exists():
            image_files = list(self.image_folder.glob("*.jpg"))
            if len(image_files) > 0:
                print(f"✓ Found {len(image_files)} images in {self.image_folder}")
                return

        # Images not found, need to download and/or extract
        print(f"Images not found in {self.image_folder}")

        # Create parent directory for extraction
        self.image_folder.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Download images.zip from HuggingFace Hub
            # If already cached, this will reuse the cached file (no re-download)
            print("Checking for images.zip in HuggingFace cache...")
            zip_path = hf_hub_download(
                repo_id="liuhaotian/LLaVA-CC3M-Pretrain-595K",
                filename="images.zip",
                repo_type="dataset",
                token=self.hf_token,
            )
            print(f"✓ Using images.zip from: {zip_path}")

            # Extract images to the target folder
            print(f"Extracting images to {self.image_folder} (this may take a few minutes)...")
            self.image_folder.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Extract all files to the images folder
                zip_ref.extractall(self.image_folder)

            # Verify extraction
            extracted_count = len(list(self.image_folder.glob("*.jpg")))
            print(f"✓ Extracted {extracted_count} images to {self.image_folder}")

            # Note: We keep the zip file in HF cache (don't delete it)
            # This allows re-extraction if needed without re-downloading

        except Exception as e:
            raise RuntimeError(
                f"Failed to download or extract images.zip: {e}\n\n"
                f"Please download manually from:\n"
                f"https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip\n"
                f"and extract to: {self.image_folder}"
            )

    def _filter_missing_images(self):
        """Filter out samples where the image file is missing.

        The CC3M dataset has some missing/broken images. We filter these
        out during initialization to avoid runtime errors during training.
        """
        print("Checking for missing images...")
        valid_indices = []

        for idx in range(len(self.hf_dataset)):
            image_filename = self.hf_dataset[idx]["image"]
            image_path = self.image_folder / image_filename

            if image_path.exists():
                valid_indices.append(idx)

        original_count = len(self.hf_dataset)
        self.hf_dataset = self.hf_dataset.select(valid_indices)
        filtered_count = original_count - len(self.hf_dataset)

        if filtered_count > 0:
            print(f"⚠ Filtered out {filtered_count} samples with missing images")
            print(f"✓ {len(self.hf_dataset)} valid samples remaining")
        else:
            print(f"✓ All {len(self.hf_dataset)} samples have valid images")

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index.

        Returns:
            Dictionary with 'image', 'text', 'label'
        """
        item = self.hf_dataset[idx]

        # Parse conversations
        # Expected format:
        # {
        #   "conversations": [
        #     {"from": "human", "value": "Describe this image.\n<image>"},
        #     {"from": "gpt", "value": "olive oil is a healthy ingredient used liberally."}
        #   ],
        #   "image": "GCC_train_002582585.jpg",
        #   "id": "GCC_train_002582585"
        # }
        conversations = item["conversations"]
        image_filename = item["image"]

        # Extract human question and gpt response
        human_msg = None
        gpt_msg = None
        for conv in conversations:
            if conv["from"] == "human":
                human_msg = conv["value"]
            elif conv["from"] == "gpt":
                gpt_msg = conv["value"]

        if human_msg is None or gpt_msg is None:
            raise ValueError(f"Invalid conversation format for sample {idx}: missing human or gpt message")

        # Remove <image> token from text (processor handles image placement)
        text = human_msg.replace("<image>", "").strip()
        label = gpt_msg.strip()

        # Load image with error handling for corrupted files
        image_path = self.image_folder / image_filename
        try:
            image = Image.open(image_path).convert("RGB")
        except (Image.UnidentifiedImageError, OSError, IOError) as e:
            # Image file exists but is corrupted/truncated/invalid
            # Skip to next sample (wraparound at end)
            print(f"⚠ Warning: Skipping corrupted image at index {idx}: {image_filename} ({e})")
            next_idx = (idx + 1) % len(self.hf_dataset)
            return self[next_idx]

        return {
            "image": image,
            "text": text,
            "label": label,
        }


def load_llava_pretrain(
    image_folder: str = "data/llava-cc3m/images",
    num_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
    auto_download: bool = True,
) -> LLaVAPretrainDataset:
    """Load LLaVA-CC3M-Pretrain-595K dataset for TheWorld training.

    Args:
        image_folder: Path to extracted images folder
        num_samples: Limit to N samples (None = use all 595K)
        hf_token: HuggingFace API token (optional, dataset is public)
        auto_download: If True, auto-download images.zip if not found

    Returns:
        LLaVAPretrainDataset instance

    Example:
        >>> # Load small subset for testing
        >>> dataset = load_llava_pretrain(num_samples=100)
        >>>
        >>> # Load full dataset for pretraining
        >>> dataset = load_llava_pretrain()
    """
    dataset = LLaVAPretrainDataset(
        image_folder=image_folder,
        num_samples=num_samples,
        hf_token=hf_token,
        auto_download=auto_download,
    )

    print(f"✓ LLaVA pretrain dataset ready")
    return dataset
