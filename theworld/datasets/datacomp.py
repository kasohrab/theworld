"""
DataComp-1B dataset loader for TheWorld model.

DataComp-1B is a large-scale image-text dataset with 1.4B samples.
Images are provided as URLs that need to be downloaded on-the-fly.

Dataset: https://huggingface.co/datasets/mlfoundations/datacomp_1b
"""

import requests
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any
import time
from torch.utils.data import Dataset as TorchDataset


def download_image(url: str, timeout: int = 5, max_retries: int = 3) -> Optional[Image.Image]:
    """Download image from URL with retry logic.

    Args:
        url: Image URL to download
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        PIL Image if successful, None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image
        except Exception as e:
            if attempt == max_retries - 1:
                # All retries failed
                return None
            # Wait before retry (exponential backoff)
            time.sleep(0.1 * (2**attempt))
    return None


class DataCompDataset(TorchDataset):
    """PyTorch Dataset wrapper for DataComp-1B.

    Handles:
    - Loading from HuggingFace datasets
    - Downloading images from URLs
    - Converting to TheWorld format (image, text, label)
    - Filtering out failed downloads

    Example:
        >>> from datasets import load_dataset
        >>> hf_dataset = load_dataset("mlfoundations/datacomp_1b", split="train", streaming=True)
        >>> dataset = DataCompDataset(hf_dataset, num_samples=1000)
    """

    def __init__(
        self,
        hf_dataset,
        num_samples: Optional[int] = None,
        question_template: str = "Describe this image in detail.",
        skip_on_error: bool = True,
    ):
        """Initialize DataComp dataset.

        Args:
            hf_dataset: HuggingFace dataset object (can be streaming)
            num_samples: Limit to N samples (None = use all)
            question_template: Question to ask about each image
            skip_on_error: If True, skip samples where image download fails
        """
        self.hf_dataset = hf_dataset
        self.num_samples = num_samples
        self.question_template = question_template
        self.skip_on_error = skip_on_error

        # For streaming datasets, we can't get len() directly
        self._is_streaming = hasattr(hf_dataset, "_head")

        # Cache for non-streaming datasets
        if not self._is_streaming and num_samples:
            # Take subset for non-streaming
            self.hf_dataset = hf_dataset.select(range(min(num_samples, len(hf_dataset))))

        # Track statistics
        self.num_failed_downloads = 0

    def __len__(self) -> int:
        """Return dataset length.

        For streaming datasets, returns num_samples if set, otherwise raises error.
        """
        if self._is_streaming:
            if self.num_samples is not None:
                return self.num_samples
            else:
                raise TypeError("Streaming dataset has no length. Set num_samples to enable.")
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item at index.

        Retries with next samples if download fails to ensure valid item is always returned.

        Returns:
            Dictionary with 'image', 'text', 'label'

        Raises:
            RuntimeError: If all retry attempts fail
        """
        max_retries = 10  # Try up to 10 samples before giving up

        # For streaming, we iterate through the dataset
        if self._is_streaming:
            # Note: This is inefficient for random access, but works for sequential iteration
            retry_count = 0
            for i, item in enumerate(self.hf_dataset):
                if i >= idx:
                    result = self._process_item(item)
                    if result is not None:
                        return result
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise RuntimeError(
                            f"Failed to load valid item after {max_retries} attempts. "
                            f"Check network connection or dataset quality."
                        )
            raise RuntimeError(f"Reached end of streaming dataset without finding valid item")
        else:
            # Non-streaming: try current index and subsequent ones
            dataset_len = len(self.hf_dataset)
            for retry in range(max_retries):
                try_idx = (idx + retry) % dataset_len
                item = self.hf_dataset[try_idx]
                result = self._process_item(item)
                if result is not None:
                    if retry > 0:
                        # Log when we had to skip samples (optional, can be removed if too verbose)
                        pass
                    return result

            # All retries failed
            raise RuntimeError(
                f"Failed to load valid item after trying {max_retries} samples starting from index {idx}. "
                f"Check network connection or dataset quality."
            )

    def _process_item(self, item: Dict) -> Optional[Dict[str, Any]]:
        """Process a single dataset item.

        Args:
            item: Raw item from HuggingFace dataset

        Returns:
            Processed item or None if image download fails
        """
        # Extract URL and caption
        url = item.get("url") or item.get("default/url")
        caption = item.get("text") or item.get("default/text", "")

        # Download image
        image = download_image(url)

        if image is None:
            self.num_failed_downloads += 1
            if self.skip_on_error:
                return None  # Collator will filter this out
            else:
                # Return a blank image as fallback
                image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        return {
            "image": image,
            "text": self.question_template,
            "label": caption,
        }


def load_datacomp(
    split: str = "train",
    num_samples: Optional[int] = None,
    streaming: bool = True,
    question_template: str = "Describe this image in detail.",
    hf_token: Optional[str] = None,
) -> DataCompDataset:
    """Load DataComp-1B dataset for TheWorld training.

    Args:
        split: Dataset split ("train" is the only available split)
        num_samples: Limit to N samples (None = use all 1.4B)
        streaming: Use streaming mode (recommended for large datasets)
        question_template: Question to ask about each image
        hf_token: HuggingFace API token (optional, dataset is public)

    Returns:
        DataCompDataset instance

    Example:
        >>> # Load small subset for testing
        >>> dataset = load_datacomp(num_samples=100, streaming=False)
        >>>
        >>> # Load for production (streaming mode)
        >>> dataset = load_datacomp(streaming=True)
    """
    from datasets import load_dataset as hf_load_dataset

    # Load HuggingFace dataset
    print(f"Loading DataComp-1B dataset (split={split}, streaming={streaming})...")

    if streaming:
        # Streaming mode: efficient for large datasets
        hf_dataset = hf_load_dataset(
            "mlfoundations/datacomp_1b",
            split=split,
            streaming=True,
            token=hf_token,
        )

        # Take subset if specified
        if num_samples:
            hf_dataset = hf_dataset.take(num_samples)
            print(f"  Taking first {num_samples} samples (streaming)")
    else:
        # Non-streaming mode: loads full dataset into memory
        if num_samples:
            # Use split slicing to load only subset
            hf_dataset = hf_load_dataset(
                "mlfoundations/datacomp_1b",
                split=f"{split}[:{num_samples}]",
                token=hf_token,
            )
            print(f"  Loaded {len(hf_dataset)} samples")
        else:
            # Load full dataset (1.4B samples - requires a lot of memory!)
            print("  ⚠️  Loading full 1.4B dataset without streaming - this may take a long time!")
            hf_dataset = hf_load_dataset(
                "mlfoundations/datacomp_1b",
                split=split,
                token=hf_token,
            )
            print(f"  Loaded {len(hf_dataset)} samples")

    # Wrap in DataCompDataset
    dataset = DataCompDataset(
        hf_dataset,
        num_samples=num_samples if streaming else None,  # Already limited above for non-streaming
        question_template=question_template,
    )

    print(f"✓ DataComp dataset ready")
    return dataset
