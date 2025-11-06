"""
Spatial-RGPT dataset loader for both evaluation and training.

**Evaluation format (SpatialRGPT-Bench):**
  - id: unique id
  - image_info: dict with 'file_path' (relative path from image_folder)
  - text_q: text-only question (without special tokens)
  - conversations: list of conversation turns with ground truth answer
  - bbox: list of bounding boxes [[x1, y1, x2, y2], ...]
  - rle: optional RLE-encoded masks
  - qa_info: dict with 'type' (qualitative/quantitative) and 'category'

**Training format (OpenSpatialDataset):**
  - filename: image filename (without extension, e.g., "img_001")
  - conversations: list of conversation turns with spatial reasoning QA
  - (no bbox field - regions are referenced in text as "Region [0]", "Region [1]", etc.)

This module provides a PyTorch/Dataset-compatible wrapper that:
  - Supports both eval and training data formats
  - Optionally draws bounding boxes on images for visual grounding (eval mode)
  - Extracts ground truth from conversations field
  - Compatible with TheWorld training pipeline
  - Uses images-first approach: only loads samples for available images
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from io import BytesIO
import time
import requests
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from .bbox_utils import draw_bounding_boxes, clamp_bbox


def download_image(url: str, timeout: int = 5, max_retries: int = 3) -> Optional[Image.Image]:
    """Download image with retry/backoff. Returns PIL.Image or None on failure."""
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(0.1 * (2**attempt))
    return None


class SpatialRGPTDataset(TorchDataset):
    """Dataset wrapper for Spatial-RGPT evaluation and training data.

    **Images-First Approach:** Only loads samples for images that exist in image_folder.
    Scans available images at initialization, then builds JSON index for those images only.

    Supports two data formats:
      1. **Evaluation (SpatialRGPT-Bench)**: Has bbox, text_q, image_info fields
      2. **Training (OpenSpatialDataset)**: Has filename, conversations only

    Accepts either:
      - a HuggingFace dataset object (map style or streaming)
      - a list of dicts loaded from a JSON/JSONL file

    Each item returned is a dict with keys:
      - id: unique identifier
      - image: PIL.Image (RGB)
      - question: text question (string)
      - answer: ground truth answer (string)
      - choices: optional choices (usually None)
      - qa_type: question type (qualitative/quantitative) for eval
      - qa_category: category for eval
      - metadata: full raw item dict
    """

    def __init__(
        self,
        data_source,
        num_samples: Optional[int] = None,
        streaming: bool = False,
        image_key_candidates: Optional[List[str]] = None,
        image_folder: Optional[str] = None,
        draw_bboxes: bool = True,
    ):
        """Initialize dataset with images-first approach.

        Args:
            data_source: HuggingFace dataset, path to JSONL, or list of dicts
            num_samples: limit samples by available images (not JSON entries)
            streaming: whether HF dataset is streaming
            image_key_candidates: preference list of image keys to look for
            image_folder: Base folder for image paths (required for training data)
            draw_bboxes: Whether to draw bounding boxes on images (default: True)
        """
        self.num_samples = num_samples
        self.streaming = streaming
        self.image_folder = Path(image_folder) if image_folder else None
        self.draw_bboxes = draw_bboxes

        if image_key_candidates is None:
            self.image_key_candidates = ["image_path", "image_url", "image", "img"]
        else:
            self.image_key_candidates = image_key_candidates

        # Accept many types for convenience
        if isinstance(data_source, str) or isinstance(data_source, Path):
            # Treat as local JSONL path
            self._load_from_jsonl(Path(data_source))
        else:
            # HuggingFace dataset or list
            self._wrap_hf_or_list(data_source)

    def _load_from_jsonl(self, path: Path) -> None:
        """Load dataset with images-first approach."""
        import json
        import ijson

        # Check if file is JSONL by reading first few chars
        with open(path, "r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                # JSON array format - use images-first approach
                print(f"  Loading JSON array with images-first approach...")
                self._load_json_array_images_first(path)
            else:
                # JSONL format (one JSON object per line) - parse line by line
                print(f"  Loading JSONL format (streaming)...")
                self._load_jsonl_images_first(path)
                
    def _load_json_array_images_first(self, path: Path) -> None:
        """Load JSON array with images-first approach: scan images, filter JSON."""
        import ijson

        start_time = time.time()

        # Step 1: Scan available images
        if self.image_folder:
            print(f"  Scanning image folder: {self.image_folder}")
            scan_start = time.time()
            available_images = set()
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for img_path in self.image_folder.glob(ext):
                    # Remove extension to match JSON "filename" field
                    available_images.add(img_path.stem)
            scan_time = time.time() - scan_start
            print(f"  ✓ Found {len(available_images):,} images in {scan_time:.1f}s")
        else:
            # No image folder specified - load all samples
            available_images = None

        # Step 2: Parse JSON and keep only samples with available images
        print(f"  Parsing JSON and filtering by available images...")
        parse_start = time.time()
        self.items = []
        with open(path, "rb") as f:
            parser = ijson.items(f, "item")
            for entry in parser:
                # Check if image is available
                if available_images is not None:
                    # Training format: "filename" field
                    image_id = entry.get("filename")
                    if image_id and image_id not in available_images:
                        continue

                # Add to items
                self.items.append(entry)

                # Stop if we have enough samples
                if self.num_samples and len(self.items) >= self.num_samples:
                    break

        parse_time = time.time() - parse_start
        total_time = time.time() - start_time
        print(f"  ✓ Loaded {len(self.items):,} samples in {parse_time:.1f}s (total: {total_time:.1f}s)")

    def _load_jsonl_images_first(self, path: Path) -> None:
        """Load JSONL format with images-first filtering."""
        import json

        start_time = time.time()

        # Step 1: Scan available images (same as JSON array)
        if self.image_folder:
            print(f"  Scanning image folder: {self.image_folder}")
            available_images = set()
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for img_path in self.image_folder.glob(ext):
                    available_images.add(img_path.stem)
            print(f"  ✓ Found {len(available_images):,} images")
        else:
            available_images = None

        # Step 2: Parse JSONL and filter
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)

                # Check if image is available
                if available_images is not None:
                    image_id = entry.get("filename")
                    if image_id and image_id not in available_images:
                        continue

                self.items.append(entry)

                # Stop if we have enough samples
                if self.num_samples and len(self.items) >= self.num_samples:
                    break

        total_time = time.time() - start_time
        print(f"  ✓ Loaded {len(self.items):,} samples in {total_time:.1f}s")

    def _wrap_hf_or_list(self, data_source) -> None:
        """Wrap HuggingFace dataset or list with images-first filtering."""
        start_time = time.time()

        # Step 1: Scan available images if image_folder is specified
        if self.image_folder:
            print(f"  Scanning image folder: {self.image_folder}")
            available_images = set()
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
                for img_path in self.image_folder.glob(ext):
                    available_images.add(img_path.stem)
            print(f"  ✓ Found {len(available_images):,} images")
        else:
            available_images = None

        # Step 2: Filter dataset by available images
        self.items = []
        try:
            length = len(data_source)  # type: ignore
            print(f"  Filtering {length:,} samples by available images...")
            for i in range(length):
                entry = data_source[i]

                # Check if image is available
                if available_images is not None:
                    image_id = entry.get("filename")
                    if image_id and image_id not in available_images:
                        continue

                self.items.append(entry)

                # Stop if we have enough samples
                if self.num_samples and len(self.items) >= self.num_samples:
                    break
        except Exception:
            # Likely streaming/iterator or list-like
            print(f"  Filtering streaming dataset by available images...")
            for entry in data_source:
                # Check if image is available
                if available_images is not None:
                    image_id = entry.get("filename")
                    if image_id and image_id not in available_images:
                        continue

                self.items.append(entry)

                # Stop if we have enough samples
                if self.num_samples and len(self.items) >= self.num_samples:
                    break

        total_time = time.time() - start_time
        print(f"  ✓ Loaded {len(self.items):,} samples in {total_time:.1f}s")

    def __len__(self) -> int:
        return len(self.items)

    def _get_image_field(self, item: Dict[str, Any]) -> Optional[str]:
        for k in self.image_key_candidates:
            if k in item and item[k]:
                return k
        # fallback: any key with url/path-like string
        for k, v in item.items():
            if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://") or Path(v).exists()):
                return k
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import ast

        # Get item from in-memory list
        raw = self.items[idx]

        # Load image
        pil_image = None

        # First check if there's a direct image field (HuggingFace format)
        if "image" in raw and raw["image"] is not None:
            # HuggingFace dataset includes PIL image directly
            if isinstance(raw["image"], Image.Image):
                pil_image = raw["image"]
            else:
                # Might be a path or other format
                try:
                    pil_image = Image.open(raw["image"]).convert("RGB")
                except Exception:
                    pil_image = None

        # Try filename field (training data format: OpenSpatialDataset)
        elif "filename" in raw:
            # Training data uses "filename" without extension
            # e.g., "img_001" -> "img_001.jpg"
            filename = raw["filename"]
            # Try common image extensions
            for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                if self.image_folder:
                    full_path = self.image_folder / f"{filename}{ext}"
                else:
                    full_path = Path(f"{filename}{ext}")

                try:
                    if full_path.exists():
                        pil_image = Image.open(full_path).convert("RGB")
                        break
                except Exception:
                    continue

        # Try image_info field (eval data format: SpatialRGPT-Bench)
        elif "image_info" in raw:
            image_info = raw["image_info"]
            # Parse if it's a string (HuggingFace serialization)
            if isinstance(image_info, str):
                try:
                    image_info = ast.literal_eval(image_info)
                except Exception:
                    image_info = {}

            if isinstance(image_info, dict) and "file_path" in image_info:
                image_path = image_info["file_path"]
                if self.image_folder:
                    full_path = self.image_folder / image_path
                else:
                    full_path = Path(image_path)

                try:
                    pil_image = Image.open(full_path).convert("RGB")
                except Exception:
                    pil_image = None

        # Fallback to original logic
        if pil_image is None:
            # Fallback to original logic
            image_field = self._get_image_field(raw)
            if image_field is not None:
                val = raw[image_field]
                if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                    pil_image = download_image(val)
                else:
                    try:
                        pil_image = Image.open(val).convert("RGB")
                    except Exception:
                        pil_image = None

        # Handle missing images - this should not happen with images-first approach
        # but keep error handling for robustness
        if pil_image is None:
            filename = raw.get("filename", raw.get("id", idx))
            raise FileNotFoundError(
                f"Image not found for sample {filename}. "
                f"This should not happen with images-first loading - please report this bug."
            )

        # Draw bounding boxes if requested and available
        if self.draw_bboxes and pil_image is not None and "bbox" in raw:
            bboxes = raw["bbox"]
            # Parse if it's a string (HuggingFace serialization)
            if isinstance(bboxes, str):
                try:
                    bboxes = ast.literal_eval(bboxes)
                except Exception:
                    bboxes = []

            if bboxes and len(bboxes) > 0:
                # Clamp bboxes to image boundaries
                img_width, img_height = pil_image.size
                clamped_bboxes = [clamp_bbox(bbox, img_width, img_height) for bbox in bboxes]

                # Draw boxes with labels
                labels = [f"Region [{i}]" for i in range(len(bboxes))]
                pil_image = draw_bounding_boxes(pil_image, clamped_bboxes, labels=labels)

        # Helper function for region token replacement
        def replace_region_tokens(text: str) -> str:
            """Replace <mask> <depth> tokens with Region [0], Region [1], etc."""
            region_idx = 0
            # Replace <mask> <depth> first (most specific)
            while "<mask> <depth>" in text:
                text = text.replace("<mask> <depth>", f"Region [{region_idx}]", 1)
                region_idx += 1
            # Replace standalone <mask> (fallback)
            while "<mask>" in text:
                text = text.replace("<mask>", f"Region [{region_idx}]", 1)
                region_idx += 1
            # Remove remaining <depth> tokens
            text = text.replace("<depth>", "").strip()
            return text

        # Parse conversations field (training data format: result_10_depth_convs.json)
        messages = None

        if "conversations" in raw:
            conversations = raw["conversations"]
            # Parse if it's a string (HuggingFace serialization)
            if isinstance(conversations, str):
                try:
                    conversations = ast.literal_eval(conversations)
                except Exception:
                    conversations = []

            # Extract ALL conversation turns (not just first Q&A pair)
            if isinstance(conversations, list) and len(conversations) >= 2:
                messages = []
                for i, conv in enumerate(conversations):
                    role = "user" if conv.get("from") == "human" else "assistant"
                    content = conv.get("value", "")

                    # Remove <image> token from first message only
                    if i == 0:
                        content = content.replace("<image>\n", "").replace("<image>", "")

                    # Replace <mask> <depth> with Region [N] sequentially
                    content = replace_region_tokens(content.strip())

                    messages.append({"role": role, "content": content})

        # Fallback to evaluation format fields (SpatialRGPT-Bench)
        if not messages:
            question = raw.get("text_q") or raw.get("question") or raw.get("prompt") or ""
            answer = raw.get("answer")
            if question:
                messages = [
                    {"role": "user", "content": question},
                ]
                if answer:
                    messages.append({"role": "assistant", "content": answer})

        # Extract question type and category from qa_info
        qa_type = None
        qa_category = None
        if "qa_info" in raw:
            qa_info = raw["qa_info"]
            # Parse if it's a string (HuggingFace serialization)
            if isinstance(qa_info, str):
                try:
                    qa_info = ast.literal_eval(qa_info)
                except Exception:
                    qa_info = {}

            if isinstance(qa_info, dict):
                qa_type = qa_info.get("type")
                qa_category = qa_info.get("category")

        return {
            "id": raw.get("id") or raw.get("example_id") or idx,
            "image": pil_image,
            "messages": messages,  # Multi-turn conversations
            "choices": raw.get("choices"),  # Usually None for SpatialRGPT-Bench
            "qa_type": qa_type,
            "qa_category": qa_category,
            "metadata": raw,
        }


def load_spatial_rgpt(
    split: str = "train",
    image_folder: str = None,
    num_samples: Optional[int] = None,
    draw_bboxes: bool = False,
    hf_token: Optional[str] = None,
) -> SpatialRGPTDataset:
    """Load SpatialRGPT OpenSpatialDataset for TheWorld training.

    Uses images-first approach: only loads samples for images that exist in image_folder.

    Args:
        split: Dataset split ("train" or "validation")
        image_folder: Path to OpenImagesV7 directory (required for training data)
        num_samples: Limit to N available samples (None = use all available)
        draw_bboxes: Draw bounding boxes on images (False for training, True for visualization)
        hf_token: HuggingFace API token (optional, dataset is public)

    Returns:
        SpatialRGPTDataset instance

    Example:
        >>> # Load training set
        >>> dataset = load_spatial_rgpt(
        ...     split="train",
        ...     image_folder="data/openimages/train",
        ...     num_samples=1000
        ... )
        >>>
        >>> # Load validation set with bboxes
        >>> dataset = load_spatial_rgpt(
        ...     split="validation",
        ...     image_folder="data/openimages/validation",
        ...     draw_bboxes=True
        ... )
    """
    from datasets import load_dataset as hf_load_dataset

    print(f"Loading SpatialRGPT OpenSpatialDataset (split={split})...")

    # Load HuggingFace dataset
    hf_dataset = hf_load_dataset(
        "a8cheng/OpenSpatialDataset",
        split=split,
        token=hf_token,
    )

    print(f"  Loaded {len(hf_dataset)} samples from HuggingFace")

    # Wrap in SpatialRGPTDataset (will filter by available images)
    dataset = SpatialRGPTDataset(
        hf_dataset,
        image_folder=image_folder,
        draw_bboxes=draw_bboxes,
        num_samples=num_samples,
    )

    print(f"✓ SpatialRGPT dataset ready ({len(dataset)} available samples)")
    return dataset
