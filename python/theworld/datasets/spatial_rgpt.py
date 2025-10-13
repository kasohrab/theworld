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
        """Initialize dataset.

        Args:
            data_source: HuggingFace dataset, path to JSONL, or list of dicts
            num_samples: limit samples (useful for testing)
            streaming: whether HF dataset is streaming
            image_key_candidates: preference list of image keys to look for
            image_folder: Base folder for image paths (if images are relative paths)
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
        import json

        items = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.num_samples and i >= self.num_samples:
                    break
                items.append(json.loads(line))
        self.items = items

    def _wrap_hf_or_list(self, data_source) -> None:
        # HuggingFace dataset: assume it supports __len__/__getitem__ or is an iterator for streaming
        try:
            length = len(data_source)  # type: ignore
            if self.num_samples:
                # take slice
                self.items = [data_source[i] for i in range(min(length, self.num_samples))]
            else:
                self.items = [data_source[i] for i in range(length)]
        except Exception:
            # Likely streaming/iterator or list-like; convert to list up to num_samples
            self.items = []
            count = 0
            for x in data_source:
                self.items.append(x)
                count += 1
                if self.num_samples and count >= self.num_samples:
                    break

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

        # Extract question (prefer text_q for SpatialRGPT-Bench)
        question = raw.get("text_q") or raw.get("question") or raw.get("prompt") or ""

        # Extract ground truth answer from conversations field
        answer = None
        if "conversations" in raw:
            conversations = raw["conversations"]
            # Parse if it's a string (HuggingFace serialization)
            if isinstance(conversations, str):
                try:
                    conversations = ast.literal_eval(conversations)
                except Exception:
                    conversations = []

            if len(conversations) >= 2:
                # Answer is at index 1 (second turn)
                answer = conversations[1].get("value", "")
        else:
            answer = raw.get("answer")

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
            "question": question,
            "choices": raw.get("choices"),  # Usually None for SpatialRGPT-Bench
            "answer": answer,
            "qa_type": qa_type,
            "qa_category": qa_category,
            "metadata": raw,
        }
