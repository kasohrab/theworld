"""
Spatial-RGPT evaluation dataset loader.

Supports local JSONL files or HuggingFace dataset ids. Each example should contain at least:
  - id: unique id
  - image_path or image_url (or image field pointing to local path)
  - question (str)
  - choices (optional list[str])
  - answer (optional ground truth)

This module provides a PyTorch/Dataset-compatible wrapper and a small downloader helper
that mirrors the pattern used by DataCompDataset in datacomp.py.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from io import BytesIO
import time
import requests
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


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
            time.sleep(0.1 * (2 ** attempt))
    return None


class SpatialRGPTDataset(TorchDataset):
    """Dataset wrapper for Spatial-RGPT eval data.

    Accepts either:
      - a HuggingFace dataset object (map style or streaming), or
      - a list of dicts loaded from a JSONL
    Each item returned is a dict with keys: id, image (PIL.Image), question, choices (optional), answer (optional), metadata
    """

    def __init__(
        self,
        data_source,
        num_samples: Optional[int] = None,
        streaming: bool = False,
        image_key_candidates: Optional[List[str]] = None,
    ):
        """Initialize dataset.

        Args:
            data_source: HuggingFace dataset, path to JSONL, or list of dicts
            num_samples: limit samples (useful for testing)
            streaming: whether HF dataset is streaming
            image_key_candidates: preference list of image keys to look for
        """
        self.num_samples = num_samples
        self.streaming = streaming

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
        raw = self.items[idx]

        image_field = self._get_image_field(raw)
        pil_image = None
        if image_field is not None:
            val = raw[image_field]
            if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                pil_image = download_image(val)
            else:
                # treat as local path
                try:
                    pil_image = Image.open(val).convert("RGB")
                except Exception:
                    pil_image = None

        return {
            "id": raw.get("id") or raw.get("example_id") or idx,
            "image": pil_image,
            "question": raw.get("question") or raw.get("prompt") or "",
            "choices": raw.get("choices"),
            "answer": raw.get("answer"),
            "metadata": raw,
        }

