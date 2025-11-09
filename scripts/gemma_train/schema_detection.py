"""
Best effort schema detection
"""
import re
from typing import Dict, List, Optional

# ranked candidates (higher first)
CANDS = {
    "instruction": ["instruction", "prompt", "question", "query", "text", "caption", "user", "input_text"],
    "input":       ["input", "context", "source", "extra", "background"],
    "response":    ["response", "output", "answer", "completion", "target", "label_text"],
    "image":       ["image", "img", "image_path", "image_url", "img_path", "img_url", "filepath"],
    "label":       ["label", "answer", "gold", "gt", "target", "targets", "answer_text"],
}

def _score_name(name: str, role: str) -> int:
    """Give a higher score if the column name equals/contains a role candidate."""
    name_l = name.lower()
    score = 0
    for i, cand in enumerate(CANDS[role]):
        if name_l == cand:
            return 1000 - i  # exact match beats everything
        if cand in name_l:
            score = max(score, 100 - i)
    return score

def _score_values(values, role: str) -> int:
    """Lightweight value-type hints to break ties."""
    if not values:
        return 0
    v0 = values[0]
    if role == "image":
        if isinstance(v0, str) and (v0.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp")) or v0.startswith(("http://","https://"))):
            return 300
        # datasets may store PIL.Image objects
        v0t = type(v0).__name__.lower()
        if "image" in v0t:
            return 300
    if role in {"instruction","input","response"}:
        if isinstance(v0, str) and len(v0) > 5:
            return 50
    if role == "label":
        if isinstance(v0, (int, bool)) or (isinstance(v0, str) and re.fullmatch(r"\d+", v0)):
            return 50
    return 0

def detect_schema(batch_or_dataset) -> Dict[str, Optional[str]]:
    """
    Returns a mapping like:
      {"instruction": "prompt", "input": None, "response": "output",
       "image": "image", "label": None}
    Works with a Dataset (will peek at first row) or a dict-like batch.
    """
    if hasattr(batch_or_dataset, "column_names"):
        cols = list(batch_or_dataset.column_names)
        # peek first example for value-type scoring
        ex0 = batch_or_dataset[0] if len(batch_or_dataset) > 0 else {}
    else:
        cols = list(batch_or_dataset.keys())
        ex0 = {k: (batch_or_dataset[k][0] if isinstance(batch_or_dataset[k], list) and batch_or_dataset[k] else None)
               for k in cols}

    mapping = {}
    used = set()

    for role in ["instruction", "response", "input", "image", "label"]:
        best_col, best_score = None, -1
        for c in cols:
            if c in used:  # avoid reusing a column for multiple roles by default
                continue
            score = _score_name(c, role) + _score_values(batch_or_dataset[c] if c in batch_or_dataset else [ex0.get(c)], role)
            if score > best_score:
                best_col, best_score = c, score
        # require a minimum score for core roles
        if role in {"instruction", "response"} and best_score < 50:
            mapping[role] = None
        else:
            mapping[role] = best_col
            if best_col is not None:
                used.add(best_col)

    return mapping
