#!/usr/bin/env python3
# run_vsr_pipeline_binary_gemma3.py
# pip install -U datasets transformers accelerate pillow torch torchvision requests tqdm

import argparse
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, pipeline
from tqdm import tqdm

# -----------------------------
# Dataset column keys
# -----------------------------
IMAGE_KEYS = ["image", "img", "image_path", "image_url", "img_path", "img_url"]
# VSR often uses "caption" as the statement; keep other aliases as fallback.
QUESTION_KEYS = ["caption", "question", "prompt", "query", "text", "statement"]
ANSWER_KEYS = ["label", "answer", "gold", "gt", "target", "targets", "answer_text"]


# -----------------------------
# Small utils
# -----------------------------
def pick_key(ex: Dict[str, Any], cands: List[str]) -> Optional[str]:
    for k in cands:
        if k in ex and ex[k] is not None:
            return k
    return None


def load_pil_image(val: Any, image_dir: Path) -> Image.Image:
    """
    Accepts PIL, URL, absolute path, or relative filename.
    - URLs are downloaded.
    - Absolute paths are opened as-is.
    - Relative paths are resolved under `image_dir`.
    """
    if isinstance(val, Image.Image):
        return val.convert("RGB")

    if isinstance(val, (str, Path)):
        s = str(val)
        if s.startswith("http://") or s.startswith("https://"):
            r = requests.get(s, timeout=20)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")

        p = Path(s)
        if p.is_absolute() and p.exists():
            return Image.open(p).convert("RGB")

        candidate = image_dir / p
        if candidate.exists():
            return Image.open(candidate).convert("RGB")

        # Last attempt: open the original string (may raise if not found)
        return Image.open(p).convert("RGB")

    raise ValueError(f"Unsupported image value type: {type(val)}")


def build_vlm_pipeline(model_name: str):
    """
    Prefer generic image-text generation; fall back to VQA if needed.
    Returns (task_name, pipeline, processor).
    """
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Ensure vision support
    if not any(hasattr(processor, a) for a in ("image_processor", "vision_processor", "image_transforms")):
        raise RuntimeError(
            f"Checkpoint '{model_name}' does not appear to be vision-capable. "
            "Pick a VLM checkpoint whose AutoProcessor includes an image processor."
        )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    common_kwargs = dict(
        model=model_name,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Try generic VLM generation first (chat-style multimodal)
    try:
        gen_pipe = pipeline(
            task="image-text-to-text",
            image_processor=getattr(processor, "image_processor", None),
            tokenizer=getattr(processor, "tokenizer", None),
            **common_kwargs,
        )
        return ("image-text-to-text", gen_pipe, processor)
    except Exception:
        pass

    # Fallback: VQA
    vqa_pipe = pipeline(
        task="visual-question-answering",
        image_processor=getattr(processor, "image_processor", None),
        tokenizer=getattr(processor, "tokenizer", None),
        **common_kwargs,
    )
    return ("visual-question-answering", vqa_pipe, processor)


# -----------------------------
# Binary prompting + parsing
# -----------------------------
BINARY_SYSTEM_INSTRUCTION = (
    "You are a vision-language model that performs binary visual entailment.\n"
    "Task: Look at the image and evaluate the statement.\n"
    "Output exactly one digit with no extra text: 1 if the statement is true, 0 if the statement is false."
)

def make_messages(pil_img: Image.Image, statement: str) -> List[Dict[str, Any]]:
    """
    Build chat-style multimodal 'messages' for the image-text-to-text pipeline.
    Each message dict MUST have 'role' and 'content'.
    The image is embedded in the user's content as {'type': 'image', 'image': PIL_IMAGE}.
    """
    return [
        {"role": "system", "content": [
            {"type": "text", "text": BINARY_SYSTEM_INSTRUCTION}
        ]},
        {"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": f"Statement: {statement}\nAnswer (only '1' or '0'):"}
        ]},
    ]


def build_binary_prompt_plain(statement: str) -> str:
    """
    Plain-text (non-chat) prompt for VQA fallback.
    """
    return (
        f"{BINARY_SYSTEM_INSTRUCTION}\n\n"
        f"Statement: {statement}\n"
        f"Answer (only '1' or '0'):"
    )


_WORD_TO_BIN = {
    "1": 1, "0": 0,
    "true": 1, "false": 0,
    "yes": 1, "no": 0,
    "correct": 1, "incorrect": 0,
    "right": 1, "wrong": 0,
    "entails": 1, "contradiction": 0,
}

def parse_binary_output(text: str) -> Tuple[Optional[int], str]:
    """
    Try hard to turn model text into {0,1}.
    1) Look for first literal 0/1.
    2) Map common words (yes/no, true/false, etc).
    Returns (parsed_int_or_None, raw_text).
    """
    raw = (text or "").strip()
    # First, literal 0/1 anywhere (but prefer at start)
    m = re.search(r"\b([01])\b", raw)
    if m:
        return int(m.group(1)), raw

    # Second, word mapping (search first word token from our map)
    tokens = re.findall(r"[A-Za-z]+", raw.lower())
    for t in tokens:
        if t in _WORD_TO_BIN:
            return _WORD_TO_BIN[t], raw

    # Nothing parsed
    return None, raw


def _normalize_pipe_out(o):
    """
    Normalize HF pipeline outputs to a dict with text.
    Handles shapes like:
      {"generated_text": "..."}
      [{"generated_text": "..."}]
      [[{"generated_text": "..."}]]
      {"answer": "..."} / {"text": "..."}
    Returns (normalized_dict, original_object).
    """
    orig = o
    # unwrap single-item lists repeatedly
    while isinstance(o, list) and len(o) == 1:
        o = o[0]
    return (o if isinstance(o, dict) else {}), orig


def batched_indices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a Gemma-3 (VLM) via HF pipeline on VSR with binary 0/1 outputs (chat-batched + tqdm)")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-it", help="HF model name or local path (VLM)")
    parser.add_argument("--image-dir", type=str,
                        default="/home/hice1/ajin37/cs8803-vlm/theworld/python/theworld/datasets/vsr/images",
                        help="Directory containing VSR images (for resolving relative names).")
    parser.add_argument("--max_new_tokens", type=int, default=4, help="Small budget to discourage verbose outputs")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16, help="Pipeline batch size for GPU efficiency")
    parser.add_argument("--output", type=str, default="preds_binary.jsonl")
    parser.add_argument("--dataset", type=str, default="cambridgeltl/vsr_random", help="HF dataset id")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    # Perf niceties
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    image_dir = Path(args.image_dir)

    # 1) Load dataset
    ds = load_dataset(args.dataset)
    dsplit = ds[args.split]
    print("Columns:", dsplit.column_names)
    print("Size:", len(dsplit))

    # 2) Build pipeline
    task_name, vlm_pipe, processor = build_vlm_pipeline(args.model_name)
    print(f"Using pipeline task: {task_name}")

    # 3) Batched inference + logging with tqdm
    out_path = Path(args.output)
    n_correct = 0
    n_total_gold = 0
    n_unparsed = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for start, end in tqdm(list(batched_indices(len(dsplit), args.batch_size)),
                               total=(len(dsplit) + args.batch_size - 1) // args.batch_size,
                               desc="Processing (batched)"):
            # Build this batch’s inputs
            ids: List[int] = []
            statements: List[str] = []
            golds: List[Optional[int]] = []

            # For image-text-to-text: a list of "messages" per sample (chat dicts)
            messages_batch: List[List[Dict[str, Any]]] = []

            # For VQA fallback: parallel lists (converted to list-of-dicts before calling)
            vqa_images: List[Image.Image] = []
            vqa_questions: List[str] = []

            for i in range(start, end):
                ex = dsplit[i]
                img_key = pick_key(ex, IMAGE_KEYS)
                q_key = pick_key(ex, QUESTION_KEYS)
                a_key = pick_key(ex, ANSWER_KEYS)
                if img_key is None or q_key is None:
                    raise KeyError(f"Missing image/statement in example {i}. Keys: {list(ex.keys())}")

                pil_img = load_pil_image(ex[img_key], image_dir=image_dir)
                statement = str(ex[q_key])

                gold = None
                if a_key is not None and a_key in ex and ex[a_key] is not None:
                    try:
                        gold = int(ex[a_key])
                    except Exception:
                        gold = int(str(ex[a_key]).strip())

                ids.append(i)
                statements.append(statement)
                golds.append(gold)

                if task_name == "image-text-to-text":
                    messages_batch.append(make_messages(pil_img, statement))
                else:
                    vqa_images.append(pil_img)
                    vqa_questions.append(build_binary_prompt_plain(statement))

            # Pipeline call
            if task_name == "image-text-to-text":
                # Pass a list of chat message-lists; this avoids the chat heuristic error
                outputs = vlm_pipe(
                    messages_batch,
                    batch_size=args.batch_size,
                    generate_kwargs={
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "do_sample": args.temperature > 0.0,
                    },
                    return_full_text=False,
                )
            else:
                # VQA fallback: pass list-of-dicts per sample
                inputs = [{"image": img, "question": q} for img, q in zip(vqa_images, vqa_questions)]
                outputs = vlm_pipe(
                    inputs,
                    batch_size=args.batch_size,
                    generate_kwargs={
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "do_sample": args.temperature > 0.0,
                    },
                )

            if not isinstance(outputs, list):
                outputs = [outputs]

            # Parse + log
            for j, out in enumerate(outputs):
                norm, orig = _normalize_pipe_out(out)
                raw = norm.get("generated_text") or norm.get("answer") or norm.get("text")
                if raw is None:
                    raw = str(orig)

                pred_bin, pred_raw = parse_binary_output(raw)

                rec = {
                    "id": ids[j],
                    "statement": statements[j],
                    "pred_raw": pred_raw,
                    "pred": pred_bin,
                }
                gold = golds[j]
                if gold is not None:
                    rec["gold"] = gold
                    n_total_gold += 1
                    if pred_bin is None:
                        n_unparsed += 1
                    if (pred_bin is not None) and (pred_bin == gold):
                        n_correct += 1

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved predictions -> {out_path}")
    if n_total_gold > 0:
        acc = n_correct / n_total_gold
        print(f"Exact-match (binary) accuracy: {acc:.3f}  ({n_correct}/{n_total_gold})")
        if n_unparsed > 0:
            print(f"Unparsed outputs (counted incorrect): {n_unparsed}")
    else:
        print("No gold labels; skipped accuracy.")


if __name__ == "__main__":
    main()
