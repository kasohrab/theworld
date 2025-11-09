#!/usr/bin/env python3
# run_vsr_theworld_binary_verify.py
# pip install -U datasets pillow torch torchvision requests tqdm safetensors

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors
from transformers import AutoProcessor

# --- Project import (mirror SpatialRGPT eval style) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from theworld import TheWorld
from theworld.constants import DEFAULT_GEMMA_MODEL

# -----------------------------
# Dataset column keys
# -----------------------------
IMAGE_KEYS = ["image", "img", "image_path", "image_url", "img_path", "img_url"]
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


def _try_open_url(url: str, timeout: float = 10.0) -> Image.Image:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


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
            return _try_open_url(s, timeout=10.0)

        p = Path(s)
        if p.is_absolute():
            # absolute path must exist
            return Image.open(p).convert("RGB")

        # relative: check under image_dir first
        candidate = image_dir / p
        if candidate.exists():
            return Image.open(candidate).convert("RGB")

        # fallback: try as-given relative to CWD
        return Image.open(p).convert("RGB")

    raise ValueError(f"Unsupported image value type: {type(val)}")


# -----------------------------
# Binary prompting + parsing
# -----------------------------
BINARY_SYSTEM_INSTRUCTION = (
    "You are a vision-language model that performs binary visual entailment.\n"
    "Task: Look at the image and evaluate the statement.\n"
    "Output exactly one digit with no extra text: 1 if the statement is true, 0 if the statement is false."
)

def build_binary_prompt(statement: str) -> str:
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
    raw = (text or "").strip()
    m = re.search(r"\b([01])\b", raw)
    if m:
        return int(m.group(1)), raw
    tokens = re.findall(r"[A-Za-z]+", raw.lower())
    for t in tokens:
        if t in _WORD_TO_BIN:
            return _WORD_TO_BIN[t], raw
    return None, raw


def batched_indices(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end


# -----------------------------
# Assistant prelude guard
# -----------------------------
# Common (Gemma3) token ids seen in logs:
# 105 = <start_of_turn>, 4368 = "model", 107 = "\n"
ASSISTANT_PFX_PATTERNS = (
    [105, 4368, 107],   # <start_of_turn> model \n
    [4368, 107],        # model \n
    [105, 4368],        # <start_of_turn> model
    [4368],             # model
)

def advance_over_assistant_prelude(seq_list: List[int], start: int) -> int:
    """If the slice at `start` begins with an assistant prelude, advance start past it."""
    for pat in ASSISTANT_PFX_PATTERNS:
        L = len(pat)
        if start + L <= len(seq_list) and seq_list[start:start+L] == pat:
            return start + L
    return start


# -----------------------------
# Verification pass
# -----------------------------
def verify_dataset_images(dsplit, image_dir: Path, max_samples: int = 0, log_path: Optional[Path] = None):
    """
    Pre-scan the dataset: ensure each image can be opened.
    Returns: (valid_indices: List[int], failures: List[dict])
    """
    total = len(dsplit) if max_samples == 0 else min(max_samples, len(dsplit))
    valid_indices: List[int] = []
    failures: List[Dict[str, Any]] = []

    for i in tqdm(range(total), desc="Verifying images"):
        ex = dsplit[i]
        img_key = pick_key(ex, IMAGE_KEYS)
        q_key = pick_key(ex, QUESTION_KEYS)

        if img_key is None or q_key is None:
            failures.append({"idx": i, "reason": "missing_required_keys", "keys": list(ex.keys())})
            continue

        try:
            _ = load_pil_image(ex[img_key], image_dir=image_dir)  # attempt to open/convert
            valid_indices.append(i)
        except Exception as e:
            failures.append({"idx": i, "reason": f"{type(e).__name__}: {e}"})

    if log_path is not None and failures:
        with log_path.open("w", encoding="utf-8") as f:
            for rec in failures:
                f.write(json.dumps(rec) + "\n")

    return valid_indices, failures


# -----------------------------
# Model loader that accepts full HF model or local delta checkpoint via --model
# -----------------------------
def load_from_model_arg(model_arg: str, enable_world: bool, device: str, **hf_kwargs) -> TheWorld:
    """
    If `model_arg` is:
      - full HF model id/path -> load directly via TheWorld.from_pretrained(...)
      - a trainable-only checkpoint dir (has checkpoint_config.json + model.safetensors/.bin, but no HF config.json)
        -> read its base gemma id from checkpoint_config.json, load base, then overlay deltas.
    """
    from theworld import TheWorld  # local import to avoid circulars
    from transformers import AutoProcessor
    import os, json
    import torch
    from safetensors.torch import load_file as load_safetensors

    ckpt_cfg = os.path.join(model_arg, "checkpoint_config.json")
    has_delta_weights = os.path.exists(os.path.join(model_arg, "model.safetensors")) or \
                        os.path.exists(os.path.join(model_arg, "pytorch_model.bin"))
    has_hf_config = os.path.exists(os.path.join(model_arg, "config.json"))

    # Trainable-only checkpoint folder?
    if os.path.isdir(model_arg) and os.path.exists(ckpt_cfg) and has_delta_weights and not has_hf_config:
        with open(ckpt_cfg, "r") as f:
            meta = json.load(f)
        mcfg = meta.get("model_config", {})
        base_gemma = mcfg.get("gemma_model_name") or DEFAULT_GEMMA_MODEL

        # Prefer hf_kwargs value if provided; otherwise use checkpoint's hint; default False to save RAM.
        ckpt_lfcp = bool(mcfg.get("load_full_cosmos_pipeline", False))
        lfcp = hf_kwargs.pop("load_full_cosmos_pipeline", ckpt_lfcp)

        # 1) Load base Gemma via TheWorld
        model = TheWorld.from_pretrained(
            base_gemma,
            enable_world=enable_world,
            device=device,
            load_full_cosmos_pipeline=lfcp,  # <-- avoid duplicate by popping above
            **hf_kwargs
        )

        # 2) Overlay trainable deltas
        st_path = os.path.join(model_arg, "model.safetensors")
        pt_path = os.path.join(model_arg, "pytorch_model.bin")
        state = load_safetensors(st_path) if os.path.exists(st_path) else torch.load(pt_path, map_source="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[ckpt] missing keys (ok for frozen): {len(missing)}")
        if unexpected:
            print(f"[ckpt] unexpected keys: {unexpected}")
        print(f"[ckpt] loaded weights from {model_arg}")

        # 3) Try to load processor from checkpoint dir (keeps any special tokens/pad identical)
        try:
            model.processor = AutoProcessor.from_pretrained(model_arg)
            print("[ckpt] loaded processor from checkpoint dir")
        except Exception:
            pass

        return model

    # Else: treat as full HF model id/path (no duplicate kw)
    return TheWorld.from_pretrained(
        model_arg,
        enable_world=enable_world,
        device=device,
        **hf_kwargs
    )



# -----------------------------
# build inputs with template
# -----------------------------
def build_inputs(model, prompts: List[str], images: List[Image.Image], debug_tokens: bool = False, debug_samples: int = 2):
    """
    IMPORTANT: For Gemma3's processor with chat images, `images` must be a list of lists:
    one inner list per conversation, even if there's only one image per item.
    """
    conversations = [[
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": p},
        ]}
    ] for p in prompts]

    # Each item has exactly one image → wrap as [[img], [img], ...]
    images_batch: List[List[Image.Image]] = [[im] for im in images]

    rendered = model.processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True
    )

    enc = model.processor(
        text=rendered,
        images=images_batch,            # <- list-of-lists to match chat placeholders
        return_tensors="pt",
        padding=True
    )

    if debug_tokens and len(rendered) > 0:
        preview_chars = 250
        k = min(len(rendered), max(1, debug_samples))
        for i in range(k):
            s = rendered[i]
            short = (s[:preview_chars] + "…") if len(s) > preview_chars else s
            print(f"[debug] rendered[{i}] (first {preview_chars} chars):\n{short}")

    return enc, images_batch, rendered


# -----------------------------
# Args / Driver
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VSR (binary 0/1) using TheWorld.generate(), with image existence verification")
    p.add_argument("--dataset", type=str, default="cambridgeltl/vsr_random", help="HF dataset id or local path")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--image-dir", type=str,
                   default="/home/hice1/ajin37/cs8803-vlm/theworld/python/theworld/datasets/vsr/images",
                   help="Directory containing VSR images (for resolving relative names).")
    p.add_argument("--model", type=str, default=DEFAULT_GEMMA_MODEL,
                   help="HF model id or local path. Can also be a local TheWorld trainable-only checkpoint dir.")
    p.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda|cpu)")
    p.add_argument("--output", type=str, default="preds_binary.jsonl")
    p.add_argument("--max-samples", type=int, default=0, help="0 = all")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=4, help="Small budget to discourage verbose outputs")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--verify-only", action="store_true", help="Only verify images and exit (no evaluation)")
    p.add_argument("--verify-log", type=str, default="missing_images.jsonl", help="Where to write missing/unreadable entries")
    # TheWorld / Cosmos knobs
    p.add_argument("--load-cosmos", action="store_true", help="Enable world model (Cosmos). Default: Gemma-only baseline.")
    p.add_argument("--num-world-steps", type=int, default=0, help="(Reserved) Number of world steps if your Cosmos path uses it")
    # HF loading knobs
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--device-map", type=str, default="auto")
    # Debug knobs
    p.add_argument("--debug-tokens", action="store_true", help="Print token-level debug info")
    p.add_argument("--debug-samples", type=int, default=2, help="How many samples per batch to print")
    p.add_argument("--debug-prompt-tail", type=int, default=32, help="How many prompt token IDs to show from the end")
    return p.parse_args()


def _str_to_dtype(s: str):
    if s == "float32":
        return torch.float32
    if s == "float16":
        return torch.float16
    return torch.bfloat16


def main():
    args = parse_args()

    # Perf niceties
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    image_dir = Path(args.image_dir)

    # 1) Load dataset
    ds = load_dataset(args.dataset)
    dsplit = ds[args.split]
    raw_total = len(dsplit)
    total = raw_total if args.max_samples == 0 else min(args.max_samples, raw_total)
    print("Columns:", dsplit.column_names)
    print("Size:", raw_total)
    print("Planned eval samples:", total)

    # 2) Pre-verify images
    verify_log_path = Path(args.verify_log) if args.verify_log else None
    valid_indices, failures = verify_dataset_images(dsplit, image_dir, max_samples=total, log_path=verify_log_path)

    if failures:
        print(f"⚠ Image verification: {len(failures)} / {total} samples failed to load. "
              f"Details written to {verify_log_path}")
    print(f"✓ Proceeding with {len(valid_indices)} valid samples")

    if args.verify_only:
        print("Verify-only mode enabled; exiting after verification.")
        return

    # 3) Load TheWorld model (supports HF id/path OR local delta checkpoint via --model)
    mode_str = "TheWorld (Cosmos enabled)" if args.load_cosmos else "Gemma-only baseline via TheWorld"
    print(f"Loading model from: {args.model}  [{mode_str}]")

    model = load_from_model_arg(
        args.model,
        enable_world=args.load_cosmos,
        device=args.device,
        load_full_cosmos_pipeline=False,   # keep RAM/GPU use modest; VAE-only if world enabled
        dtype=_str_to_dtype(args.dtype),
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        torch_dtype=_str_to_dtype(args.dtype),
    )

    model.eval()
    device_str = args.device
    print(f"✓ Model ready (eval mode)")

    # 4) Run (batched) inference with model.generate()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_correct = 0
    n_total_gold = 0
    n_unparsed = 0
    n_skipped_runtime = 0

    tokenizer = model.processor.tokenizer  # convenience
    if tokenizer.pad_token_id is None:
        # Gemma3 typically uses 0 as pad; ensure set so generate() can pad
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    print(f"[tokenizer] pad_token_id={tokenizer.pad_token_id} eos_token_id={tokenizer.eos_token_id} "
          f"pad_side={getattr(tokenizer, 'padding_side', 'unknown')}")

    with out_path.open("w", encoding="utf-8") as fout:
        steps = (len(valid_indices) + args.batch_size - 1) // args.batch_size
        for bi_start, bi_end in tqdm(
            list(batched_indices(len(valid_indices), args.batch_size)),
            total=steps,
            desc="Evaluating"
        ):
            ids: List[int] = []
            images: List[Image.Image] = []
            prompts: List[str] = []
            golds: List[Optional[int]] = []

            # Build batch (re-open images to handle flakiness gracefully)
            for k in range(bi_start, bi_end):
                i = valid_indices[k]
                ex = dsplit[i]
                img_key = pick_key(ex, IMAGE_KEYS)
                q_key = pick_key(ex, QUESTION_KEYS)
                a_key = pick_key(ex, ANSWER_KEYS)

                if img_key is None or q_key is None:
                    n_skipped_runtime += 1
                    continue

                try:
                    pil_img = load_pil_image(ex[img_key], image_dir=image_dir)
                except Exception:
                    n_skipped_runtime += 1
                    continue

                statement = str(ex[q_key])

                gold = None
                if a_key is not None and a_key in ex and ex[a_key] is not None:
                    try:
                        gold = int(ex[a_key])
                    except Exception:
                        try:
                            gold = int(str(ex[a_key]).strip())
                        except Exception:
                            gold = None

                ids.append(i)
                images.append(pil_img)
                prompts.append(build_binary_prompt(statement))
                golds.append(gold)

            if not images:
                continue  # whole batch collapsed due to skips

            # build inputs
            try:
                inputs, images_batch, rendered = build_inputs(
                    model, prompts, images,
                    debug_tokens=args.debug_tokens, debug_samples=args.debug_samples
                )
                # move tensors to model's device(s) if single device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(model.device if hasattr(model, "device") else device_str)

                if args.debug_tokens:
                    attn = inputs["attention_mask"]
                    print(f"[debug] batch input_ids shape={tuple(inputs['input_ids'].shape)} "
                          f"attention_mask shape={tuple(attn.shape)}")
                    for di in range(min(args.debug_samples, len(images_batch))):
                        prompt_len_i = int(attn[di].sum().item())
                        print(f"[debug] sample#{di}: images={len(images_batch[di])}, "
                              f"rendered_len={len(rendered[di])}, prompt_len(tokens)={prompt_len_i}")

                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=(args.temperature > 0.0),
                    pad_token_id=tokenizer.pad_token_id,
                    # eos_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                    return_dict_in_generate=True,
                )

                if args.load_cosmos:
                    gen_out = model.generate(**inputs, images=images_batch, **gen_kwargs)
                else:
                    gen_out = model.generate(**inputs, **gen_kwargs)

                # ---- Per-row boundary based on attention mask + assistant-prelude guard ----
                seq = gen_out.sequences            # (B, T_in + T_new) — true prompt (unpadded) + generated
                attn = inputs["attention_mask"]    # (B, T_in)        — padded mask for inputs
                inp_ids = inputs["input_ids"]      # (B, T_in)        — padded input ids (for debug only)

                new_token_rows: List[torch.Tensor] = []
                cut_indices: List[int] = []

                for i in range(seq.size(0)):
                    pl = int(attn[i].sum().item())         # true input length for this row
                    start = pl                              # boundary
                    seq_list = seq[i].tolist()

                    # If the immediate tokens after `start` are the assistant prelude,
                    # advance the boundary past them so they stay with the prompt.
                    start = advance_over_assistant_prelude(seq_list, start)

                    row_new_ids = seq_list[start:]
                    new_token_rows.append(torch.tensor(row_new_ids, dtype=seq.dtype, device=seq.device))
                    cut_indices.append(start)

                # Decode row-by-row
                preds_text = [
                    model.processor.decode(r.tolist(), skip_special_tokens=True)
                    for r in new_token_rows
                ]

                # Optional, detailed token-level debug for first N samples
                if args.debug_tokens:
                    vocab_to_str = tokenizer.convert_ids_to_tokens
                    for i in range(min(args.debug_samples, len(new_token_rows))):
                        prompt_len_i = int(attn[i].sum().item())
                        inp_ids_i = inp_ids[i].tolist()
                        tail_k = min(args.debug_prompt_tail, prompt_len_i)
                        prompt_tail_ids = inp_ids_i[-prompt_len_i:][-tail_k:]
                        prompt_tail_txt = tokenizer.decode(prompt_tail_ids, skip_special_tokens=False)

                        gen_ids = new_token_rows[i].tolist()
                        gen_tok_strs = [vocab_to_str(tid) for tid in gen_ids]

                        cut_index = cut_indices[i]
                        seq_head = seq[i].tolist()[:min(8, cut_index)]
                        print(f"\n[debug] >>> sample {i}")
                        print(f"[debug] id={ids[i]}  gold={golds[i]}  max_new_tokens={args.max_new_tokens}")
                        print(f"[debug] prompt_len={prompt_len_i}  prompt_tail_ids(last {tail_k}): {prompt_tail_ids}")
                        print(f"[debug] prompt_tail_text:\n{repr(prompt_tail_txt)}")
                        print(f"[debug] cut_index={cut_index}  seq_head={seq_head}")
                        print(f"[debug] generated_token_ids: {gen_ids}")
                        print(f"[debug] generated_tokens (strings): {gen_tok_strs}")
                        print(f"[debug] generated_text:{repr(preds_text[i])}")

            except Exception as e:
                preds_text = [f"<ERROR: {e}>"] * len(images)
                raise e

            # Parse + log
            for j, raw in enumerate(preds_text):
                pred_bin, pred_raw = parse_binary_output(str(raw))

                rec = {
                    "id": ids[j],
                    "statement": prompts[j],  # includes the statement; useful for debugging
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

    print(f"\n✓ Saved predictions -> {out_path}")
    if n_total_gold > 0:
        acc = n_correct / n_total_gold
        print(f"Exact-match (binary) accuracy: {acc:.3f}  ({n_correct}/{n_total_gold})")
        if n_unparsed > 0:
            print(f"Unparsed outputs (counted incorrect): {n_unparsed}")
    else:
        print("No gold labels; skipped accuracy.")

    if n_skipped_runtime > 0:
        print(f"Note: skipped {n_skipped_runtime} samples during evaluation due to runtime image load failures.")


if __name__ == "__main__":
    main()
