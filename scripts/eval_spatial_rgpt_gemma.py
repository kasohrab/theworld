"""
Run Gemma baseline on Spatial-RGPT evaluation data and save results as JSONL.

Usage (example):
    python scripts/eval_spatial_rgpt_gemma.py \
        --data-path data/spatial_rgpt_eval.jsonl \
        --model google/gemma-3-4b-it \
        --device cuda \
        --output outputs/spatial_rgpt_gemma_results.jsonl \
        --max-samples 200

This script uses TheWorld with load_cosmos=False for Gemma-only baseline evaluation.
It intentionally does NOT compute or use depth maps — it is a baseline
that uses Gemma-only inputs so you can compare against Spatial-RGPT depth-based outputs.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# Add project root to path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from theworld.datasets.spatial_rgpt import SpatialRGPTDataset
from theworld import TheWorld


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True, help="Path to local JSONL or HF dataset id")
    p.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="outputs/spatial_rgpt_gemma_results.jsonl")
    p.add_argument("--max-samples", type=int, default=0, help="Limit samples for quick runs (0 = all)")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def parse_choice_from_text(generated_text: str, choices: Optional[list]) -> str:
    # Very small heuristic parser used in demo scripts
    if choices is None:
        return generated_text.strip()
    text = generated_text.strip()
    # If single letter
    if len(text) == 1 and text.upper() in [chr(ord("A") + i) for i in range(len(choices))]:
        return text.upper()
    # Letter at start
    import re

    m = re.match(r"^([A-Da-d])[\):\.\s]", text)
    if m:
        return m.group(1).upper()
    # Match full choice text
    tl = text.lower()
    for i, c in enumerate(choices):
        if c.lower() in tl:
            return chr(ord("A") + i)
    # Fallback: empty
    return ""


def run_eval(
    data_path: str,
    model_name: str,
    device: str,
    output_path: str,
    max_samples: int,
    max_new_tokens: int,
    temperature: float,
):
    # Load dataset: support local JSONL path or HuggingFace dataset id
    data_path_obj = Path(data_path)
    if data_path_obj.exists():
        ds = SpatialRGPTDataset(data_path_obj, num_samples=max_samples if max_samples > 0 else None)
    else:
        # Try loading as a HuggingFace dataset id. Try common eval splits, then fallback to no split.
        from datasets import load_dataset as hf_load_dataset

        hf_dataset = None
        tried = []
        for split in ("validation", "test", "train"):
            try:
                tried.append(split)
                hf_dataset = hf_load_dataset(data_path, split=split)
                break
            except Exception:
                hf_dataset = None
        if hf_dataset is None:
            try:
                hf_dataset = hf_load_dataset(data_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load HuggingFace dataset '{data_path}' (tried splits {tried}): {e}")

        ds = SpatialRGPTDataset(hf_dataset, num_samples=max_samples if max_samples > 0 else None)

    # Init TheWorld in Gemma-only mode (no Cosmos loading)
    print(f"Loading TheWorld model (Gemma-only, load_cosmos=False): {model_name}")
    model = TheWorld(
        gemma_model_name=model_name,
        device=device,
        load_cosmos=False,  # Skip Cosmos loading for baseline evaluation
    )
    model.eval()  # Set to evaluation mode
    print(f"✓ Model loaded in evaluation mode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for i in tqdm(range(len(ds)), desc="Evaluating"):
            ex = ds[i]
            img = ex["image"]
            prompt = ex["question"]

            try:
                response = model.generate(
                    image=img,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    skip_world_tokens=True,  # Gemma-only baseline mode
                )
            except Exception as e:
                response = f"<ERROR: {e}>"

            parsed = parse_choice_from_text(response, ex.get("choices"))

            out = {
                "id": ex.get("id"),
                "prompt": prompt,
                "gemma_response": response,
                "parsed_choice": parsed,
                "ground_truth": ex.get("answer"),
                "has_choices": bool(ex.get("choices")),
            }
            fout.write(json.dumps(out) + "\n")

    print(f"Saved results to {output_path}")


def main():
    args = parse_args()
    run_eval(
        data_path=args.data_path,
        model_name=args.model,
        device=args.device,
        output_path=args.output,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
