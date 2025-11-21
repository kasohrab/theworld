#!/usr/bin/env python3
"""Analyze prediction files and correlate with model training config.

Reads prediction files, fetches model configs from HuggingFace Hub,
and reports empty prediction rates alongside training configuration.

Usage:
    python scripts/spatial/analyze_predictions.py
    python scripts/spatial/analyze_predictions.py --predictions-dir outputs/spatial_results/predictions
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args():
    p = argparse.ArgumentParser(description="Analyze predictions vs model config")
    p.add_argument(
        "--predictions-dir",
        type=str,
        default="outputs/spatial_results/predictions",
        help="Directory containing prediction JSONL files",
    )
    p.add_argument(
        "--hub-prefix",
        type=str,
        default="kasohrab",
        help="HuggingFace Hub username prefix (default: kasohrab)",
    )
    return p.parse_args()


def count_empty_predictions(filepath: Path) -> tuple[int, int]:
    """Count empty and total predictions in a JSONL file."""
    empty = 0
    total = 0
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            total += 1
            if data.get("prediction", "") == "":
                empty += 1
    return empty, total


def get_hub_config(repo_id: str) -> dict | None:
    """Fetch config.json from HuggingFace Hub."""
    try:
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not fetch config for {repo_id}: {e}")
        return None


def get_trained_components(config: dict) -> str:
    """Determine which components were trained based on freeze flags."""
    if config is None:
        return "unknown"

    components = []

    # Check freeze flags (False = trained)
    if not config.get("freeze_gemma_language", True):
        components.append("LM")
    if not config.get("freeze_gemma_vision", True):
        components.append("Vision")
    if not config.get("freeze_cosmos_vae", True):
        components.append("VAE")

    # Projection is always trained
    components.append("Proj")

    return "+".join(components)


def main():
    args = parse_args()
    predictions_dir = Path(args.predictions_dir)

    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        return

    # Find all prediction files
    prediction_files = sorted(predictions_dir.glob("*.jsonl"))
    if not prediction_files:
        print(f"No prediction files found in {predictions_dir}")
        return

    print("=" * 100)
    print("PREDICTION ANALYSIS - Model Config vs Empty Predictions")
    print("=" * 100)
    print()

    results = []

    for pred_file in prediction_files:
        model_name = pred_file.stem  # e.g., "theworld-spatial-channel-2"

        # Count empty predictions
        empty, total = count_empty_predictions(pred_file)
        if total == 0:
            print(f"Skipping {model_name}: empty file")
            continue

        empty_pct = (empty / total) * 100

        # Try to fetch config from Hub
        # Handle both theworld-* and gemma-* models
        if model_name.startswith("gemma-3"):
            # Base Gemma model - no custom config
            repo_id = f"google/{model_name}"
            config = None
            trained = "N/A (base)"
            proj_mode = "N/A"
            proj_arch = "N/A"
            enable_world = False
        else:
            repo_id = f"{args.hub_prefix}/{model_name}"
            config = get_hub_config(repo_id)
            trained = get_trained_components(config)
            proj_mode = config.get("world_projection_mode", "?") if config else "?"
            proj_arch = config.get("projection_architecture", "?") if config else "?"
            enable_world = config.get("enable_world", False) if config else False

        results.append(
            {
                "model": model_name,
                "empty": empty,
                "total": total,
                "empty_pct": empty_pct,
                "trained": trained,
                "proj_mode": proj_mode,
                "proj_arch": proj_arch,
                "enable_world": enable_world,
            }
        )

    # Sort by empty percentage (descending)
    results.sort(key=lambda x: x["empty_pct"], reverse=True)

    # Print table
    print(f"{'Model':<50} {'Empty %':>10} {'Empty/Total':>15} {'Trained':>20} {'Proj Mode':>12} {'World':>8}")
    print("-" * 120)

    for r in results:
        world_str = "✓" if r["enable_world"] else "✗"
        print(
            f"{r['model']:<50} {r['empty_pct']:>9.1f}% {r['empty']:>6}/{r['total']:<6} {r['trained']:>20} {r['proj_mode']:>12} {world_str:>8}"
        )

    print()
    print("=" * 100)
    print("LEGEND:")
    print("  Trained: LM=Language Model, Vision=SigLIP encoder, VAE=Cosmos VAE, Proj=Projection layers")
    print("  Proj Mode: spatial (4096 tokens) or channel (16 tokens)")
    print("  World: ✓ = TheWorld with Cosmos, ✗ = Gemma-only baseline")
    print("=" * 100)

    # Summary
    print()
    print("OBSERVATIONS:")
    working = [r for r in results if r["empty_pct"] < 5]
    broken = [r for r in results if r["empty_pct"] >= 50]

    if working:
        print(f"  Working models (<5% empty): {len(working)}")
        for r in working:
            print(f"    - {r['model']} ({r['trained']})")

    if broken:
        print(f"  Broken models (≥50% empty): {len(broken)}")
        for r in broken:
            print(f"    - {r['model']} ({r['trained']})")


if __name__ == "__main__":
    main()
