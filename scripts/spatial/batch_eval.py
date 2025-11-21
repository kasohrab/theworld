#!/usr/bin/env python3
"""Batch evaluate models on SpatialRGPT-Bench.

Reads model IDs from a text file and runs evaluation for any that don't
already have prediction files.

Usage:
    python scripts/spatial/batch_eval.py models.txt
    python scripts/spatial/batch_eval.py models.txt --output-dir outputs/spatial_results
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Default paths
DEFAULT_OUTPUT_DIR = Path("outputs/spatial_results")
DEFAULT_DATA_PATH = "a8cheng/SpatialRGPT-Bench"  # HuggingFace dataset
DEFAULT_IMAGE_FOLDER = ""  # Images from HF dataset


def model_id_to_filename(model_id: str) -> str:
    """Convert HuggingFace model ID to a safe filename.

    Examples:
        kasohrab/theworld-spatial-channel -> theworld-spatial-channel.jsonl
        google/gemma-3-4b-it -> gemma-3-4b-it.jsonl
    """
    # Take the part after the slash (or the whole thing if no slash)
    name = model_id.split("/")[-1]
    return f"{name}.jsonl"


def parse_args():
    p = argparse.ArgumentParser(description="Batch evaluate models on SpatialRGPT-Bench")
    p.add_argument("models_file", type=str, help="Text file with model IDs (one per line)")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"HuggingFace dataset or path to JSONL (default: {DEFAULT_DATA_PATH})",
    )
    p.add_argument(
        "--image-folder",
        type=str,
        default=DEFAULT_IMAGE_FOLDER,
        help="Path to images folder (empty = from HF dataset)",
    )
    p.add_argument("--max-samples", type=int, default=0, help="Max samples per model (0=all)")
    p.add_argument("--dry-run", action="store_true", help="Print what would be run without executing")
    return p.parse_args()


def main():
    args = parse_args()

    # Read model IDs
    models_file = Path(args.models_file)
    if not models_file.exists():
        print(f"Error: Models file not found: {models_file}")
        sys.exit(1)

    model_ids = []
    with open(models_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                model_ids.append(line)

    if not model_ids:
        print("No model IDs found in file")
        sys.exit(1)

    print(f"Found {len(model_ids)} model(s) in {models_file}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    skipped = []
    evaluated = []
    failed = []

    for model_id in model_ids:
        filename = model_id_to_filename(model_id)
        output_path = predictions_dir / filename

        if output_path.exists():
            print(f"[SKIP] {model_id} -> {filename} (already exists)")
            skipped.append(model_id)
            continue

        print(f"\n[EVAL] {model_id} -> {filename}")

        cmd = [
            "python",
            "scripts/spatial/eval_spatial_rgpt.py",
            "--data-path",
            args.data_path,
            "--image-folder",
            args.image_folder,
            "--model",
            model_id,
            "--skip-judging",
            "--output",
            str(output_path),
        ]

        if args.max_samples > 0:
            cmd.extend(["--max-samples", str(args.max_samples)])

        if args.dry_run:
            print(f"  Would run: {' '.join(cmd)}")
            continue

        try:
            subprocess.run(cmd, check=True)
            evaluated.append(model_id)
        except subprocess.CalledProcessError as e:
            print(f"  [FAILED] {model_id}: {e}")
            failed.append(model_id)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Evaluated: {len(evaluated)}")
    print(f"  Skipped:   {len(skipped)} (already had predictions)")
    print(f"  Failed:    {len(failed)}")

    if failed:
        print("\nFailed models:")
        for m in failed:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
