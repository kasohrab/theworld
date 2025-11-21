#!/usr/bin/env python3
"""Batch judge predictions that are missing judgments.

Scans the predictions directory and runs judging for any models
that don't have a judged results file yet.

Usage:
    python scripts/spatial/batch_judge.py
    python scripts/spatial/batch_judge.py --judge gpt-oss --judge-model openai/gpt-oss-20b
    python scripts/spatial/batch_judge.py --output-dir outputs/spatial_results
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("outputs/spatial_results")
DEFAULT_JUDGE = "gpt-oss"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-20b"


def parse_args():
    p = argparse.ArgumentParser(description="Batch judge spatial predictions")
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--judge",
        type=str,
        default=DEFAULT_JUDGE,
        choices=["gemma", "gpt4", "gpt-oss"],
        help=f"Judge to use (default: {DEFAULT_JUDGE})",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Judge model ID (default: {DEFAULT_JUDGE_MODEL})",
    )
    p.add_argument("--batch-size", type=int, default=56, help="Batch size for judging")
    p.add_argument("--dry-run", action="store_true", help="Print what would be run without executing")
    return p.parse_args()


def calculate_metrics_from_jsonl(jsonl_path: Path) -> dict:
    """Calculate metrics from a judged JSONL file."""
    results = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not results:
        return {"total": 0, "correct": 0, "accuracy": 0.0}

    # Calculate accuracy
    correct = sum(1 for r in results if r.get("correct", False))
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    # By type
    by_type = {}
    for r in results:
        qa_type = r.get("qa_type", "unknown")
        if qa_type not in by_type:
            by_type[qa_type] = {"correct": 0, "total": 0}
        by_type[qa_type]["total"] += 1
        if r.get("correct", False):
            by_type[qa_type]["correct"] += 1

    by_type_acc = {k: v["correct"] / v["total"] if v["total"] > 0 else 0.0 for k, v in by_type.items()}

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_type": by_type_acc,
    }


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    judged_dir = output_dir / "judged"

    if not predictions_dir.exists():
        print(f"Error: Predictions directory not found: {predictions_dir}")
        sys.exit(1)

    # Find all prediction files
    prediction_files = list(predictions_dir.glob("*.jsonl"))
    if not prediction_files:
        print(f"No prediction files found in {predictions_dir}")
        sys.exit(0)

    print(f"Found {len(prediction_files)} prediction file(s)")
    print(f"Judge: {args.judge} ({args.judge_model})")

    # Track results
    skipped = []
    judged = []
    failed = []

    for pred_file in prediction_files:
        model_name = pred_file.stem  # e.g., "theworld-spatial-channel"
        model_judged_dir = judged_dir / model_name
        judge_output = model_judged_dir / f"{args.judge}.jsonl"

        if judge_output.exists():
            print(f"[SKIP] {model_name} (already judged with {args.judge})")
            skipped.append(model_name)
            continue

        print(f"\n[JUDGE] {model_name} with {args.judge}")

        cmd = [
            "python",
            "scripts/spatial/judge_predictions.py",
            "--predictions",
            str(pred_file),
            "--judge",
            args.judge,
            "--judge-model",
            args.judge_model,
            "--batch-size",
            str(args.batch_size),
            "--output",
            str(judge_output),
        ]

        if args.dry_run:
            print(f"  Would run: {' '.join(cmd)}")
            continue

        # Create output directory
        model_judged_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(cmd, check=True)
            judged.append(model_name)

            # Calculate and save metrics
            if judge_output.exists():
                metrics = calculate_metrics_from_jsonl(judge_output)
                metrics_file = model_judged_dir / "metrics.json"

                # Load existing metrics if present
                existing_metrics = {}
                if metrics_file.exists():
                    with open(metrics_file, "r") as f:
                        existing_metrics = json.load(f)

                # Add/update this judge's metrics
                existing_metrics[args.judge] = metrics

                with open(metrics_file, "w") as f:
                    json.dump(existing_metrics, f, indent=2)
                print(f"  Saved metrics to {metrics_file}")

        except subprocess.CalledProcessError as e:
            print(f"  [FAILED] {model_name}: {e}")
            failed.append(model_name)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Judged:  {len(judged)}")
    print(f"  Skipped: {len(skipped)} (already had {args.judge} judgments)")
    print(f"  Failed:  {len(failed)}")

    if failed:
        print("\nFailed models:")
        for m in failed:
            print(f"  - {m}")

    # Print accuracy table for all completed models
    if not args.dry_run:
        print("\n" + "=" * 60)
        print("ACCURACY TABLE")
        print("=" * 60)
        print(f"{'Model':<50} {'Accuracy':>10}")
        print("-" * 60)

        for pred_file in sorted(predictions_dir.glob("*.jsonl")):
            model_name = pred_file.stem
            metrics_file = judged_dir / model_name / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                if args.judge in metrics:
                    acc = metrics[args.judge]["accuracy"]
                    print(f"{model_name:<50} {acc*100:>9.2f}%")


if __name__ == "__main__":
    main()
