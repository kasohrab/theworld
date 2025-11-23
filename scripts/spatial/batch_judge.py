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
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm

# Enable verbose logging for model loading
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["HF_HUB_VERBOSITY"] = "info"


DEFAULT_OUTPUT_DIR = Path("outputs/spatial_results")
DEFAULT_JUDGE = "gpt-oss"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-20b"


def make_judge_slug(model_id: str) -> str:
    """Create a slug from model ID for filenames.

    Examples:
        openai/gpt-oss-120b -> openai-gpt-oss-120b
        unsloth/gpt-oss-120b-unsloth-bnb-4bit -> unsloth-gpt-oss-120b-unsloth-bnb-4bit
    """
    return model_id.replace("/", "-").replace("_", "-")


def make_judge_key(judge_type: str, model_id: str) -> str:
    """Create unique key for metrics JSON.

    Examples:
        ("gpt-oss", "openai/gpt-oss-120b") -> "gpt-oss--openai-gpt-oss-120b"
        ("gpt-oss", "unsloth/...") -> "gpt-oss--unsloth-gpt-oss-120b-unsloth-bnb-4bit"
    """
    return f"{judge_type}--{make_judge_slug(model_id)}"


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
        choices=["gemma", "gpt4", "gpt-oss", "deepseek"],
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


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def initialize_judge(judge_type: str, judge_model: str, max_tokens: int = 150, cache_dir: str = "/tmp/hf_cache"):
    """Initialize judge based on type."""
    if judge_type == "gemma":
        from theworld import TheWorld
        from theworld.evaluation import GemmaJudge

        print(f"Initializing Gemma judge: {judge_model}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TheWorld(
            gemma_model_name=judge_model,
            device=device,
            load_cosmos=False,
        )
        model.eval()
        print(f"✓ Model loaded on {device}")
        return GemmaJudge(model=model, max_new_tokens=max_tokens)

    elif judge_type == "gpt4":
        from theworld.evaluation import GPT4Judge

        print(f"Initializing GPT-4 judge: {judge_model}")
        judge = GPT4Judge(model=judge_model, max_tokens=max_tokens)
        print(f"✓ GPT-4 judge initialized")
        return judge

    elif judge_type == "gpt-oss":
        from theworld.evaluation import GPTOSSJudge

        print(f"Initializing GPT-OSS judge: {judge_model}")
        print(f"Cache directory: {cache_dir}")
        judge = GPTOSSJudge(
            model_id=judge_model,
            max_new_tokens=max_tokens,
            cache_dir=cache_dir,
        )
        print(f"✓ GPT-OSS judge initialized")
        return judge

    elif judge_type == "deepseek":
        from theworld.evaluation import DeepSeekJudge

        print(f"Initializing DeepSeek judge: {judge_model}")
        judge = DeepSeekJudge(
            model=judge_model,
            max_tokens=max_tokens,
        )
        print(f"✓ DeepSeek judge initialized")
        return judge

    else:
        raise ValueError(f"Unknown judge type: {judge_type}")


def run_judging_direct(
    predictions: List[Dict[str, Any]],
    judge,
    batch_size: int,
    output_path: Path,
) -> None:
    """Run judging on predictions and save results."""
    # Prepare output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"  Judging {len(predictions)} predictions (batch_size={batch_size})...")

    with open(output_path, "w", encoding="utf-8") as fout:
        # Process in batches
        for batch_start in tqdm(range(0, len(predictions), batch_size), desc="  Progress"):
            batch_end = min(batch_start + batch_size, len(predictions))
            batch = predictions[batch_start:batch_end]

            # Extract fields for judging
            questions = [p["question"] for p in batch]
            preds = [p["prediction"] for p in batch]
            ground_truths = [p["ground_truth"] for p in batch]
            qa_types = [p["qa_type"] for p in batch]

            # Judge batch
            try:
                judge_results = judge.judge(
                    questions=questions,
                    predictions=preds,
                    ground_truths=ground_truths,
                    qa_types=qa_types,
                )
            except Exception as e:
                # Error during judging
                judge_results = [
                    {
                        "score": 0.0,
                        "correct": False,
                        "judge_response": f"<ERROR: {e}>",
                        "judge_prompt": "",
                    }
                ] * len(batch)

            # Merge judge results with predictions
            for pred, judge_result in zip(batch, judge_results):
                result = {
                    **pred,  # Include all original fields
                    "score": judge_result["score"],
                    "correct": judge_result["correct"],
                    "judge_response": judge_result["judge_response"],
                }
                results.append(result)
                fout.write(json.dumps(result) + "\n")

    print(f"  ✓ Saved judged results to {output_path}")
    return results


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

    # Filter to only files that need judging
    judge_key = make_judge_key(args.judge, args.judge_model)
    files_to_judge = []
    skipped = []

    for pred_file in prediction_files:
        model_name = pred_file.stem
        model_judged_dir = judged_dir / model_name
        judge_output = model_judged_dir / f"{judge_key}.jsonl"

        if judge_output.exists():
            print(f"[SKIP] {model_name} (already judged with {judge_key})")
            skipped.append(model_name)
        else:
            files_to_judge.append(pred_file)

    if not files_to_judge:
        print(f"\nAll models already judged with {judge_key}")
        sys.exit(0)

    print(f"\nFound {len(prediction_files)} prediction file(s)")
    print(f"  To judge: {len(files_to_judge)}")
    print(f"  Skipped: {len(skipped)}")
    print(f"Judge: {args.judge} ({args.judge_model})")

    if args.dry_run:
        print("\n[DRY RUN] Would judge:")
        for pred_file in files_to_judge:
            print(f"  - {pred_file.stem}")
        sys.exit(0)

    # Initialize judge once
    print(f"\n{'='*60}")
    print("INITIALIZING JUDGE")
    print(f"{'='*60}")
    judge = initialize_judge(args.judge, args.judge_model, cache_dir="/tmp/hf_cache")

    # Track results
    judged = []
    failed = []

    # Process all files with the same judge instance
    print(f"\n{'='*60}")
    print("JUDGING PREDICTIONS")
    print(f"{'='*60}")

    for pred_file in files_to_judge:
        model_name = pred_file.stem  # e.g., "theworld-spatial-channel"
        model_judged_dir = judged_dir / model_name
        judge_output = model_judged_dir / f"{judge_key}.jsonl"

        print(f"\n[JUDGE] {model_name}")

        # Create output directory
        model_judged_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load predictions
            predictions = load_predictions(pred_file)

            # Run judging
            run_judging_direct(
                predictions=predictions,
                judge=judge,
                batch_size=args.batch_size,
                output_path=judge_output,
            )

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

                # Add metadata to metrics
                metrics_with_metadata = {
                    **metrics,
                    "metadata": {
                        "judge_type": args.judge,
                        "judge_model": args.judge_model,
                        "timestamp": datetime.now().isoformat(),
                        "batch_size": args.batch_size,
                    },
                }

                # Add/update this judge's metrics using unique key
                existing_metrics[judge_key] = metrics_with_metadata

                with open(metrics_file, "w") as f:
                    json.dump(existing_metrics, f, indent=2)
                print(f"  ✓ Saved metrics to {metrics_file}")

        except Exception as e:
            print(f"  [FAILED] {model_name}: {e}")
            failed.append(model_name)

    # Clean up judge model to free GPU memory
    del judge
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n✓ Freed GPU memory")

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
        print(f"{'Model':<50} {'Judge':<40} {'Accuracy':>10}")
        print("-" * 100)

        judge_key = make_judge_key(args.judge, args.judge_model)

        for pred_file in sorted(predictions_dir.glob("*.jsonl")):
            model_name = pred_file.stem
            metrics_file = judged_dir / model_name / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)

                # Try new-style key first, then fall back to old-style
                if judge_key in metrics:
                    acc = metrics[judge_key]["accuracy"]
                    judge_display = judge_key
                    print(f"{model_name:<50} {judge_display:<40} {acc*100:>9.2f}%")
                elif args.judge in metrics:
                    # Backward compatibility: old-style key
                    acc = metrics[args.judge]["accuracy"]
                    judge_display = args.judge
                    print(f"{model_name:<50} {judge_display:<40} {acc*100:>9.2f}%")


if __name__ == "__main__":
    main()
