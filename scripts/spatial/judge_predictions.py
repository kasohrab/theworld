"""Judge spatial reasoning predictions using various judge models.

This script takes predictions from eval_spatial_rgpt.py and judges them
using Gemma, GPT-4, or GPT-OSS. This allows re-judging predictions without
re-running expensive inference.

Usage:
    # Judge with Gemma (default model)
    python scripts/spatial/judge_predictions.py \
        --predictions outputs/predictions.jsonl \
        --judge gemma \
        --output outputs/results_judged_gemma.jsonl

    # Judge with GPT-4 (requires OPENAI_API_KEY env var)
    python scripts/spatial/judge_predictions.py \
        --predictions outputs/predictions.jsonl \
        --judge gpt4 \
        --output outputs/results_judged_gpt4.jsonl

    # Judge with GPT-OSS (custom model)
    python scripts/spatial/judge_predictions.py \
        --predictions outputs/predictions.jsonl \
        --judge gpt-oss \
        --judge-model openai/gpt-oss-20b \
        --output outputs/results_judged_gptoss.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm


def load_predictions(path: str) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file.

    Args:
        path: Path to predictions JSONL file

    Returns:
        List of prediction dictionaries
    """
    predictions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Judge spatial reasoning predictions")

    p.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSONL file (from eval_spatial_rgpt.py --skip-judging)",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for judged results JSONL",
    )
    p.add_argument(
        "--judge",
        type=str,
        choices=["gemma", "gpt4", "gpt-oss"],
        required=True,
        help="Judge model to use",
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model ID (defaults: gemma=google/gemma-3-4b-it, gpt4=gpt-4, gpt-oss=openai/gpt-oss-120b)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens for judge response",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=56,
        help="Batch size for judging",
    )

    return p.parse_args()


def run_judging(
    predictions: List[Dict[str, Any]],
    judge,
    batch_size: int,
    output_path: str,
) -> None:
    """Run judging on predictions and save results.

    Args:
        predictions: List of prediction dicts
        judge: Judge instance (GemmaJudge or GPT4Judge)
        batch_size: Batch size for processing
        output_path: Output path for results
    """
    # Prepare output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"Judging {len(predictions)} predictions...")
    print(f"Batch size: {batch_size}")

    with open(output_file, "w", encoding="utf-8") as fout:
        # Process in batches
        for batch_start in tqdm(range(0, len(predictions), batch_size), desc="Judging"):
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

    print(f"✓ Saved judged results to {output_path}")

    # Calculate and print metrics
    print("\n" + "=" * 60)
    print("JUDGING RESULTS")
    print("=" * 60)

    from theworld.evaluation import calculate_spatial_accuracy

    metrics = calculate_spatial_accuracy(results)
    print(f"\nOverall:")
    print(f"  Total: {metrics['total']}")
    print(f"  Correct: {metrics['correct']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

    if metrics["by_type"]:
        print(f"\nBy Question Type:")
        for qa_type, acc in metrics["by_type"].items():
            print(f"  {qa_type}: {acc:.4f} ({acc*100:.2f}%)")

    if metrics["by_category"]:
        print(f"\nBy Category:")
        sorted_categories = sorted(metrics["by_category"].items(), key=lambda x: x[1], reverse=True)
        for qa_category, acc in sorted_categories[:10]:  # Top 10 categories
            print(f"  {qa_category}: {acc:.4f} ({acc*100:.2f}%)")

    print("=" * 60)


def get_default_model(judge_type: str) -> str:
    """Get default model for judge type."""
    defaults = {
        "gemma": "google/gemma-3-4b-it",
        "gpt4": "gpt-4",
        "gpt-oss": "openai/gpt-oss-120b",
    }
    return defaults[judge_type]


def main():
    """Main entry point."""
    args = parse_args()

    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    predictions = load_predictions(args.predictions)
    print(f"✓ Loaded {len(predictions)} predictions")

    # Get model ID
    model_id = args.judge_model or get_default_model(args.judge)

    # Initialize judge
    if args.judge == "gemma":
        print(f"Initializing Gemma judge: {model_id}")
        from theworld import TheWorld
        from theworld.evaluation import GemmaJudge

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = TheWorld(
            gemma_model_name=model_id,
            device=device,
            load_cosmos=False,
        )
        model.eval()
        print(f"✓ Model loaded on {device}")

        judge = GemmaJudge(model=model, max_new_tokens=args.max_tokens)

    elif args.judge == "gpt4":
        print(f"Initializing GPT-4 judge: {model_id}")
        from theworld.evaluation import GPT4Judge

        judge = GPT4Judge(model=model_id, max_tokens=args.max_tokens)
        print(f"✓ GPT-4 judge initialized")

    elif args.judge == "gpt-oss":
        print(f"Initializing GPT-OSS judge: {model_id}")
        from theworld.evaluation import GPTOSSJudge

        judge = GPTOSSJudge(model_id=model_id, max_new_tokens=args.max_tokens)
        print(f"✓ GPT-OSS judge initialized")

    else:
        raise ValueError(f"Unknown judge: {args.judge}")

    # Run judging
    run_judging(
        predictions=predictions,
        judge=judge,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
