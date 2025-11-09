"""Judge spatial reasoning predictions using various judge models.

This script takes predictions from eval_spatial_rgpt.py and judges them
using either Gemma or GPT-4. This allows re-judging predictions without
re-running expensive inference.

Usage:
    # Judge with Gemma
    python scripts/spatial/judge_predictions.py \
        --predictions outputs/predictions.jsonl \
        --judge gemma \
        --model google/gemma-3-4b-it \
        --output outputs/results_judged_gemma.jsonl

    # Judge with GPT-4
    export OPENAI_API_KEY=sk-...
    python scripts/spatial/judge_predictions.py \
        --predictions outputs/predictions.jsonl \
        --judge gpt4 \
        --output outputs/results_judged_gpt4.jsonl

    # Judge with GPT-OSS
    python scripts/spatial/judge_predictions.py \
        --predictions outputs/predictions.jsonl \
        --judge gpt-oss \
        --gpt-oss-model openai/gpt-oss-120b \
        --output outputs/results_judged_gptoss.jsonl
"""

import argparse
import json
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

    # Input/output
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

    # Judge selection
    p.add_argument(
        "--judge",
        type=str,
        choices=["gemma", "gpt4", "gpt-oss"],
        required=True,
        help="Judge model to use",
    )

    # Gemma-specific args
    p.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-4b-it",
        help="Gemma model to use (only for --judge gemma)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for Gemma (only for --judge gemma)",
    )

    # GPT-4 specific args
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    p.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-4",
        help="GPT model to use (only for --judge gpt4)",
    )

    # GPT-OSS specific args
    p.add_argument(
        "--gpt-oss-model",
        type=str,
        default="openai/gpt-oss-120b",
        help="GPT-OSS model to use (only for --judge gpt-oss)",
    )
    p.add_argument(
        "--gpt-oss-device-map",
        type=str,
        default="auto",
        help="Device map for GPT-OSS (only for --judge gpt-oss)",
    )
    p.add_argument(
        "--gpt-oss-dtype",
        type=str,
        default="auto",
        help="Torch dtype for GPT-OSS (only for --judge gpt-oss)",
    )

    # Judge parameters
    p.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Max tokens for judge response",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for judge",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for judging (only for Gemma)",
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


def main():
    """Main entry point."""
    args = parse_args()

    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    predictions = load_predictions(args.predictions)
    print(f"✓ Loaded {len(predictions)} predictions")

    # Initialize judge
    if args.judge == "gemma":
        print(f"Initializing Gemma judge: {args.model}")
        from theworld import TheWorld
        from theworld.evaluation import GemmaJudge

        # Load model
        model = TheWorld(
            gemma_model_name=args.model,
            device=args.device,
            load_cosmos=False,  # Don't need Cosmos for judging
        )
        model.eval()
        print(f"✓ Model loaded on {args.device}")

        judge = GemmaJudge(
            model=model,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    elif args.judge == "gpt4":
        print(f"Initializing GPT-4 judge: {args.gpt_model}")
        from theworld.evaluation import GPT4Judge

        judge = GPT4Judge(
            api_key=args.api_key,
            model=args.gpt_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"✓ GPT-4 judge initialized")

    elif args.judge == "gpt-oss":
        print(f"Initializing GPT-OSS judge: {args.gpt_oss_model}")
        from theworld.evaluation import GPTOSSJudge

        judge = GPTOSSJudge(
            model_id=args.gpt_oss_model,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            device_map=args.gpt_oss_device_map,
            torch_dtype=args.gpt_oss_dtype,
        )
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
