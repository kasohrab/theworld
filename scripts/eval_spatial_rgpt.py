"""
Evaluate TheWorld on SpatialRGPT-Bench with Gemma-as-judge evaluation.

Usage (example):
    python scripts/eval_spatial_rgpt.py \
        --data-path /path/to/SpatialRGPT-Bench/val_SpatialRGPT-Bench.jsonl \
        --image-folder /path/to/SpatialRGPT-Bench/images \
        --model username/theworld-model \
        --output outputs/spatial_rgpt_results.jsonl \
        --max-samples 10

This script:
1. Loads SpatialRGPT-Bench data (with bounding box visualization)
2. Generates answers using TheWorld or Gemma baseline
3. Uses Gemma-as-judge to evaluate free-form answers
4. Saves results and calculates accuracy metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from theworld import TheWorld
from theworld.constants import DEFAULT_GEMMA_MODEL
from theworld.datasets.spatial_rgpt import SpatialRGPTDataset
from theworld.evaluation import evaluate_with_gemma, calculate_spatial_accuracy


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Evaluate on SpatialRGPT-Bench")
    p.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to SpatialRGPT-Bench JSONL file",
    )
    p.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Base folder for images (contains relative paths from dataset)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GEMMA_MODEL,
        help="Model to evaluate (HF model ID or local path)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda, cpu)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/spatial_rgpt_results.jsonl",
        help="Output path for results",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to evaluate (0 = all)",
    )
    p.add_argument(
        "--load-cosmos",
        action="store_true",
        help="Load Cosmos world model (TheWorld mode). Default: Gemma-only baseline.",
    )
    p.add_argument(
        "--num-world-steps",
        type=int,
        default=0,
        help="Number of world prediction steps (only if --load-cosmos)",
    )
    p.add_argument(
        "--draw-bboxes",
        action="store_true",
        default=True,
        help="Draw bounding boxes on images (default: True)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per answer",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    return p.parse_args()


def run_eval(
    data_path: str,
    image_folder: str,
    model_name: str,
    device: str,
    output_path: str,
    max_samples: int,
    load_cosmos: bool,
    num_world_steps: int,
    draw_bboxes: bool,
    max_new_tokens: int,
    temperature: float,
) -> None:
    """Run evaluation on SpatialRGPT-Bench.

    Args:
        data_path: Path to JSONL file
        image_folder: Base folder for images
        model_name: Model to evaluate
        device: Device to run on
        output_path: Output path for results
        max_samples: Max samples (0 = all)
        load_cosmos: Whether to load Cosmos (TheWorld mode)
        num_world_steps: Number of world steps
        draw_bboxes: Whether to draw bounding boxes
        max_new_tokens: Max tokens per answer
        temperature: Sampling temperature
    """
    # Load dataset
    print(f"Loading dataset from: {data_path}")
    print(f"Image folder: {image_folder}")

    # Check if it's a HuggingFace dataset ID (contains '/')
    data_path_obj = Path(data_path)
    if not data_path_obj.exists() and "/" in data_path:
        # Try loading as HuggingFace dataset
        print("Loading from HuggingFace...")
        from datasets import load_dataset

        try:
            hf_dataset = load_dataset(data_path, split="val")
            ds = SpatialRGPTDataset(
                hf_dataset,
                num_samples=max_samples if max_samples > 0 else None,
                image_folder=image_folder if image_folder else None,
                draw_bboxes=draw_bboxes,
            )
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("Trying as local path...")
            ds = SpatialRGPTDataset(
                data_path,
                num_samples=max_samples if max_samples > 0 else None,
                image_folder=image_folder if image_folder else None,
                draw_bboxes=draw_bboxes,
            )
    else:
        # Local file path
        ds = SpatialRGPTDataset(
            data_path,
            num_samples=max_samples if max_samples > 0 else None,
            image_folder=image_folder if image_folder else None,
            draw_bboxes=draw_bboxes,
        )
    print(f"✓ Loaded {len(ds)} samples")

    # Load model
    mode_str = "TheWorld" if load_cosmos else "Gemma-only baseline"
    print(f"Loading model ({mode_str}): {model_name}")
    model = TheWorld(
        gemma_model_name=model_name,
        device=device,
        load_cosmos=load_cosmos,
    )
    model.eval()
    print(f"✓ Model loaded in evaluation mode")

    # Prepare output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []

    with open(output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(len(ds)), desc="Evaluating"):
            ex = ds[i]
            img = ex["image"]
            question = ex["question"]
            ground_truth = ex["answer"]
            qa_type = ex["qa_type"]
            qa_category = ex["qa_category"]

            # Generate answer
            try:
                if load_cosmos:
                    # TheWorld mode with world model
                    prediction = model.generate(
                        image=img,
                        prompt=question,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        skip_world_tokens=False,
                        num_world_steps=num_world_steps,
                    )
                else:
                    # Baseline mode (Gemma-only)
                    prediction = model.generate(
                        image=img,
                        prompt=question,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        skip_world_tokens=True,
                    )
            except Exception as e:
                prediction = f"<ERROR: {e}>"

            # Evaluate with Gemma-as-judge
            try:
                eval_result = evaluate_with_gemma(
                    model,
                    question=question,
                    prediction=prediction,
                    ground_truth=ground_truth,
                )
            except Exception as e:
                eval_result = {
                    "score": 0.0,
                    "correct": False,
                    "judge_response": f"<ERROR: {e}>",
                    "judge_prompt": "",
                }

            # Save result
            result = {
                "id": ex["id"],
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "qa_type": qa_type,
                "qa_category": qa_category,
                "score": eval_result["score"],
                "correct": eval_result["correct"],
                "judge_response": eval_result["judge_response"],
            }
            results.append(result)
            fout.write(json.dumps(result) + "\n")

    print(f"\n✓ Saved results to {output_path}")

    # Calculate and print metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

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
        for qa_category, acc in metrics["by_category"].items():
            print(f"  {qa_category}: {acc:.4f} ({acc*100:.2f}%)")

    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()
    run_eval(
        data_path=args.data_path,
        image_folder=args.image_folder,
        model_name=args.model,
        device=args.device,
        output_path=args.output,
        max_samples=args.max_samples,
        load_cosmos=args.load_cosmos,
        num_world_steps=args.num_world_steps,
        draw_bboxes=args.draw_bboxes,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
