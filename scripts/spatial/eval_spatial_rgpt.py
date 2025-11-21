"""
Evaluate TheWorld on SpatialRGPT-Bench with Gemma-as-judge evaluation.

Usage (example):
    python scripts/spatial/eval_spatial_rgpt.py \
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
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from theworld import TheWorld
from theworld.constants import DEFAULT_GEMMA_MODEL
from theworld.datasets.spatial_rgpt import SpatialRGPTDataset
from theworld.evaluation import evaluate_with_gemma, calculate_spatial_accuracy


def generate_output_filename(
    model_name: str,
    load_cosmos: bool,
    num_world_steps: int,
    max_samples: int,
) -> str:
    """Generate a descriptive output filename based on config.

    Examples:
        - spatial_bench_gemma-3-4b-it_baseline.jsonl
        - spatial_bench_theworld-checkpoint_world-4steps.jsonl
        - spatial_bench_gemma-3-4b-it_baseline_50samples.jsonl
    """
    # Extract model name (strip path and special chars)
    model_slug = model_name.replace("/", "-").replace("_", "-")
    if "google-gemma" in model_slug or model_slug.startswith("gemma-"):
        model_slug = model_slug.split("google-")[-1]  # Remove google/ prefix

    # Add world config
    if load_cosmos:
        if num_world_steps > 0:
            config = f"world-{num_world_steps}steps"
        else:
            config = "world-0steps"
    else:
        config = "baseline"

    # Add sample limit if specified
    sample_suffix = f"_{max_samples}samples" if max_samples > 0 else ""

    return f"outputs/spatial_bench/{model_slug}_{config}{sample_suffix}.jsonl"


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
        default=None,
        help="Output path for results (default: auto-generated based on model and config)",
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
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1 for no batching)",
    )
    p.add_argument(
        "--save-visualizations",
        action="store_true",
        default=False,
        help="Save bbox images and responses for each sample",
    )
    p.add_argument(
        "--lenient-judge",
        action="store_true",
        default=False,
        help="Use lenient judge (semantic equivalence). Default: strict judge",
    )
    p.add_argument(
        "--official-judge",
        action="store_true",
        default=False,
        help="Use official SpatialRGPT-Bench judge prompts (GPT-4 style). Overrides --lenient-judge.",
    )
    p.add_argument(
        "--skip-judging",
        action="store_true",
        default=False,
        help="Skip judging step - only generate and save predictions. Use judge_predictions.py to judge later.",
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
    batch_size: int = 1,
    save_visualizations: bool = False,
    judge_mode: str = "strict",
    skip_judging: bool = False,
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
        batch_size: Batch size for processing (default: 1)
        save_visualizations: Save bbox images and responses (default: False)
        judge_mode: Judging mode - "strict", "lenient", or "official" (default: "strict")
        skip_judging: Skip judging step - only save predictions (default: False)
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

    model = TheWorld.from_pretrained(
        model_name,
        enable_world=load_cosmos,
        dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else device,
    )
    model.eval()
    print(f"✓ Model loaded in evaluation mode")

    # Prepare output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create visualization directory if requested
    viz_dir = None
    if save_visualizations:
        viz_dir = output_file.parent / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {viz_dir}")

    results = []

    # Print batching info
    if batch_size > 1:
        print(f"Using batch size: {batch_size}")
    else:
        print("Processing samples one at a time (no batching)")

    with open(output_file, "w", encoding="utf-8") as fout:
        # Process in batches
        for batch_start in tqdm(range(0, len(ds), batch_size), desc="Evaluating"):
            batch_end = min(batch_start + batch_size, len(ds))
            batch_indices = range(batch_start, batch_end)

            # Collect batch data
            batch_examples = [ds[i] for i in batch_indices]
            batch_images = [ex["image"] for ex in batch_examples]
            # Extract question from messages (first message is always user question)
            batch_questions = [ex["messages"][0]["content"] for ex in batch_examples]
            # Extract ground truth answer (second message if exists)
            batch_ground_truths = [
                ex["messages"][1]["content"] if len(ex["messages"]) > 1 else "" for ex in batch_examples
            ]

            # Generate answers using standard HuggingFace approach
            try:
                # Prepare batch of messages
                messages_batch = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": question},
                            ],
                        }
                    ]
                    for img, question in zip(batch_images, batch_questions)
                ]

                # Process batch with processor (handles batching automatically)
                inputs = model.processor.apply_chat_template(
                    messages_batch, tokenize=True, return_dict=True, return_tensors="pt", padding=True
                ).to(device)

                # Generate (HuggingFace handles batching) - always use greedy decoding
                input_length = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

                # Decode only the newly generated tokens (skip input prompt)
                batch_predictions = []
                for output in outputs:
                    generated_tokens = output[input_length:]
                    response = model.processor.decode(generated_tokens, skip_special_tokens=True).strip()
                    batch_predictions.append(response)

            except Exception as e:
                # Error during generation
                batch_predictions = [f"<ERROR: {e}>"] * len(batch_examples)

            # Collect qa_types for judge (needed for official mode)
            batch_qa_types = [ex["qa_type"] for ex in batch_examples]

            # Evaluate with Gemma-as-judge (batched if batch_size > 1)
            if not skip_judging:
                try:
                    if batch_size == 1:
                        # Single sample
                        batch_eval_results = [
                            evaluate_with_gemma(
                                model,
                                question=batch_questions[0],
                                prediction=batch_predictions[0],
                                ground_truth=batch_ground_truths[0],
                                judge_mode=judge_mode,
                                qa_type=batch_qa_types[0],
                            )
                        ]
                    else:
                        # Batched
                        batch_eval_results = evaluate_with_gemma(
                            model,
                            question=batch_questions,
                            prediction=batch_predictions,
                            ground_truth=batch_ground_truths,
                            judge_mode=judge_mode,
                            qa_type=batch_qa_types,
                        )
                except Exception as e:
                    # Error during evaluation
                    batch_eval_results = [
                        {
                            "score": 0.0,
                            "correct": False,
                            "judge_response": f"<ERROR: {e}>",
                            "judge_prompt": "",
                        }
                    ] * len(batch_examples)
            else:
                # Skip judging - create placeholder results
                batch_eval_results = [
                    {
                        "score": None,
                        "correct": None,
                        "judge_response": None,
                    }
                ] * len(batch_examples)

            # Save results for this batch
            for idx_in_batch, (ex, question, ground_truth, prediction, eval_result) in enumerate(
                zip(batch_examples, batch_questions, batch_ground_truths, batch_predictions, batch_eval_results)
            ):
                sample_idx = batch_start + idx_in_batch

                result = {
                    "id": ex["id"],
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "qa_type": ex["qa_type"],
                    "qa_category": ex["qa_category"],
                }

                # Add judge results if not skipping judging
                if not skip_judging:
                    result["score"] = eval_result["score"]
                    result["correct"] = eval_result["correct"]
                    result["judge_response"] = eval_result["judge_response"]
                results.append(result)
                fout.write(json.dumps(result) + "\n")

                # Save visualization if requested
                if save_visualizations and viz_dir is not None:
                    sample_dir = viz_dir / f"sample_{sample_idx:04d}_{ex['id']}"
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    # Save image with bboxes
                    if ex["image"] is not None:
                        image_path = sample_dir / "image_with_bboxes.png"
                        ex["image"].save(image_path)

                    # Save response.txt
                    response_path = sample_dir / "response.txt"
                    with open(response_path, "w", encoding="utf-8") as rf:
                        rf.write(f"Question: {ex['question']}\n\n")
                        rf.write(f"Ground Truth: {ex['answer']}\n\n")
                        rf.write(f"Gemma Prediction:\n{prediction}\n\n")
                        rf.write(f"Judge Result: {'Correct' if eval_result['correct'] else 'Incorrect'}\n")
                        rf.write(f"Judge Response: {eval_result['judge_response']}\n")

                    # Save metadata.json
                    metadata_path = sample_dir / "metadata.json"
                    with open(metadata_path, "w", encoding="utf-8") as mf:
                        metadata = {
                            "id": ex["id"],
                            "question": question,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                            "qa_type": ex["qa_type"],
                            "qa_category": ex["qa_category"],
                            "score": eval_result["score"],
                            "correct": eval_result["correct"],
                            "judge_response": eval_result["judge_response"],
                        }
                        # Add bbox coordinates if available
                        if "metadata" in ex and "bbox" in ex["metadata"]:
                            metadata["bounding_boxes"] = ex["metadata"]["bbox"]
                        json.dump(metadata, mf, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved results to {output_path}")

    # Calculate and print metrics (skip if judging was skipped)
    if not skip_judging:
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

        # Display quantitative-specific metrics
        if metrics.get("quantitative_metrics"):
            qm = metrics["quantitative_metrics"]
            print(f"\nQuantitative Metrics (Multi-Threshold):")
            print(f"  Total quantitative: {qm['total']}")
            if qm.get("success_rates"):
                print(f"  Success rates:")
                for threshold, rate in qm["success_rates"].items():
                    marker = " ← official" if threshold == "±25%" else ""
                    print(f"    {threshold}: {rate:.4f} ({rate*100:.2f}%){marker}")
            if qm.get("absolute_relative_error") is not None:
                print(f"  Absolute Relative Error: {qm['absolute_relative_error']:.4f}")
            if qm.get("unparseable") > 0:
                print(f"  Unparseable responses: {qm['unparseable']} ({qm['unparseable_rate']*100:.1f}%)")

        if metrics["by_category"]:
            print(f"\nBy Category:")
            for qa_category, acc in metrics["by_category"].items():
                print(f"  {qa_category}: {acc:.4f} ({acc*100:.2f}%)")

        print("=" * 60)

    # Save summary to file (skip if judging was skipped)
    if not skip_judging:
        summary_path = output_path.replace(".jsonl", "_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("SPATIAL RGPT-BENCH EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Load Cosmos: {load_cosmos}\n")
            if load_cosmos:
                f.write(f"World Steps: {num_world_steps}\n")
            f.write(f"Results File: {output_path}\n")
            f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("OVERALL ACCURACY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total Samples: {metrics['total']}\n")
            f.write(f"Correct: {metrics['correct']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write("\n")

            if metrics["by_type"]:
                f.write("=" * 60 + "\n")
                f.write("BY QUESTION TYPE\n")
                f.write("=" * 60 + "\n")
                for qa_type, acc in metrics["by_type"].items():
                    f.write(f"{qa_type}: {acc:.4f} ({acc*100:.2f}%)\n")
                f.write("\n")

            # Write quantitative-specific metrics
            if metrics.get("quantitative_metrics"):
                qm = metrics["quantitative_metrics"]
                f.write("=" * 60 + "\n")
                f.write("QUANTITATIVE METRICS (MULTI-THRESHOLD)\n")
                f.write("=" * 60 + "\n")
                f.write(f"Total Quantitative: {qm['total']}\n")
                if qm.get("success_rates"):
                    f.write("\nSuccess Rates:\n")
                    for threshold, rate in qm["success_rates"].items():
                        marker = " ← official metric" if threshold == "±25%" else ""
                        f.write(f"  {threshold}: {rate:.4f} ({rate*100:.2f}%){marker}\n")
                if qm.get("absolute_relative_error") is not None:
                    f.write(f"\nAbsolute Relative Error: {qm['absolute_relative_error']:.4f}\n")
                if qm.get("unparseable") > 0:
                    f.write(f"\nUnparseable Responses: {qm['unparseable']} ({qm['unparseable_rate']*100:.1f}%)\n")
                f.write("\n")

            if metrics["by_category"]:
                f.write("=" * 60 + "\n")
                f.write("BY CATEGORY (DETAILED BREAKDOWN)\n")
                f.write("=" * 60 + "\n")
                # Sort by accuracy descending
                sorted_categories = sorted(metrics["by_category"].items(), key=lambda x: x[1], reverse=True)
                for qa_category, acc in sorted_categories:
                    f.write(f"{qa_category}: {acc:.4f} ({acc*100:.2f}%)\n")
                f.write("\n")

            f.write("=" * 60 + "\n")

        print(f"✓ Saved summary to {summary_path}")


def main():
    """Main entry point."""
    args = parse_args()

    # Generate output filename if not specified
    output_path = args.output
    if output_path is None:
        output_path = generate_output_filename(
            model_name=args.model,
            load_cosmos=args.load_cosmos,
            num_world_steps=args.num_world_steps,
            max_samples=args.max_samples,
        )
        print(f"Auto-generated output path: {output_path}")

    # Determine judge mode from flags
    if args.official_judge:
        judge_mode = "official"
    elif args.lenient_judge:
        judge_mode = "lenient"
    else:
        judge_mode = "strict"

    run_eval(
        data_path=args.data_path,
        image_folder=args.image_folder,
        model_name=args.model,
        device=args.device,
        output_path=output_path,
        max_samples=args.max_samples,
        load_cosmos=args.load_cosmos,
        num_world_steps=args.num_world_steps,
        draw_bboxes=args.draw_bboxes,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        save_visualizations=args.save_visualizations,
        judge_mode=judge_mode,
        skip_judging=args.skip_judging,
    )


if __name__ == "__main__":
    main()
