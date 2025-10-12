"""
Evaluation script for BLINK benchmark.

Supports evaluating TheWorld models on BLINK tasks to measure
visual perception capabilities, especially spatial and depth understanding.

Example usage:
    # Evaluate on Relative_Depth
    python scripts/evaluate_blink.py \
        --task Relative_Depth \
        --model username/theworld-datacomp

    # Evaluate on both tasks with ablation
    python scripts/evaluate_blink.py \
        --tasks Relative_Depth,Spatial_Relation \
        --model username/theworld-datacomp \
        --num_world_steps 0,4 \
        --output results/blink_ablation.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, cast
from tqdm import tqdm
import re

from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld
from theworld.constants import DEFAULT_GEMMA_MODEL


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate TheWorld on BLINK benchmark")

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        default="Relative_Depth",
        choices=["Relative_Depth", "Spatial_Relation"],
        help="BLINK task to evaluate on",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of tasks (overrides --task)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token for private models",
    )

    # Dataset configuration
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)",
    )

    # Generation configuration
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Maximum tokens to generate (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0 for deterministic)",
    )
    parser.add_argument(
        "--num_world_steps",
        type=str,
        default="0",
        help="World steps to use (comma-separated for ablation, e.g., '0,4')",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON (default: results/blink_{task}.json)",
    )
    parser.add_argument(
        "--save_errors",
        action="store_true",
        help="Save incorrect predictions to results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    return parser.parse_args()


def format_question(example: Dict[str, Any]) -> str:
    """
    Format question with choices for model input.

    Args:
        example: BLINK dataset example

    Returns:
        Formatted prompt string

    Example:
        "Question: Which object is closer to the camera?
         A) The tree
         B) The building
         Answer:"
    """
    prompt = f"Question: {example['question']}\n"

    for i, choice in enumerate(example["choices"]):
        letter = chr(ord("A") + i)
        prompt += f"{letter}) {choice}\n"

    prompt += "Answer:"
    return prompt


def parse_choice(generated_text: str, choices: List[str]) -> str:
    """
    Parse model output to extract choice letter.

    Handles various formats:
    - "A"
    - "The answer is A"
    - "A) The tree"
    - "A."
    - Full text: "The tree" -> match to choices

    Args:
        generated_text: Model's generated response
        choices: List of choice texts

    Returns:
        Choice letter (A, B, C, or D)
    """
    text = generated_text.strip()

    # Check for single letter (case insensitive)
    if text.upper() in ["A", "B", "C", "D"]:
        return text.upper()

    # Check for letter at start: "A)", "A.", "A:"
    match = re.match(r"^([A-Da-d])[):\.\s]", text)
    if match:
        return match.group(1).upper()

    # Check for "answer is X" pattern
    match = re.search(r"answer is ([A-Da-d])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Try to match full choice text
    text_lower = text.lower()
    for i, choice in enumerate(choices):
        if choice.lower() in text_lower:
            return chr(ord("A") + i)

    # Extract first letter if it's A-D
    for char in text.upper():
        if char in ["A", "B", "C", "D"]:
            return char

    # Fallback: return "A"
    return "A"


def compute_metrics(
    predictions: List[str],
    references: List[str],
    choices: List[str],
) -> Dict[str, Any]:
    """
    Compute evaluation metrics.

    Args:
        predictions: List of predicted choices
        references: List of ground truth choices
        choices: List of all possible choices (e.g., ["A", "B"])

    Returns:
        Dictionary with accuracy, F1 scores, confusion matrix
    """
    # Accuracy
    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(predictions)
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    # F1 scores
    f1_macro = f1_score(references, predictions, labels=choices, average="macro", zero_division="warn") * 100

    f1_weighted = f1_score(references, predictions, labels=choices, average="weighted", zero_division="warn") * 100

    # Confusion matrix
    cm = confusion_matrix(references, predictions, labels=choices)

    # Per-choice metrics
    report = classification_report(
        references, predictions, labels=choices, target_names=choices, output_dict=True, zero_division="warn"
    )

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "per_choice": report,
    }


def evaluate_task(
    task: str,
    model: TheWorld,
    split: str = "test",
    num_samples: Optional[int] = None,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
    num_world_steps: int = 0,
    save_errors: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on a single BLINK task.

    Args:
        task: Task name (e.g., "Relative_Depth")
        model: TheWorld model instance
        split: Dataset split ("test" or "val")
        num_samples: Number of samples to evaluate (None = all)
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature
        num_world_steps: Number of world model steps
        save_errors: Whether to save incorrect predictions
        verbose: Print detailed progress

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {task} (split={split})")
    print(f"{'='*60}")

    # Load dataset
    print(f"Loading BLINK/{task} dataset...")
    dataset_raw = load_dataset("BLINK-Benchmark/BLINK", task, split=split, trust_remote_code=True)
    dataset = cast(Dataset, dataset_raw)

    if num_samples is not None:
        dataset = cast(Dataset, dataset.select(range(min(num_samples, len(dataset)))))

    print(f"Dataset size: {len(dataset)} examples")

    # Get unique choices from first example
    first_example = dataset[0]
    choices = sorted(list(set(first_example["choices"])))
    print(f"Choices: {choices}")

    # Run inference
    predictions = []
    references = []
    errors = []

    for idx, example_raw in enumerate(tqdm(dataset, desc="Evaluating")):
        # Cast example to proper type
        example = cast(Dict[str, Any], example_raw)

        # Format question
        prompt = format_question(example)

        # Generate answer
        try:
            response = model.generate(
                example["image_1"],
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_world_steps=num_world_steps,
            )
        except Exception as e:
            print(f"\n⚠ Error on example {idx}: {e}")
            response = "A"  # Default fallback

        # Parse choice
        predicted_choice = parse_choice(response, example["choices"])
        ground_truth = example["answer"]

        predictions.append(predicted_choice)
        references.append(ground_truth)

        # Track errors
        if save_errors and predicted_choice != ground_truth:
            errors.append(
                {
                    "idx": idx,
                    "question": example["question"],
                    "choices": example["choices"],
                    "predicted": predicted_choice,
                    "ground_truth": ground_truth,
                    "generated_text": response,
                }
            )

        if verbose and idx < 5:
            print(f"\nExample {idx}:")
            print(f"  Question: {example['question']}")
            print(f"  Predicted: {predicted_choice}")
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Correct: {'✓' if predicted_choice == ground_truth else '✗'}")

    # Compute metrics
    metrics = compute_metrics(predictions, references, choices)

    # Add errors if requested
    if save_errors:
        metrics["errors"] = errors

    # Print results
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Accuracy:      {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")
    print(f"F1 (macro):    {metrics['f1_macro']:.2f}%")
    print(f"F1 (weighted): {metrics['f1_weighted']:.2f}%")

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Get HF token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Determine tasks to evaluate
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = [args.task]

    # Parse world steps (support ablation with multiple values)
    world_steps_list = [int(s.strip()) for s in args.num_world_steps.split(",")]

    # Load model
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    # Detect model type and load accordingly
    if args.model.lower() in ["gemma3-baseline", "gemma3", DEFAULT_GEMMA_MODEL]:
        # Load Gemma baseline (TheWorld without Cosmos)
        if args.model.lower() in ["gemma3-baseline", "gemma3"]:
            model_name = DEFAULT_GEMMA_MODEL
        else:
            model_name = args.model

        print(f"Loading Gemma baseline (TheWorld with load_cosmos=False)")
        model = TheWorld(
            gemma_model_name=model_name,
            device=args.device,
            load_cosmos=False,  # Gemma-only baseline mode
        )
    else:
        # Load TheWorld model from Hub
        model = TheWorld.from_pretrained(
            args.model,
            device=args.device,
            hf_token=hf_token,
        )

    # Run evaluations
    all_results = {}

    for task in tasks:
        task_results = {}

        for num_world_steps in world_steps_list:
            config_name = f"world_steps_{num_world_steps}"

            print(f"\n{'#'*60}")
            print(f"Configuration: {config_name}")
            print(f"{'#'*60}")

            metrics = evaluate_task(
                task=task,
                model=model,
                split=args.split,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_world_steps=num_world_steps,
                save_errors=args.save_errors,
                verbose=args.verbose,
            )

            task_results[config_name] = metrics

        all_results[task] = task_results

    # Compute summary
    summary = {
        "mean_accuracy": sum(
            result[config]["accuracy"] for task_result in all_results.values() for config, result in task_result.items()
        )
        / sum(len(task_result) for task_result in all_results.values()),
        "tasks_evaluated": len(tasks),
        "configurations": len(world_steps_list),
    }

    # Save results
    output_path = args.output or f"results/blink_{'_'.join(tasks)}.json"
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    results_dict = {
        "model": args.model,
        "config": {
            "split": args.split,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        "results": all_results,
        "summary": summary,
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Mean Accuracy: {summary['mean_accuracy']:.2f}%")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
