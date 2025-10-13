"""
Evaluation script for VSR (Visual Spatial Reasoning) dataset.

VSR is a binary visual entailment benchmark that evaluates spatial understanding.
The model must predict whether a spatial statement correctly describes an image (0/1).

Example usage:
    # Evaluate TheWorld checkpoint from Hub
    python scripts/evaluate_vsr.py \
        --model kasohrab/theworld-vsr \
        --output results/vsr_eval.json

    # Evaluate local checkpoint
    python scripts/evaluate_vsr.py \
        --model checkpoints/theworld-vsr/final \
        --output results/vsr_eval.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from datasets import load_dataset
from PIL import Image
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld
from theworld.constants import DEFAULT_GEMMA_MODEL
from theworld.data import create_theworld_collator
import torch


# System instruction for binary visual entailment task
BINARY_SYSTEM_INSTRUCTION = (
    "You are a vision-language model that performs binary visual entailment.\n"
    "Task: Look at the image and evaluate the statement.\n"
    "Output exactly one digit with no extra text: 1 if the statement is true, 0 if the statement is false."
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate TheWorld on VSR dataset")

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace Hub ID (e.g., kasohrab/theworld-vsr or google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate baseline Gemma without world model (enable_world=False)",
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
        "--dataset",
        type=str,
        default="cambridgeltl/vsr_random",
        choices=["cambridgeltl/vsr_random", "cambridgeltl/vsr_zeroshot"],
        help="VSR dataset variant (default: vsr_random)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/home/hice1/ksohrab3/scratch/theworld/data/images",
        help="Path to VSR images folder",
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
        default=4,
        help="Maximum tokens to generate (default: 4)",
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
        default="results/vsr_eval.json",
        help="Output path for results JSON",
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


def format_question(pil_image: Image.Image, caption: str) -> List[Dict[str, Any]]:
    """
    Format VSR caption as chat messages for the model.

    Args:
        pil_image: PIL Image object
        caption: Spatial reasoning statement (e.g., "The cat is left of the dog")

    Returns:
        List of message dicts with system and user roles (chat template format)
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": BINARY_SYSTEM_INSTRUCTION}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": f"Statement: {caption}\nAnswer (only '1' or '0'):"},
            ],
        },
    ]


def parse_binary_output(generated_text: str) -> Tuple[Optional[int], str]:
    """
    Parse model output to extract binary prediction (0 or 1).

    Handles various formats:
    - "0" or "1"
    - "The answer is 0"
    - "0."
    - "False" / "True"

    Args:
        generated_text: Model's generated response

    Returns:
        Tuple of (parsed_int_or_None, raw_text)
    """
    text = generated_text.strip()

    # Check for single digit 0 or 1
    if text in ["0", "1"]:
        return int(text), text

    # Check for digit at start: "0)", "1.", "0:"
    match = re.match(r"^([01])[):\.\s]", text)
    if match:
        return int(match.group(1)), text

    # Check for "answer is X" pattern
    match = re.search(r"answer is ([01])", text, re.IGNORECASE)
    if match:
        return int(match.group(1)), text

    # Look for first occurrence of 0 or 1
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1)), text

    # Map common words
    text_lower = text.lower()
    if any(word in text_lower for word in ["false", "no", "incorrect", "wrong"]):
        return 0, text
    if any(word in text_lower for word in ["true", "yes", "correct", "right"]):
        return 1, text

    # Fallback: return None
    return None, text


def compute_metrics(
    predictions: List[int],
    references: List[int],
) -> Dict[str, Any]:
    """
    Compute evaluation metrics.

    Args:
        predictions: List of predicted labels (0 or 1)
        references: List of ground truth labels (0 or 1)

    Returns:
        Dictionary with accuracy, F1 scores, confusion matrix
    """
    # Accuracy
    correct = sum(p == r for p, r in zip(predictions, references))
    total = len(predictions)
    accuracy = (correct / total) * 100 if total > 0 else 0.0

    # F1 scores
    labels = [0, 1]
    f1_macro = f1_score(references, predictions, labels=labels, average="macro", zero_division=0) * 100
    f1_weighted = f1_score(references, predictions, labels=labels, average="weighted", zero_division=0) * 100

    # Per-class F1
    f1_per_class = f1_score(references, predictions, labels=labels, average=None, zero_division=0) * 100

    # Confusion matrix
    cm = confusion_matrix(references, predictions, labels=labels)

    # Classification report
    report = classification_report(
        references,
        predictions,
        labels=labels,
        target_names=["False (0)", "True (1)"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": {
            "false_0": f1_per_class[0],
            "true_1": f1_per_class[1],
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def evaluate_vsr(
    model: TheWorld,
    dataset,
    image_folder: str,
    num_samples: Optional[int] = None,
    max_new_tokens: int = 4,
    temperature: float = 0.0,
    num_world_steps: int = 0,
    save_errors: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on VSR dataset.

    Args:
        model: TheWorld model instance
        dataset: HuggingFace dataset
        image_folder: Path to VSR images
        num_samples: Number of samples to evaluate (None = all)
        max_new_tokens: Maximum tokens to generate
        temperature: Generation temperature
        num_world_steps: Number of world model steps
        save_errors: Whether to save incorrect predictions
        verbose: Print detailed progress

    Returns:
        Dictionary with evaluation results
    """
    image_folder_path = Path(image_folder)

    # Limit dataset if requested
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)} examples")

    # Create collator for preprocessing
    collate_fn = create_theworld_collator(model)
    device = next(model.parameters()).device

    # Run inference
    predictions = []
    references = []
    errors = []
    unparsed_count = 0

    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        # Get image path
        image_link = example["image_link"]
        image_filename = Path(image_link).name
        image_path = image_folder_path / image_filename

        if not image_path.exists():
            print(f"\n⚠ Image not found: {image_path}")
            # Skip this sample
            continue

        # Load image
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # Format question
        caption = example["caption"]
        prompt = format_question(caption)

        # Preprocess with collator
        try:
            batch = [{"image": image, "text": prompt, "label": None}]
            inputs = collate_fn(batch)

            # Move tensors to device
            inputs["input_ids"] = inputs["input_ids"].to(device)
            inputs["attention_mask"] = inputs["attention_mask"].to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(device)

            # Generate answer with proper tensors
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
            }
            if temperature > 0.0:
                generation_kwargs["temperature"] = temperature

            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
                **generation_kwargs,
            )

            # Decode response (skip prompt tokens)
            prompt_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0][prompt_length:]
            response = model.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            print(f"\n⚠ Error on example {idx}: {e}")
            response = "0"  # Default fallback

        # Parse prediction
        predicted_label, pred_raw = parse_binary_output(response)
        ground_truth = int(example["label"])

        if predicted_label is None:
            unparsed_count += 1
            predicted_label = 0  # Default to 0 if unparsed

        predictions.append(predicted_label)
        references.append(ground_truth)

        # Track errors
        if save_errors and predicted_label != ground_truth:
            errors.append(
                {
                    "idx": idx,
                    "caption": caption,
                    "predicted": predicted_label,
                    "ground_truth": ground_truth,
                    "generated_text": response,
                    "image_path": str(image_path),
                }
            )

        if verbose and idx < 5:
            print(f"\nExample {idx}:")
            print(f"  Caption: {caption}")
            print(f"  Predicted: {predicted_label}")
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Correct: {'✓' if predicted_label == ground_truth else '✗'}")

    # Compute metrics
    metrics = compute_metrics(predictions, references)

    # Add unparsed count
    metrics["unparsed_count"] = unparsed_count

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
    print(f"F1 (False/0):  {metrics['f1_per_class']['false_0']:.2f}%")
    print(f"F1 (True/1):   {metrics['f1_per_class']['true_1']:.2f}%")
    if unparsed_count > 0:
        print(f"Unparsed outputs: {unparsed_count}")

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Get HF token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Parse world steps (support ablation with multiple values)
    world_steps_list = [int(s.strip()) for s in args.num_world_steps.split(",")]

    # Load model
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    if args.baseline:
        # Baseline Gemma (no world model)
        print("Mode: Baseline Gemma (enable_world=False)")
        model = TheWorld.from_pretrained(
            args.model,
            enable_world=False,
            device=args.device,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        # TheWorld checkpoint (with world model)
        print("Mode: TheWorld (trained checkpoint)")
        # Check if model is from Hub or local path
        if Path(args.model).exists():
            # Local checkpoint
            print(f"Loading from local checkpoint: {args.model}")
            model = TheWorld.from_checkpoint(args.model, device=args.device)
        else:
            # HuggingFace Hub
            print(f"Loading from HuggingFace Hub: {args.model}")
            model = TheWorld.from_checkpoint_hub(args.model, device=args.device, hf_token=hf_token)

    # Load dataset
    print(f"\n{'='*60}")
    print("Loading dataset...")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")

    dataset = load_dataset(args.dataset, split=args.split, trust_remote_code=True)

    # Run evaluations for each world step configuration
    all_results = {}

    if args.baseline:
        # Baseline mode: single evaluation without world model
        config_name = "baseline_gemma"

        print(f"\n{'#'*60}")
        print(f"Configuration: {config_name}")
        print(f"{'#'*60}")

        metrics = evaluate_vsr(
            model=model,
            dataset=dataset,
            image_folder=args.image_folder,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_world_steps=0,  # No world steps for baseline
            save_errors=args.save_errors,
            verbose=args.verbose,
        )

        all_results[config_name] = metrics
    else:
        # TheWorld mode: evaluate for each world step configuration
        for num_world_steps in world_steps_list:
            config_name = f"world_steps_{num_world_steps}"

            print(f"\n{'#'*60}")
            print(f"Configuration: {config_name}")
            print(f"{'#'*60}")

            metrics = evaluate_vsr(
                model=model,
                dataset=dataset,
                image_folder=args.image_folder,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_world_steps=num_world_steps,
                save_errors=args.save_errors,
                verbose=args.verbose,
            )

            all_results[config_name] = metrics

    # Compute summary
    summary = {
        "mean_accuracy": sum(result["accuracy"] for result in all_results.values()) / len(all_results),
        "configurations": len(world_steps_list),
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "model": args.model,
        "dataset": args.dataset,
        "config": {
            "split": args.split,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "image_folder": args.image_folder,
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
