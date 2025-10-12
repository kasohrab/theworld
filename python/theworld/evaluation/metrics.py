"""Metrics calculation utilities for evaluation results."""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def calculate_accuracy(results_path: str) -> Dict[str, Any]:
    """Calculate accuracy metrics from evaluation results JSONL file.

    Args:
        results_path: Path to JSONL file with evaluation results.
                     Each line should be a JSON object with at minimum:
                     - parsed_choice: The model's predicted answer
                     - ground_truth: The correct answer

    Returns:
        Dictionary with metrics:
            - total: Total number of samples evaluated
            - correct: Number of correctly answered samples
            - accuracy: Accuracy as a float (0.0 to 1.0)
            - skipped: Number of samples skipped (missing data)

    Example:
        >>> metrics = calculate_accuracy("outputs/results.jsonl")
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
        Accuracy: 85.50%
    """
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    total = 0
    correct = 0
    skipped = 0

    with open(results_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} (invalid JSON): {e}")
                skipped += 1
                continue

            # Extract parsed choice and ground truth
            parsed_choice = result.get("parsed_choice")
            ground_truth = result.get("ground_truth")

            # Skip if either field is missing or None
            if parsed_choice is None or ground_truth is None:
                skipped += 1
                continue

            total += 1

            # Compare (case-insensitive for robustness)
            parsed = str(parsed_choice).strip().upper()
            truth = str(ground_truth).strip()

            # Handle two comparison modes:
            # 1. Direct match: parsed_choice == ground_truth (e.g., both are "A")
            # 2. Choice letter match: parsed is "A" and ground_truth is "A cat" or "A) Cat"
            if parsed == truth.upper():
                correct += 1
            elif len(parsed) == 1 and parsed.isalpha():
                # parsed is a single letter (A, B, C, D)
                # Check if ground_truth starts with this letter
                if truth.upper().startswith(parsed) or truth.upper().startswith(f"{parsed})"):
                    correct += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "skipped": skipped,
    }
