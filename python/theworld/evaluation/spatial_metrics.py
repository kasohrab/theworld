"""Evaluation metrics for SpatialRGPT-Bench using Gemma-as-judge.

This module uses Gemma itself to evaluate free-form spatial reasoning answers
by comparing predictions against ground truth. This is a self-contained, free
alternative to expensive GPT-4-based evaluation.
"""

import re
import torch
from typing import Dict, Any, Optional
from PIL import Image


def create_judge_prompt(question: str, ground_truth: str, prediction: str) -> str:
    """Create prompt for Gemma to judge answer correctness.

    Args:
        question: The original question asked
        ground_truth: The correct answer
        prediction: The model's predicted answer

    Returns:
        Formatted prompt for judging

    Example:
        >>> prompt = create_judge_prompt(
        ...     "What color is the car?",
        ...     "Red",
        ...     "The car appears to be red."
        ... )
    """
    prompt = f"""You are evaluating spatial reasoning answers. Compare the prediction to the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {prediction}

Does the prediction correctly answer the question based on the ground truth?
Consider semantic equivalence - the prediction doesn't need exact wording.

Respond with ONLY "Yes" or "No" (one word).
"""
    return prompt.strip()


def parse_yes_no_response(response: str) -> bool:
    """Parse Yes/No from model response.

    Args:
        response: Model's text response

    Returns:
        True if "Yes", False if "No" or unclear

    Example:
        >>> parse_yes_no_response("Yes")
        True
        >>> parse_yes_no_response("No, the answer is incorrect.")
        False
    """
    response_lower = response.strip().lower()

    # Check first word
    first_word = response_lower.split()[0] if response_lower else ""

    if first_word == "yes":
        return True
    elif first_word == "no":
        return False

    # Check if "yes" appears prominently
    if response_lower.startswith("yes"):
        return True
    if response_lower.startswith("no"):
        return False

    # Default to False (strict evaluation)
    return False


def evaluate_with_gemma(
    model,
    question: str,
    prediction: str,
    ground_truth: str,
    max_new_tokens: int = 10,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate a spatial reasoning answer using Gemma-as-judge.

    Args:
        model: TheWorld model instance (or any model with generate method)
        question: Original question
        prediction: Model's predicted answer
        ground_truth: Correct answer
        max_new_tokens: Max tokens for judge response (default: 10, just need "Yes"/"No")
        temperature: Sampling temperature (default: 0.0 for deterministic)

    Returns:
        Dictionary with evaluation results:
            - score: Float score (1.0 if correct, 0.0 if incorrect)
            - correct: Boolean
            - judge_response: Raw response from Gemma judge
            - judge_prompt: The prompt used for judging

    Example:
        >>> from theworld import TheWorld
        >>> model = TheWorld("google/gemma-3-4b-it", load_cosmos=False)
        >>> result = evaluate_with_gemma(
        ...     model,
        ...     "What color is the car?",
        ...     "The car is red.",
        ...     "Red"
        ... )
        >>> result['correct']
        True
    """
    # Create judge prompt
    judge_prompt = create_judge_prompt(question, ground_truth, prediction)

    # Generate judgment (text-only, no image needed for judging)
    try:
        # Use Gemma directly for text-only generation
        inputs = model.processor(
            text=judge_prompt,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = model.gemma.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        # Decode (skip input prompt)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]
        judge_response = model.processor.decode(generated_ids[0], skip_special_tokens=True)

    except Exception as e:
        # If generation fails, return error
        return {
            "score": 0.0,
            "correct": False,
            "judge_response": f"<ERROR: {e}>",
            "judge_prompt": judge_prompt,
        }

    # Parse Yes/No
    correct = parse_yes_no_response(judge_response)
    score = 1.0 if correct else 0.0

    return {
        "score": score,
        "correct": correct,
        "judge_response": judge_response,
        "judge_prompt": judge_prompt,
    }


def calculate_spatial_accuracy(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy metrics from spatial evaluation results.

    Args:
        results: List of evaluation results (each with 'score' field)

    Returns:
        Dictionary with metrics:
            - total: Total number of samples
            - correct: Number of correct predictions
            - accuracy: Overall accuracy (0.0 to 1.0)
            - by_type: Accuracy broken down by question type (if available)
            - by_category: Accuracy broken down by category (if available)

    Example:
        >>> results = [
        ...     {"score": 1.0, "qa_type": "qualitative"},
        ...     {"score": 0.0, "qa_type": "qualitative"},
        ...     {"score": 1.0, "qa_type": "quantitative"},
        ... ]
        >>> metrics = calculate_spatial_accuracy(results)
        >>> metrics['accuracy']
        0.6666666666666666
    """
    if len(results) == 0:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "by_type": {},
            "by_category": {},
        }

    total = len(results)
    correct = sum(1 for r in results if r.get("score", 0.0) == 1.0)
    accuracy = correct / total

    # Calculate accuracy by question type
    by_type = {}
    type_counts = {}
    for r in results:
        qa_type = r.get("qa_type")
        if qa_type:
            if qa_type not in by_type:
                by_type[qa_type] = 0.0
                type_counts[qa_type] = 0
            by_type[qa_type] += r.get("score", 0.0)
            type_counts[qa_type] += 1

    for qa_type in by_type:
        by_type[qa_type] = by_type[qa_type] / type_counts[qa_type]

    # Calculate accuracy by category
    by_category = {}
    category_counts = {}
    for r in results:
        qa_category = r.get("qa_category")
        if qa_category:
            if qa_category not in by_category:
                by_category[qa_category] = 0.0
                category_counts[qa_category] = 0
            by_category[qa_category] += r.get("score", 0.0)
            category_counts[qa_category] += 1

    for qa_category in by_category:
        by_category[qa_category] = by_category[qa_category] / category_counts[qa_category]

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_type": by_type,
        "by_category": by_category,
    }
