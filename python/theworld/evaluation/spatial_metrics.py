"""Evaluation metrics for SpatialRGPT-Bench using Gemma-as-judge.

This module uses Gemma itself to evaluate free-form spatial reasoning answers
by comparing predictions against ground truth. This is a self-contained, free
alternative to expensive GPT-4-based evaluation.
"""

import re
import torch
from typing import Dict, Any, Optional, Union
from PIL import Image


def create_judge_prompt(
    question: str,
    ground_truth: str,
    prediction: str,
    judge_mode: str = "official",
    qa_type: Optional[str] = None,
) -> str:
    """Create prompt for Gemma to judge answer correctness.

    Args:
        question: The original question asked
        ground_truth: The correct answer
        prediction: The model's predicted answer
        judge_mode: Judging mode - "strict", "lenient", or "official"
        qa_type: Question type ("qualitative" or "quantitative") - required for "official" mode

    Returns:
        Formatted prompt for judging

    Example:
        >>> prompt = create_judge_prompt(
        ...     "What color is the car?",
        ...     "Red",
        ...     "The car appears to be red.",
        ...     judge_mode="strict"
        ... )
    """
    if judge_mode == "official":
        # Official SpatialRGPT-Bench prompts (Tables 12 & 13 from paper)
        if qa_type == "qualitative":
            # Table 12: Qualitative evaluation (output 0 or 1)
            prompt = f"""You are a helpful assistant designed to output JSON.
You should help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 0 and 1.
(1) means that the response perfectly matches the answer.
(0) means that the response is completely different from the answer.

Question: {question}
Correct Answer: {ground_truth}
Response: {prediction}

Output a single integer (0 or 1):"""
        else:  # quantitative
            # Table 13: Quantitative evaluation (convert to meters and compare)
            prompt = f"""You are a helpful assistant designed to output JSON.
You should help me to evaluate the response given the question and the correct answer.
You need to convert the distance of the correct answer and response to meters.
The conversion factors are as follows:
1 inch = 0.0254 meters. 1 foot = 0.3048 meters. 1 centimeter (cm) = 0.01 meters.
You should output two floats in meters, one for the answer, and one for the response.

Question: {question}
Correct Answer: {ground_truth}
Response: {prediction}

Output two floats in meters (answer, response):"""
    elif judge_mode == "strict":
        # Strict judging: prediction MUST contain ground truth
        criteria = """Does the prediction contain the correct answer from the ground truth?
The prediction MUST include the key information from the ground truth to be marked correct.
Be strict - if the prediction contradicts or lacks the ground truth answer, mark it as incorrect."""

        prompt = f"""You are evaluating spatial reasoning answers. Compare the prediction to the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {prediction}

{criteria}

Respond with ONLY "Yes" or "No" (one word).
"""
    else:  # lenient
        # Lenient judging: semantic equivalence allowed
        criteria = """Does the prediction correctly answer the question based on the ground truth?
Consider semantic equivalence - the prediction doesn't need exact wording."""

        prompt = f"""You are evaluating spatial reasoning answers. Compare the prediction to the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Predicted Answer: {prediction}

{criteria}

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


def parse_official_qualitative_response(response: str) -> bool:
    """Parse 0/1 from official qualitative judge response.

    Args:
        response: Model's text response (should be 0 or 1)

    Returns:
        True if 1 (correct), False if 0 (incorrect) or unparseable

    Example:
        >>> parse_official_qualitative_response("1")
        True
        >>> parse_official_qualitative_response("0")
        False
    """
    response_clean = response.strip()

    # Try to extract first digit
    import re

    match = re.search(r"[01]", response_clean)
    if match:
        return match.group() == "1"

    # Default to False
    return False


def parse_official_quantitative_response(response: str) -> tuple[Optional[float], Optional[float]]:
    """Parse two floats from official quantitative judge response.

    Args:
        response: Model's text response (should contain two floats)

    Returns:
        Tuple of (ground_truth_meters, prediction_meters) or (None, None) if unparseable

    Example:
        >>> parse_official_quantitative_response("1.5 2.0")
        (1.5, 2.0)
        >>> parse_official_quantitative_response("Answer: 1.5 meters, Response: 2.0 meters")
        (1.5, 2.0)
    """
    import re

    # Try to extract all floats from response
    floats = re.findall(r"[-+]?\d*\.?\d+", response)

    if len(floats) >= 2:
        try:
            return (float(floats[0]), float(floats[1]))
        except ValueError:
            return (None, None)

    return (None, None)


def evaluate_with_gemma(
    model,
    question: Union[str, list],
    prediction: Union[str, list],
    ground_truth: Union[str, list],
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    judge_mode: str = "strict",
    qa_type: Union[str, list, None] = None,
) -> Union[Dict[str, Any], list]:
    """Evaluate a spatial reasoning answer using Gemma-as-judge.

    Supports both single and batch inputs implicitly.

    Args:
        model: TheWorld model instance (or any model with generate method)
        question: Original question (str or list of str)
        prediction: Model's predicted answer (str or list of str)
        ground_truth: Correct answer (str or list of str)
        max_new_tokens: Max tokens for judge response (default: 10)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        judge_mode: Judging mode - "strict", "lenient", or "official" (default: "strict")
        qa_type: Question type (str or list of str) - required for "official" mode

    Returns:
        Dictionary with evaluation results (single input) or list of dicts (batch input):
            - score: Float score (1.0 if correct, 0.0 if incorrect)
            - correct: Boolean
            - judge_response: Raw response from Gemma judge
            - judge_prompt: The prompt used for judging

    Example:
        >>> from theworld import TheWorld
        >>> model = TheWorld("google/gemma-3-4b-it", load_cosmos=False)
        >>>
        >>> # Single input
        >>> result = evaluate_with_gemma(
        ...     model,
        ...     "What color is the car?",
        ...     "The car is red.",
        ...     "Red"
        ... )
        >>> result['correct']
        True
        >>>
        >>> # Batch input
        >>> results = evaluate_with_gemma(
        ...     model,
        ...     ["Q1", "Q2"],
        ...     ["Pred1", "Pred2"],
        ...     ["GT1", "GT2"]
        ... )
        >>> len(results)
        2
    """
    # Detect if batch or single input
    is_batch = isinstance(question, list) or isinstance(prediction, list) or isinstance(ground_truth, list)

    if not is_batch:
        # Wrap single items in lists for uniform processing
        questions = [question]
        predictions = [prediction]
        ground_truths = [ground_truth]
        qa_types = [qa_type] if qa_type else [None]
    else:
        # Handle mixed single/batch inputs
        questions = question if isinstance(question, list) else [question] * max(
            len(prediction) if isinstance(prediction, list) else 1,
            len(ground_truth) if isinstance(ground_truth, list) else 1
        )
        predictions = prediction if isinstance(prediction, list) else [prediction] * len(questions)
        ground_truths = ground_truth if isinstance(ground_truth, list) else [ground_truth] * len(questions)
        qa_types = qa_type if isinstance(qa_type, list) else [qa_type] * len(questions)

    # Create judge prompts for all samples
    judge_prompts = [
        create_judge_prompt(q, gt, pred, judge_mode=judge_mode, qa_type=qtype)
        for q, gt, pred, qtype in zip(questions, ground_truths, predictions, qa_types)
    ]

    # Generate judgments (text-only, no image needed for judging)
    try:
        # Use Gemma directly for text-only generation - processor handles padding
        inputs = model.processor(
            text=judge_prompts,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = model.gemma.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=model.processor.tokenizer.pad_token_id,
                eos_token_id=model.processor.tokenizer.eos_token_id,
            )

        # Decode all responses (skip input prompts)
        input_length = inputs["input_ids"].shape[1]
        batch_size = output_ids.shape[0]

        judge_responses = []
        for i in range(batch_size):
            generated_ids = output_ids[i, input_length:]
            response = model.processor.decode(generated_ids, skip_special_tokens=True)
            judge_responses.append(response.strip())

    except Exception as e:
        # If generation fails, return errors for all samples
        error_results = [
            {
                "score": 0.0,
                "correct": False,
                "judge_response": f"<ERROR: {e}>",
                "judge_prompt": prompt,
            }
            for prompt in judge_prompts
        ]
        return error_results[0] if not is_batch else error_results

    # Parse responses based on judge mode
    results = []
    for judge_response, judge_prompt, qtype in zip(judge_responses, judge_prompts, qa_types):
        if judge_mode == "official":
            # Official mode: parse based on qa_type
            if qtype == "qualitative":
                # Parse 0/1
                correct = parse_official_qualitative_response(judge_response)
                score = 1.0 if correct else 0.0
            else:  # quantitative
                # Parse two floats and compare
                gt_meters, pred_meters = parse_official_quantitative_response(judge_response)
                if gt_meters is not None and pred_meters is not None:
                    # Calculate relative error: |pred - gt| / gt
                    if gt_meters > 0:
                        relative_error = abs(pred_meters - gt_meters) / gt_meters
                    else:
                        relative_error = float('inf') if pred_meters != 0 else 0.0

                    # Official threshold: ±25% (from SpatialRGPT-Bench paper)
                    correct = relative_error <= 0.25
                    score = 1.0 if correct else 0.0
                else:
                    # Failed to parse - mark as incorrect
                    correct = False
                    score = 0.0
                    relative_error = None
        else:
            # Strict/lenient modes: parse Yes/No
            correct = parse_yes_no_response(judge_response)
            score = 1.0 if correct else 0.0

        result = {
            "score": score,
            "correct": correct,
            "judge_response": judge_response,
            "judge_prompt": judge_prompt,
        }

        # Add relative_error for quantitative questions in official mode
        if judge_mode == "official" and qtype == "quantitative":
            result["relative_error"] = relative_error
            result["gt_meters"] = gt_meters
            result["pred_meters"] = pred_meters

        results.append(result)

    # Return single dict or list based on input
    return results[0] if not is_batch else results


def calculate_spatial_accuracy(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy metrics from spatial evaluation results.

    For quantitative questions (when relative_error is available), calculates:
    - Success rates at multiple thresholds (±10%, ±25%, ±50%)
    - Absolute relative error (average)
    - Number of unparseable responses

    For qualitative questions, calculates traditional accuracy.

    Args:
        results: List of evaluation results (each with 'score' field, and optionally 'relative_error')

    Returns:
        Dictionary with metrics:
            - total: Total number of samples
            - correct: Number of correct predictions
            - accuracy: Overall accuracy (0.0 to 1.0)
            - by_type: Accuracy broken down by question type (if available)
            - by_category: Accuracy broken down by category (if available)
            - quantitative_metrics: Dict with multi-threshold metrics (if quantitative results exist)

    Example:
        >>> results = [
        ...     {"score": 1.0, "qa_type": "qualitative"},
        ...     {"score": 0.0, "qa_type": "qualitative"},
        ...     {"score": 1.0, "qa_type": "quantitative", "relative_error": 0.15},
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
            "quantitative_metrics": {},
        }

    total = len(results)
    correct = sum(1 for r in results if r.get("score", 0.0) == 1.0)
    accuracy = correct / total

    # Separate quantitative and qualitative results
    quant_results = [r for r in results if r.get("qa_type") == "quantitative"]
    qual_results = [r for r in results if r.get("qa_type") == "qualitative"]

    # Calculate quantitative-specific metrics
    quantitative_metrics = {}
    if quant_results:
        # Extract relative errors (excluding None values for unparseable responses)
        relative_errors = [r.get("relative_error") for r in quant_results if r.get("relative_error") is not None]
        unparseable = sum(1 for r in quant_results if r.get("relative_error") is None)

        # Calculate success rates at different thresholds
        thresholds = [0.10, 0.25, 0.50]  # ±10%, ±25%, ±50%
        success_rates = {}
        for threshold in thresholds:
            within_threshold = sum(1 for err in relative_errors if err <= threshold)
            success_rates[f"±{int(threshold*100)}%"] = within_threshold / len(quant_results) if len(quant_results) > 0 else 0.0

        # Calculate absolute relative error (average of successful parses)
        abs_relative_error = sum(relative_errors) / len(relative_errors) if len(relative_errors) > 0 else None

        quantitative_metrics = {
            "total": len(quant_results),
            "success_rates": success_rates,
            "absolute_relative_error": abs_relative_error,
            "unparseable": unparseable,
            "unparseable_rate": unparseable / len(quant_results) if len(quant_results) > 0 else 0.0,
        }

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
        "quantitative_metrics": quantitative_metrics,
    }
