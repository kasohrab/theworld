"""Judge implementations for evaluating spatial reasoning predictions.

This module provides a flexible architecture for judging model predictions
using different judge models (Gemma, GPT-4, Claude, etc.).

Each judge handles qualitative and quantitative questions separately with
appropriate prompts and parsing logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import torch
from PIL import Image


class BaseJudge(ABC):
    """Abstract base class for prediction judges."""

    @abstractmethod
    def judge_qualitative(
        self,
        questions: Union[str, List[str]],
        predictions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Judge qualitative predictions (Yes/No, comparisons).

        Args:
            questions: Question(s) that was asked
            predictions: Model's prediction(s)
            ground_truths: Correct answer(s)

        Returns:
            Dictionary or list of dictionaries with:
                - score: Float score (1.0 if correct, 0.0 if incorrect)
                - correct: Boolean
                - judge_response: Raw response from judge
                - judge_prompt: The prompt used for judging
        """
        pass

    @abstractmethod
    def judge_quantitative(
        self,
        questions: Union[str, List[str]],
        predictions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Judge quantitative predictions (measurements, distances).

        Args:
            questions: Question(s) that was asked
            predictions: Model's prediction(s)
            ground_truths: Correct answer(s)

        Returns:
            Dictionary or list of dictionaries with:
                - score: Float score (1.0 if correct, 0.0 if incorrect)
                - correct: Boolean
                - judge_response: Raw response from judge
                - judge_prompt: The prompt used for judging
        """
        pass

    def _create_qualitative_prompt(self, question: str, ground_truth: str, prediction: str) -> str:
        """Create qualitative judge prompt (Table 12 from SpatialRGPT-Bench paper).

        This method is shared by all judges and uses the official prompt format.
        """
        return f"""You are a helpful assistant designed to output JSON.
You should help me to evaluate the response given the question and the correct answer.
To mark a response, you should output a single integer between 0 and 1.
(1) means that the response perfectly matches the answer.
(0) means that the response is completely different from the answer.

Question: {question}
Correct Answer: {ground_truth}
Response: {prediction}

Output a single integer (0 or 1):"""

    def _create_quantitative_prompt(self, question: str, ground_truth: str, prediction: str) -> str:
        """Create quantitative judge prompt (Table 13 from SpatialRGPT-Bench paper).

        This method is shared by all judges and uses the official prompt format.
        """
        return f"""You are a helpful assistant designed to output JSON.
You should help me to evaluate the response given the question and the correct answer.
You need to convert the distance of the correct answer and response to meters.
The conversion factors are as follows:
1 inch = 0.0254 meters. 1 foot = 0.3048 meters. 1 centimeter (cm) = 0.01 meters.
You should output two floats in meters, one for the answer, and one for the response.

Question: {question}
Correct Answer: {ground_truth}
Response: {prediction}

Output two floats in meters (answer, response):"""

    def judge(
        self,
        questions: Union[str, List[str]],
        predictions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
        qa_types: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Judge predictions with automatic routing to qualitative or quantitative.

        Args:
            questions: Question(s) that was asked
            predictions: Model's prediction(s)
            ground_truths: Correct answer(s)
            qa_types: Question type(s) - "qualitative" or "quantitative"

        Returns:
            Dictionary or list of dictionaries with judge results
        """
        # Detect if batch or single input
        is_batch = isinstance(questions, list)

        if not is_batch:
            # Single input
            if qa_types == "qualitative":
                return self.judge_qualitative(questions, predictions, ground_truths)
            else:
                return self.judge_quantitative(questions, predictions, ground_truths)
        else:
            # Batch input - group by qa_type for efficiency
            results = [None] * len(questions)

            # Find qualitative indices
            qual_indices = [i for i, qt in enumerate(qa_types) if qt == "qualitative"]
            if qual_indices:
                qual_questions = [questions[i] for i in qual_indices]
                qual_predictions = [predictions[i] for i in qual_indices]
                qual_ground_truths = [ground_truths[i] for i in qual_indices]
                qual_results = self.judge_qualitative(
                    qual_questions, qual_predictions, qual_ground_truths
                )
                for idx, result in zip(qual_indices, qual_results):
                    results[idx] = result

            # Find quantitative indices
            quant_indices = [i for i, qt in enumerate(qa_types) if qt == "quantitative"]
            if quant_indices:
                quant_questions = [questions[i] for i in quant_indices]
                quant_predictions = [predictions[i] for i in quant_indices]
                quant_ground_truths = [ground_truths[i] for i in quant_indices]
                quant_results = self.judge_quantitative(
                    quant_questions, quant_predictions, quant_ground_truths
                )
                for idx, result in zip(quant_indices, quant_results):
                    results[idx] = result

            return results


class GemmaJudge(BaseJudge):
    """Judge using Gemma model (official SpatialRGPT-Bench prompts)."""

    def __init__(
        self,
        model,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ):
        """Initialize Gemma judge.

        Args:
            model: TheWorld model instance (uses Gemma for text generation)
            max_new_tokens: Max tokens for judge response
            temperature: Sampling temperature (0.0 = greedy)
        """
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def judge_qualitative(
        self,
        questions: Union[str, List[str]],
        predictions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Judge qualitative predictions using official SpatialRGPT prompt."""
        from .spatial_metrics import parse_official_qualitative_response

        # Ensure lists for uniform processing
        is_batch = isinstance(questions, list)
        if not is_batch:
            questions = [questions]
            predictions = [predictions]
            ground_truths = [ground_truths]

        # Create judge prompts (Table 12 from paper)
        judge_prompts = [
            self._create_qualitative_prompt(q, gt, pred)
            for q, gt, pred in zip(questions, ground_truths, predictions)
        ]

        # Generate judgments
        judge_responses = self._generate_batch(judge_prompts)

        # Parse responses
        results = []
        for judge_response, judge_prompt in zip(judge_responses, judge_prompts):
            correct = parse_official_qualitative_response(judge_response)
            score = 1.0 if correct else 0.0
            results.append(
                {
                    "score": score,
                    "correct": correct,
                    "judge_response": judge_response,
                    "judge_prompt": judge_prompt,
                }
            )

        return results[0] if not is_batch else results

    def judge_quantitative(
        self,
        questions: Union[str, List[str]],
        predictions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Judge quantitative predictions using official SpatialRGPT prompt."""
        from .spatial_metrics import parse_official_quantitative_response

        # Ensure lists for uniform processing
        is_batch = isinstance(questions, list)
        if not is_batch:
            questions = [questions]
            predictions = [predictions]
            ground_truths = [ground_truths]

        # Create judge prompts (Table 13 from paper)
        judge_prompts = [
            self._create_quantitative_prompt(q, gt, pred)
            for q, gt, pred in zip(questions, ground_truths, predictions)
        ]

        # Generate judgments
        judge_responses = self._generate_batch(judge_prompts)

        # Parse responses
        results = []
        for judge_response, judge_prompt in zip(judge_responses, judge_prompts):
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
                gt_meters = None
                pred_meters = None

            results.append(
                {
                    "score": score,
                    "correct": correct,
                    "judge_response": judge_response,
                    "judge_prompt": judge_prompt,
                    "relative_error": relative_error,
                    "gt_meters": gt_meters,
                    "pred_meters": pred_meters,
                }
            )

        return results[0] if not is_batch else results

    # def _generate_batch(self, prompts: List[str]) -> List[str]:
    #     """Generate judge responses using text-only generation from Gemma."""
    #     try:
    #         # Use tokenizer only (avoid image processor)
    #         tokenizer = self.model.processor.tokenizer

    #         # Tokenize prompts (TEXT ONLY, no images)
    #         inputs = tokenizer(
    #             prompts,
    #             return_tensors="pt",
    #             padding=True,
    #             truncation=True,
    #         ).to(self.model.device)

    #         # Generate responses
    #         with torch.no_grad():
    #             output_ids = self.model.generate(
    #                 **inputs,
    #                 max_new_tokens=self.max_new_tokens,
    #                 do_sample=self.temperature > 0,
    #                 temperature=self.temperature if self.temperature > 0 else None,
    #                 pad_token_id=tokenizer.pad_token_id,
    #                 eos_token_id=tokenizer.eos_token_id,
    #             )

    #         # Decode and remove the input prompt
    #         input_len = inputs["input_ids"].shape[1]
    #         responses = []
    #         for i in range(output_ids.shape[0]):
    #             generated = output_ids[i, input_len:]
    #             text = tokenizer.decode(generated, skip_special_tokens=True)
    #             responses.append(text.strip())

    #         return responses

    #     except Exception as e:
    #         # Return error for every prompt in batch
    #         return [f"<ERROR: {e}>"] * len(prompts)
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Fast batched generation for Gemma judge."""
        try:
            tokenizer = self.model.processor.tokenizer

            # Batch tokenize
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            # Fast greedy generation
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,          # important for speed
                    num_beams=1,
                )

            # Decode new tokens only
            input_len = inputs["input_ids"].shape[1]
            responses = []
            for i in range(output_ids.size(0)):
                gen = output_ids[i, input_len:]
                text = tokenizer.decode(gen, skip_special_tokens=True).strip()
                responses.append(text)

            return responses

        except Exception as e:
            return [f"<ERROR: {e}>"] * len(prompts)


class GPT4Judge(BaseJudge):
    """Judge using OpenAI GPT-4 API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        max_tokens: int = 50,
        temperature: float = 0.0,
    ):
        """Initialize GPT-4 judge.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-4, gpt-4-turbo, etc.)
            max_tokens: Max tokens for judge response
            temperature: Sampling temperature
        """
        import os

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Lazy import OpenAI client
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def judge_qualitative(
        self,
        questions: Union[str, List[str]],
        predictions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Judge qualitative predictions using GPT-4."""
        from .spatial_metrics import parse_official_qualitative_response

        # Ensure lists
        is_batch = isinstance(questions, list)
        if not is_batch:
            questions = [questions]
            predictions = [predictions]
            ground_truths = [ground_truths]

        # Create prompts
        prompts = [
            self._create_qualitative_prompt(q, gt, pred)
            for q, gt, pred in zip(questions, ground_truths, predictions)
        ]

        # Call GPT-4 API
        responses = [self._call_gpt4(prompt) for prompt in prompts]

        # Parse responses
        results = []
        for response, prompt in zip(responses, prompts):
            correct = parse_official_qualitative_response(response)
            score = 1.0 if correct else 0.0
            results.append(
                {
                    "score": score,
                    "correct": correct,
                    "judge_response": response,
                    "judge_prompt": prompt,
                }
            )

        return results[0] if not is_batch else results

    def judge_quantitative(
        self,
        questions: Union[str, List[str]],
        predictions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Judge quantitative predictions using GPT-4."""
        from .spatial_metrics import parse_official_quantitative_response

        # Ensure lists
        is_batch = isinstance(questions, list)
        if not is_batch:
            questions = [questions]
            predictions = [predictions]
            ground_truths = [ground_truths]

        # Create prompts
        prompts = [
            self._create_quantitative_prompt(q, gt, pred)
            for q, gt, pred in zip(questions, ground_truths, predictions)
        ]

        # Call GPT-4 API
        responses = [self._call_gpt4(prompt) for prompt in prompts]

        # Parse responses
        results = []
        for response, prompt in zip(responses, prompts):
            gt_meters, pred_meters = parse_official_quantitative_response(response)
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
                correct = False
                score = 0.0
                relative_error = None
                gt_meters = None
                pred_meters = None

            results.append(
                {
                    "score": score,
                    "correct": correct,
                    "judge_response": response,
                    "judge_prompt": prompt,
                    "relative_error": relative_error,
                    "gt_meters": gt_meters,
                    "pred_meters": pred_meters,
                }
            )

        return results[0] if not is_batch else results

    def _call_gpt4(self, prompt: str) -> str:
        """Call GPT-4 API with a prompt."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"<ERROR: {e}>"


class GPTOSSJudge(BaseJudge):
    """Judge using OpenAI GPT-OSS model via transformers pipeline."""

    def __init__(
        self,
        model_id: str = "openai/gpt-oss-120b",
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        """Initialize GPT-OSS judge.

        Args:
            model_id: HuggingFace model ID for GPT-OSS (default: openai/gpt-oss-120b)
            max_new_tokens: Max tokens for judge response
            temperature: Sampling temperature (0.0 = greedy)
            device_map: Device map for model loading (auto = automatic placement)
            torch_dtype: Torch dtype for model (auto = automatic selection)
        """
        try:
            from transformers import pipeline

            self.model_id = model_id
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature

            # Load pipeline
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")

    def judge_qualitative(
        self,
        questions: Union[str, List[str]],
        predictions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Judge qualitative predictions using GPT-OSS."""
        from .spatial_metrics import parse_official_qualitative_response

        # Ensure lists
        is_batch = isinstance(questions, list)
        if not is_batch:
            questions = [questions]
            predictions = [predictions]
            ground_truths = [ground_truths]

        # Create prompts
        prompts = [
            self._create_qualitative_prompt(q, gt, pred)
            for q, gt, pred in zip(questions, ground_truths, predictions)
        ]

        # Generate responses
        responses = self._generate_batch(prompts)

        # Parse responses
        results = []
        for response, prompt in zip(responses, prompts):
            correct = parse_official_qualitative_response(response)
            score = 1.0 if correct else 0.0
            results.append(
                {
                    "score": score,
                    "correct": correct,
                    "judge_response": response,
                    "judge_prompt": prompt,
                }
            )

        return results[0] if not is_batch else results

    def judge_quantitative(
        self,
        questions: Union[str, List[str]],
        predictions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Judge quantitative predictions using GPT-OSS."""
        from .spatial_metrics import parse_official_quantitative_response

        # Ensure lists
        is_batch = isinstance(questions, list)
        if not is_batch:
            questions = [questions]
            predictions = [predictions]
            ground_truths = [ground_truths]

        # Create prompts
        prompts = [
            self._create_quantitative_prompt(q, gt, pred)
            for q, gt, pred in zip(questions, ground_truths, predictions)
        ]

        # Generate responses
        responses = self._generate_batch(prompts)

        # Parse responses
        results = []
        for response, prompt in zip(responses, prompts):
            gt_meters, pred_meters = parse_official_quantitative_response(response)
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
                correct = False
                score = 0.0
                relative_error = None
                gt_meters = None
                pred_meters = None

            results.append(
                {
                    "score": score,
                    "correct": correct,
                    "judge_response": response,
                    "judge_prompt": prompt,
                    "relative_error": relative_error,
                    "gt_meters": gt_meters,
                    "pred_meters": pred_meters,
                }
            )

        return results[0] if not is_batch else results

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate judge responses for a batch of prompts.

        Note: Processes prompts sequentially since pipeline doesn't support
        efficient batching for chat interface.
        """
        responses = []
        for prompt in prompts:
            try:
                # Format as chat message
                messages = [{"role": "user", "content": prompt}]

                # Generate response
                outputs = self.pipe(
                    messages,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                )

                # Extract generated text from last message
                response = outputs[0]["generated_text"][-1]["content"]
                responses.append(response.strip())

            except Exception as e:
                responses.append(f"<ERROR: {e}>")

        return responses
