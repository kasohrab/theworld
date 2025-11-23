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

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate judge responses for a batch of prompts."""
        try:
            # Use Gemma directly for text-only generation
            inputs = self.model.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                output_ids = self.model.gemma.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.model.processor.tokenizer.pad_token_id,
                    eos_token_id=self.model.processor.tokenizer.eos_token_id,
                )

            # Decode all responses (skip input prompts)
            input_length = inputs["input_ids"].shape[1]
            batch_size = output_ids.shape[0]

            responses = []
            for i in range(batch_size):
                generated_ids = output_ids[i, input_length:]
                response = self.model.processor.decode(generated_ids, skip_special_tokens=True)
                responses.append(response.strip())

            return responses

        except Exception as e:
            # If generation fails, return errors for all prompts
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
        max_new_tokens: int = 150,
        temperature: float = 0.0,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        """Initialize GPT-OSS judge.

        Args:
            model_id: HuggingFace model ID for GPT-OSS (default: openai/gpt-oss-120b)
            max_new_tokens: Max tokens for judge response
            temperature: Sampling temperature (0.0 = greedy)
            device_map: Device map for model loading (auto = automatic placement)
            torch_dtype: Torch dtype for model (auto = automatic selection)
            cache_dir: Directory to cache model weights (default: None = use default cache)
        """
        try:
            from transformers import pipeline
            import os

            self.model_id = model_id
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature

            # Download model to cache_dir if specified
            if cache_dir:
                print(f"Downloading {model_id} to {cache_dir}...")
                os.makedirs(cache_dir, exist_ok=True)

            # Load pipeline
            self.pipe = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                model_kwargs={"cache_dir": cache_dir} if cache_dir else {},
            )
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")

    def _create_qualitative_prompt(self, question: str, ground_truth: str, prediction: str) -> str:
        """Create qualitative judge prompt with few-shot examples for GPT-OSS.

        Overrides base class to add examples that guide GPT-OSS to output concise responses.
        """
        # Get base prompt from parent class
        base_prompt = super()._create_qualitative_prompt(question, ground_truth, prediction)

        # Add few-shot examples before the actual question
        examples = """
Example 1:
Question: Can you confirm if Region [0] is smaller than Region [1]?
Correct Answer: Yes, Region [0] is smaller than Region [1].
Response: Indeed, Region [0] has a smaller size compared to Region [1].
Output: 1

Example 2:
Question: Is Region [0] behind Region [1]?
Correct Answer: No, Region [0] is not behind Region [1].
Response: Region [0] appears to be in front of Region [1].
Output: 0

Your Turn:
"""
        # Insert examples before the question section of the base prompt
        # Base prompt format: "You are...Output a single integer (0 or 1):"
        # We'll add examples right before "Question:"
        prompt_parts = base_prompt.split("Question:", 1)
        if len(prompt_parts) == 2:
            return prompt_parts[0] + examples + "Question:" + prompt_parts[1]
        else:
            # Fallback: just append examples
            return base_prompt + "\n" + examples

    def _create_quantitative_prompt(self, question: str, ground_truth: str, prediction: str) -> str:
        """Create quantitative judge prompt with few-shot examples for GPT-OSS.

        Overrides base class to add examples that guide GPT-OSS to output concise responses.
        """
        # Get base prompt from parent class
        base_prompt = super()._create_quantitative_prompt(question, ground_truth, prediction)

        # Add few-shot examples before the actual question
        examples = """
Example 1:
Question: What is the approximate distance between Region [0] and Region [1]?
Correct Answer: The distance is approximately 2 feet.
Response: I estimate the distance to be around 24 inches.
Output: 0.6096, 0.6096

Example 2:
Question: How far apart are these two regions?
Correct Answer: About 1.5 meters.
Response: Roughly 150 centimeters.
Output: 1.5, 1.5

Your Turn:
"""
        # Insert examples before the question section
        prompt_parts = base_prompt.split("Question:", 1)
        if len(prompt_parts) == 2:
            return prompt_parts[0] + examples + "Question:" + prompt_parts[1]
        else:
            # Fallback: just append examples
            return base_prompt + "\n" + examples

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


class DeepSeekJudge(BaseJudge):
    """Judge using DeepSeek Chat API with JSON output mode and async batching."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_tokens: int = 150,
        temperature: float = 0.0,
    ):
        """Initialize DeepSeek judge.

        Args:
            api_key: DeepSeek API key (or set DEEPSEEK_API_KEY env var)
            model: Model to use (default: deepseek-chat)
            max_tokens: Max tokens for judge response
            temperature: Sampling temperature (0.0 = greedy)
        """
        import os

        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            from openai import AsyncOpenAI

            # Initialize async client for concurrent batching
            self.async_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com",
            )
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def _create_qualitative_prompt(self, question: str, ground_truth: str, prediction: str) -> str:
        """Create qualitative judge prompt with JSON output format.

        Overrides base class to include JSON example for DeepSeek's JSON output mode.
        """
        return f"""You are an expert judge evaluating spatial reasoning answers.

Task: Compare a model's response to the correct answer and determine if they match semantically.

Question: {question}
Correct Answer: {ground_truth}
Response: {prediction}

Instructions:
- Output 1 if the response matches the correct answer (semantically equivalent)
- Output 0 if the response does NOT match the correct answer
- Ignore minor wording differences if the meaning is the same
- Focus on the core spatial relationship being described

Output your judgment as JSON in this exact format:
{{"output": 0}}

or

{{"output": 1}}"""

    def _create_quantitative_prompt(self, question: str, ground_truth: str, prediction: str) -> str:
        """Create quantitative judge prompt with JSON output format.

        Overrides base class to include JSON example for DeepSeek's JSON output mode.
        """
        return f"""You are an expert judge evaluating quantitative spatial reasoning answers.

Task: Extract distance/size measurements in meters from both the correct answer and the model's response.

Question: {question}
Correct Answer: {ground_truth}
Response: {prediction}

Instructions:
- Extract the numeric value in METERS from the correct answer
- Extract the numeric value in METERS from the model's response
- Convert to meters if needed (e.g., "5 feet" → 1.524 meters, "2 km" → 2000 meters)
- If no measurement is present, use null

Output your judgment as JSON in this exact format:
{{
    "ground_truth_meters": 10.5,
    "predicted_meters": 12.3
}}

If a value cannot be extracted, use null:
{{
    "ground_truth_meters": null,
    "predicted_meters": null
}}"""

    def judge_qualitative(
        self,
        questions: Union[str, List[str]],
        predictions: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Judge qualitative predictions using DeepSeek API with JSON output."""
        import json
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

        # Call DeepSeek API with async batching
        responses = self._generate_batch_async(prompts)

        # Parse responses
        results = []
        for response, prompt in zip(responses, prompts):
            # Try JSON parsing first
            try:
                response_json = json.loads(response)
                output = response_json.get("output")
                if output in [0, 1]:
                    correct = bool(output)
                else:
                    # Fallback to string parsing
                    correct = parse_official_qualitative_response(response)
            except (json.JSONDecodeError, ValueError):
                # Fallback to string parsing
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
        """Judge quantitative predictions using DeepSeek API with JSON output."""
        import json
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

        # Call DeepSeek API with async batching
        responses = self._generate_batch_async(prompts)

        # Parse responses
        results = []
        for response, prompt in zip(responses, prompts):
            # Try JSON parsing first
            try:
                response_json = json.loads(response)
                gt_meters = response_json.get("ground_truth_meters")
                pred_meters = response_json.get("predicted_meters")

                # Validate parsed values
                if gt_meters is None or pred_meters is None:
                    raise ValueError("Missing meters in JSON response")

            except (json.JSONDecodeError, ValueError, KeyError):
                # Fallback to string parsing
                gt_meters, pred_meters = parse_official_quantitative_response(response)

            # Calculate metrics
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

    def _generate_batch_async(self, prompts: List[str]) -> List[str]:
        """Generate judge responses for a batch of prompts using async concurrency.

        Uses asyncio.gather() to send all prompts concurrently to DeepSeek API.
        This is much faster than sequential processing.

        Args:
            prompts: List of prompts to process

        Returns:
            List of responses in same order as prompts
        """
        import asyncio

        async def async_batch():
            """Async function to handle concurrent requests."""
            tasks = [self._call_deepseek_async(prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)

        # Run async batch in event loop
        try:
            # Check if event loop is already running (e.g., in Jupyter)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use nest_asyncio for compatibility with running loops
                try:
                    import nest_asyncio

                    nest_asyncio.apply()
                except ImportError:
                    pass
        except RuntimeError:
            pass

        return asyncio.run(async_batch())

    async def _call_deepseek_async(self, prompt: str) -> str:
        """Call DeepSeek API asynchronously with a single prompt.

        Args:
            prompt: The prompt to send to DeepSeek

        Returns:
            Response text from DeepSeek API
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},  # Enable JSON output mode
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"<ERROR: {e}>"
