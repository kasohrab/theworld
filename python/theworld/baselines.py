"""
Baseline models for comparison with TheWorld.

This module provides baseline implementations to evaluate whether
the world model fusion provides benefits:

1. Gemma3Baseline - Standard Gemma 3 vision-language model (no world model)
2. Additional baselines can be added here
"""

from typing import Optional, Union
from PIL import Image
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq


class Gemma3Baseline:
    """
    Gemma 3 vision-language baseline (no world model).

    This provides a fair comparison to TheWorld by using the same
    base vision-language model but without world model components.

    Usage:
        # Load pretrained Gemma 3
        baseline = Gemma3Baseline.from_pretrained("google/gemma-3-4b-it")

        # Generate response
        response = baseline.generate(
            image=pil_image,
            prompt="What is in this image?",
            max_new_tokens=50
        )

    Note: This baseline can be used with evaluation scripts like
    evaluate_blink.py by passing --model gemma3-baseline and implementing
    custom loading logic.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize Gemma 3 baseline.

        Args:
            model_name: HuggingFace model ID
            device: Device to run on
            torch_dtype: Data type for model weights
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype

        print(f"Loading Gemma 3 baseline: {model_name}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )

        # Load model
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        print(f"✓ Loaded Gemma 3 baseline on {device}")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "cuda",
        hf_token: Optional[str] = None,
        **kwargs,
    ) -> "Gemma3Baseline":
        """
        Load Gemma 3 baseline from pretrained model.

        Args:
            model_name_or_path: HuggingFace model ID (e.g., "google/gemma-3-4b-it")
            device: Device to run on
            hf_token: HuggingFace API token (if needed)
            **kwargs: Additional arguments (for compatibility)

        Returns:
            Gemma3Baseline instance
        """
        # For compatibility with TheWorld interface, accept but ignore
        # world-model-specific parameters
        _ = kwargs.pop("num_world_steps", None)

        return cls(
            model_name=model_name_or_path,
            device=device,
        )

    def _prepare_image(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Convert image to PIL format."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            # Assume (C, H, W) or (B, C, H, W)
            if image.ndim == 4:
                image = image[0]  # Take first image if batched
            # Convert to numpy (C, H, W) -> (H, W, C)
            img_np = image.permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed (assume [0, 1] range)
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            return Image.fromarray(img_np).convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def generate(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        num_world_steps: int = 0,  # Ignored (for compatibility)
        **kwargs,
    ) -> str:
        """
        Generate response for image and prompt.

        Args:
            image: Input image (PIL, numpy array, or tensor)
            prompt: Text prompt/question
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            num_world_steps: Ignored (baseline has no world model)
            **kwargs: Additional generation parameters

        Returns:
            Generated text response
        """
        # Prepare image
        pil_image = self._prepare_image(image)

        # Format with chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        # Move to device
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        elif isinstance(inputs, dict):
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            if temperature == 0.0:
                # Greedy decoding
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
            else:
                # Sampling
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

        # Decode output
        # Get only the generated tokens (skip input)
        if isinstance(inputs, dict) and "input_ids" in inputs:
            input_len = inputs["input_ids"].shape[1]  # type: ignore[union-attr]
        else:
            input_len = inputs.shape[1]  # type: ignore[union-attr]

        generated_ids = outputs[:, input_len:]
        response = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        return response.strip()

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "type": "Gemma3Baseline",
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "has_world_model": False,
        }

    def __repr__(self) -> str:
        return f"Gemma3Baseline(model={self.model_name}, device={self.device})"
