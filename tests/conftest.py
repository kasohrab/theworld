"""Shared test fixtures and utilities for pytest."""

import pytest
import torch


@pytest.fixture
def get_main_device():
    """Factory fixture that returns a function to get the main compute device for models.

    With device_map='auto', models use Accelerate's dispatch mechanism which
    places tensors via hooks. The actual .weight.device might show 'cpu', but
    runtime dispatch moves data to the correct device. We need to check the
    hf_device_map to find where inputs should go.

    Returns:
        Function that takes a model and returns its main compute device
    """

    def _get_device(model):
        """Get the main compute device for a model.

        Args:
            model: Model instance to get device from

        Returns:
            torch.device: The main compute device
        """
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            device_id = None
            for key in ["model.language_model.embed_tokens", "lm_head", "model.embed_tokens"]:
                if key in model.hf_device_map:
                    device_id = model.hf_device_map[key]
                    break

            if device_id is not None:
                if isinstance(device_id, int):
                    return torch.device(f"cuda:{device_id}")
                else:
                    return torch.device(device_id)

        if hasattr(model, "lm_head") and model.lm_head is not None:
            return model.lm_head.weight.device
        elif hasattr(model, "model") and hasattr(model.model, "language_model"):
            return next(model.model.language_model.parameters()).device
        else:
            return next(model.parameters()).device

    return _get_device
