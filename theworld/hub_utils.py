"""
HuggingFace Hub utilities for TheWorld model.

Provides model card generation and Hub upload helpers.
"""

from typing import Optional


def generate_model_card(
    gemma_model_name: str,
    cosmos_model_name: str,
    dataset_name: str,
    num_samples: Optional[int],
    trainable_params: int,
    total_params: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    freeze_gemma_vision: bool = True,
    freeze_gemma_language: bool = True,
    freeze_cosmos_vae: bool = True,
) -> str:
    """Generate a model card for HuggingFace Hub.

    Args:
        gemma_model_name: Name of Gemma model used
        cosmos_model_name: Name of Cosmos model used
        dataset_name: Name of dataset used for training
        num_samples: Number of training samples (None = all)
        trainable_params: Number of trainable parameters
        total_params: Total number of parameters
        learning_rate: Learning rate used
        batch_size: Effective batch size
        num_epochs: Number of training epochs
        freeze_gemma_vision: Whether Gemma vision encoder was frozen
        freeze_gemma_language: Whether Gemma language model was frozen
        freeze_cosmos_vae: Whether Cosmos VAE was frozen

    Returns:
        Model card content in Markdown format
    """

    # Calculate percentage of trainable parameters
    trainable_percentage = (trainable_params / total_params) * 100

    # Determine which components were trained
    trained_components = []
    if not freeze_gemma_vision:
        trained_components.append("Gemma Vision Encoder")
    if not freeze_gemma_language:
        trained_components.append("Gemma Language Model")
    if not freeze_cosmos_vae:
        trained_components.append("Cosmos VAE Encoder")
    trained_components.append("Projection Layers")  # Always trained

    components_str = ", ".join(trained_components)

    # Format dataset info
    dataset_info = f"{dataset_name}"
    if num_samples:
        dataset_info += f" ({num_samples:,} samples)"

    model_card = f"""---
license: apache-2.0
library_name: transformers
tags:
- vision
- multimodal
- world-model
- gemma
- cosmos
- video-prediction
---

# TheWorld Model

TheWorld is a fused vision-language-world model that combines Google's Gemma 3 vision-language model with NVIDIA's Cosmos world model to enable reasoning about both static visual understanding and temporal dynamics.

## Model Description

This model fuses three components:
1. **Gemma 3 Vision-Language Model** ({gemma_model_name}) - Provides static visual understanding via SigLIP encoder + language reasoning
2. **Cosmos World Model** ({cosmos_model_name}) - Provides temporal dynamics and future state prediction via VAE encoder
3. **Projection Layers** (trainable) - Bridges Cosmos latent space (16-dim) to Gemma embedding space (2304-dim)

### Architecture

```
Input Image (PIL/tensor)
    ↓
[Gemma Vision Processing]
    → SigLIP encoder produces ~264 vision tokens
    ↓
[Cosmos World Processing]
    → VAE encode + optional autoregressive rollout
    → Project to Gemma dimension: 16→2304
    → Produces world tokens with temporal embeddings
    ↓
[Token Combination]
    → Concatenate: [Gemma vision tokens | Cosmos world tokens]
    → Feed combined sequence to Gemma language model
    ↓
Output: Language model logits for next token prediction
```

## Training Details

### Training Data

- **Dataset**: {dataset_info}
- **Objective**: Next-token prediction for text generation with vision+world context

### Training Configuration

- **Base Models**:
  - Gemma: {gemma_model_name}
  - Cosmos: {cosmos_model_name}
- **Trainable Components**: {components_str}
- **Trainable Parameters**: {trainable_params:,} / {total_params:,} ({trainable_percentage:.4f}%)
- **Learning Rate**: {learning_rate}
- **Effective Batch Size**: {batch_size}
- **Epochs**: {num_epochs}

### Training Procedure

This checkpoint was trained with the following frozen/unfrozen configuration:
- Gemma Vision Encoder: {"Frozen ❄️" if freeze_gemma_vision else "Trainable 🔥"}
- Gemma Language Model: {"Frozen ❄️" if freeze_gemma_language else "Trainable 🔥"}
- Cosmos VAE Encoder: {"Frozen ❄️" if freeze_cosmos_vae else "Trainable 🔥"}
- Projection Layers: Trainable 🔥 (always)

## Usage

```python
from theworld import TheWorld

# Load model
model = TheWorld.from_pretrained("YOUR_USERNAME/YOUR_MODEL_NAME")

# Single-step (current frame only, fastest)
outputs = model.forward(image, text, num_world_steps=0)

# Multi-step (predict 4 future frames)
outputs = model.forward(image, text, num_world_steps=4)

# Generate text
generated_text = model.generate(image, question="What is in this image?")
print(generated_text)
```

### Input Format

The model accepts three input formats:
- **PIL Image**: `Image.open(path)`
- **NumPy array**: `(H, W, C)` uint8 array
- **PyTorch tensor**: `(B, C, H, W)` normalized tensor

## Intended Use

This model is designed for:
- Visual question answering with temporal understanding
- Future frame prediction and reasoning
- Multimodal understanding combining static and dynamic cues

## Limitations

- Cosmos pipeline requires PIL Image input for autoregressive rollout
- Single-step mode (num_world_steps=0) is much faster than multi-step
- Memory usage scales with num_world_steps (each frame adds ~784 tokens)
- Currently only supports single image input (no video sequences yet)

## Citation

If you use this model, please cite the original Gemma and Cosmos papers:

```bibtex
@article{{gemma3,
  title={{Gemma 3: A Family of Highly Capable Multimodal Models}},
  author={{Gemma Team}},
  year={{2024}},
  publisher={{Google}}
}}

@article{{cosmos,
  title={{Cosmos: A Foundation Model for Physical AI}},
  author={{NVIDIA}},
  year={{2024}},
  journal={{arXiv preprint arXiv:2503.15558}}
}}
```

## License

This model is released under the Apache 2.0 license. Please ensure compliance with the licenses of the base models (Gemma 3 and Cosmos).
"""

    return model_card


def create_model_card_from_config(config, model) -> str:
    """Create model card from TrainingConfig and TheWorld model.

    Args:
        config: TrainingConfig instance
        model: TheWorld model instance

    Returns:
        Model card content in Markdown format
    """
    trainable, total, _ = model.get_trainable_parameters()

    effective_batch_size = config.batch_size * config.gradient_accumulation_steps

    return generate_model_card(
        gemma_model_name=config.model_name,
        cosmos_model_name=config.cosmos_model_name,
        dataset_name=config.dataset_name,
        num_samples=config.num_samples,
        trainable_params=trainable,
        total_params=total,
        learning_rate=config.learning_rate,
        batch_size=effective_batch_size,
        num_epochs=config.num_epochs,
        freeze_gemma_vision=config.freeze_gemma_vision,
        freeze_gemma_language=config.freeze_gemma_language,
        freeze_cosmos_vae=config.freeze_cosmos_vae,
    )
