Rough in some places codebase but was our CS8803 VLM project.
# 🌍 TheWorld

**A Fused Vision-Language-World Model for Temporal Reasoning**

TheWorld combines the power of **Google Gemma 3** (vision-language understanding) with **NVIDIA Cosmos** (world dynamics modeling) to create a model that doesn't just see and understand—it reasons about *what will happen next*.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 What Makes TheWorld Special?

| Traditional VLMs | TheWorld |
|------------------|----------|
| "What is this?" | "What is this *and what happens next*?" |
| Static visual understanding | Static + temporal dynamics |
| Sees current frame | Predicts future states |

**Example**: Given an image of a ball mid-air, TheWorld can reason about:
- **Static**: "A red ball in the air"
- **Temporal**: "The ball is falling and will hit the ground"

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/theworld.git
cd theworld

# 2. One-command setup (installs dependencies + creates directories)
bash scripts/setup.sh

# 3. Set your HuggingFace token (for model downloads)
export HF_TOKEN=hf_your_token_here
```

### Basic Usage

```python
from theworld import TheWorld
from PIL import Image
import torch

# Load model with world reasoning enabled
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Prepare your image and question
image = Image.open("example.jpg")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "What will happen next in this scene?"}
    ]
}]

# Generate response (standard HuggingFace interface)
inputs = model.processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=100)
response = model.processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

**That's it!** TheWorld uses the standard HuggingFace API—if you know how to use Gemma3, you know how to use TheWorld.

---

## 🎯 Running Examples

**Inference example:**
```bash
export HF_TOKEN=hf_your_token_here
PYTHONPATH=python:$PYTHONPATH uv run python examples/inference.py
```

**Test baseline equivalence** (TheWorld with `enable_world=False` == pure Gemma3):
```bash
PYTHONPATH=python:$PYTHONPATH uv run pytest tests/test_baseline_equivalence.py -v
```

---

## 📦 Installation Options

<details>
<summary><b>Automated Setup (Recommended)</b></summary>

```bash
bash scripts/setup.sh          # Full setup with dev tools
bash scripts/setup.sh --skip-dev  # Skip dev dependencies (faster)
```

The setup script:
- ✅ Installs core dependencies with `uv`
- ✅ Installs Cosmos safety checker
- ✅ Creates necessary directories
- ✅ Verifies installation
</details>

<details>
<summary><b>Manual Setup</b></summary>

```bash
# Install with uv (recommended package manager)
uv sync --dev

# Install Cosmos safety checker
uv pip install cosmos_guardrail

# Verify installation
make smoke-test
```
</details>

---

## ✨ Key Features

### 🏗️ Architecture
- **Fused Design**: Gemma 3 (4B) + Cosmos (2B) connected via learnable projection layers
- **Standard HF Interface**: Drop-in replacement for Gemma3 - same `from_pretrained()` and `generate()` API
- **Validated**: Logits are **numerically identical** to pure Gemma3 when `enable_world=False`

### ⚡ Efficient Training
- **Parameter Efficient**: Train only 0.07% of parameters by default (2.9M out of 4.3B)
- **Flexible Unfreezing**: Choose which components to train (projection, vision, language, world)
- **Multi-Stage Training**: Start fast with projections, progressively unfreeze as needed

### 🎛️ Configuration
- **Enable/Disable World Model**: Compare with and without temporal reasoning
- **Component Freezing**: Fine-grained control over which parts train
- **Baseline Comparison**: Perfect Gemma3 baseline for ablation studies

---

## 📚 Usage Examples

### Basic: World-Enabled Reasoning

```python
from theworld import TheWorld
import torch

# Load with world model enabled
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

### Baseline: Gemma3-Only Mode

```python
# Perfect Gemma3 baseline for comparison
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=False,  # No world model
    dtype=torch.bfloat16,
    device_map="auto"
)
# This produces identical outputs to pure Gemma3
```

### Training: Custom Freeze Configuration

```python
# Train only projection layers (fastest, 0.07% params)
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_gemma_vision=True,    # Freeze SigLIP (346M)
    freeze_gemma_language=True,  # Freeze Gemma LLM (3.95B)
    freeze_cosmos_vae=True,      # Freeze Cosmos VAE
    dtype=torch.bfloat16,
    device_map="auto"
)

# Train vision + projection (domain adaptation)
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_gemma_vision=False,   # Train SigLIP
    freeze_gemma_language=True,
    freeze_cosmos_vae=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Train language + projection (task-specific generation)
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_gemma_vision=True,
    freeze_gemma_language=False,  # Train Gemma LLM
    freeze_cosmos_vae=True,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

### Advanced: Custom Cosmos Model

```python
# Use different Cosmos variant
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    cosmos_model_name="nvidia/Cosmos-Predict2-2B-Video2World",
    dtype=torch.bfloat16,
    device_map="auto"
)
```

## Training

**Local/Interactive:**
```bash
# Multi-GPU with Accelerate
accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \
    scripts/train_hf.py --config configs/spatial_rgpt_training.json
```

**SLURM (HPC Clusters):**

See detailed guides:
- [SLURM Training Guide](docs/training/slurm-ice.md)
- [SpatialRGPT Training Guide](docs/training/spatial-rgpt.md)
- [Multi-GPU Training Guide](docs/training/distributed.md)

## Architecture

```
Input Image � [Gemma Vision (SigLIP)] � Vision Tokens (256)
           �
           � [Cosmos VAE Encoder] � World Latents (16-dim)
                                  �
                                  [Projection 16�2304] � World Tokens (784)
                                  �
Combined: [BOS, SOW, WORLD�784, EOW, TEXT, IMAGE�256]
           �
       [Gemma Language Model] � Output Logits
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Complete development guide and API reference
- [Training Guide](docs/training_infrastructure_design.md) - Training configuration and best practices
- [Logit Validation](docs/logit_validation_investigation.md) - Initialization investigation and solution
- [Hub Upload Guide](docs/huggingface_hub_upload.md) - Publishing models to HuggingFace Hub
- [Multi-Stage Training](docs/multi_stage_training.md) - Progressive training workflow
- [Evaluation Guide](docs/evaluation.md) - Evaluation on BLINK benchmark

## Citation

```bibtex
@misc{theworld2025,
  title={TheWorld: Fused Vision-Language-World Model},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/theworld}
}
```

## License

[Add your license here]

## Acknowledgments

- Google Gemma 3: Vision-language foundation
- NVIDIA Cosmos: World model foundation
- HuggingFace Transformers: Model infrastructure
