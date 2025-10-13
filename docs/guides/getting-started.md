# Getting Started with TheWorld

Welcome! This guide will help you set up and run TheWorld for the first time.

## What is TheWorld?

TheWorld is a vision-language-world model that combines:
- **Gemma 3** (4B params) - Vision-language understanding
- **Cosmos** (2B params) - World model for temporal dynamics

It can understand images, reason about spatial relationships, and predict future states.

## Quick Start (5 minutes)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/theworld.git
cd theworld
```

### 2. Install Dependencies

We use **uv** for fast Python package management:

```bash
# One-command setup (recommended)
make setup

# Or run setup script directly
bash scripts/setup.sh
```

This automatically:
- Installs dependencies
- Creates required directories
- Verifies installation

### 3. Set HuggingFace Token

Required for downloading models:

```bash
export HF_TOKEN=hf_your_token_here
```

Get your token at: https://huggingface.co/settings/tokens

### 4. Run Smoke Test

Verify everything works:

```bash
make smoke-test
```

This trains on 2 samples (~3 minutes) to ensure your setup is correct.

## What's Next?

**For inference:**
- [Inference Guide](inference.md) - Run inference with TheWorld

**For training:**
- [Training Guide](../training/README.md) - Train your own model

**To learn more:**
- [Architecture Overview](../architecture/overview.md) - Understand how it works

---

## Installation Options

### Option 1: Automated Setup (Recommended)

```bash
# Full setup with all dependencies
make setup
```

### Option 2: Manual Setup

```bash
# Install core dependencies
uv sync

# Install dev dependencies (black, wandb, etc.)
uv sync --dev

# Install Cosmos safety checker
uv pip install cosmos_guardrail

# Create directories
mkdir -p checkpoints logs
```

### Option 3: Minimal Setup

```bash
# Skip dev tools (faster)
bash scripts/setup.sh --skip-dev
```

---

## System Requirements

### Minimum (Inference Only)

- **GPU**: 24GB VRAM (RTX 3090, RTX 4090, A5000)
- **RAM**: 32GB
- **Disk**: 50GB (for model weights)
- **Python**: 3.10+

### Recommended (Training)

- **GPU**: 40GB+ VRAM (A100, H100)
- **RAM**: 64GB+
- **Disk**: 100GB+
- **Multi-GPU**: Helpful for large-scale training

### For Multi-GPU Training

- 2-4 GPUs (40GB+ each)
- See [Distributed Training Guide](../training/distributed.md)

---

## Verify Installation

### Check Python Environment

```bash
python --version  # Should be 3.10+
```

### Check CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
python -c "import torch; print(torch.cuda.device_count())"  # Number of GPUs
```

### Check Package Installation

```bash
python -c "from theworld import TheWorld; print('✓ TheWorld installed')"
```

### Run Quick Test

```python
from theworld import TheWorld
import torch

# Load model (will download on first run)
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

print("✓ TheWorld loaded successfully!")
print(f"Trainable parameters: {model.get_trainable_parameters()[0]:,}")
```

---

## Common Issues

### "CUDA out of memory"

**Solution:** Reduce batch size or use smaller model

```bash
# In your training config
{
  "batch_size": 2,  # ← Reduce this
  "gradient_accumulation_steps": 8  # ← Increase this
}
```

### "HuggingFace token required"

**Solution:** Set your HF token

```bash
export HF_TOKEN=hf_your_token_here
```

### "Model download slow"

**First download:** Models are ~10-15GB total, may take 10-30 minutes
**Subsequent runs:** Uses cached models (fast)

### "Permission denied"

**Solution:** Check directory permissions

```bash
chmod +x scripts/setup.sh
bash scripts/setup.sh
```

---

## Next Steps

1. **Try inference**: [Inference Guide](inference.md)
2. **Train a model**: [Training Guide](../training/README.md)
3. **Learn the architecture**: [Architecture Overview](../architecture/overview.md)
4. **Evaluate a model**: [Evaluation Guide](../evaluation/overview.md)

## Need Help?

- **Troubleshooting**: [Troubleshooting Guide](troubleshooting.md)
- **Documentation**: [Main README](../README.md)
- **Issues**: [GitHub Issues](https://github.com/your-username/theworld/issues)

Welcome to TheWorld! 🌍
