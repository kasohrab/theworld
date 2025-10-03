# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow Rules

**IMPORTANT**: When working in this repository:
1. **Always use Makefile commands** when available instead of running `uv run` directly
2. **Always run `make format`** after completing code changes before marking tasks as done
3. Run `make check` to verify formatting before committing

## Project Overview

**TheWorld** is a novel fused vision-language-world model that combines Google's Gemma 3 vision-language model with NVIDIA's Cosmos world model to enable reasoning about both static visual understanding and temporal dynamics.

### Core Architecture

The model fuses three components:
1. **Gemma 3 Vision-Language Model** (4B params) - Provides static visual understanding via SigLIP encoder + language reasoning
2. **Cosmos World Model** (2B params) - Provides temporal dynamics and future state prediction via VAE encoder
3. **Projection Layers** (trainable) - Bridges Cosmos latent space (16-dim) to Gemma embedding space (2304-dim)

### Token Flow Architecture

```
Input Image (PIL/tensor)
    ↓
[Gemma Vision Processing]
    → Gemma processor applies chat template with <start_of_image> token
    → SigLIP encoder produces ~264 vision tokens
    ↓
[Cosmos World Processing]
    → If num_world_steps=0: VAE encode current frame only
    → If num_world_steps>0: Autoregressive rollout predicts future frames
    → Extract latent.mean (16-dim, deterministic)
    → Add temporal embeddings to distinguish t=0, t=1, t=2...
    → Project to Gemma dimension: 16→2304
    → Produces (T × H × W) world tokens where T=frames, H×W=28×28
    ↓
[Token Combination]
    → Concatenate: [Gemma vision tokens | Cosmos world tokens]
    → Feed combined sequence to Gemma language model
    ↓
Output: Language model logits for next token prediction
```

## Running Commands

### Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Install with dev dependencies (includes black formatter)
uv sync --dev
```

### Development
```bash
# Format code with black (120 char line length)
make format

# Check formatting without modifying
make check

# Run inference example (demonstrates 3 scenarios)
make run

# Run training setup (demonstrates e2e forward/backward)
make train

# Clean cache and temporary files
make clean
```

### Model Usage

**Inference:**
```python
from model import TheWorld

# Single-step (current frame only, fastest)
model = TheWorld("google/gemma-3-4b-it", num_world_steps=0)

# Multi-step (predict 4 future frames)
model = TheWorld("google/gemma-3-4b-it", num_world_steps=4)

# Override at runtime
outputs = model.forward(image, text, num_world_steps=8)
```

**Training Configuration:**
```python
# Train only projection layers (default, 0.07% params)
model = TheWorld("google/gemma-3-4b-it")

# Train VAE encoder + projections
model = TheWorld("google/gemma-3-4b-it", freeze_cosmos_vae=False)

# Train Gemma vision + projections
model = TheWorld("google/gemma-3-4b-it", freeze_gemma_vision=False)

# Train Gemma language model + projections
model = TheWorld("google/gemma-3-4b-it", freeze_gemma_language=False)
```

## Key Implementation Details

### Input Format Compatibility

The model accepts three input formats (HuggingFace datasets compatible):
- **PIL Image** (default from HF datasets): `Image.open(path)`
- **NumPy array**: `(H, W, C)` uint8 array
- **PyTorch tensor**: `(B, C, H, W)` normalized tensor

### Gemma 3 Vision Processing

**Critical**: Gemma 3 requires proper chat template format with image tokens:
```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},  # PIL image directly
            {"type": "text", "text": "What is in this image?"}
        ]
    }
]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
```

This handles `<start_of_image>` and `<end_of_image>` token insertion automatically.

### Cosmos Latent Space

**Why we use `latent_dist.mean` instead of raw encoder output:**

The Cosmos VAE encoder outputs 32 channels (split into 16 mean + 16 logvar). We use the **mean** of the latent distribution because:
1. **Deterministic**: Same input always produces same output (critical for reproducibility)
2. **Normalized**: Uses pre-computed `latents_mean` and `latents_std` for stable training
3. **Semantic**: The 16-dim space is where the world model truly lives (decoder trained on this)
4. **Efficient**: 50% fewer parameters in projection layer (16→2304 vs 32→2304)

See `docs/world_model_latent_space.md` for detailed comparison of three extraction options.

### Autoregressive World Rollout

When `num_world_steps > 0`, Cosmos predicts future frames autoregressively:
```python
# Single-step: Just encode current frame
latent_dist = cosmos_pipe.vae.encode(image).latent_dist
latents = latent_dist.mean  # (B, 16, 1, H, W)

# Multi-step: Predict future states
output = cosmos_pipe(
    image=pil_image,
    num_frames=1 + num_world_steps,  # Current + future
    num_inference_steps=10,           # Diffusion steps
    output_type="latent"              # Skip decoding to pixels
)
latents = output.frames  # (B, 16, T, H, W) where T=1+num_world_steps
```

Temporal embeddings are added to distinguish between timesteps t=0 (current), t=1, t=2, etc.

See `docs/autoregressive_world_rollout.md` for architecture details.

### Training Label Alignment

During training, labels must align with the combined embedding sequence:
```python
combined_embeds = [Gemma vision tokens | Cosmos world tokens]
```

The forward pass automatically handles this by:
1. Using Gemma's `input_ids` for vision+text tokens
2. Padding world tokens with `-100` (ignore index)
3. Loss computed only on text tokens (vision and world tokens ignored)

### Device Management

The model uses `device_map="auto"` for Gemma to enable:
- **Tensor Parallelism**: Automatically splits across multiple GPUs
- **Memory Efficiency**: CPU offloading if needed on single GPU

Because of this, tensor devices are dynamically detected:
```python
target_device = gemma_vision_embeds.device  # May be cuda:0, cuda:7, etc.
projected_world_embeds = projected_world_embeds.to(target_device)
```

**Important**: Don't hardcode `.to("cuda")` for Gemma-related tensors.

### Model Loading Optimizations

Both models use:
- `torch_dtype=torch.bfloat16` - Memory efficient, good for training
- `low_cpu_mem_usage=True` - Faster loading
- `local_files_only=True` - Skip network checks (assumes cached models)

Remove `local_files_only=True` on first run to download models.

## File Structure

```
theworld/
├── model.py           # TheWorld class - main model architecture
├── main.py            # Inference examples (3 scenarios)
├── train.py           # Training setup example
├── docs/
│   ├── world_model_latent_space.md      # Cosmos latent extraction details
│   └── autoregressive_world_rollout.md  # Temporal prediction architecture
└── pyproject.toml     # Dependencies (uv)
```

## Common Development Patterns

### Adding New Training Configurations

To add a new freeze configuration:
1. Add parameter to `__init__`: `freeze_new_component=True`
2. Store in `self.freeze_new_component`
3. Add logic to `_apply_freezing()` method
4. Update docstring

### Changing Temporal Context

To modify how many future frames are predicted:
- **At initialization**: `TheWorld(..., num_world_steps=N)`
- **At inference**: `model.forward(..., num_world_steps=N)`
- **Maximum**: Limited by `max_world_steps` (default 16, affects temporal embedding table size)

### Debugging Device Mismatches

If you see "Expected all tensors to be on the same device" errors:
1. Check that you're using `target_device = gemma_vision_embeds.device` as reference
2. Don't use `self.device` for Gemma-related tensors (they may be on different GPUs)
3. Ensure Cosmos pipeline is moved to device: `self.cosmos_pipe.to(self.device)`

### Checking Trainable Parameters

```python
trainable, total, percentage = model.get_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({percentage:.4f}%)")
```

Expected percentages:
- Projection only (default): ~0.07%
- + VAE encoder: ~25%
- + Gemma vision: ~30%
- + Gemma language: ~50%

## Architecture Notes

### Why This Fusion Works

1. **Gemma provides**: Static visual understanding (objects, scenes, text in images)
2. **Cosmos provides**: Temporal dynamics (motion, physics, future states)
3. **Combined**: Model can reason about "what will happen" not just "what is"

### Training Strategy

Default configuration freezes everything except the projection layers. This is a fusion of two pretrained models, not training from scratch. The projection layers learn to:
- Map Cosmos's 16-dim world latent space to Gemma's 2304-dim token space
- Add temporal position information via learned embeddings

For domain-specific tasks, consider unfreezing components in this order:
1. Start: Train projection only (fastest, 0.07% params)
2. Next: Unfreeze Gemma vision for domain-specific visual features
3. Last: Unfreeze language model only if task differs drastically from pre-training

### Known Limitations

- Cosmos pipeline requires PIL Image input for autoregressive rollout
- Single-step mode (num_world_steps=0) is much faster than multi-step
- Memory usage scales with num_world_steps (each frame adds ~784 tokens)
- Currently only supports single image input (no video sequences yet)

## References

- Gemma 3: https://huggingface.co/docs/transformers/model_doc/gemma3
- Cosmos: https://arxiv.org/pdf/2503.15558
- Proper multimodal training format: https://cloud.google.com/blog/topics/developers-practitioners/building-a-production-multimodal-fine-tuning-pipeline
