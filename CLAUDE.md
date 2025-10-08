# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow Rules

**IMPORTANT**: When working in this repository:
1. **Always use Makefile commands** when available instead of running `uv run` directly
2. **Always run `make format`** after completing code changes before marking tasks as done
3. Run `make check` to verify formatting before committing
4. **Run smoke test first** after setup to verify everything works: `make smoke-test`

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

**One-Command Setup (Recommended):**
```bash
# Automated setup script - handles everything
make setup

# Or run directly:
bash scripts/setup.sh
```

This script automatically:
- Installs all dependencies (core + dev tools)
- Installs Cosmos safety checker (cosmos_guardrail)
- Creates necessary directories (checkpoints/, logs/)
- Verifies installation
- Checks for HuggingFace token

**Manual Setup:**
```bash
# Install dependencies (uses uv package manager)
uv sync

# Install with dev dependencies (includes black formatter, wandb, tensorboard)
uv sync --dev

# Install Cosmos safety checker (required for Cosmos pipeline)
uv pip install cosmos_guardrail
```

**Setup Script Options:**
```bash
bash scripts/setup.sh --help         # Show help
bash scripts/setup.sh --skip-dev     # Skip dev tools (faster)
bash scripts/setup.sh --skip-models  # Skip model caching
```

### Development
```bash
# Format code with black (120 char line length)
make format

# Check formatting without modifying
make check

# Run inference example (demonstrates 3 scenarios)
make run

# Run simple training demo (forward/backward pass demonstration)
make train-simple

# Run production training with HuggingFace Trainer
make train-hf

# Clean cache and temporary files
make clean
```

### Smoke Test

**Quick verification** that your setup works correctly (recommended after initial setup):

```bash
# Set your HuggingFace token (required for downloading models)
export HF_TOKEN=hf_your_token_here

# Run smoke test (2 samples, ~3 minutes)
make smoke-test
```

The smoke test:
- Trains on just 2 samples from DataComp-1B
- Tests the entire pipeline (model loading, training, checkpointing, Hub upload)
- Completes in ~3 minutes
- Uploads to a private Hub repo for verification
- Uses config: `configs/smoke_test.json`

**Alternative (without make):**
```bash
export HF_TOKEN=hf_your_token_here
python scripts/train_hf.py --config configs/smoke_test.json
```

### Model Usage

**Loading from HuggingFace Hub:**
```python
from theworld import TheWorld

# Load a trained model from Hub (for inference or continued training)
model = TheWorld.from_pretrained("username/theworld-datacomp")

# Use for inference
response = model.generate(image, "What is in this image?")

# Load specific checkpoint
model = TheWorld.from_pretrained(
    "username/theworld-datacomp",
    checkpoint_name="checkpoint-1000/pytorch_model.bin"
)
```

**Initializing new model:**
```python
from theworld import TheWorld

# Single-step (current frame only, fastest)
model = TheWorld("google/gemma-3-4b-it", num_world_steps=0)

# Multi-step (predict 4 future frames)
model = TheWorld("google/gemma-3-4b-it", num_world_steps=4)

# Override at runtime
outputs = model.forward(image, text, num_world_steps=8)
```

**Training Configuration:**
```python
from theworld import TheWorld

# Train only projection layers (default, 0.07% params)
model = TheWorld("google/gemma-3-4b-it")

# Use a different Cosmos model variant
model = TheWorld(
    "google/gemma-3-4b-it",
    cosmos_model_name="nvidia/Cosmos-Predict2-2B-Video2World"
)

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
├── theworld/                           # Core package
│   ├── __init__.py                    # Package exports
│   ├── modeling.py                    # TheWorld model class
│   ├── generation.py                  # Text generation utilities
│   ├── config.py                      # TrainingConfig dataclass
│   ├── data.py                        # Dataset + collator for HF Trainer
│   ├── hub_utils.py                   # HuggingFace Hub utilities (model cards)
│   └── datasets/                      # Dataset loaders
│       ├── __init__.py
│       └── datacomp.py                # DataComp-1B dataset loader
├── examples/                           # Simple examples and demos
│   ├── inference.py                   # Inference demo (3 scenarios)
│   ├── simple_training.py             # Basic training demo (forward/backward)
│   └── load_from_hub.py              # Load model from HuggingFace Hub
├── scripts/                            # Production scripts
│   └── train_hf.py                    # HuggingFace Trainer-based training
├── configs/                            # Training configurations
│   ├── default.json                   # Default training config
│   ├── smoke_test.json               # Smoke test (2 samples, ~3 min verification)
│   ├── datacomp_test.json            # DataComp quick test (100 samples)
│   ├── datacomp_production.json      # DataComp production (streaming)
│   └── eval_blink.json               # BLINK benchmark evaluation config
├── docs/                               # Documentation
│   ├── world_model_latent_space.md   # Cosmos latent extraction details
│   ├── autoregressive_world_rollout.md  # Temporal prediction architecture
│   ├── training_infrastructure_design.md  # Training design doc
│   ├── deepspeed_zero_analysis.md    # DeepSpeed ZeRO analysis
│   ├── huggingface_hub_upload.md     # HuggingFace Hub upload guide
│   ├── multi_stage_training.md       # Multi-stage/progressive training guide
│   └── loss_function_and_evaluation.md  # Loss function & evaluation metrics
├── pyproject.toml                      # Dependencies (uv)
├── Makefile                            # Development commands
└── CLAUDE.md                           # Project guidance (this file)
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

## Training with HuggingFace Trainer

### Quick Start

**Simple training (projection layers only):**
```bash
# Uses default config (configs/default.json)
make train-hf
```

**Custom configuration:**
```bash
# Create custom config
cp configs/default.json configs/my_config.json
# Edit configs/my_config.json with your settings

# Train with custom config
python scripts/train_hf.py --config configs/my_config.json
```

### Training Configuration

Edit `configs/default.json` or create a new config file:

```json
{
  "model_name": "google/gemma-3-4b-it",
  "cosmos_model_name": "nvidia/Cosmos-Predict2-2B-Video2World",
  "num_world_steps": 0,

  "freeze_gemma_vision": true,
  "freeze_gemma_language": true,
  "freeze_cosmos_vae": true,

  "learning_rate": 0.0001,
  "batch_size": 4,
  "gradient_accumulation_steps": 4,
  "num_epochs": 3,

  "use_gradient_checkpointing": false,
  "mixed_precision": "bf16",

  "output_dir": "./checkpoints",
  "save_steps": 500,
  "log_to_wandb": false,

  "push_to_hub": false,
  "hub_model_id": "your-username/theworld-model"
}
```

**Key Configuration Options:**

- `model_name`: HuggingFace model ID for Gemma 3 (e.g., `google/gemma-3-4b-it`)
- `cosmos_model_name`: HuggingFace model ID for Cosmos (e.g., `nvidia/Cosmos-Predict2-2B-Video2World`)
- `dataset_name`: Dataset to use (`"datacomp"`, `"custom"`, etc.)
- `push_to_hub`: Upload checkpoints to HuggingFace Hub (see [Hub Upload Guide](docs/huggingface_hub_upload.md))
- `hub_model_id`: Repository name on Hub (e.g., `"username/theworld-datacomp"`)
- `output_dir`: Local directory for checkpoints (default: `./checkpoints`)

### Implementing Your Dataset

You must implement the `load_datasets()` function in `scripts/train_hf.py`:

```python
from theworld import TheWorldDataset, HFDatasetWrapper
from datasets import load_dataset

def load_datasets(config):
    # Option 1: Use HuggingFace datasets
    hf_dataset = load_dataset("your_dataset")
    train_dataset = HFDatasetWrapper(
        hf_dataset["train"],
        image_key="image",
        text_key="question",
        label_key="answer"
    )

    # Option 2: Custom data
    train_data = [
        {
            "image": "path/to/image.jpg",  # or PIL Image
            "text": "What is this?",
            "label": "A cat"
        },
        # ... more examples
    ]
    train_dataset = TheWorldDataset(train_data)

    return train_dataset, eval_dataset
```

### Gradient Checkpointing

For training large portions of the model (e.g., unfreezing Gemma language model):

```json
{
  "use_gradient_checkpointing": true,
  "freeze_gemma_language": false
}
```

Benefits:
- Reduces activation memory by 4-8×
- Enables training on smaller GPUs (24GB instead of 40GB+)
- Cost: 30-40% slower training

### Checkpoint Management

**Automatic checkpointing:**
```json
{
  "output_dir": "./checkpoints",
  "save_steps": 500,           // Save every 500 steps
  "save_total_limit": 3        // Keep only last 3 checkpoints
}
```

**Resume from checkpoint:**
```bash
# From local checkpoint
python scripts/train_hf.py --resume_from checkpoints/checkpoint-1000

# From HuggingFace Hub
python scripts/train_hf.py --resume_from username/theworld-datacomp
```

**Manual save/load:**
```python
from theworld import TheWorld

model = TheWorld("google/gemma-3-4b-it")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Save checkpoint
model.save_checkpoint("my_checkpoint.pt", optimizer=optimizer, epoch=5)

# Load checkpoint
info = model.load_checkpoint("my_checkpoint.pt", optimizer=optimizer)
print(f"Resuming from epoch {info['epoch']}")
```

### Logging

**TensorBoard (default):**
```bash
# Start training (logs saved to ./checkpoints/logs)
make train-hf

# View in TensorBoard
tensorboard --logdir checkpoints/logs
```

**Weights & Biases:**
```json
{
  "log_to_wandb": true,
  "wandb_project": "theworld-training",
  "wandb_run_name": "projection-only-v1"
}
```

### HuggingFace Hub Upload

Automatically upload checkpoints to the HuggingFace Hub during training:

```json
{
  "push_to_hub": true,
  "hub_model_id": "your-username/theworld-datacomp",
  "hub_strategy": "every_save",
  "hub_private_repo": false,
  "hf_token": null
}
```

**Provide your HuggingFace token:**
```bash
# Environment variable (recommended)
export HF_TOKEN="hf_your_token_here"
python scripts/train_hf.py --config configs/datacomp_production.json

# Or via command line
python scripts/train_hf.py --hf_token hf_your_token_here
```

**Hub upload features:**
- Automatic checkpoint uploads during training
- Auto-generated model card with training details
- Public or private repositories
- Version control for checkpoints

See [HuggingFace Hub Upload Guide](docs/huggingface_hub_upload.md) for detailed instructions.

### Distributed Training

HuggingFace Trainer automatically handles distributed training:

```bash
# Single GPU
python scripts/train_hf.py

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 scripts/train_hf.py

# Multi-node
# See HuggingFace Accelerate documentation
```

### Memory Optimization Strategies

| Scenario | Config | Memory/GPU | Speed |
|----------|--------|------------|-------|
| **Projection only** | Default | 20-24GB | Fast |
| **+ Vision encoder** | `freeze_gemma_vision=false` | 35-40GB | Medium |
| **+ Vision + GradChkpt** | + `use_gradient_checkpointing=true` | 25-30GB | Slower |
| **Full model** | All `false` + checkpointing | 56-60GB | Slow |

For full model training beyond single GPU capacity, see `docs/deepspeed_zero_analysis.md` for DeepSpeed ZeRO strategies.

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
1. **Stage 1**: Train projection only (fastest, 0.07% params, ~20-24GB VRAM)
2. **Stage 2**: Unfreeze Gemma vision for domain-specific visual features (~30% params, ~35-40GB VRAM)
3. **Stage 3**: Unfreeze language model for task-specific generation (~50% params, ~56-60GB VRAM with gradient checkpointing)

**Multi-stage training** allows you to progressively unfreeze components, starting fast and cheap with projection-only, then expanding only if needed. Each stage resumes from the previous checkpoint seamlessly.

See [Multi-Stage Training Guide](docs/multi_stage_training.md) for detailed workflow, configuration examples, and best practices.

### Known Limitations

- Cosmos pipeline requires PIL Image input for autoregressive rollout
- Single-step mode (num_world_steps=0) is much faster than multi-step
- Memory usage scales with num_world_steps (each frame adds ~784 tokens)
- Currently only supports single image input (no video sequences yet)

## Loss Function and Evaluation

**Training objective:** Causal language modeling (next-token prediction)
- Cross-entropy loss on text tokens only (vision/world tokens masked with -100)
- Projection layer learns to map Cosmos → Gemma embedding space

**Evaluation strategy:** Compare against baselines to measure world model contribution:
1. Gemma3 baseline (no world model)
2. Random projection (tests if pretrained Cosmos helps)
3. World token ablation (tests if world tokens contribute)

See [Evaluation Guide](docs/evaluation.md) for quick start and [Loss Function Details](docs/loss_function_and_evaluation.md) for mathematical formulation.

## Evaluation

**Quick evaluation commands:**
```bash
# Evaluate TheWorld on BLINK
make eval-blink MODEL=username/theworld-datacomp

# Evaluate Gemma baseline
make eval-gemma

# Compare results
make compare-results
```

**Manual evaluation:**
```bash
# TheWorld on BLINK Relative_Depth
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp \
  --num_world_steps 0,4

# Interactive demo
python scripts/inference_demo.py \
  --model username/theworld-datacomp \
  --task Relative_Depth
```

See [Evaluation Guide](docs/evaluation.md) for:
- Baseline comparisons (Gemma3, random projection, ablation)
- BLINK benchmark details
- Metrics interpretation
- Troubleshooting

## References

- Gemma 3: https://huggingface.co/docs/transformers/model_doc/gemma3
- Cosmos: https://arxiv.org/pdf/2503.15558
- Proper multimodal training format: https://cloud.google.com/blog/topics/developers-practitioners/building-a-production-multimodal-fine-tuning-pipeline
