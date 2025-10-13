# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow Rules

**IMPORTANT**: When working in this repository:
1. **Always use Makefile commands** when available instead of running `uv run` directly
2. **Always run `make format`** after completing code changes before marking tasks as done
3. Run `make check` to verify formatting before committing
4. **Run `make typecheck`** to verify type correctness (catches method signature mismatches, etc.)
5. **Run smoke test first** after setup to verify everything works: `make smoke-test`

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
    → Gemma processor.apply_chat_template() returns:
      - input_ids with image placeholders (token ID 262144)
      - pixel_values (preprocessed for SigLIP)
    → Get text embeddings from input_ids
    → SigLIP encoder processes pixel_values → vision features (~256 tokens)
    → Replace placeholders in embeddings with vision features
    → Result: multimodal embeddings (text + vision)
    ↓
[Cosmos World Processing]
    → If num_world_steps=0: VAE encode current frame only
    → If num_world_steps>0: Autoregressive rollout predicts future frames
    → Extract latent.mean (16-dim, deterministic)
    → Add temporal embeddings to distinguish t=0, t=1, t=2...
    → Project to Gemma dimension: 16→2304
    → Produces (T × H × W) world tokens where T=frames, H×W=28×28
    ↓
[Token Combination via EmbeddingFusion]
    → Locate <start_of_world> and <end_of_world> special tokens
    → Insert world embeddings between brackets
    → Final sequence: [BOS, SOW, WORLD×784, EOW, ..., IMG×256, ...]
    → World-first ordering ensures temporal context precedes static vision
    ↓
[Language Model Processing]
    → Feed combined embeddings to Gemma language model
    → Causal language modeling loss on text tokens only
    → Vision and world tokens masked with -100 (ignored in loss)
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

# Run type checking with pyright (strict mode)
make typecheck

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
- Trains on just 2 samples from DataComp-Small
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

# Load a trained checkpoint from Hub (for inference or continued training)
model = TheWorld.from_checkpoint_hub("username/theworld-datacomp")

# Use for inference
response = model.generate(image, "What is in this image?")

# Load specific checkpoint
model = TheWorld.from_checkpoint_hub(
    "username/theworld-datacomp",
    checkpoint_name="checkpoint-1000/pytorch_model.bin"
)
```

**Initializing new model:**
```python
from theworld import TheWorld

# IMPORTANT: Always use from_pretrained(), not the constructor
# This properly handles dtype, device_map, and weight loading

# Standard initialization with world model
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Gemma-only baseline (no world model)
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=False,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

**Training Configuration:**
```python
from theworld import TheWorld
import torch

# Train only projection layers (default, 0.07% params)
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Use a different Cosmos model variant
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    cosmos_model_name="nvidia/Cosmos-Predict2-2B-Video2World",
    dtype=torch.bfloat16,
    device_map="auto"
)

# Train VAE encoder + projections
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_cosmos_vae=False,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Train Gemma vision + projections
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_gemma_vision=False,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Train Gemma language model + projections
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_gemma_language=False,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

## Key Implementation Details

### Initialization Pattern

**IMPORTANT**: TheWorld uses the standard HuggingFace initialization pattern:

```python
# ✅ Correct - use from_pretrained()
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

# ❌ Incorrect - do NOT use constructor directly
model = TheWorld(config)  # Only for internal use
```

**Why `from_pretrained()` is required:**
1. **Proper weight loading**: Parent's `from_pretrained()` handles dtype conversion automatically
2. **Buffer preservation**: Non-persistent buffers (like `inv_freq` in rotary embeddings) stay float32
3. **Device mapping**: Automatically distributes model across GPUs with `device_map="auto"`
4. **Standard pattern**: Follows HuggingFace conventions (same as LLaVA, other multimodal models)

The `__init__(config)` constructor only creates model structure and is called internally by `from_pretrained()`.

See `docs/logit_validation_investigation.md` for detailed explanation.

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
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,  # Returns dict with input_ids, attention_mask, pixel_values
    return_tensors="pt"
)
```

**Key implementation details:**
- `apply_chat_template()` handles `<start_of_image>` and `<end_of_image>` tokens automatically
- Returns `pixel_values` already preprocessed for SigLIP (no manual preprocessing needed)
- Vision processing happens inline in `forward()`: 6 lines that call `embed_tokens()`, `get_image_features()`, and `get_placeholder_mask()`
- SigLIP vision tower is loaded once on initialization and reused
- No separate `GemmaVisionEncoder` class - logic is inlined directly in TheWorld.forward()

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
combined_embeds = [BOS, SOW, WORLD×784, EOW, ..., IMG×256, ...]
```

The forward pass automatically handles this by:
1. Using Gemma's `input_ids` for text tokens (includes SOW/EOW special tokens)
2. World tokens are inserted dynamically via `EmbeddingFusion` module
3. Loss computed only on text tokens (vision and world tokens masked with -100)
4. World-first ordering: world embeddings appear before vision tokens in sequence

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
├── external/
│   └── SpatialRGPT/                   # Git submodule (SpatialRGPT reference implementation)
├── python/theworld/                    # Core package
│   ├── __init__.py                    # Package exports
│   ├── config.py                      # TrainingConfig dataclass
│   ├── constants.py                   # Special token IDs (BOS, IMAGE_SOFT_TOKEN, etc.)
│   ├── data.py                        # Dataset + collator for HF Trainer
│   ├── hub_utils.py                   # HuggingFace Hub utilities (model cards)
│   ├── modeling/                      # Model architecture modules
│   │   ├── __init__.py
│   │   ├── theworld.py                # Main TheWorld model class
│   │   ├── cosmos_encoder.py          # CosmosEncoder (VAE → projection)
│   │   ├── fusion.py                  # EmbeddingFusion (insert world tokens)
│   │   └── outputs.py                 # Output dataclasses
│   └── datasets/                      # Dataset loaders
│       ├── __init__.py
│       ├── datacomp.py                # DataComp-1B dataset loader
│       └── spatial_rgpt.py            # SpatialRGPT dataset (eval + training)
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
│   ├── spatial_rgpt_training.json    # SpatialRGPT OpenSpatialDataset (~900K samples)
│   └── eval_blink.json               # BLINK benchmark evaluation config
├── tests/                              # Test suite
│   ├── test_cosmos_encoder.py        # CosmosEncoder integration tests
│   └── validation/                   # Ad-hoc validation scripts (NOT run by pytest)
│       ├── check_model_structure.py  # Model architecture verification
│       ├── test_gradient_flow.py     # Gradient flow debugging
│       └── ...                       # Other validation/debugging scripts
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

### Creating Test and Validation Scripts

**Where to put test scripts:**
- **Formal tests**: Add to `tests/` directory (run by pytest)
  - Example: `tests/test_cosmos_encoder.py`
  - Use pytest fixtures and assertions
  - Should be part of CI/CD pipeline

- **Validation/debugging scripts**: Add to `tests/validation/` directory
  - Example: `tests/validation/test_gradient_flow.py`
  - Ad-hoc scripts for debugging, verification, experiments
  - NOT run by pytest (use `if __name__ == "__main__"` pattern)
  - Quick throwaway scripts to validate behavior

**Important**: NEVER create test/validation scripts in the repository root. Always use the appropriate tests subdirectory.

### Using Global Constants

**ALWAYS use `python/theworld/constants.py` for global constants** instead of hardcoding values throughout the codebase.

**What belongs in constants.py:**
- Special token IDs (BOS, EOS, PAD, image tokens, world tokens)
- Model names and HuggingFace IDs
- Configuration values used across multiple modules
- Any magic numbers that appear in multiple places

**Current constants available:**
```python
from theworld.constants import (
    # Token IDs
    BOS_TOKEN_ID,              # 2 - Beginning of sequence
    EOS_TOKEN_ID,              # 1 - End of sequence
    PAD_TOKEN_ID,              # 0 - Padding token
    IMAGE_SOFT_TOKEN_ID,       # 262144 - Vision encoder placeholder

    # Custom token slots and names
    CUSTOM_TOKEN_SLOT_SOW,     # 0 - <start_of_world> slot
    CUSTOM_TOKEN_SLOT_EOW,     # 1 - <end_of_world> slot
    CUSTOM_TOKEN_SOW,          # "<start_of_world>" string
    CUSTOM_TOKEN_EOW,          # "<end_of_world>" string

    # Default models
    DEFAULT_GEMMA_MODEL,       # "google/gemma-3-4b-it"
    DEFAULT_COSMOS_MODEL,      # "nvidia/Cosmos-Predict2-2B-Video2World"
)
```

**Example usage:**
```python
# Good - uses constants
from theworld.constants import BOS_TOKEN_ID, IMAGE_SOFT_TOKEN_ID

if token_id == IMAGE_SOFT_TOKEN_ID:
    # Handle image token

# Bad - hardcoded magic number
if token_id == 262144:  # Don't do this!
    # Handle image token
```

### Type Checking with Pyright

The project uses **Pyright in strict mode** to catch type errors, including method signature mismatches when overriding base class methods.

**Run type checking:**
```bash
make typecheck
```

**Configuration:**
- `pyrightconfig.json` enables strict type checking with `reportIncompatibleMethodOverride`
- Catches parameter count mismatches, parameter name mismatches, and type incompatibilities
- Especially important when inheriting from Transformers models (e.g., `Gemma3ForConditionalGeneration`)

**Example - Method Override Checking:**

If you override a base class method, pyright will verify the signature matches exactly:

```python
# Base class (Gemma3ForConditionalGeneration)
def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, inputs_embeds=None,
    cache_position=None, position_ids=None, pixel_values=None,
    attention_mask=None, token_type_ids=None, use_cache=True,
    logits_to_keep=None, labels=None, **kwargs
):
    ...

# Override in TheWorld - must match exactly!
def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, inputs_embeds=None,
    cache_position=None, position_ids=None,  # <- parameter 6 must be position_ids
    pixel_values=None, attention_mask=None, token_type_ids=None,
    use_cache=True, logits_to_keep=None, labels=None,
    images=None,  # <- TheWorld-specific parameter added AFTER base params
    **kwargs
):
    ...
```

**Important:** When overriding methods from Transformers models:
1. Match the base class signature exactly (same parameter names and order)
2. Add custom parameters AFTER all base parameters (before `**kwargs`)
3. Run `make typecheck` to verify compatibility

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

### SpatialRGPT Training Dataset

TheWorld includes built-in support for the **OpenSpatialDataset** (~900K spatial reasoning examples).

**What is it?**
- **Dataset**: [a8cheng/OpenSpatialDataset](https://huggingface.co/datasets/a8cheng/OpenSpatialDataset)
- **Content**: Spatial reasoning QA pairs with region references (e.g., "Is Region [0] behind Region [1]?")
- **Images**: OpenImagesV7 (requires separate download)
- **Size**: ~900K training examples
- **Purpose**: Teaches models about 3D spatial relationships, object positions, and visual grounding

**Quick start:**

```python
from datasets import load_dataset
from theworld.datasets import SpatialRGPTDataset

# Load dataset metadata from HuggingFace
hf_dataset = load_dataset("a8cheng/OpenSpatialDataset")

# Wrap with SpatialRGPT loader
train_dataset = SpatialRGPTDataset(
    hf_dataset["train"],
    image_folder="data/openimages/train",  # Point to your local OpenImages directory
    draw_bboxes=False,  # Training data doesn't need bbox overlay
)
```

**Full setup:**

1. **Download OpenImagesV7 images**:
   ```bash
   mkdir -p data/openimages
   # Download from: https://storage.googleapis.com/openimages/web/download_v7.html
   # Follow their instructions to download train/validation splits
   ```

2. **Update training config**:
   ```bash
   # Edit configs/spatial_rgpt_training.json
   vim configs/spatial_rgpt_training.json
   # Set "image_folder": "data/openimages/train"
   ```

3. **Run training**:
   ```bash
   python scripts/train_hf.py --config configs/spatial_rgpt_training.json
   ```

**Integration details:**
- SpatialRGPT repo is included as a git submodule at `external/SpatialRGPT`
- Dataset loader automatically handles both training and evaluation formats
- No need to draw bounding boxes for training (regions are referenced in text)
- Compatible with standard TheWorld training pipeline

**See also**:
- `docs/eval/spatial_rgpt_bench.md` - Evaluation on SpatialRGPT-Bench
- `configs/spatial_rgpt_training.json` - Training configuration

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

### Simplified Vision Processing (January 2025 Refactor)

The architecture was recently simplified to eliminate redundant code:

**Before:** Separate `GemmaVisionEncoder` class (~97 lines) that wrapped vision processing
**After:** Vision processing inlined (6 lines) directly in `TheWorld.forward()`:
```python
# Get text embeddings
inputs_embeds = self.gemma.model.language_model.embed_tokens(input_ids)

# Process vision through SigLIP
image_features = self.gemma.model.get_image_features(pixel_values)

# Replace image placeholders with vision features
special_image_mask = self.gemma.model.get_placeholder_mask(input_ids, inputs_embeds, image_features)
multimodal_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
```

**Benefits:**
- Simpler code (removed entire class)
- Reuses already-loaded SigLIP vision tower from `gemma.model.vision_tower`
- No double-loading of vision models
- `pixel_values` comes from `apply_chat_template()` - no manual preprocessing needed

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

### Known Issues

**RetinaFace Gradient Bug** (Fixed with workaround):
The `retinaface` library (a dependency of `cosmos_guardrail`) globally disables PyTorch gradients at import time (`torch.set_grad_enabled(False)` in `retinaface/inference_framework.py:4`). This would break all training, but `TheWorld.__init__` automatically re-enables gradients as a workaround. If you import Cosmos components directly, you must manually call `torch.set_grad_enabled(True)` after import. See [docs/retinaface_gradient_bug.md](docs/retinaface_gradient_bug.md) for detailed explanation.

**Memory Requirements** (80GB+ GPU):
The full model (Gemma 3 4B + Cosmos 2B + projection) requires ~47GB at initialization and can exceed 80GB during training due to activation memory. For GPUs with <80GB:
- Use gradient checkpointing (`use_gradient_checkpointing: true`) to reduce activation memory
- Consider smaller Gemma model variants (2B instead of 4B)
- Note: `load_full_cosmos_pipeline: false` doesn't work - Cosmos VAE uses custom architecture and must be loaded via full pipeline

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
