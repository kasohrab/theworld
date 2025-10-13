# Multi-Stage Training Guide

This guide explains how to progressively train TheWorld model by unfreezing components in stages, from fast projection-only training to full model fine-tuning.

## Overview

**Multi-stage training** (also called progressive unfreezing or gradual unfreezing) is a training strategy where you:

1. **Start small**: Train only the lightweight projection layers (50K params, ~0.07%)
2. **Expand gradually**: Progressively unfreeze larger components (vision encoder, language model)
3. **Resume seamlessly**: Each stage builds on the previous one using checkpoints

### Why Use Multi-Stage Training?

**Benefits:**
- ✅ **Faster iteration**: Quick experiments with projection-only (minutes instead of days)
- ✅ **Lower compute cost**: Start with small GPUs, scale up only if needed
- ✅ **Better convergence**: Pretrained components stay stable while new connections learn
- ✅ **Reduced risk**: Test your pipeline before committing to expensive full training
- ✅ **Flexibility**: Stop at any stage if performance is good enough

**When to use:**
- Starting with a new dataset (validate pipeline quickly)
- Limited compute resources (train what you can afford)
- Uncertain if full fine-tuning is needed
- Want to compare different unfreezing strategies

## How It Works

TheWorld's checkpoint system is designed for multi-stage training:

### Checkpoint Behavior

**What gets saved:**
- ✅ Trainable parameter weights (only unfrozen components)
- ✅ Optimizer state for trainable parameters
- ✅ Model configuration (freeze settings, model names)
- ✅ Training metadata (epoch, step, etc.)

**What doesn't get saved:**
- ❌ Frozen component weights (they stay at pretrained values)

### Resuming with Different Freeze Settings

When you resume from a checkpoint with different freeze settings:

```python
# Stage 1 checkpoint contains:
# - projection_layer weights (trained)
# - freeze_config: {vision: true, language: true, vae: true}

# Stage 2 resume with vision unfrozen:
# 1. Load projection_layer weights from checkpoint ✓
# 2. Gemma vision encoder starts from pretrained weights (not in checkpoint) ✓
# 3. Both projection + vision become trainable ✓
# 4. Training continues from where stage 1 left off ✓
```

**Key insight**: Frozen components always use their original pretrained weights, so unfreezing them later is safe and seamless.

## Recommended Training Stages

### Stage 1: Projection Layers Only

**Configuration:**
```json
{
  "freeze_gemma_vision": true,
  "freeze_gemma_language": true,
  "freeze_cosmos_vae": true
}
```

**Characteristics:**
- **Trainable params**: 50K (0.07%)
- **Memory**: 20-24 GB VRAM
- **Speed**: Fast (~1-2 seconds/step)
- **Use case**: Quick iteration, pipeline validation

**What's being learned:**
- Mapping from Cosmos 16-dim latent space to Gemma 2304-dim embedding space
- Temporal position embeddings for multi-step rollouts

### Stage 2: Projection + Vision Encoder

**Configuration:**
```json
{
  "freeze_gemma_vision": false,    // Unfrozen!
  "freeze_gemma_language": true,
  "freeze_cosmos_vae": true,

  "resume_from_checkpoint": "username/theworld-stage1"
}
```

**Characteristics:**
- **Trainable params**: ~1.5B (~30%)
- **Memory**: 35-40 GB VRAM
- **Speed**: Slower (~3-4 seconds/step)
- **Use case**: Domain-specific visual features (medical images, satellite imagery, etc.)

**What's being learned:**
- Domain-specific visual representations (SigLIP fine-tuning)
- Better projection layer alignment with new visual features

### Stage 3: Projection + Vision + Language

**Configuration:**
```json
{
  "freeze_gemma_vision": false,
  "freeze_gemma_language": false,   // Unfrozen!
  "freeze_cosmos_vae": true,

  "use_gradient_checkpointing": true,  // Required for memory!
  "resume_from_checkpoint": "username/theworld-stage2"
}
```

**Characteristics:**
- **Trainable params**: ~3B (~50%)
- **Memory**: 56-60 GB VRAM (with gradient checkpointing)
- **Speed**: Much slower (~8-10 seconds/step)
- **Use case**: Task-specific language patterns, reasoning styles

**What's being learned:**
- Task-specific language generation
- Better integration of world model information with language

### Stage 4: Full Model (All Unfrozen)

**Configuration:**
```json
{
  "freeze_gemma_vision": false,
  "freeze_gemma_language": false,
  "freeze_cosmos_vae": false,       // Unfrozen!

  "use_gradient_checkpointing": true,
  "resume_from_checkpoint": "username/theworld-stage3"
}
```

**Characteristics:**
- **Trainable params**: ~6B (100%)
- **Memory**: 80-84 GB VRAM (single GPU) or distributed training
- **Speed**: Slowest (~15-20 seconds/step)
- **Use case**: Maximum adaptation, but rarely needed

**What's being learned:**
- Domain-specific world model dynamics
- End-to-end optimization of entire pipeline

## Step-by-Step Workflow

### Complete Example: DataComp-1B Training

This example shows how to train TheWorld on DataComp-1B dataset across multiple stages.

#### Stage 1: Projection Only

**Create config: `configs/stage1_projection.json`**
```json
{
  "_comment": "Stage 1: Projection layers only (fast iteration)",

  "model_name": "google/gemma-3-4b-it",
  "cosmos_model_name": "nvidia/Cosmos-Predict2-2B-Video2World",
  "num_world_steps": 0,
  "max_world_steps": 16,

  "freeze_gemma_vision": true,
  "freeze_gemma_language": true,
  "freeze_cosmos_vae": true,

  "learning_rate": 0.0001,
  "batch_size": 4,
  "gradient_accumulation_steps": 4,
  "num_epochs": 1,
  "warmup_steps": 500,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,

  "use_gradient_checkpointing": false,
  "mixed_precision": "bf16",

  "output_dir": "./checkpoints/stage1_projection",
  "save_steps": 1000,
  "save_total_limit": 5,

  "eval_steps": 5000,
  "do_eval": true,

  "logging_steps": 100,
  "log_to_wandb": true,
  "wandb_project": "theworld-multistage",
  "wandb_run_name": "stage1-projection",

  "dataset_name": "datacomp",
  "num_samples": null,
  "streaming": true,

  "push_to_hub": true,
  "hub_model_id": "your-username/theworld-stage1-projection",
  "hub_strategy": "every_save"
}
```

**Train:**
```bash
export HF_TOKEN="hf_your_token_here"
python scripts/train_hf.py --config configs/stage1_projection.json
```

**Monitor and decide:**
- Watch validation loss in wandb/tensorboard
- When loss plateaus (usually after 10K-50K steps), move to stage 2
- Or complete full epoch and evaluate results

#### Stage 2: Add Vision Encoder

**Create config: `configs/stage2_vision.json`**
```json
{
  "_comment": "Stage 2: Projection + Vision (domain-specific features)",

  "model_name": "google/gemma-3-4b-it",
  "cosmos_model_name": "nvidia/Cosmos-Predict2-2B-Video2World",
  "num_world_steps": 0,
  "max_world_steps": 16,

  "freeze_gemma_vision": false,      // CHANGED: Unfreeze vision
  "freeze_gemma_language": true,
  "freeze_cosmos_vae": true,

  "learning_rate": 0.00005,          // CHANGED: Lower LR for fine-tuning
  "batch_size": 4,
  "gradient_accumulation_steps": 4,
  "num_epochs": 1,
  "warmup_steps": 100,               // CHANGED: Shorter warmup
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,

  "use_gradient_checkpointing": false,
  "mixed_precision": "bf16",

  "output_dir": "./checkpoints/stage2_vision",
  "save_steps": 1000,
  "save_total_limit": 5,
  "resume_from_checkpoint": "your-username/theworld-stage1-projection",  // RESUME!

  "eval_steps": 5000,
  "do_eval": true,

  "logging_steps": 100,
  "log_to_wandb": true,
  "wandb_project": "theworld-multistage",
  "wandb_run_name": "stage2-vision",

  "dataset_name": "datacomp",
  "num_samples": null,
  "streaming": true,

  "push_to_hub": true,
  "hub_model_id": "your-username/theworld-stage2-vision",  // Different repo
  "hub_strategy": "every_save"
}
```

**Train:**
```bash
export HF_TOKEN="hf_your_token_here"
python scripts/train_hf.py --config configs/stage2_vision.json

# The script will:
# 1. Download stage1 checkpoint from Hub
# 2. Load projection layer weights
# 3. Unfreeze vision encoder (starts from pretrained Gemma)
# 4. Continue training both projection + vision
```

**Monitor:**
- Validation loss should initially drop (vision learning)
- Watch for gradient explosion (if happens, reduce learning rate)
- Continue until performance plateaus

#### Stage 3: Add Language Model (Optional)

**Create config: `configs/stage3_language.json`**
```json
{
  "_comment": "Stage 3: Projection + Vision + Language (full fine-tuning)",

  "model_name": "google/gemma-3-4b-it",
  "cosmos_model_name": "nvidia/Cosmos-Predict2-2B-Video2World",
  "num_world_steps": 0,
  "max_world_steps": 16,

  "freeze_gemma_vision": false,
  "freeze_gemma_language": false,    // CHANGED: Unfreeze language
  "freeze_cosmos_vae": true,

  "learning_rate": 0.00001,          // CHANGED: Even lower LR
  "batch_size": 4,
  "gradient_accumulation_steps": 4,
  "num_epochs": 1,
  "warmup_steps": 100,
  "weight_decay": 0.01,
  "max_grad_norm": 1.0,

  "use_gradient_checkpointing": true,  // CHANGED: Enable for memory
  "mixed_precision": "bf16",

  "output_dir": "./checkpoints/stage3_language",
  "save_steps": 1000,
  "save_total_limit": 5,
  "resume_from_checkpoint": "your-username/theworld-stage2-vision",  // RESUME!

  "eval_steps": 5000,
  "do_eval": true,

  "logging_steps": 100,
  "log_to_wandb": true,
  "wandb_project": "theworld-multistage",
  "wandb_run_name": "stage3-language",

  "dataset_name": "datacomp",
  "num_samples": null,
  "streaming": true,

  "push_to_hub": true,
  "hub_model_id": "your-username/theworld-stage3-language",
  "hub_strategy": "every_save"
}
```

**Train:**
```bash
export HF_TOKEN="hf_your_token_here"
python scripts/train_hf.py --config configs/stage3_language.json
```

## Configuration Best Practices

### Learning Rates by Stage

| Stage | Components Trained | Recommended LR | Reasoning |
|-------|-------------------|----------------|-----------|
| Stage 1 | Projection only | 1e-4 | Random init, can use higher LR |
| Stage 2 | + Vision | 5e-5 | Fine-tuning pretrained, lower LR |
| Stage 3 | + Language | 1e-5 | Very careful fine-tuning |
| Stage 4 | + VAE | 5e-6 | Full model, minimal LR |

### Warmup Steps

- **Stage 1**: 500-1000 steps (projections need warmup)
- **Stage 2+**: 100-200 steps (most params already trained)

### Batch Size and Memory

| Stage | GPU Memory (bf16) | Suggested Batch Config |
|-------|-------------------|------------------------|
| Stage 1 | 20-24 GB | batch=4, grad_accum=4 |
| Stage 2 | 35-40 GB | batch=4, grad_accum=4 |
| Stage 3 | 56-60 GB | batch=2, grad_accum=8 + grad_checkpoint |
| Stage 4 | 80-84 GB | Use distributed training or DeepSpeed |

### Save Strategy

**Option A: Separate Hub repositories (Recommended)**
```
your-username/theworld-stage1-projection
your-username/theworld-stage2-vision
your-username/theworld-stage3-language
```
- ✅ Easy rollback to previous stage
- ✅ Compare different stages
- ✅ Share intermediate results

**Option B: Single repository with tags**
```
your-username/theworld-multistage
  ├── checkpoint-1000  (stage 1)
  ├── checkpoint-5000  (stage 1 complete)
  ├── checkpoint-6000  (stage 2)
  └── checkpoint-10000 (stage 2 complete)
```
- ✅ All in one place
- ✅ Simpler management
- ⚠️ Harder to track which is which

## When to Move to Next Stage

### Quantitative Signals

**Move to next stage when:**
- ✅ Validation loss plateaus for 2000+ steps
- ✅ Completed target number of steps/epochs
- ✅ Loss improvement < 1% over last 5K steps

**Don't move yet if:**
- ❌ Loss still decreasing significantly
- ❌ Haven't reached minimum steps (10K for stage 1, 5K for others)
- ❌ Validation metrics still improving

### Qualitative Evaluation

Before moving stages, test model quality:

```python
from theworld import TheWorld

# Load latest checkpoint
model = TheWorld.from_pretrained("your-username/theworld-stage1-projection")

# Test on example images
response = model.generate(test_image, "Describe this image")
print(response)

# Compare with baseline
# If quality is acceptable, consider skipping further stages!
```

**Remember**: More stages = more compute. If stage 1 works well enough, you're done!

## Troubleshooting

### Gradient Explosion When Unfreezing

**Symptoms:**
- Loss suddenly jumps to NaN
- Gradients > 1000
- Training becomes unstable

**Solutions:**
```json
{
  "learning_rate": 0.00001,      // Reduce by 10x
  "max_grad_norm": 0.5,          // Lower gradient clipping
  "warmup_steps": 500            // Longer warmup
}
```

### Memory Issues

**Stage 2 OOM:**
```json
{
  "batch_size": 2,                // Reduce from 4
  "gradient_accumulation_steps": 8  // Increase to maintain effective batch
}
```

**Stage 3 OOM:**
```json
{
  "use_gradient_checkpointing": true,  // Enable
  "batch_size": 1,                     // Further reduce if needed
}
```

**Stage 4 OOM:**
- Use distributed training (multi-GPU)
- See `docs/deepspeed_zero_analysis.md` for DeepSpeed ZeRO

### Performance Degradation

If later stages perform worse than earlier stages:

**Possible causes:**
1. **Learning rate too high**: Reduce by 5-10x
2. **Catastrophic forgetting**: Use lower LR and more warmup
3. **Overfitting**: Reduce number of epochs or add regularization
4. **Bad hyperparameters**: Try different batch sizes, weight decay

**Solution:**
```json
{
  "learning_rate": 0.000005,     // Very low
  "weight_decay": 0.05,          // Higher regularization
  "warmup_steps": 1000,          // Longer warmup
  "num_epochs": 0.5              // Train for less
}
```

## Quick Testing

Before committing to full multi-stage training, test with small dataset:

**Test Config: `configs/multistage_test.json`**
```json
{
  "dataset_name": "datacomp",
  "num_samples": 1000,           // Only 1K samples
  "streaming": false,
  "num_epochs": 3,               // Multiple epochs on small data

  "save_steps": 100,
  "push_to_hub": false           // Don't upload test runs
}
```

Run through all stages quickly to verify the workflow works.

## Summary

**Multi-stage training workflow:**

1. **Stage 1**: Train projection layers (fast, cheap)
   - Validate pipeline works
   - Get baseline performance

2. **Evaluate**: Is performance good enough?
   - **Yes**: Done! Deploy stage 1 model
   - **No**: Continue to stage 2

3. **Stage 2**: Add vision encoder (medium cost)
   - Learn domain-specific visual features
   - Improve visual understanding

4. **Evaluate**: Is performance good enough?
   - **Yes**: Done! Deploy stage 2 model
   - **No**: Continue to stage 3

5. **Stage 3**: Add language model (expensive)
   - Task-specific language patterns
   - Full fine-tuning

6. **Deploy**: Use best performing stage

**Key advantages:**
- Iterate quickly in early stages
- Scale compute only when needed
- Each stage builds on previous work
- Easy rollback if later stages don't help

Start with projection-only training and expand only if you need to!
