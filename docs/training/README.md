# Training Documentation

Complete guide to training TheWorld models.

## Quick Start

**Simplest training** (projection layers only):

```bash
# 1. Set HuggingFace token
export HF_TOKEN=hf_your_token_here

# 2. Run training with default config
make train-hf

# Or with custom config
python scripts/train_hf.py --config configs/my_config.json
```

## Quick Links

- **[Infrastructure](infrastructure.md)** - Training design and setup
- **[Multi-Stage Training](multi-stage.md)** - Progressive unfreezing strategy
- **[Distributed Training](distributed.md)** - Accelerate and multi-GPU
- **[Hub Upload](hub-upload.md)** - Publishing to HuggingFace
- **[Datasets](datasets/)** - DataComp, SpatialRGPT, and custom datasets

## Training Configurations

### Configuration 1: Projection Only (Default)

**Best for**: Initial training, limited compute

```python
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

- **Trainable**: 76.29M (1.72%)
- **Memory**: ~20-24GB GPU
- **Speed**: Fast

### Configuration 2: + Vision Encoder

**Best for**: Domain adaptation (medical, satellite images)

```json
{
  "freeze_gemma_vision": false
}
```

- **Trainable**: 493.16M (11.14%)
- **Memory**: ~35-40GB GPU
- **Speed**: Medium

### Configuration 3: + Language Model

**Best for**: Task-specific generation

```json
{
  "freeze_gemma_language": false,
  "use_gradient_checkpointing": true
}
```

- **Trainable**: 4.03B (91.04%)
- **Memory**: ~56-60GB GPU
- **Speed**: Slow

## Training Workflow

### 1. Prepare Dataset

Choose your dataset:
- [DataComp](datasets/datacomp.md) - Large-scale image-caption data
- [SpatialRGPT](datasets/spatial-rgpt.md) - Spatial reasoning data
- Custom dataset (see [Infrastructure](infrastructure.md))

### 2. Create Config

```json
{
  "model_name": "google/gemma-3-4b-it",
  "enable_world": true,
  "learning_rate": 0.0001,
  "batch_size": 4,
  "gradient_accumulation_steps": 4,
  "num_epochs": 3,
  "output_dir": "./checkpoints",
  "push_to_hub": false
}
```

### 3. Run Training

```bash
python scripts/train_hf.py --config configs/my_config.json
```

### 4. Monitor Progress

```bash
# View TensorBoard logs
tensorboard --logdir checkpoints/logs

# Or use Weights & Biases
# (set "log_to_wandb": true in config)
```

### 5. Evaluate

```bash
# Evaluate on BLINK
make eval-blink MODEL=username/theworld-my-model

# Compare against baseline
make eval-gemma
make compare-results
```

## Memory Requirements

| Config | Trainable | Memory | GPUs Needed |
|--------|-----------|--------|-------------|
| Projection only | 1.72% | 20-24GB | 1x 3090/4090 |
| + Vision | 11.14% | 35-40GB | 1x A100 |
| + Vision + GradChkpt | 11.14% | 25-30GB | 1x 3090/4090 |
| + Language | 91.04% | 56-60GB | 1x A100 |
| + Language + GradChkpt | 91.04% | 30-35GB | 1x A100 |
| Full model | 100% | 80GB+ | 2x A100 with FSDP |

## Multi-GPU Training

All multi-GPU training uses HuggingFace Accelerate for simplicity and flexibility:

```bash
# Auto-detect GPUs and use optimal strategy
accelerate launch scripts/train_hf.py --config config.json

# Explicit config (DDP for small models)
accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \
    scripts/train_hf.py --config config.json

# FSDP for larger models (shards across GPUs)
accelerate launch --config_file configs/accelerate/multi_gpu_fsdp.yaml \
    scripts/train_hf.py --config config.json
```

See [Distributed Training](distributed.md) for detailed memory calculations and configuration options.

## Progressive Training Strategy

**Recommended approach** for best results:

1. **Stage 1**: Train projection only (cheap, fast)
2. **Stage 2**: If needed, unfreeze vision encoder
3. **Stage 3**: If needed, unfreeze language model

Each stage resumes from previous checkpoint.

See [Multi-Stage Training](multi-stage.md) for detailed guide.

## HuggingFace Hub Integration

Upload checkpoints to Hub during training:

```json
{
  "push_to_hub": true,
  "hub_model_id": "username/theworld-my-model",
  "hub_strategy": "every_save"
}
```

See [Hub Upload Guide](hub-upload.md) for details.

## Common Issues

**Training loss not decreasing:**
- Check learning rate (try increasing)
- Verify gradient flow
- Check labels (text tokens should have valid labels)

**Out of memory:**
- Reduce batch size
- Enable gradient checkpointing
- Use fewer world steps
- Use Accelerate FSDP (multi-GPU)

**Training too slow:**
- Increase batch size (if memory allows)
- Use multiple GPUs
- Reduce world steps

See [Troubleshooting](../guides/troubleshooting.md) for more.

## Related Documentation

- [Architecture Overview](../architecture/overview.md) - Understanding the model
- [Evaluation Guide](../evaluation/overview.md) - Evaluating trained models
- [Inference Guide](../guides/inference.md) - Using trained models
