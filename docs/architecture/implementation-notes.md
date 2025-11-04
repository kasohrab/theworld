# TheWorld Implementation Notes

This document covers important technical implementation details, design decisions, and lessons learned during TheWorld development.

## Table of Contents

1. [Model Initialization Pattern](#model-initialization-pattern)
2. [Parameter Breakdown](#parameter-breakdown)
3. [Memory Optimization](#memory-optimization)

---

## Model Initialization Pattern

### The from_pretrained() Pattern

**Key Principle**: TheWorld uses the standard HuggingFace initialization pattern. Always use `from_pretrained()`, never call the constructor directly.

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

### Why This Matters

**Investigation Summary** (January 2025):

We discovered that manual weight loading via `load_state_dict()` produced incorrect outputs, even when all parameter values were identical. The issue was traced to initialization order and internal HuggingFace state management.

**The Problem**:
```python
def __init__(self, gemma_model_name, ...):
    # 1. Load config
    config = Gemma3Config.from_pretrained(gemma_model_name, ...)

    # 2. Initialize parent with random weights
    super().__init__(config)  # ← Problem!

    # 3. Load pretrained model and copy weights
    pretrained = Gemma3ForConditionalGeneration.from_pretrained(...)
    self.load_state_dict(pretrained.state_dict(), strict=False)
```

This approach:
- Created random weights first, then overwrote them
- Missed non-persistent buffers (like `inv_freq` in rotary embeddings)
- Lost device mapping configuration
- Broke internal HuggingFace initialization order

**The Solution**:

Override `from_pretrained()` classmethod to let parent handle weight loading:

```python
class TheWorld(Gemma3ForConditionalGeneration):

    def __init__(self, config: Gemma3Config):
        """Simple structural init - called by from_pretrained()."""
        super().__init__(config)
        # Just initialize attributes, no weight loading
        self.cosmos_pipe = None
        self.cosmos_encoder = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        enable_world: bool = True,
        **kwargs  # dtype, device_map, etc.
    ):
        # Parent's from_pretrained handles EVERYTHING correctly
        model = super(TheWorld, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Now add Cosmos components
        if enable_world:
            model.cosmos_pipe = Cosmos2VideoToWorldPipeline.from_pretrained(...)
            model.cosmos_encoder = CosmosEncoder(...)

        return model
```

### Benefits of This Pattern

1. **Proper weight loading**: Parent's `from_pretrained()` handles dtype conversion automatically
2. **Buffer preservation**: Non-persistent buffers (like `inv_freq`) stay float32 as expected
3. **Device mapping**: Automatically distributes model across GPUs with `device_map="auto"`
4. **Standard pattern**: Follows HuggingFace conventions (same as LLaVA, other multimodal models)
5. **No double-loading**: Gemma loaded once (not twice)

### Validation Results

After fixing the initialization:

```
Max absolute diff:  0.00e+00
Mean absolute diff: 0.00e+00
✅ PASS: TheWorld(enable_world=False) is identical to pure Gemma3
```

### Key Learnings

1. **Weight equality ≠ Behavioral equality**: Same parameter values don't guarantee same outputs
2. **Initialization matters**: How you initialize matters as much as what values you load
3. **State beyond parameters**: Models have state beyond `state_dict()` (buffers, RNG, cache)
4. **Buffers not in state_dict**: Non-persistent buffers aren't in state_dict, created during init
5. **`__init__()` always creates float32**: HuggingFace's `__init__(config)` ignores `config.torch_dtype`
6. **`from_pretrained()` is the way**: Parent's `from_pretrained()` handles dtype, device_map, buffers correctly

---

## Parameter Breakdown

Complete breakdown of model parameters by component for different training configurations.

**Model**: TheWorld (Gemma 3 4B + Cosmos 2B)

### Summary

| Configuration | Total Params | Trainable | Frozen | % Trainable | Memory (GPU) |
|--------------|--------------|-----------|--------|-------------|--------------|
| **Default (Projection Only)** | 4.43B | 76.29M | 4.35B | 1.72% | ~20-24GB |
| Projection + Vision | 4.43B | 493.16M | 3.93B | 11.14% | ~35-40GB |
| Projection + Language | 4.43B | 4.03B | 397M | 91.04% | ~56-60GB* |
| Full Model | 4.43B | 4.43B | 0 | 100% | ~80GB+ |

*With gradient checkpointing

### Component Details

#### 1. Gemma Vision Encoder (SigLIP)

- **Total Parameters**: 416,866,032 (416.87M)
- **Default State**: ❄️ Frozen
- **When to unfreeze**: Domain-specific visual features (medical, satellite, etc.)
- **Memory impact**: +15-20GB

Pre-trained SigLIP vision encoder that converts images to 256 visual tokens.

#### 2. Gemma Language Model

- **Total Parameters**: 3,880,107,008 (3.88B)
- **Default State**: ❄️ Frozen
- **When to unfreeze**: Task-specific generation, instruction following
- **Memory impact**: +30-35GB (requires gradient checkpointing)

Core Gemma 3 transformer layers for language understanding and generation.

#### 3. LM Head (Output Projection)

- **Total Parameters**: 671,096,320 (671.10M)
- **Default State**: ❄️ Frozen
- **Tied to**: Language model embeddings
- **When to unfreeze**: When unfreezing language model

Projects hidden states to vocabulary logits. Tied weights with embedding layer.

#### 4. Cosmos VAE Encoder

- **Total Parameters**: 53,595,872 (53.60M)
- **Default State**: ❄️ Frozen
- **When to unfreeze**: Rarely (specialized world modeling)
- **Memory impact**: +5-10GB

Encodes images to 16-dim latent space for temporal dynamics.

#### 5. World Projection Layer

- **Total Parameters**: 43,520 (0.04M)
- **Default State**: ✅ Trainable
- **Dimensionality**: 16 → 2304
- **Memory impact**: Negligible

Linear layer `nn.Linear(16, 2304)` that maps Cosmos latent space to Gemma embedding space. Always trainable (learns the fusion).

#### 6. Temporal Embeddings

- **Total Parameters**: 76,246,931 (76.25M)
- **Default State**: ✅ Trainable
- **Purpose**: Distinguish timesteps (t=0, t=1, t=2, ...)
- **Memory impact**: Negligible

Learned positional embeddings for time, added to world tokens to indicate temporal position. Critical for multi-step world prediction.

### Training Configurations

#### Configuration 1: Projection Only (Default)

**Best for**: Initial training, quick experiments, limited compute

```python
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_gemma_vision=True,
    freeze_gemma_language=True,
    freeze_cosmos_vae=True,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

- **Trainable**: 76.29M (1.72%)
- **Memory**: ~20-24GB GPU
- **Training speed**: Fast
- **Use case**: Learning to fuse Gemma + Cosmos representations

#### Configuration 2: Projection + Vision

**Best for**: Domain adaptation (medical, satellite, specialized imagery)

```python
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_gemma_vision=False,  # ← Unfreeze
    freeze_gemma_language=True,
    freeze_cosmos_vae=True,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

- **Trainable**: 493.16M (11.14%)
- **Memory**: ~35-40GB GPU
- **Training speed**: Medium
- **Use case**: Adapting vision encoder to domain-specific visual features

#### Configuration 3: Projection + Language

**Best for**: Task-specific generation, instruction following

```python
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_gemma_vision=True,
    freeze_gemma_language=False,  # ← Unfreeze
    freeze_cosmos_vae=True,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

- **Trainable**: 4.03B (91.04%)
- **Memory**: ~56-60GB GPU (with gradient checkpointing)
- **Training speed**: Slow
- **Use case**: Fine-tuning language generation for specific tasks

### Verification

To verify parameter counts:

```python
from theworld import TheWorld

model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Get trainable parameter count
trainable, total, percentage = model.get_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({percentage:.2f}%)")

# Expected output (default config):
# Trainable: 76,290,451 / 4,426,859,363 (1.72%)
```

---

## Memory Optimization

### Gradient Checkpointing

Reduces activation memory by 4-8× at cost of 30-40% slower training:

```python
# Enable in training config
{
  "use_gradient_checkpointing": true
}
```

**Impact**:
- Config 3 (Language): 56-60GB → ~30-35GB
- Config 4 (Full): 80GB+ → ~45-50GB

### Mixed Precision Training

Already enabled by default (`dtype=torch.bfloat16`):
- Params stored in bfloat16
- Gradients accumulated in float32
- ~50% memory savings vs full float32

### Multi-GPU Training

For training beyond single GPU capacity, use Accelerate:
- **DDP**: Each GPU has full model copy (best for projection-only and + vision)
- **FSDP**: Shards model across GPUs (required for full model training)

See [Distributed Training Guide](../training/distributed.md) for detailed configuration.

### Memory Requirements by Configuration

| Scenario | Config | Memory/GPU | Speed |
|----------|--------|------------|-------|
| **Projection only** | Default | 20-24GB | Fast |
| **+ Vision encoder** | `freeze_gemma_vision=false` | 35-40GB | Medium |
| **+ Vision + GradChkpt** | + `use_gradient_checkpointing=true` | 25-30GB | Slower |
| **Full model** | All `false` + checkpointing | 56-60GB | Slow |

---

## Key Takeaways

1. **Always use from_pretrained()**: Never call constructor directly
2. **Start small**: Default config (1.72% params) is usually sufficient
3. **Progressive unfreezing**: Only unfreeze components if projection-only doesn't work
4. **Memory aware**: Each component unfrozen adds significant memory requirements
5. **Vision before language**: Unfreeze vision (11.14%) before language (91.04%) for efficiency
6. **Gradient checkpointing**: Essential for training language model on <80GB GPUs

---

## Related Documentation

- [Architecture Overview](overview.md) - Core architecture concepts
- [Multi-Stage Training Guide](../training/multi-stage.md) - Progressive unfreezing workflow
- [Distributed Training](../training/distributed.md) - Accelerate and multi-GPU training
