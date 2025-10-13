# TheWorld Parameter Breakdown

Complete breakdown of model parameters by component, showing total count, trainable parameters, and frozen parameters for different training configurations.

**Generated**: January 13, 2025
**Model**: TheWorld (Gemma 3 4B + Cosmos 2B)

---

## Summary

| Configuration | Total Params | Trainable | Frozen | % Trainable | Memory (GPU) |
|--------------|--------------|-----------|--------|-------------|--------------|
| **Default (Projection Only)** | 4.43B | 76.29M | 4.35B | 1.72% | ~20-24GB |
| Projection + Vision | 4.43B | 493.16M | 3.93B | 11.14% | ~35-40GB |
| Projection + Language | 4.43B | 4.03B | 397M | 91.04% | ~56-60GB* |
| Full Model | 4.43B | 4.43B | 0 | 100% | ~80GB+ |

*With gradient checkpointing

---

## Component Breakdown

### 1. Gemma Vision Encoder (SigLIP)

| Metric | Value |
|--------|-------|
| **Total Parameters** | 416,866,032 (416.87M) |
| **Default State** | ❄️ Frozen |
| **When to unfreeze** | Domain-specific visual features (medical, satellite, etc.) |
| **Memory impact** | +15-20GB |

**Details**:
- Pre-trained SigLIP vision encoder
- Converts images to 256 visual tokens
- Already trained on diverse image data

### 2. Gemma Language Model

| Metric | Value |
|--------|-------|
| **Total Parameters** | 3,880,107,008 (3,880.11M or 3.88B) |
| **Default State** | ❄️ Frozen |
| **When to unfreeze** | Task-specific generation, instruction following |
| **Memory impact** | +30-35GB (requires gradient checkpointing) |

**Details**:
- Core Gemma 3 transformer layers
- Already trained on massive text corpus
- Handles language understanding and generation

### 3. LM Head (Output Projection)

| Metric | Value |
|--------|-------|
| **Total Parameters** | 671,096,320 (671.10M) |
| **Default State** | ❄️ Frozen |
| **Tied to** | Language model embeddings |
| **When to unfreeze** | When unfreezing language model |

**Details**:
- Projects hidden states to vocabulary logits
- Tied weights with embedding layer
- Automatically unfrozen when language model trains

### 4. Cosmos VAE Encoder

| Metric | Value |
|--------|-------|
| **Total Parameters** | 53,595,872 (53.60M) |
| **Default State** | ❄️ Frozen |
| **When to unfreeze** | Rarely (specialized world modeling) |
| **Memory impact** | +5-10GB |

**Details**:
- Encodes images to 16-dim latent space
- Pre-trained on world dynamics
- Usually kept frozen

### 5. World Projection Layer

| Metric | Value |
|--------|-------|
| **Total Parameters** | 43,520 (0.04M) |
| **Default State** | ✅ Trainable |
| **Dimensionality** | 16 → 2304 |
| **Memory impact** | Negligible |

**Details**:
- Linear layer: `nn.Linear(16, 2304)`
- Maps Cosmos latent space to Gemma embedding space
- Always trainable (learns the fusion)

### 6. Temporal Embeddings

| Metric | Value |
|--------|-------|
| **Total Parameters** | 76,246,931 (76.25M) |
| **Default State** | ✅ Trainable |
| **Purpose** | Distinguish timesteps (t=0, t=1, t=2, ...) |
| **Memory impact** | Negligible |

**Details**:
- Learned positional embeddings for time
- Added to world tokens to indicate temporal position
- Critical for multi-step world prediction

---

## Training Configurations

### Configuration 1: Projection Only (Default)

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

| Component | State | Parameters |
|-----------|-------|------------|
| Gemma Vision | ❄️ Frozen | 416.87M |
| Gemma Language | ❄️ Frozen | 3,880.11M |
| LM Head | ❄️ Frozen | 671.10M |
| Cosmos VAE | ❄️ Frozen | 53.60M |
| **World Projection** | ✅ Trainable | 0.04M |
| **Temporal Embeddings** | ✅ Trainable | 76.25M |
| **Total Trainable** | | **76.29M (1.72%)** |

**Memory**: ~20-24GB GPU
**Training speed**: Fast (only 1.72% params to update)
**Use case**: Learning to fuse Gemma + Cosmos representations

### Configuration 2: Projection + Vision

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

| Component | State | Parameters |
|-----------|-------|------------|
| **Gemma Vision** | ✅ Trainable | 416.87M |
| Gemma Language | ❄️ Frozen | 3,880.11M |
| LM Head | ❄️ Frozen | 671.10M |
| Cosmos VAE | ❄️ Frozen | 53.60M |
| **World Projection** | ✅ Trainable | 0.04M |
| **Temporal Embeddings** | ✅ Trainable | 76.25M |
| **Total Trainable** | | **493.16M (11.14%)** |

**Memory**: ~35-40GB GPU
**Training speed**: Medium
**Use case**: Adapting vision encoder to domain-specific visual features

### Configuration 3: Projection + Language

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

| Component | State | Parameters |
|-----------|-------|------------|
| Gemma Vision | ❄️ Frozen | 416.87M |
| **Gemma Language** | ✅ Trainable | 3,880.11M |
| **LM Head** | ✅ Trainable | 671.10M |
| Cosmos VAE | ❄️ Frozen | 53.60M |
| **World Projection** | ✅ Trainable | 0.04M |
| **Temporal Embeddings** | ✅ Trainable | 76.25M |
| **Total Trainable** | | **4.03B (91.04%)** |

**Memory**: ~56-60GB GPU (with gradient checkpointing)
**Training speed**: Slow (91% params to update)
**Use case**: Fine-tuning language generation for specific tasks

### Configuration 4: Full Model

**Best for**: Research, understanding limits (rarely needed)

```python
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_gemma_vision=False,  # ← All unfrozen
    freeze_gemma_language=False,
    freeze_cosmos_vae=False,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

| Component | State | Parameters |
|-----------|-------|------------|
| **Gemma Vision** | ✅ Trainable | 416.87M |
| **Gemma Language** | ✅ Trainable | 3,880.11M |
| **LM Head** | ✅ Trainable | 671.10M |
| **Cosmos VAE** | ✅ Trainable | 53.60M |
| **World Projection** | ✅ Trainable | 0.04M |
| **Temporal Embeddings** | ✅ Trainable | 76.25M |
| **Total Trainable** | | **4.43B (100%)** |

**Memory**: ~80GB+ GPU (requires multi-GPU or DeepSpeed)
**Training speed**: Very slow
**Use case**: Research purposes, rarely needed in practice

---

## Memory Optimization Strategies

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

### DeepSpeed ZeRO

For training beyond single GPU capacity:
- ZeRO Stage 2: Shard optimizer states + gradients
- ZeRO Stage 3: Shard parameters + optimizer + gradients

See `docs/deepspeed_zero_analysis.md` for detailed configuration.

---

## Verification

To verify parameter counts in your environment:

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

## Key Takeaways

1. **Start small**: Default config (1.72% params) is usually sufficient
2. **Progressive unfreezing**: Only unfreeze components if projection-only doesn't work
3. **Memory aware**: Each component unfrozen adds significant memory requirements
4. **Vision before language**: Unfreeze vision (11.14%) before language (91.04%) for efficiency
5. **Gradient checkpointing**: Essential for training language model on <80GB GPUs

---

## Related Documentation

- [Multi-Stage Training Guide](multi_stage_training.md) - Progressive unfreezing workflow
- [Training Infrastructure Design](training_infrastructure_design.md) - Complete training setup
- [DeepSpeed ZeRO Analysis](deepspeed_zero_analysis.md) - Multi-GPU training strategies
