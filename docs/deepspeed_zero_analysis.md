# DeepSpeed ZeRO Strategy Analysis for TheWorld

## Executive Summary

**TL;DR:** For training the full 6B TheWorld model, **DeepSpeed ZeRO-3 on 2 GPUs is the best solution**, saving ~40GB per GPU compared to naive training. For projection-only training, ZeRO provides minimal benefit.

## Memory Breakdown

### Baseline: Full Model Training (All 6B Parameters Unfrozen)

| Component | Memory (GB) | Notes |
|-----------|-------------|-------|
| Model parameters (bf16) | 12 | 6B × 2 bytes |
| Gradients (bf16) | 12 | Same size as params |
| Optimizer states (AdamW fp32) | 48 | 2× params (momentum + variance) in fp32 |
| Activations | 8-12 | Depends on batch size, sequence length |
| **Total** | **80-84 GB** | **Won't fit on single A100 80GB!** |

### Projection-Only Training (50K Parameters Trainable)

| Component | Memory (GB) | Notes |
|-----------|-------------|-------|
| Model parameters (bf16) | 12 | Entire 6B model loaded (frozen) |
| Gradients (bf16) | ~0.0001 | Only 50K params |
| Optimizer states (AdamW fp32) | ~0.0004 | Only 50K params |
| Activations | 8-12 | **This is the bottleneck** |
| **Total** | **20-24 GB** | **Fits on single GPU easily** |

**Key Insight:** For projection-only, optimizer/gradient memory is negligible. Bottleneck is activations (solved by gradient checkpointing, not ZeRO).

## DeepSpeed ZeRO Stages

### ZeRO-1: Shard Optimizer States Only
- Optimizer states distributed across GPUs
- **Memory saved:** 48GB / N_GPUs

### ZeRO-2: Shard Optimizer States + Gradients
- Optimizer states + gradients distributed
- **Memory saved:** (48GB + 12GB) / N_GPUs = 60GB / N_GPUs

### ZeRO-3: Shard Everything (Optimizer + Gradients + Parameters)
- All model state distributed across GPUs
- **Memory saved:** (48GB + 12GB + 12GB) / N_GPUs = 72GB / N_GPUs
- **Trade-off:** Extra communication overhead for parameter gathering

### ZeRO-Offload: CPU Offloading
- Offload optimizer states to CPU RAM
- **Memory saved:** 48GB moved from GPU to CPU
- **Trade-off:** CPU-GPU transfer overhead (~40-50% slower)

## Comparison Table: Full Model Training Strategies

| Strategy | GPUs | Memory/GPU | Speed | Fits A100 40GB? | Complexity |
|----------|------|------------|-------|-----------------|------------|
| **Naive** | 1 | 80-84 GB | 1.0× | ❌ No | Low |
| **Gradient Checkpointing** | 1 | 56-60 GB | 0.6-0.7× | ❌ No | Low |
| **ZeRO-1** | 2 | 56-60 GB | 1.7× | ❌ No | Medium |
| **ZeRO-2** | 2 | 50-54 GB | 1.75× | ❌ Barely | Medium |
| **ZeRO-3** | 2 | 44-48 GB | 1.8× | ⚠️ Tight | Medium |
| **ZeRO-3** | 4 | 26-30 GB | 3.5× | ✅ Yes | Medium |
| **ZeRO-3 + GradChkpt** | 2 | 36-38 GB | 1.1× | ✅ Yes | Medium |
| **ZeRO-3 + GradChkpt** | 4 | 20-22 GB | 2.2× | ✅ Yes | Medium |
| **ZeRO-Offload** | 1 | 32-36 GB | 0.4-0.5× | ✅ Yes | Medium |
| **ZeRO-Offload + GradChkpt** | 1 | 24-26 GB | 0.3× | ✅ Yes | Medium |

### Detailed Calculations

#### ZeRO-3 on 2 GPUs (No Gradient Checkpointing)
```
Per GPU:
- Model parameters: 12GB / 2 = 6GB
- Gradients: 12GB / 2 = 6GB
- Optimizer states: 48GB / 2 = 24GB
- Activations: 8-12GB (not sharded by ZeRO)
Total: 44-48 GB per GPU
```

#### ZeRO-3 on 4 GPUs (No Gradient Checkpointing)
```
Per GPU:
- Model parameters: 12GB / 4 = 3GB
- Gradients: 12GB / 4 = 3GB
- Optimizer states: 48GB / 4 = 12GB
- Activations: 8-12GB
Total: 26-30 GB per GPU ✅ Comfortable fit on A100 40GB
```

#### ZeRO-3 on 2 GPUs + Gradient Checkpointing
```
Per GPU:
- Model parameters: 6GB
- Gradients: 6GB
- Optimizer states: 24GB
- Activations: 2-3GB (reduced 4-8× by checkpointing)
Total: 38-39 GB per GPU ✅ Fits on A100 40GB!
```

#### ZeRO-Offload on 1 GPU + Gradient Checkpointing
```
GPU:
- Model parameters: 12GB (streamed in/out)
- Gradients: 12GB
- Optimizer states: 0GB (on CPU)
- Activations: 2-3GB (with checkpointing)
Total GPU: 26-27 GB ✅ Fits easily

CPU RAM:
- Optimizer states: 48GB
```

## Recommendations by Use Case

### Use Case 1: Projection-Only Training (Default)
**Scenario:** Training only the 50K projection layers (0.07% of model)

**Recommendation:** **Single GPU + No ZeRO**
- Memory: 20-24GB (fits on RTX 4090, A100 40GB)
- Speed: Fast (no overhead)
- Setup: Simple

**Why not ZeRO:** Optimizer/gradient memory is negligible (~400KB). ZeRO adds complexity with no benefit.

**Optional:** Add gradient checkpointing if activations are large (long sequences)

### Use Case 2: Full Model Training, 2× A100 40GB Available
**Scenario:** Training all 6B parameters, have 2 GPUs

**Recommendation:** **ZeRO-3 + Gradient Checkpointing**
- Memory: 36-38GB per GPU (fits!)
- Speed: ~1.1× baseline (reasonable)
- Setup: Medium complexity (DeepSpeed config)

**Why this works:**
- ZeRO-3 shards optimizer (24GB per GPU instead of 48GB)
- Gradient checkpointing reduces activations (2-3GB instead of 8-12GB)
- Combined savings: 44GB → 38GB per GPU

### Use Case 3: Full Model Training, 4+ GPUs Available
**Scenario:** Training all 6B parameters, have 4 GPUs

**Recommendation:** **ZeRO-3 (No Gradient Checkpointing)**
- Memory: 26-30GB per GPU (plenty of headroom)
- Speed: ~3.5× baseline (excellent scaling)
- Setup: Medium complexity

**Why skip gradient checkpointing:** Enough memory without it, so skip the 30% speed penalty.

### Use Case 4: Full Model Training, Only 1 GPU Available
**Scenario:** Training all 6B parameters, single GPU (A100 40GB or 80GB)

**Recommendation:** **ZeRO-Offload + Gradient Checkpointing**
- Memory: 26-27GB GPU, 48GB CPU RAM
- Speed: ~0.3× baseline (slow but works)
- Setup: Medium complexity

**Trade-off:** Very slow due to CPU-GPU transfers, but enables training on single GPU.

**Alternative:** Don't train full model; train projection + vision encoder only (~30% of params, fits without ZeRO)

### Use Case 5: Partial Training (Projection + Vision Encoder)
**Scenario:** Training projection layers + Gemma vision encoder (~2B params)

**Recommendation:** **Single GPU + Gradient Checkpointing**
- Memory: ~35-40GB (fits on A100 40GB)
- Speed: 0.6-0.7× baseline
- Setup: Simple

**Why not ZeRO:** Single GPU is sufficient with gradient checkpointing.

## DeepSpeed Configuration Example

### ZeRO-3 + Gradient Checkpointing (2 GPUs)

```json
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,

  "bf16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  },

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": true,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  }
}
```

### ZeRO-Offload + Gradient Checkpointing (1 GPU)

```json
{
  "train_batch_size": 8,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,
  "gradient_clipping": 1.0,

  "bf16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true
  }
}
```

## Implementation Changes

### 1. Update `training_config.py`

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...

    # DeepSpeed configuration
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None  # Path to DeepSpeed JSON
    zero_stage: int = 0  # 0 (disabled), 1, 2, or 3
    offload_optimizer: bool = False  # CPU offload for ZeRO-Offload
    offload_param: bool = False  # Parameter offload
```

### 2. Update `train_hf.py`

```python
from transformers import Trainer, TrainingArguments

def train():
    config = TrainingConfig()

    training_args = TrainingArguments(
        # ... existing args ...
        deepspeed=config.deepspeed_config if config.use_deepspeed else None,
    )

    # HuggingFace Trainer auto-handles DeepSpeed integration
    trainer = Trainer(...)
    trainer.train()
```

### 3. Add DeepSpeed Configs

```
theworld/
├── configs/
│   ├── deepspeed_zero3_2gpu.json
│   ├── deepspeed_zero3_4gpu.json
│   └── deepspeed_offload_1gpu.json
```

## HuggingFace Trainer + DeepSpeed Integration

**Good news:** HuggingFace Trainer has **native DeepSpeed support**. No manual integration needed!

```python
training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="configs/deepspeed_zero3_2gpu.json",  # Just pass config path
    # ... other args ...
)
```

Trainer automatically:
- Initializes DeepSpeed engine
- Wraps model with ZeRO
- Handles distributed training
- Saves/loads ZeRO checkpoints

## Performance Expectations

### Training Speed (Relative to Baseline Single GPU)

| Strategy | Expected Speed | Notes |
|----------|----------------|-------|
| Single GPU | 1.0× | Baseline |
| Single GPU + GradChkpt | 0.6-0.7× | 30-40% slower |
| ZeRO-3, 2 GPUs | 1.8× | Good scaling |
| ZeRO-3, 4 GPUs | 3.5× | Excellent scaling |
| ZeRO-3 + GradChkpt, 2 GPUs | 1.1× | Some slowdown from checkpointing |
| ZeRO-Offload, 1 GPU | 0.4-0.5× | CPU-GPU transfer bottleneck |

### Wall Clock Time Example (1 Epoch, 10K Steps)

| Strategy | Time | Cost (A100 $2/hr) |
|----------|------|-------------------|
| Single GPU (if it fit) | 10 hours | $20 |
| ZeRO-3, 2 GPUs | 5.5 hours | $22 |
| ZeRO-3, 4 GPUs | 2.9 hours | $23 |
| ZeRO-Offload, 1 GPU | 25 hours | $50 |

**Insight:** ZeRO-3 on 4 GPUs is fastest and most cost-effective for multi-epoch training.

## Cost-Benefit Analysis

### Projection-Only Training
- **Current approach:** Single GPU, no ZeRO ✅ Optimal
- **ZeRO benefit:** None (overhead with no gain)

### Full Model Training
- **Current approach:** Can't train on single GPU ❌
- **ZeRO-3 on 2 GPUs:** Enables training ✅
- **Alternative (gradient checkpointing only):** Still doesn't fit (56-60GB > 40GB) ❌

**Verdict:** For full model training, **DeepSpeed ZeRO-3 is necessary**, not optional.

## Recommendation: Hybrid Approach

### Phase 1: Projection-Only (Simple)
```python
# No DeepSpeed needed
model = TheWorld(..., freeze_all=True)
trainer = Trainer(...)  # Single GPU, simple
```

### Phase 2: Full Model Training (Advanced)
```python
# Use DeepSpeed ZeRO-3
training_args = TrainingArguments(
    deepspeed="configs/deepspeed_zero3_2gpu.json",
    gradient_checkpointing=True,
)
```

### Implementation Priority
1. **Phase 1-3 from original design:** Basic training (projection-only, single GPU)
2. **Add DeepSpeed configs:** For users who want full model training
3. **Document both paths:** Simple (projection) vs. advanced (full model + ZeRO)

## Conclusion

**For TheWorld model:**

| Training Scenario | Recommendation | Memory/GPU | GPUs |
|-------------------|----------------|------------|------|
| Projection-only | No ZeRO | 20-24GB | 1 |
| Projection + Vision | GradChkpt only | 35-40GB | 1 |
| Full model | ZeRO-3 + GradChkpt | 36-38GB | 2 |
| Full model (faster) | ZeRO-3 | 26-30GB | 4 |
| Full model (1 GPU only) | ZeRO-Offload + GradChkpt | 26GB GPU + 48GB CPU | 1 |

**Bottom line:** DeepSpeed ZeRO-3 saves **~40GB per GPU** for full model training, making it **essential** for training beyond projection layers.
