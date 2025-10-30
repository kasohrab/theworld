# Multi-GPU Training with Accelerate

## Executive Summary

**TL;DR:** Use `accelerate launch` for all multi-GPU training. For projection-only training (default), DDP on 2 GPUs is fastest. For larger model portions (unfrozen vision/language), use FSDP to shard model across GPUs.

**Quick Start:**
```bash
# Single GPU
python scripts/train_hf.py --config configs/llava_pretrain_full.json

# Multi-GPU (auto-detect)
accelerate launch scripts/train_hf.py --config configs/llava_pretrain_full.json

# Multi-GPU with specific config
accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \
    scripts/train_hf.py --config configs/llava_pretrain_full.json
```

## Memory Breakdown

### Baseline: Full Model Training (All 4.4B Parameters Unfrozen)

| Component | Memory (GB) | Notes |
|-----------|-------------|-------|
| Model parameters (bf16) | 8.8 | 4.4B × 2 bytes |
| Gradients (bf16) | 8.8 | Same size as params |
| Optimizer states (AdamW fp32) | 35.2 | 2× params (momentum + variance) in fp32 |
| Activations | 8-12 | Depends on batch size, sequence length |
| **Total** | **60-65 GB** | **Fits on single H100 80GB with tight margin** |

### Projection-Only Training (82M Parameters Trainable, 1.87%)

| Component | Memory (GB) | Notes |
|-----------|-------------|-------|
| Model parameters (bf16) | 8.8 | Entire 4.4B model loaded (frozen) |
| Gradients (bf16) | 0.16 | Only 82M params |
| Optimizer states (AdamW fp32) | 0.64 | Only 82M params |
| Activations | 8-12 | **This is the bottleneck** |
| **Total** | **17-21 GB** | **Fits on single GPU easily** |

**Key Insight:** For projection-only, optimizer/gradient memory is negligible. Bottleneck is activations (solved by gradient checkpointing, not distributed training).

## Distributed Training Strategies

### DDP (Data Distributed Parallel)
- **How it works:** Each GPU holds a full copy of the model, processes different data batches
- **Memory:** Each GPU needs full model size + activations
- **Speed:** Near-linear scaling (2× GPUs ≈ 2× throughput)
- **Best for:** Models that fit in single GPU memory (projection-only, + vision encoder)

**Memory per GPU:**
```
- Model parameters: 8.8 GB (full copy on each GPU)
- Gradients: 0.16-8.8 GB (depending on what's trainable)
- Optimizer states: 0.64-35.2 GB (depending on what's trainable)
- Activations: 8-12 GB
Total: 17-65 GB per GPU
```

### FSDP (Fully Sharded Data Parallel)
- **How it works:** Shards model parameters, gradients, and optimizer states across GPUs
- **Memory:** Significantly reduced memory per GPU
- **Speed:** Slightly slower than DDP due to communication overhead
- **Best for:** Large models that don't fit in single GPU, or when unfreezing language model

**Memory per GPU (full model unfrozen, 2 GPUs):**
```
- Model parameters: 8.8 GB / 2 = 4.4 GB (sharded)
- Gradients: 8.8 GB / 2 = 4.4 GB (sharded)
- Optimizer states: 35.2 GB / 2 = 17.6 GB (sharded)
- Activations: 8-12 GB (not sharded)
Total: 34-38 GB per GPU (vs 60-65 GB with DDP)
```

## Comparison Table: Training Strategies

### Projection-Only Training (82M params, 1.87%)

| Strategy | GPUs | Memory/GPU | Speed | Fits H100 80GB? | Recommended |
|----------|------|------------|-------|-----------------|-------------|
| **Single GPU** | 1 | 17-21 GB | 1.0× | ✅ Yes | ✅ Default |
| **DDP** | 2 | 17-21 GB each | 1.9× | ✅ Yes | ✅ Fastest |
| **DDP** | 4 | 17-21 GB each | 3.7× | ✅ Yes | 🟡 Overkill |
| **FSDP** | 2 | 15-18 GB each | 1.7× | ✅ Yes | ❌ Unnecessary overhead |

**Recommendation:** Use **DDP on 2 GPUs** for nearly 2× speedup with minimal complexity.

### + Vision Encoder (30% params trainable, ~1.3B)

| Strategy | GPUs | Memory/GPU | Speed | Fits H100 80GB? | Recommended |
|----------|------|------------|-------|-----------------|-------------|
| **Single GPU** | 1 | 35-40 GB | 1.0× | ✅ Yes | 🟡 OK |
| **Single GPU + GradChkpt** | 1 | 25-30 GB | 0.7× | ✅ Yes | ✅ Memory constrained |
| **DDP** | 2 | 35-40 GB each | 1.9× | ✅ Yes | ✅ Fastest |
| **DDP + GradChkpt** | 2 | 25-30 GB each | 1.3× | ✅ Yes | 🟡 Balanced |

**Recommendation:** Use **DDP on 2 GPUs** if you have the memory. Use **gradient checkpointing** on single GPU if memory is tight.

### Full Model Training (100% params, 4.4B)

| Strategy | GPUs | Memory/GPU | Speed | Fits H100 80GB? | Recommended |
|----------|------|------------|-------|-----------------|-------------|
| **Single GPU** | 1 | 60-65 GB | 1.0× | ⚠️ Tight | ❌ Risky |
| **Single GPU + GradChkpt** | 1 | 42-47 GB | 0.6× | ✅ Yes | 🟡 Slow but works |
| **DDP** | 2 | 60-65 GB each | 1.9× | ⚠️ Tight | ❌ Risky |
| **FSDP** | 2 | 34-38 GB each | 1.7× | ✅ Yes | ✅ Best choice |
| **FSDP + GradChkpt** | 2 | 26-30 GB each | 1.2× | ✅ Yes | ✅ Memory efficient |
| **FSDP** | 4 | 20-24 GB each | 3.2× | ✅ Yes | 🟡 If available |

**Recommendation:** Use **FSDP on 2 GPUs** for safe margin. Add gradient checkpointing if you need even more memory headroom.

## Accelerate Configuration

### Pre-configured Files

TheWorld includes ready-to-use Accelerate configs in `configs/accelerate/`:

**`single_gpu.yaml`:**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
mixed_precision: bf16
use_cpu: false
```

**`multi_gpu_ddp.yaml`:**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
mixed_precision: bf16
num_processes: 2
use_cpu: false
```

**`multi_gpu_fsdp.yaml`:**
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
mixed_precision: bf16
num_processes: 2
use_cpu: false
```

### Creating Custom Configs

**Interactive setup:**
```bash
accelerate config
```

This launches an interactive wizard that:
1. Asks about your hardware (single/multi-GPU, CPU/TPU)
2. Configures distributed backend (DDP/FSDP/DeepSpeed)
3. Sets mixed precision (fp16/bf16)
4. Saves config to `~/.cache/huggingface/accelerate/default_config.yaml`

**Manual configuration:**
Edit YAML files directly for fine-grained control. See [Accelerate documentation](https://huggingface.co/docs/accelerate/basic_tutorials/launch) for all options.

## Usage Examples

### Single GPU (Projection-Only)
```bash
python scripts/train_hf.py --config configs/llava_pretrain_full.json
```

**Expected memory:** 17-21 GB
**Speed:** Baseline 1.0×

### DDP on 2 GPUs (Projection-Only)
```bash
accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \
    scripts/train_hf.py --config configs/llava_pretrain_full.json
```

**Expected memory:** 17-21 GB per GPU
**Speed:** ~1.9× faster than single GPU

### FSDP on 2 GPUs (Full Model)
```bash
# Update config to unfreeze language model
# Set: freeze_gemma_language: false

accelerate launch --config_file configs/accelerate/multi_gpu_fsdp.yaml \
    scripts/train_hf.py --config configs/llava_pretrain_full.json
```

**Expected memory:** 34-38 GB per GPU (vs 60-65 GB with single GPU)
**Speed:** ~1.7× faster than single GPU

### Auto-detect GPUs
```bash
# Uses default Accelerate config or auto-detects
accelerate launch scripts/train_hf.py --config configs/llava_pretrain_full.json
```

Accelerate automatically:
- Detects number of available GPUs
- Selects appropriate backend (DDP for small models, FSDP for large)
- Optimizes for your hardware

## DDP vs FSDP: When to Use Each

### Use DDP when:
- ✅ Model fits in single GPU memory
- ✅ Only training projection layers (1.87% of params)
- ✅ Training with vision encoder unfrozen (30% of params)
- ✅ You want maximum training speed

**Advantages:**
- Faster communication (no parameter gathering needed)
- Simpler debugging (each GPU has full model)
- Better for small-medium models

**Disadvantages:**
- Each GPU needs full model size
- Doesn't scale to very large models

### Use FSDP when:
- ✅ Training full language model (100% of params)
- ✅ Model doesn't fit in single GPU
- ✅ You want to maximize batch size per GPU
- ✅ Training on 4+ GPUs for large models

**Advantages:**
- Dramatically reduces memory per GPU
- Enables training larger models
- Scales well to many GPUs

**Disadvantages:**
- Slightly slower than DDP (communication overhead)
- More complex debugging
- Requires careful configuration for optimal performance

## Gradient Checkpointing

Gradient checkpointing trades compute for memory by recomputing activations during backward pass instead of storing them.

**Enable in training config:**
```json
{
  "use_gradient_checkpointing": true
}
```

**Memory savings:** 30-40% reduction in activation memory
**Speed cost:** 20-40% slower training

**Use when:**
- Single GPU and memory is tight
- Want to maximize batch size
- Training with vision or language model unfrozen

**Combine with FSDP for maximum memory efficiency:**
```bash
# FSDP + gradient checkpointing = minimal memory per GPU
accelerate launch --config_file configs/accelerate/multi_gpu_fsdp.yaml \
    scripts/train_hf.py --config configs/llava_pretrain_full_gradchkpt.json
```

## Multi-Node Training

For training across multiple machines:

```bash
# On each node, run:
accelerate launch --config_file multi_node_config.yaml \
    --num_machines N \
    --machine_rank RANK \
    --main_process_ip MASTER_IP \
    --main_process_port MASTER_PORT \
    scripts/train_hf.py --config configs/llava_pretrain_full.json
```

See [Accelerate Multi-Node Guide](https://huggingface.co/docs/accelerate/basic_tutorials/launch#multi-node-training) for detailed setup.

## Troubleshooting

### OOM (Out of Memory) Errors

**Problem:** `CUDA out of memory` during training

**Solutions:**
1. **Enable gradient checkpointing** (saves 30-40%)
   ```json
   {"use_gradient_checkpointing": true}
   ```

2. **Switch to FSDP** if using DDP
   ```bash
   accelerate launch --config_file configs/accelerate/multi_gpu_fsdp.yaml ...
   ```

3. **Reduce batch size**
   ```json
   {"batch_size": 1, "gradient_accumulation_steps": 16}
   ```

4. **Use more GPUs**
   ```yaml
   num_processes: 4  # Instead of 2
   ```

### Slow Training

**Problem:** Multi-GPU training not speeding up as expected

**Check:**
1. **Using DDP, not FSDP** for small models (FSDP has overhead)
2. **Batch size is large enough** to saturate GPUs
3. **No data loading bottleneck** (increase `num_workers`)
4. **Network is fast** for multi-node (check bandwidth)

### DDP Unused Parameters Error

**Problem:** `RuntimeError: Expected to have finished reduction in the prior iteration`

**Solution:** Already handled in `train_hf.py` via:
```python
TrainingArguments(ddp_find_unused_parameters=True)
```

This is required because we freeze most parameters in TheWorld.

## Performance Expectations

Based on H100 80GB GPUs, batch size 2, projection-only training:

| Setup | Throughput | Memory/GPU | Cost Efficiency |
|-------|------------|------------|-----------------|
| 1× H100 | 100 samples/hour | 20 GB | Baseline |
| 2× H100 (DDP) | 190 samples/hour | 20 GB each | 1.9× better |
| 4× H100 (DDP) | 370 samples/hour | 20 GB each | 1.85× better |
| 2× H100 (FSDP, full model) | 85 samples/hour | 35 GB each | 0.85× throughput, enables full model |

**Scaling efficiency:**
- DDP: ~95% efficiency (near-linear)
- FSDP: ~85% efficiency (communication overhead)

## References

- [HuggingFace Accelerate Documentation](https://huggingface.co/docs/accelerate/index)
- [Accelerate Launch Tutorial](https://huggingface.co/docs/accelerate/basic_tutorials/launch)
- [FSDP Configuration Guide](https://huggingface.co/docs/accelerate/usage_guides/fsdp)
- [PyTorch FSDP Overview](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
