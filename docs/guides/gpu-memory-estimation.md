# GPU Memory Estimation for Deep Learning Training

A practical guide to estimating memory usage before training. No fluff.

---

## Core Memory Components

Training a deep learning model requires memory for 4 main components:

```
Total Memory = Model Weights + Activations + Gradients + Optimizer States
```

**Precision matters:** Most training uses mixed precision (bfloat16/float16 for most, float32 for optimizer).
- bfloat16/float16: 2 bytes per parameter
- float32: 4 bytes per parameter

---

## 1. Model Weights

**Formula:**
```python
memory_gb = num_parameters * bytes_per_param / (1024**3)
```

**Example (Gemma 3 4B in bfloat16):**
```python
params = 4_300_079_472
memory = params * 2 / (1024**3)  # 8.01 GB
```

**How to find num_parameters:**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("model_name")
total_params = sum(p.numel() for p in model.parameters())
```

**Reference:** [Kaplan et al., 2020 - Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

---

## 2. Activations (Forward Pass)

Activations are intermediate tensors kept in memory for backpropagation.

**Without gradient checkpointing:** ALL intermediate tensors stored.
**With gradient checkpointing:** Only checkpointed layers stored, rest recomputed during backward.

### Transformer Layer Activations

For a single transformer layer:

```python
batch_size = B
seq_len = S
hidden_dim = D
intermediate_size = I  # Typically 4×D
precision = 2  # bytes (bfloat16)

# Attention activations
attn_output = B * S * D * precision

# MLP activations (gate + up + intermediate + down)
mlp_activations = B * S * I * precision * 3 + B * S * D * precision

# LayerNorm + residuals
norm_residual = B * S * D * precision * 4  # 2 LN + 2 residuals

# Q/K/V projections
num_heads = H
head_dim = D // H
qkv = B * S * H * head_dim * precision * 3  # Q + K + V

# Per layer total
per_layer = attn_output + mlp_activations + norm_residual + qkv

# All layers
total_activations = per_layer * num_layers
```

**Simplified formula (Transformer):**
```
Activations per layer ≈ B * S * D * 2 * (4I/D + 6)
For D=2560, I=10240: ≈ B * S * 2560 * 2 * 22 = B * S * 112,640 bytes
```

**Reference:** [Chen et al., 2016 - Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (gradient checkpointing)

---

## 3. Gradients

Gradients have the same size as the parameters they correspond to.

### Full Training (all params trainable)
```python
gradient_memory = num_trainable_params * 2  # bfloat16 gradients
```

### Partial Training (frozen layers)
```python
# Only need gradients for trainable params
gradient_memory = num_trainable_params * 2

# BUT: Also need activation gradients through frozen layers!
# These are the same size as activations
activation_gradient_memory = total_activations  # Same as forward pass
```

**Key insight:** Frozen layers don't have weight gradients, but you still need **activation gradients** to backprop through them.

**Example (TheWorld):**
- Trainable params: 6.6M (projection only)
- Weight gradients: 6.6M * 2 = 12.6 MB
- **Activation gradients: 32.26 GB** (through frozen Gemma!)

---

## 4. Optimizer States

### SGD (no momentum)
```python
optimizer_memory = 0  # No extra state
```

### SGD with momentum
```python
optimizer_memory = num_trainable_params * 4  # float32 momentum
```

### Adam/AdamW
```python
# First moment (momentum)
first_moment = num_trainable_params * 4  # float32

# Second moment (variance)
second_moment = num_trainable_params * 4  # float32

optimizer_memory = first_moment + second_moment
```

**Reference:** [Kingma & Ba, 2014 - Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

---

## 5. Attention Memory (Special Case)

Attention has **O(n²)** memory in sequence length.

### Self-Attention Components

```python
batch_size = B
num_heads = H
seq_len = S
head_dim = D_h
precision = 2  # bfloat16

# Q, K, V projections
qkv_memory = 3 * B * S * H * D_h * precision

# Attention scores (Q @ K.T)
attn_scores = B * H * S * S * precision  # This is O(n²)!

# KV cache (stored across all layers)
kv_cache_per_layer = 2 * B * H * S * D_h * precision  # K + V
kv_cache_total = kv_cache_per_layer * num_layers
```

**Critical:** Attention scores scale **quadratically** with sequence length!
- 1K tokens: ~0.015 GB per layer
- 5K tokens: ~0.77 GB per layer
- 10K tokens: ~3.05 GB per layer

**Reference:** [Dao et al., 2022 - FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

---

## 6. KV Cache

The KV cache stores keys and values for all previous tokens (for autoregressive generation).

### Formula

```python
# Per layer
kv_per_layer = 2 * batch_size * num_kv_heads * seq_len * head_dim * precision

# All layers
kv_total = kv_per_layer * num_layers
```

**Example (Gemma 3 4B, 5076 tokens):**
```python
B = 2
num_kv_heads = 4  # Grouped-query attention
seq_len = 5076
head_dim = 256
num_layers = 34

kv_per_layer = 2 * B * num_kv_heads * seq_len * head_dim * 2 / (1024**3)
# = 0.039 GB per layer

kv_total = kv_per_layer * num_layers  # = 1.32 GB
```

**Reference:** [Shazeer, 2019 - Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (multi-query attention)

---

## Estimation Workflow

### Step 1: Get Model Architecture

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("model_name")

# For multimodal models, check for nested configs
if hasattr(config, 'text_config'):
    config = config.text_config

print(f"Hidden dim: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Intermediate size: {config.intermediate_size}")
print(f"Num attention heads: {config.num_attention_heads}")
print(f"Num KV heads: {config.num_key_value_heads}")
```

### Step 2: Calculate Each Component

```python
batch_size = 2
seq_len = 1000
hidden_dim = config.hidden_size
num_layers = config.num_hidden_layers
intermediate_size = config.intermediate_size

# 1. Weights
weights_gb = num_params * 2 / (1024**3)

# 2. Activations (per layer, all layers)
act_per_layer = (
    batch_size * seq_len * intermediate_size * 2 * 3 +  # MLP
    batch_size * seq_len * hidden_dim * 2 * 6  # attn + LN + residuals
) / (1024**3)
activations_gb = act_per_layer * num_layers

# 3. Gradients
gradients_gb = num_trainable_params * 2 / (1024**3)

# 4. Optimizer (AdamW)
optimizer_gb = num_trainable_params * 4 * 2 / (1024**3)

# 5. KV cache
kv_gb = 2 * batch_size * num_kv_heads * seq_len * head_dim * 2 * num_layers / (1024**3)

# Total
total_gb = weights_gb + activations_gb + gradients_gb + optimizer_gb + kv_gb
```

### Step 3: Add Overhead

```python
# Empirical: Add 15-20% for PyTorch overhead, intermediate tensors
overhead_factor = 1.2
estimated_peak_gb = total_gb * overhead_factor
```

---

## Common Pitfalls

### 1. Forgetting Activation Gradients for Frozen Layers

**Wrong:**
```python
# Only count gradients for trainable params
gradients = num_trainable_params * 2
```

**Correct:**
```python
# Weight gradients + activation gradients
gradients = num_trainable_params * 2 + activation_memory
```

### 2. Underestimating Sequence Length

**Watch out:**
- Text-only models: Sequence length is just token count
- Vision-language models: Sequence includes image tokens!
- **TheWorld:** 725 text → 5076 after adding vision (256) + world (4096)

### 3. Ignoring Batch Size in Memory Estimates

All activation formulas scale **linearly** with batch size. Doubling batch size doubles activation memory.

### 4. O(n²) Attention Memory

Attention memory scales **quadratically**. Be very careful with long sequences:
- 512 tokens: manageable
- 2048 tokens: 16× more attention memory
- 8192 tokens: 256× more attention memory

---

## Memory Reduction Techniques

### 1. Gradient Checkpointing

**Savings:** Reduce activation memory by 4-8×
**Cost:** 30-40% slower training (recompute activations during backward)

```python
model.gradient_checkpointing_enable()
```

**Reference:** [Chen et al., 2016](https://arxiv.org/abs/1604.06174)

### 2. Mixed Precision Training

**Savings:** 2× memory reduction for weights/activations
**Implementation:** Use `torch.cuda.amp` or HuggingFace Trainer with `fp16=True` or `bf16=True`

**Reference:** [Micikevicius et al., 2017 - Mixed Precision Training](https://arxiv.org/abs/1710.03740)

### 3. Reduce Batch Size

**Linear savings:** Half batch size → half activation memory

### 4. Reduce Sequence Length

**Quadratic savings for attention:** Half sequence → 4× less attention memory

**For vision models:**
- Reduce input resolution: 512×512 → 224×224 (TheWorld: 4096 → 784 world tokens)
- Use patch merging/pooling

### 5. LoRA (Low-Rank Adaptation)

Instead of training all weights, train low-rank adapters.

**Memory savings:**
```python
# Full fine-tuning
trainable_params = 4_300_000_000

# LoRA (rank=8)
lora_params = num_layers * hidden_dim * 8 * 2  # Up and down projections
# ≈ 34 * 2560 * 8 * 2 = 1,397,760 (0.03% of full!)
```

**Reference:** [Hu et al., 2021 - LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

## Verification: Profiling Your Training

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # Run training step
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# Print memory summary
print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

# Get peak memory
import torch
peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")
```

### CUDA Memory Stats

```python
import torch

# Before training
torch.cuda.reset_peak_memory_stats()

# After training step
peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

print(f"Peak allocated: {peak_allocated:.2f} GB")
print(f"Peak reserved: {peak_reserved:.2f} GB")
print(f"Fragmentation: {peak_reserved - peak_allocated:.2f} GB")
```

---

## Case Study: Gemma 3 4B vs TheWorld

See `scripts/profile/PROFILING_COMPARISON.md` for a complete worked example comparing:
- Gemma 3 4B (baseline, 725 tokens)
- TheWorld (projection training, 5076 tokens)

**Key findings:**
1. Activation gradients (32 GB) dominate when training through frozen layers
2. MLP activations (30 GB) scale with sequence length
3. Attention is O(n²): 0.77 GB per layer for 5076 tokens
4. Always measure - estimates are 15-20% off due to overhead

---

## Further Reading

### Foundational Papers

1. **Memory Optimization:**
   - [Chen et al., 2016 - Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (gradient checkpointing)
   - [Dao et al., 2022 - FlashAttention](https://arxiv.org/abs/2205.14135) (efficient attention)

2. **Scaling Laws:**
   - [Kaplan et al., 2020 - Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
   - [Hoffmann et al., 2022 - Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)

3. **Efficient Training:**
   - [Hu et al., 2021 - LoRA](https://arxiv.org/abs/2106.09685) (parameter-efficient fine-tuning)
   - [Micikevicius et al., 2017 - Mixed Precision Training](https://arxiv.org/abs/1710.03740)

4. **Attention Mechanisms:**
   - [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762)
   - [Shazeer, 2019 - Multi-Query Attention](https://arxiv.org/abs/1911.02150)

### Tools

1. **PyTorch Profiler:** https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
2. **NVIDIA NSight Systems:** https://developer.nvidia.com/nsight-systems
3. **DeepSpeed Memory Estimator:** https://deepspeed.ai/

---

## Quick Reference Formulas

```python
# Transformer memory estimation
def estimate_transformer_memory(
    batch_size,
    seq_len,
    hidden_dim,
    num_layers,
    intermediate_size,
    num_params,
    num_trainable_params,
    num_kv_heads,
    head_dim,
    full_training=False,
    gradient_checkpointing=False
):
    """
    All outputs in GB.
    Assumes bfloat16 for weights/activations, float32 for optimizer.
    """
    # Weights
    weights = num_params * 2 / (1024**3)

    # Activations
    act_per_layer = (
        batch_size * seq_len * intermediate_size * 2 * 3 +
        batch_size * seq_len * hidden_dim * 2 * 6
    ) / (1024**3)

    if gradient_checkpointing:
        activations = act_per_layer * num_layers * 0.2  # Checkpoint every 5 layers
    else:
        activations = act_per_layer * num_layers

    # Gradients
    weight_gradients = num_trainable_params * 2 / (1024**3)

    if full_training:
        activation_gradients = 0  # No need to store activation gradients
    else:
        activation_gradients = activations  # Need gradients through frozen layers

    gradients = weight_gradients + activation_gradients

    # Optimizer (AdamW)
    optimizer = num_trainable_params * 4 * 2 / (1024**3)

    # KV cache
    kv_cache = 2 * batch_size * num_kv_heads * seq_len * head_dim * 2 * num_layers / (1024**3)

    # Total (without overhead)
    total = weights + activations + gradients + optimizer + kv_cache

    # Add 20% overhead
    total_with_overhead = total * 1.2

    return {
        "weights": weights,
        "activations": activations,
        "gradients": gradients,
        "optimizer": optimizer,
        "kv_cache": kv_cache,
        "total": total,
        "total_with_overhead": total_with_overhead,
    }
```

**Usage:**
```python
memory = estimate_transformer_memory(
    batch_size=2,
    seq_len=5076,
    hidden_dim=2560,
    num_layers=34,
    intermediate_size=10240,
    num_params=4_300_000_000,
    num_trainable_params=6_600_000,
    num_kv_heads=4,
    head_dim=256,
    full_training=False,
    gradient_checkpointing=False
)

print(f"Estimated memory: {memory['total_with_overhead']:.2f} GB")
```
