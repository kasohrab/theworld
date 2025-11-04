# Training Performance Optimizations

This document tracks performance bottlenecks and optimization strategies for TheWorld training.

## Current Performance Baseline

**As of:** October 2025
**Configuration:** LLaVA-CC3M pretrain, projection-only training (1.87% params)

- **Batch size:** 2 per GPU
- **Gradient accumulation:** 4 steps
- **GPUs:** 4x A100 (144GB each) with DDP
- **Observed speed:** ~18 sec/step
- **Total steps:** 14,148 steps/epoch
- **Projected time:** ~70 hours/epoch

## Implemented Optimizations

### ✅ 1. Fixed Double Image Processing (October 2025)

**Location:** `python/theworld/data.py:200-212`

**Problem:**
The collate function was calling `processor.apply_chat_template()` twice per sample:
1. Full conversation with image (for training data)
2. Prompt-only with image (for label masking position)

Each call processed the image through SigLIP preprocessing, causing 2x overhead.

**Solution:**
Use `processor.tokenizer.apply_chat_template()` directly for prompt length calculation, avoiding redundant image processing.

**Impact:** ~2-3x speedup on data loading

**Code change:**
```python
# Before: Re-processed image
prompt_tokenized = processor.apply_chat_template(
    messages_prompt,  # Contains image!
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

# After: Text-only tokenization
prompt_ids = processor.tokenizer.apply_chat_template(
    messages_prompt_text,  # Text only, no image
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
```

### ✅ 2. Memory Optimization for Evaluation

**Configuration:** `configs/llava_pretrain_full.json`

**Changes:**
- `eval_batch_size: 1` (prevents OOM at step 100)
- `eval_steps: 150` (staggers eval from checkpoint saves at step 100)

**Problem:** Training OOM'd at step 100 when evaluation + checkpointing + Hub upload happened simultaneously.

**Impact:** Prevents training crashes, allows evaluation to proceed

---

## Remaining High-Impact Optimizations

### 🔥 Priority 1: Pre-compute Cosmos Latents

**Estimated speedup:** 5-10x

**Problem:**
Every training step calls `CosmosEncoder.forward(images)` which:
- Resizes PIL images to 512x512
- Converts to tensors
- Runs VAE encoding through WanEncoder3d (~50-100ms per image)
- This happens **even though Cosmos VAE is frozen**

With 595K samples × 2 epochs = 1.2M encodings at ~50-100ms each = **16-33 hours just on VAE encoding**.

**Solution:**
1. **Pre-compute script** (run once offline):
   ```python
   # Pseudocode
   for image_id, image_path in dataset:
       latent = cosmos_vae.encode(image).latent_dist.mean
       save_to_disk(f"cache/{image_id}.pt", latent)
   ```

2. **Modified dataset:**
   ```python
   def __getitem__(self, idx):
       image_id = self.data[idx]["image_id"]
       latent = torch.load(f"cache/{image_id}.pt")  # Pre-computed
       return {"latent": latent, "text": ..., "label": ...}
   ```

3. **Modified forward pass:**
   - Skip `CosmosEncoder.forward()` entirely
   - Load pre-computed latents directly from batch
   - Only project: `latent (16-dim) → projection → embedding (2304-dim)`

**Implementation notes:**
- Cache directory: `data/llava-cc3m/cosmos_latents/`
- Cache files: `{image_id}.pt` with shape `(1, 16, 1, 28, 28)`
- Estimated cache size: 595K × 16 × 28 × 28 × 2 bytes ≈ 17 GB
- Pre-computation time: ~8-16 hours one-time cost
- Net speedup: 5-10x on training after cache built

**Status:** Not implemented (high priority, significant engineering effort)

---

### 🔶 Priority 2: Dataloader Optimizations

**Estimated speedup:** 1.5-2x

**Current config:**
```python
TrainingArguments(
    dataloader_num_workers=4,
    # Missing optimizations:
)
```

**Recommended config:**
```python
TrainingArguments(
    dataloader_num_workers=8,  # Increase from 4
    dataloader_pin_memory=True,  # Faster CPU→GPU transfer
    dataloader_persistent_workers=True,  # Avoid worker respawn
    dataloader_prefetch_factor=2,  # Prefetch next batch
)
```

**Impact:**
- Reduces GPU idle time waiting for data
- Better CPU/GPU overlap
- Lower overhead from worker management

**Status:** Not implemented (easy config change, medium impact)

---

### 🔶 Priority 3: Investigate Gradient Explosion

**Estimated speedup:** 1.2-1.5x (via better convergence)

**Problem:**
Training logs show abnormally high gradient norms:
```
{'loss': 425.1857, 'grad_norm': 13120.0}
{'loss': 430.5656, 'grad_norm': 14528.0}
{'loss': 416.8567, 'grad_norm': 16384.0}
```

**Expected gradient norms for projection-only training:** 1-100
**Observed:** 13,000-16,000 (100x higher!)

**Possible causes:**
1. Poor initialization of projection layers
2. More parameters accidentally unfrozen than expected
3. Gradient clipping not working (`max_grad_norm=1.0` in config)
4. Numerical instability in projection computation

**Investigation steps:**
1. Verify only projection layers have `requires_grad=True`:
   ```python
   for name, param in model.named_parameters():
       if param.requires_grad:
           print(f"{name}: {param.shape}")
   ```

2. Check projection layer initialization:
   ```python
   # Current: torch.nn.Linear default init (may be unstable)
   # Consider: Xavier or Kaiming initialization
   ```

3. Verify gradient clipping is applied:
   ```python
   # Check trainer.py logs for "gradient clipping" messages
   ```

4. Test with lower learning rate for projections:
   ```python
   # Current: 1e-4 (same as base LR)
   # Try: 1e-5 or 1e-6 for projections only
   ```

**Status:** Needs investigation (medium priority, may indicate training instability)

---

### 🔷 Priority 4: Increase Batch Size

**Estimated speedup:** 1.2-1.5x

**Current:**
- `batch_size=2` per GPU
- `gradient_accumulation_steps=4`
- Effective batch size: 2 × 4 × 4 = 32 samples

**Recommendation:**
```json
{
  "batch_size": 4,  // or even 8
  "gradient_accumulation_steps": 2,  // Keep effective batch = 32
}
```

**Benefits:**
- Better GPU utilization (batch_size=2 is very small)
- Fewer gradient accumulation steps = less overhead
- More efficient forward passes

**Tradeoff:**
- Need to verify memory fits
- With projection-only + gradient checkpointing, should have plenty of headroom
- Monitor for OOM, adjust accordingly

**Status:** Not tested (low risk, medium reward)

---

### 🔷 Priority 5: Image Loading Optimization

**Estimated speedup:** 1.3-1.5x

**Location:** `python/theworld/datasets/llava_pretrain.py:215-224`

**Problem:**
- `Image.open(image_path).convert("RGB")` happens in `__getitem__`
- Blocking I/O on each sample
- No prefetching or caching
- Corrupted images trigger wraparound logic (expensive)

**Solutions:**

**Option A: Use HuggingFace datasets caching**
```python
from datasets import load_dataset, Image as HFImage

dataset = load_dataset("...", split="train")
dataset = dataset.cast_column("image", HFImage())
dataset.set_format("torch")  # Automatic caching
```

**Option B: Implement LRU cache**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def load_image_cached(image_path):
    return Image.open(image_path).convert("RGB")
```

**Option C: Pre-filter corrupted images**
- Already done during initialization (line 223 wraparound logic)
- Could be optimized to fail faster

**Status:** Not implemented (nice-to-have optimization)

---

## Lower Priority Optimizations

### Mixed Precision for Cosmos Encoding

**Current:** Cosmos VAE runs in bfloat16 (inherited from pipeline)
**Possible:** Could use float16 for encoding (slightly faster)
**Impact:** Minimal (VAE already frozen, only projection trainable)

### Gradient Checkpointing Tuning

**Current:** Enabled globally with `use_gradient_checkpointing=true`
**Possible:** Selective checkpointing (only on language model, not vision)
**Impact:** 5-10% speedup, but adds complexity

### FSDP vs DDP Comparison

**Current:** Using DDP (Data Distributed Parallel)
**Alternative:** FSDP (Fully Sharded Data Parallel)
**When useful:** Training larger portions of model (e.g., unfreezing Gemma)
**Current status:** Not needed for projection-only training

---

## Not Recommended

### ❌ Flash Attention

**Status:** Already optimized by default

Modern PyTorch (2.0+) with Transformers automatically selects the best attention implementation via `torch.nn.functional.scaled_dot_product_attention`, which dispatches to Flash Attention if available. Manual specification is not needed.

**Verification:**
```python
# PyTorch automatically uses best available:
# - Flash Attention 2 (if installed)
# - SDPA (Scaled Dot-Product Attention)
# - Fallback to standard attention
```

---

## Performance Tracking

### Before Optimizations (October 2025)
- **Speed:** ~18 sec/step
- **Time/epoch:** ~70 hours
- **Bottleneck:** Cosmos VAE encoding + double image processing

### After Double Image Processing Fix
- **Expected speed:** ~6-9 sec/step
- **Expected time/epoch:** ~24-35 hours
- **Status:** Implemented, awaiting benchmark

### After Full Optimization (Target)
- **Target speed:** ~1-2 sec/step
- **Target time/epoch:** ~4-8 hours
- **Requirements:** Pre-computed latents + dataloader opts + batch size tuning

---

## Implementation Roadmap

1. ✅ **Fix double image processing** (Done)
2. ✅ **Add eval_batch_size** (Done)
3. 🔥 **Pre-compute Cosmos latents** (High priority, ~1-2 days engineering)
4. 🔶 **Dataloader optimizations** (Easy config change, <1 hour)
5. 🔶 **Investigate gradient explosion** (Research task, ~half day)
6. 🔷 **Increase batch size** (Test + tune, ~1 hour)
7. 🔷 **Image loading optimization** (Optional, ~half day)

**Recommended order:** 1 → 2 → 3 → 4 → 5 → 6 → 7

---

## Profiling Training Performance

TheWorld includes integrated PyTorch profiling support to identify performance bottlenecks empirically.

### Quick Start

**Profile training for 10 steps:**
```bash
accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \
    scripts/train_hf.py \
    --config configs/llava_pretrain_full.json \
    --profile \
    --profile_steps 10 \
    --max_steps 15
```

This will:
- Skip first step (initialization)
- Warmup for 2 steps
- Profile next 10 steps
- Save traces to `checkpoints/llava_pretrain_full/profile_traces/`

### View Results

**Option 1: TensorBoard (recommended)**
```bash
tensorboard --logdir checkpoints/llava_pretrain_full/profile_traces
```

Navigate to http://localhost:6006 and click the "PROFILE" tab to see:
- Timeline view of GPU/CPU operations
- Operation breakdown by time
- Memory usage patterns
- Kernel launch sequences

**Option 2: Chrome Trace Viewer**
1. Open `chrome://tracing` in Chrome browser
2. Load `checkpoints/llava_pretrain_full/profile_traces/*.pt.trace.json`
3. Visualize exact timeline with microsecond precision

**Option 3: Text Report**
```bash
python scripts/analyze_profile.py \
    --trace_dir checkpoints/llava_pretrain_full/profile_traces \
    --output profile_report.md
```

Generates markdown report with:
- Top 20 operations by total time
- Bottleneck categorization (Cosmos, Gemma, Data Loading, etc.)
- Optimization recommendations
- Per-trace breakdown

### Profiler Configuration

The profiler is configured with:
- **Activities:** CPU operations + CUDA kernels
- **Schedule:** wait=1, warmup=2, active=10 (configurable)
- **record_shapes:** Captures tensor shapes for each operation
- **profile_memory:** Tracks memory allocations/deallocations
- **with_stack:** Records call stacks for debugging

### What to Look For

**1. Cosmos VAE Encoding Time**
- Look for: `WanEncoder3d`, `vae.encode`, `latent_dist`
- Expected: 50-100ms per image (if this is high, pre-compute latents!)
- Percentage: Should be <10% of total time with caching

**2. Data Loading Gaps**
- GPU idle time between steps
- Indicates dataloader is bottleneck
- Fix: Increase num_workers, add pin_memory, persistent_workers

**3. Image Preprocessing**
- Look for: `apply_chat_template`, SigLIP preprocessing
- Expected: Should be fast after double-processing fix
- If still high: Image loading or PIL operations are slow

**4. Attention Computation**
- Look for: `scaled_dot_product_attention`, `flash_attention`
- Should use hardware-accelerated attention automatically
- If using fallback attention: Check PyTorch/CUDA versions

**5. Forward/Backward Pass Balance**
- Forward should be ~60-70% of step time
- Backward should be ~30-40%
- If imbalanced: Check gradient computation settings

### Example Output

After profiling completes, you'll see a summary like:
```
==============================================================================
PROFILING SUMMARY (Top 20 operations by CUDA time)
==============================================================================
-----------------------  ------------  ------------  ------------  ------------
Name                     Self CPU %    Self CPU     Self CUDA %   Self CUDA
-----------------------  ------------  ------------  ------------  ------------
aten::conv3d             5.2%          450.2ms      8.1%          720.5ms
aten::linear             12.3%         1.05s        15.7%         1.40s
apply_chat_template      2.1%          180.3ms      1.2%          105.8ms
...
==============================================================================
```

### Advanced Usage

**Profile specific components:**
```bash
# Profile only first 5 steps (faster)
--profile --profile_steps 5 --max_steps 8

# Profile later in training (skip warmup)
--profile --profile_steps 10 --max_steps 110 --resume_from checkpoint-100
```

**Multi-GPU profiling:**
- Profiler runs on rank 0 only (main process)
- Captures DDP communication overhead
- Shows GPU synchronization points

**Memory profiling:**
- Check "MEMORY" view in TensorBoard
- Identifies memory leaks and allocation hotspots
- Useful for OOM debugging

### Integration Details

The profiler is implemented as a `ProfilerCallback` that:
1. Steps after each training step (`on_step_end`)
2. Prints summary on completion (`on_train_end`)
3. Saves traces via `tensorboard_trace_handler`

**Source files:**
- `scripts/train_hf.py`: ProfilerCallback class + integration
- `scripts/analyze_profile.py`: Trace analysis script

---

## Measurement & Verification

To verify optimizations, track:
- **Training speed:** steps/second (via TensorBoard)
- **Throughput:** samples/second = (batch_size × grad_accum × num_gpus) / time_per_step
- **GPU utilization:** `nvidia-smi dmon -s u` (should be >80%)
- **Data loading time:** Add timing to collate function
- **Convergence:** Loss should decrease at same rate (optimizations shouldn't hurt quality)

**Benchmark command:**
```bash
# Run for 100 steps, measure time
time PYTHONPATH=python:$PYTHONPATH accelerate launch \
  --config_file configs/accelerate/multi_gpu_ddp.yaml \
  scripts/train_hf.py \
  --config configs/llava_pretrain_full.json \
  --max_steps 100
```

---

## Questions / TODO

- [ ] Verify Flash Attention is actually being used (check model config)
- [ ] Profile with `py-spy` to confirm Cosmos encoding is the bottleneck
- [ ] Test pre-computed latents on small subset (100 samples)
- [ ] Benchmark dataloader optimizations independently
- [ ] Investigate why gradient norms are so high
- [ ] Test memory headroom for larger batch sizes

---

*Last updated: October 2025*
