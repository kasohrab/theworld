# Troubleshooting Guide

Common issues and their solutions when working with TheWorld.

## Installation Issues

### "uv: command not found"

**Problem:** uv package manager not installed

**Solution:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart shell or reload profile
source ~/.bashrc  # or ~/.zshrc
```

### "Permission denied" during setup

**Problem:** Script not executable

**Solution:**
```bash
chmod +x scripts/setup.sh
bash scripts/setup.sh
```

### "cosmos_guardrail installation failed"

**Problem:** Cosmos safety checker has complex dependencies

**Solution:**
```bash
# Install manually
uv pip install cosmos_guardrail

# If still fails, try with pip
pip install cosmos_guardrail
```

---

## Memory Issues

### "CUDA out of memory" during inference

**Problem:** Model too large for GPU

**Solution 1: Use bfloat16**
```python
model = TheWorld.from_pretrained(
    "username/theworld-datacomp",
    dtype=torch.bfloat16,  # ← Saves ~50% memory
    device_map="auto"
)
```

**Solution 2: Reduce world steps**
```python
# Use fewer world tokens
outputs = model.generate(**inputs, num_world_steps=0)  # Minimal tokens
```

**Solution 3: Clear cache**
```python
import torch
torch.cuda.empty_cache()
```

**Solution 4: Reduce batch size**
```python
# Process one image at a time
for image in images:
    outputs = model.generate(...)
```

### "CUDA out of memory" during training

**Problem:** Training requires more memory than inference

**Solution 1: Reduce batch size**
```json
{
  "batch_size": 2,  # ← Reduce
  "gradient_accumulation_steps": 8  # ← Increase to maintain effective batch size
}
```

**Solution 2: Enable gradient checkpointing**
```json
{
  "use_gradient_checkpointing": true
}
```

**Solution 3: Use smaller model**
- Try Gemma 2B instead of 4B
- Reduce max_world_steps

---

## Model Loading Issues

### "HuggingFace token required"

**Problem:** Need authentication to download models

**Solution:**
```bash
export HF_TOKEN=hf_your_token_here
```

Get token at: https://huggingface.co/settings/tokens

### "Model download slow"

**Problem:** Models are large (~10-15GB)

**Solutions:**
- **First time:** Be patient, may take 10-30 minutes
- **Subsequent runs:** Uses cached models (fast)
- **Check connection:** Ensure stable internet
- **Use local files:** Set `local_files_only=True` after first download

### "Cannot load checkpoint"

**Problem:** Checkpoint format incompatible

**Solution: Use correct loading method**
```python
# For full model checkpoints
model = TheWorld.from_pretrained("./checkpoints/checkpoint-1000")

# For Hub checkpoints
model = TheWorld.from_checkpoint_hub("username/theworld-datacomp")

# For base model + trained projection
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True
)
model.load_checkpoint("projection_weights.pt")
```

---

## Training Issues

### "Training loss not decreasing"

**Problem:** Model not learning

**Possible causes & solutions:**

**1. Learning rate too low**
```json
{
  "learning_rate": 0.0005  # Try increasing
}
```

**2. Projection layer not trainable**
```python
# Check trainable parameters
trainable, total, pct = model.get_trainable_parameters()
print(f"Trainable: {trainable:,} ({pct:.2f}%)")
# Should be > 0
```

**3. Labels masked incorrectly**
- Check that text tokens have valid labels
- World/vision tokens should be -100

**4. Gradient flow blocked**
```bash
# Check gradients during training
python tests/validation/test_gradient_flow.py
```

### "NaN loss during training"

**Problem:** Loss becomes NaN (not a number)

**Solutions:**

**1. Lower learning rate**
```json
{
  "learning_rate": 0.00001  # Try 10x smaller
}
```

**2. Check for inf/nan in data**
```python
# Add to training loop
import torch
if torch.isnan(loss) or torch.isinf(loss):
    print("WARNING: NaN/Inf loss detected!")
```

**3. Use mixed precision carefully**
```json
{
  "mixed_precision": "bf16"  # bfloat16 more stable than fp16
}
```

### "Training very slow"

**Problem:** Training taking too long

**Solutions:**

**1. Increase batch size** (if memory allows)
```json
{
  "batch_size": 8,  # ← Increase
  "gradient_accumulation_steps": 2  # ← Decrease
}
```

**2. Reduce world steps**
```json
{
  "num_world_steps": 0  # Single frame is faster
}
```

**3. Use multiple GPUs**
```bash
# Data parallel
torchrun --nproc_per_node=4 scripts/train_hf.py --config config.json

# DeepSpeed
deepspeed --num_gpus=4 scripts/train_hf.py --config config_deepspeed.json
```

---

## Inference Issues

### "Outputs are nonsensical"

**Problem:** Model generating gibberish

**Possible causes & solutions:**

**1. Model not trained**
```python
# Make sure you're loading a trained model
model = TheWorld.from_pretrained("username/theworld-datacomp")  # ← Trained
# NOT:
model = TheWorld.from_pretrained("google/gemma-3-4b-it")  # ← Base model
```

**2. Incorrect prompt format**
```python
# Use standard Gemma chat template
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "What is this?"}
    ]
}]
inputs = processor.apply_chat_template(messages, return_tensors="pt")
```

**3. Temperature too high**
```python
outputs = model.generate(
    **inputs,
    temperature=0.7,  # ← Lower if needed (0.1-0.5)
)
```

### "Inference very slow"

**Problem:** Generation taking too long

**Solutions:**

**1. Reduce world steps**
```python
outputs = model.generate(**inputs, num_world_steps=0)  # Faster
```

**2. Use greedy decoding**
```python
outputs = model.generate(
    **inputs,
    do_sample=False,  # Faster than sampling
)
```

**3. Reduce max_new_tokens**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,  # ← Lower if possible
)
```

### "Different results each time"

**Problem:** Non-deterministic outputs

**Solution: Use deterministic generation**
```python
import torch

# Set seed
torch.manual_seed(42)

# Use greedy decoding
outputs = model.generate(
    **inputs,
    do_sample=False,  # Deterministic
)
```

---

## Known Issues

### RetinaFace Gradient Bug

**Problem:** `cosmos_guardrail` dependency disables gradients globally

**Impact:** Would break training, but TheWorld automatically re-enables gradients

**Workaround:** Already handled in `TheWorld.__init__()`, but be aware if using Cosmos directly

**See:** `docs/architecture/implementation-notes.md` for details

### World Model Not Helping

**Problem:** TheWorld performs similarly to Gemma baseline

**Possible causes:**

1. **Task doesn't need world model** - Static image understanding may not benefit
2. **Need more training** - Projection layer needs sufficient data
3. **Need to unfreeze components** - Try Stage 2/3 training

**Solutions:**
- Test on temporal/spatial tasks (BLINK, SpatialRGPT)
- Train longer or on more data
- See [Multi-Stage Training](../training/multi-stage.md)

---

## Getting Help

### Check Diagnostics

```bash
# Get diagnostics info
make check

# Check CUDA
nvidia-smi

# Check Python/torch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### File an Issue

If you can't resolve the issue:

1. Check [GitHub Issues](https://github.com/your-username/theworld/issues)
2. Search for similar problems
3. File new issue with:
   - Error message
   - Steps to reproduce
   - System info (GPU, Python version, etc.)
   - Minimal code example

---

## Related Documentation

- [Getting Started](getting-started.md) - Initial setup
- [Inference Guide](inference.md) - Running inference
- [Training Guide](../training/README.md) - Training models
- [Architecture Overview](../architecture/overview.md) - Understanding the model
