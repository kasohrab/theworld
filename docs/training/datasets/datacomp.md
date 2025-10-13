# DataComp Production Training Plan

## Overview

Plan to run full-scale training on DataComp-Small dataset (100k samples) with TheWorld model (Gemma 3 4B + Cosmos 2B).

**Date Created**: 2025-10-11
**Status**: Ready to Execute
**Branch**: `users/kasra/canary`

## Current Configuration Analysis

### Dataset
- **Source**: DataComp-Small (`mlfoundations/datacomp_small`)
- **Samples**: 100,000 (configurable, full dataset has 12.8M)
- **Format**: Image URLs + text captions
- **Mode**: Non-streaming (load into memory)
- **Question template**: "Describe this image simply."

### Model Configuration
- **Architecture**: TheWorld (Gemma 3 4B + Cosmos 2B + projection layers)
- **Trainable params**: 76.3M / 4.43B (1.72% - projection + temporal embeddings only)
- **Frozen components**:
  - Gemma vision encoder (SigLIP)
  - Gemma language model (4B)
  - Cosmos VAE encoder (2B)

### Training Hyperparameters
- **Batch size per GPU**: 4
- **Gradient accumulation steps**: 4
- **Effective batch size**: 16
- **Learning rate**: 1e-4
- **Epochs**: 3
- **Total updates**: ~18,750 steps (100k samples × 3 epochs / 16 batch size)
- **Warmup steps**: 500
- **Weight decay**: 0.01
- **Gradient clipping**: 1.0
- **Mixed precision**: bfloat16

### Checkpointing & Logging
- **Save frequency**: Every 50 steps
- **Keep checkpoints**: Last 3 only
- **Logging frequency**: Every 50 steps
- **Weights & Biases**: Enabled (`theworld-datacomp` project)
- **TensorBoard**: Enabled
- **Hub uploads**: Every save to `kasohrab/theworld-datacomp-projection`

## Required Configuration Updates

### 1. Update `configs/datacomp_production.json`

**Critical changes needed**:
```json
{
  "device": "cuda:1",  // Place Cosmos on GPU 1 (Gemma auto-splits across both)
  "load_full_cosmos_pipeline": true,  // Required - custom VAE architecture
  "num_samples": 100000,  // Or null for full 12.8M dataset
  "streaming": false,  // Non-streaming mode as requested
  "use_gradient_checkpointing": false  // Optional: set to true if OOM occurs
}
```

**Rationale**:
- `device: "cuda:1"` ensures Cosmos loads on GPU 1 while Gemma splits across both GPUs (verified in smoke test)
- `load_full_cosmos_pipeline: true` is mandatory - Cosmos VAE uses custom architecture
- `streaming: false` loads dataset into memory (faster but requires ~20-30GB RAM)
- Gradient checkpointing disabled by default since we're only training 1.72% of params

### 2. Create Training Launch Script

**File**: `scripts/launch_datacomp_training.sh`

```bash
#!/bin/bash
# DataComp production training launcher
# Expected runtime: ~48-60 hours for 100k samples

set -e

# Configuration
export HF_TOKEN="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"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Training settings
CONFIG="configs/datacomp_production.json"
LOG_FILE="logs/datacomp_training_$(date +%Y%m%d_%H%M%S).log"

# Create log directory
mkdir -p logs

# Launch training
echo "Starting DataComp production training..."
echo "Config: $CONFIG"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"

uv run python scripts/train_hf.py \
    --config $CONFIG \
    2>&1 | tee $LOG_FILE

echo "Training completed at: $(date)"
```

## Execution Plan

### Step 1: Update Configuration
1. Edit `configs/datacomp_production.json`:
   - Set `device: "cuda:1"`
   - Add `load_full_cosmos_pipeline: true`
   - Verify `num_samples: 100000` and `streaming: false`

### Step 2: Create Launch Script
1. Create `scripts/launch_datacomp_training.sh` with content above
2. Make executable: `chmod +x scripts/launch_datacomp_training.sh`

### Step 3: Launch Training
```bash
# Option A: Direct execution (blocks terminal)
bash scripts/launch_datacomp_training.sh

# Option B: Background with tmux (recommended for long jobs)
tmux new -s datacomp
bash scripts/launch_datacomp_training.sh
# Ctrl+B, D to detach
# tmux attach -t datacomp to reattach

# Option C: Background with nohup
nohup bash scripts/launch_datacomp_training.sh &
# Monitor: tail -f logs/datacomp_training_*.log
```

### Step 4: Monitor Training
- **Weights & Biases**: https://wandb.ai/[username]/theworld-datacomp
- **Local logs**: `tail -f logs/datacomp_training_*.log`
- **GPU usage**: `watch -n 1 nvidia-smi`
- **Checkpoints**: `ls -lh checkpoints/datacomp_production/`

### Step 5: Post-Training Verification
1. Check final checkpoint uploaded to Hub
2. Verify model loads correctly: `TheWorld.from_pretrained("kasohrab/theworld-datacomp-projection")`
3. Run evaluation on BLINK benchmark (see `docs/evaluation.md`)
4. Document results and commit

## Resource Requirements

### Hardware
- **GPUs**: 2× NVIDIA H200 (143GB each) ✅ Available
- **GPU Memory Usage**:
  - GPU 0: ~60-70GB (Gemma vision + first 11 layers)
  - GPU 1: ~60-70GB (Cosmos + remaining 23 layers)
- **System RAM**: ~30-40GB (dataset loading + dataloaders)
- **Disk Space**: ~50GB (checkpoints + logs)

### Time Estimates
- **100k samples × 3 epochs**: ~48-60 hours (~3 seconds per step × 18,750 steps)
- **Checkpointing overhead**: ~5-10 minutes every 50 steps
- **Hub upload time**: ~2-3 minutes per checkpoint (153MB each)
- **Total estimated time**: 2-3 days

### Network Requirements
- **Dataset download**: ~15-20GB (image downloads on-the-fly)
- **Hub uploads**: ~1-2GB total (checkpoints + training args)
- **W&B logging**: ~100-200MB (metrics and logs)

## Expected Outcomes

### Training Metrics
- **Initial loss**: ~20-25 (untrained projection)
- **Expected final loss**: ~5-10 (well-trained projection)
- **Gradient norms**: Should stabilize around 100-500
- **Learning rate schedule**: Linear warmup (500 steps) → linear decay

### Checkpoints
- **Location**: `checkpoints/datacomp_production/checkpoint-{step}/`
- **Hub location**: `kasohrab/theworld-datacomp-projection`
- **Size per checkpoint**: ~153MB (projection + temporal embeddings + optimizer state)
- **Total checkpoints saved**: Last 3 only (rotating)

### Model Capabilities (After Training)
- Better world-aware image captioning
- Improved understanding of spatial relationships
- Enhanced temporal reasoning (even with num_world_steps=0, learned from projection)

## Troubleshooting

### If OOM Occurs
1. Set `use_gradient_checkpointing: true` in config
2. Reduce `batch_size` from 4 to 2
3. Reduce `max_seq_length` from 2048 to 1536

### If Training Stalls
- Check W&B for loss spikes
- Verify dataloaders aren't stuck on image downloads
- Check `nvidia-smi` for GPU utilization

### If Checkpoints Fail to Upload
- Verify `HF_TOKEN` is set correctly
- Check Hub repo exists and is accessible
- Verify network connectivity

## Scaling Options

### For Full 12.8M Dataset
```json
{
  "num_samples": null,  // Use all samples
  "num_epochs": 1,      // Single epoch sufficient
  "save_steps": 500,    // Less frequent saves
  "logging_steps": 100  // Less frequent logging
}
```
- **Expected time**: ~25-30 days
- **Total steps**: ~800k steps
- **Requires**: Stable long-running environment (SLURM job, cloud VM)

### For Faster Iteration (10k samples)
```json
{
  "num_samples": 10000,
  "num_epochs": 5,      // More epochs for smaller dataset
  "save_steps": 25      // More frequent saves
}
```
- **Expected time**: ~5-6 hours
- **Use case**: Quick testing, hyperparameter tuning

## Next Steps After Training

1. **Evaluate on BLINK**: Run `scripts/evaluate_blink.py` on trained model
2. **Compare to baselines**: Gemma3 alone, random projection, ablations
3. **Document results**: Update `docs/results.md` with metrics
4. **Create model card**: Generate detailed card for Hub
5. **Prepare for Stage 2**: Plan unfreezing Gemma vision (if needed)

## References

- Architecture docs: `CLAUDE.md` and `docs/tokenization_and_special_tokens.md`
- Evaluation guide: `docs/evaluation.md`
- Dataset: https://huggingface.co/datasets/mlfoundations/datacomp_small
- Model Hub: https://huggingface.co/kasohrab/theworld-datacomp-projection
