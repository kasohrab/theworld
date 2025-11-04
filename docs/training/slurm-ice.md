# SLURM Training on ICE Cluster

## Quick Start

**Submit training job:**
```bash
# Set HuggingFace token (required for Hub uploads)
export HF_TOKEN=hf_your_token_here

# Submit job with config (uses FSDP by default)
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json

# Or specify custom accelerate config
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json configs/accelerate/multi_gpu_ddp.yaml
```

**Check job status:**
```bash
# View your jobs
squeue -u ksohrab3

# View detailed job info
pace-job-summary <job-id>
```

**Monitor training:**
```bash
# View live logs (replace with your job ID)
tail -f logs/slurm-<job-id>.out
```

**Cancel job:**
```bash
scancel <job-id>
```

## Automatic Checkpoint Resumption

The SLURM script automatically finds and resumes from the latest checkpoint:

1. **First run**: Starts training from scratch
2. **Subsequent runs**: Automatically detects latest checkpoint in `output_dir` and resumes
3. **Checkpoint format**: `checkpoint-100`, `checkpoint-200`, etc.

**Example workflow:**
```bash
# First submission - trains from scratch
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json

# Job times out after 3 hours...
# Checkpoint saved at: checkpoints/llava_pretrain_full/checkpoint-7088

# Resubmit - automatically resumes from checkpoint-7088
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json
```

## Resource Allocation

The script requests:
- **Nodes**: 1 node
- **GPUs**: 4x H200 (140GB each)
- **Memory**: 560GB total (140GB per GPU)
- **Time**: 3 hours
- **Tasks**: 4 (one per GPU for FSDP)

## Configuration

The SLURM script accepts two arguments:
1. **Training config** (required): JSON file with model/training settings
2. **Accelerate config** (optional): YAML file with distributed training settings

**Default**: Uses `configs/accelerate/multi_gpu_fsdp.yaml` (FSDP with 4 GPUs)

**Available accelerate configs:**
- `configs/accelerate/multi_gpu_fsdp.yaml` - FSDP (4 GPUs, memory efficient)
- `configs/accelerate/multi_gpu_ddp.yaml` - DDP (4 GPUs, faster but more memory)
- `configs/accelerate/single_gpu.yaml` - Single GPU (for testing)

**Examples:**
```bash
# Default (FSDP)
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json

# Use DDP instead
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json configs/accelerate/multi_gpu_ddp.yaml

# Single GPU test
sbatch scripts/train_slurm.sbatch configs/smoke_test.json configs/accelerate/single_gpu.yaml
```

## Email Notifications

The script sends emails for:
- Job start
- Job completion
- Job failure

**Update email address:**
Edit `scripts/train_slurm.sbatch` line 8:
```bash
#SBATCH --mail-user=your-email@gatech.edu
```

## Customizing Time Limit

**Change time limit** in `scripts/train_slurm.sbatch` line 5:
```bash
#SBATCH -t 3:00:00   # Format: HH:MM:SS or D-HH:MM:SS
```

Examples:
- 1 hour: `-t 1:00:00`
- 6 hours: `-t 6:00:00`
- 12 hours: `-t 12:00:00`
- 1 day: `-t 1-00:00:00`

**Note**: Maximum is 16 hours for GPU jobs on ICE.

## Training Multiple Configs

**Sequential training** (one after another):
```bash
# Submit multiple jobs
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json
sbatch scripts/train_slurm.sbatch configs/vsr_training.json
```

**Chain jobs** (second starts after first completes):
```bash
# Submit first job
JOB1=$(sbatch --parsable scripts/train_slurm.sbatch configs/llava_pretrain_full.json)

# Submit second job dependent on first
sbatch --dependency=afterok:$JOB1 scripts/train_slurm.sbatch configs/vsr_training.json
```

## Troubleshooting

### Job won't start
- **Check queue**: `squeue -u ksohrab3`
- **Check allocation**: ICE should auto-assign partitions
- **H200 availability**: H200s may be in high demand

### Job failed
```bash
# View job summary
pace-job-summary <job-id>

# Check logs
cat logs/slurm-<job-id>.out
```

### Out of memory
- Reduce `batch_size` in config
- Enable `use_gradient_checkpointing: true`
- Reduce `--mem-per-gpu` if needed

### Checkpoint not found
- Check `output_dir` in your config JSON
- Ensure previous job saved checkpoints (`save_steps` in config)
- Verify path: `ls checkpoints/llava_pretrain_full/`

## Example: Long Training Run

For training that takes >3 hours, split into multiple SLURM jobs:

```bash
# Initial submission
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json

# Wait for completion or timeout...

# Resubmit (auto-resumes from latest checkpoint)
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json

# Repeat until training completes
```

**Tip**: Monitor progress via logs or HuggingFace Hub to know when to resubmit.

## Advanced: Job Chaining for Full Training

Create a wrapper script to auto-resubmit until training completes:

```bash
#!/bin/bash
# submit_chain.sh - Auto-resubmit for long training

CONFIG="$1"
MAX_RESUBMITS=10  # Safety limit

for i in $(seq 1 $MAX_RESUBMITS); do
    echo "Submission $i/$MAX_RESUBMITS"
    JOB_ID=$(sbatch --parsable scripts/train_slurm.sbatch "$CONFIG")
    echo "Submitted job: $JOB_ID"

    # Wait for job to complete
    while squeue -j $JOB_ID 2>/dev/null | grep -q $JOB_ID; do
        sleep 60
    done

    # Check if training completed successfully
    # (This is simplified - add more robust completion checking)
    if grep -q "Training completed successfully" logs/slurm-$JOB_ID.out; then
        echo "Training completed!"
        exit 0
    fi

    echo "Job finished. Resubmitting..."
done

echo "Reached maximum resubmits. Check logs."
```

Usage:
```bash
bash submit_chain.sh configs/llava_pretrain_full.json
```
