# SLURM Training on ICE Cluster

## Quick Start

**One-time setup:**
```bash
# Store HuggingFace token (recommended for automatic Hub uploads)
echo 'hf_your_token_here' > ~/.hf_token
chmod 600 ~/.hf_token
```

**Submit training job:**
```bash
# Basic: H100 with 2 GPUs, default settings (4 hour time limit)
sbatch scripts/train_slurm.sh --gpu-type H100 configs/spatial_rgpt_training.json

# H200 with 4 GPUs and longer time limit
sbatch scripts/train_slurm.sh \
  --gpu-type H200 \
  --gpu-count 4 \
  --time 8:00:00 \
  configs/spatial_rgpt_training.json

# Custom accelerate config (FSDP for large models)
sbatch scripts/train_slurm.sh \
  --gpu-type H100 \
  configs/my_config.json \
  configs/accelerate/multi_gpu_fsdp.yaml
```

**Check job status:**
```bash
# View your jobs
squeue -u $USER

# View detailed job info
pace-job-summary <job-id>
```

**Monitor training:**
```bash
# View live logs (check job output for exact path)
tail -f logs/2025-11-19/11/theworld-spatial_rgpt_training-h100x2-*.out
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
sbatch scripts/train_slurm.sh --gpu-type H100 configs/spatial_rgpt_training.json

# Job times out after 4 hours...
# Checkpoint saved at: checkpoints/spatial_rgpt_training/checkpoint-7088

# Resubmit - automatically resumes from checkpoint-7088
sbatch scripts/train_slurm.sh --gpu-type H100 configs/spatial_rgpt_training.json
# Script automatically finds and resumes from latest checkpoint!
```

No need to specify `--resume_from` - the script handles this automatically.

## Configuration Options

### Command-Line Arguments

The `train_slurm.sh` wrapper accepts these arguments:

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--gpu-type TYPE` | **Yes** | - | GPU type (H100, H200, A100, etc.) |
| `--gpu-count N` | No | 2 | Number of GPUs |
| `--time HH:MM:SS` | No | 4:00:00 | Time limit (max 16 hours on ICE) |
| `--mem SIZE` | No | 256G | Memory allocation |
| `--email ADDRESS` | No | ksohrab3@gatech.edu | Email for notifications |
| `<config.json>` | **Yes** | - | Training configuration file |
| `[accelerate.yaml]` | No | multi_gpu_ddp.yaml | Accelerate config |

### Resource Allocation Examples

**Default (2 GPUs, 4 hours):**
```bash
sbatch scripts/train_slurm.sh --gpu-type H100 configs/my_config.json
# Allocates: 2x H100, 256GB RAM, 4 hour time limit
```

**Large job (4 GPUs, 12 hours, 512GB RAM):**
```bash
sbatch scripts/train_slurm.sh \
  --gpu-type H200 \
  --gpu-count 4 \
  --time 12:00:00 \
  --mem 512G \
  configs/my_config.json
```

**Maximum ICE time limit (16 hours):**
```bash
sbatch scripts/train_slurm.sh \
  --gpu-type H100 \
  --gpu-count 2 \
  --time 16:00:00 \
  configs/my_config.json
```

### Accelerate Configuration

**Default**: Uses `configs/accelerate/multi_gpu_ddp.yaml` (Data Distributed Parallel)

**Available accelerate configs:**
- `configs/accelerate/multi_gpu_ddp.yaml` - DDP (2-4 GPUs, faster, more memory)
- `configs/accelerate/multi_gpu_fsdp.yaml` - FSDP (4+ GPUs, memory efficient for large models)
- `configs/accelerate/single_gpu.yaml` - Single GPU (for testing)

**Examples:**
```bash
# Default (DDP with 2 GPUs)
sbatch scripts/train_slurm.sh --gpu-type H100 configs/my_config.json

# Use FSDP for larger models
sbatch scripts/train_slurm.sh \
  --gpu-type H100 \
  --gpu-count 4 \
  configs/my_config.json \
  configs/accelerate/multi_gpu_fsdp.yaml

# Single GPU test
sbatch scripts/train_slurm.sh \
  --gpu-type H100 \
  --gpu-count 1 \
  configs/smoke_test.json \
  configs/accelerate/single_gpu.yaml
```

### Email Notifications

The script sends emails for job events (start, completion, failure).

**Change email address:**
```bash
sbatch scripts/train_slurm.sh \
  --email your-email@gatech.edu \
  --gpu-type H100 \
  configs/my_config.json
```

**Default email:** `ksohrab3@gatech.edu` (can be customized via `--email` flag)

## Training Multiple Configs

**Sequential training** (one after another):
```bash
# Submit multiple jobs (will queue independently)
sbatch scripts/train_slurm.sh --gpu-type H100 configs/spatial_rgpt_training.json
sbatch scripts/train_slurm.sh --gpu-type H100 configs/vsr_training.json
```

**Chain jobs** (second starts after first completes):
```bash
# Submit first job, capture job ID
JOB1=$(sbatch --parsable scripts/train_slurm.sh \
  --gpu-type H100 \
  configs/spatial_rgpt_training.json)

# Submit second job dependent on first
sbatch --dependency=afterok:$JOB1 \
  scripts/train_slurm.sh \
  --gpu-type H100 \
  configs/vsr_training.json
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
- Verify path: `ls checkpoints/spatial_rgpt_training/`

## Example: Long Training Run

For training that exceeds the time limit, simply resubmit the same command:

```bash
# Initial submission (trains for up to 4 hours by default)
sbatch scripts/train_slurm.sh --gpu-type H100 configs/spatial_rgpt_training.json

# Wait for completion or timeout...

# Resubmit (automatically resumes from latest checkpoint!)
sbatch scripts/train_slurm.sh --gpu-type H100 configs/spatial_rgpt_training.json

# Repeat until training completes
```

**No changes needed** - the script automatically finds and resumes from the latest checkpoint.