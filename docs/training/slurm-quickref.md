# SLURM Quick Reference

## Setup

```bash
# Store HF token (one-time)
echo 'hf_your_token_here' > ~/.hf_token
chmod 600 ~/.hf_token

# Load token
export HF_TOKEN=$(cat ~/.hf_token)
```

## Submit Jobs

```bash
# Default (FSDP, 4 GPUs)
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json

# With DDP
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json configs/accelerate/multi_gpu_ddp.yaml

# Single GPU test
sbatch scripts/train_slurm.sbatch configs/smoke_test.json configs/accelerate/single_gpu.yaml

# Pass HF_TOKEN explicitly
sbatch --export=HF_TOKEN=$(cat ~/.hf_token) scripts/train_slurm.sbatch configs/llava_pretrain_full.json
```

## Monitor Jobs

```bash
# Check status
squeue -u ksohrab3

# Job details
pace-job-summary <job-id>

# Live logs
tail -f logs/slurm-<job-id>.out

# Cancel job
scancel <job-id>
```

## Resume Training

```bash
# Just resubmit (auto-resumes from latest checkpoint)
sbatch scripts/train_slurm.sbatch configs/llava_pretrain_full.json
```

## Job Chaining

```bash
# Submit second job after first completes
JOB1=$(sbatch --parsable scripts/train_slurm.sbatch configs/llava_pretrain_full.json)
sbatch --dependency=afterok:$JOB1 scripts/train_slurm.sbatch configs/vsr_training.json
```

## Check Resources

```bash
# GPU info on login node
nvidia-smi

# Check partition availability
pace-check-queue ice-gpu

# View job history
sacct -u ksohrab3 -S 2025-01-01
```
