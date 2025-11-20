# SLURM Quick Reference

## Setup

```bash
# Store HF token (one-time)
echo 'hf_your_token_here' > ~/.hf_token
chmod 600 ~/.hf_token
```

## Submit Jobs

```bash
# Custom accelerate config (DDP), make sure gpu count matches
sbatch scripts/train_slurm.sh \
  --gpu-type H100 \
  configs/my_config.json \
  configs/accelerate/multi_gpu_ddp.yaml

# All options
./scripts/train_slurm.sh \
  --gpu-type H100 \
  --gpu-count 2 \
  --time 12:00:00 \
  --mem 512G \
  --email your-email@gatech.edu \
  configs/my_config.json
```

## Monitor Jobs

```bash
# Check status
squeue -u $USER

# Job details
pace-job-summary <job-id>

# Live logs
tail -f logs/2025-11-19/11/theworld-*-<job-id>.out

# Cancel job
scancel <job-id>
```

## Resume Training

```bash
# Just resubmit same command (auto-resumes!)
./scripts/train_slurm.sh --gpu-type H100 configs/spatial_rgpt_channel_training_all_fixed_mlp.json configs/accelerate/multi_gpu_ddp.yaml 
```

## Check Resources

```bash
# GPU info
nvidia-smi

# Storage quota
pace-quota

# Job history
sacct -u $USER -S 2025-01-01
```

---

**See [SLURM Training Guide](slurm-ice.md) for detailed documentation.**
