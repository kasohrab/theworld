#!/bin/bash
# Launch script for full DataComp training in 3-hour windows
#
# This script is designed for environments with 3-hour job limits (e.g., SLURM).
# It automatically resumes from the latest checkpoint and saves frequently.
#
# Training configuration:
# - Full DataComp-Small dataset (12.8M samples)
# - 1 epoch = ~800,000 steps total
# - Checkpoint every 250 steps (~12.5 minutes)
# - Each 3-hour run completes ~14 checkpoints (~230 runs needed for full training)
#
# Usage:
#   export HF_TOKEN=hf_your_token_here
#   bash scripts/launch_datacomp_3h_window.sh
#
# For SLURM:
#   sbatch --time=03:00:00 --gres=gpu:2 scripts/launch_datacomp_3h_window.sh

set -e  # Exit on error

echo "=========================================="
echo "TheWorld - DataComp Full Training"
echo "3-Hour Window Mode"
echo "=========================================="
echo ""

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN environment variable not set!"
    echo "Please set it with: export HF_TOKEN=hf_your_token_here"
    exit 1
fi

echo "✓ HF_TOKEN is set"

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
echo "✓ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""

# Configuration
CONFIG_FILE="configs/datacomp_production.json"
CHECKPOINT_DIR="./checkpoints/datacomp_production"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/datacomp_3h_$(date +%Y%m%d_%H%M%S).log"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Log file: $LOG_FILE"
echo "  Hub model: kasohrab/theworld-datacomp-projection"
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠ WARNING: nvidia-smi not found, cannot verify GPU availability"
else
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
fi

# Check for existing checkpoints
LATEST_CHECKPOINT=""
if [ -d "$CHECKPOINT_DIR" ]; then
    # Find the latest checkpoint by looking for checkpoint-* directories
    LATEST_CHECKPOINT=$(ls -td "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | head -1 || echo "")

    if [ -n "$LATEST_CHECKPOINT" ]; then
        # Extract step number from checkpoint name
        STEP_NUM=$(basename "$LATEST_CHECKPOINT" | sed 's/checkpoint-//')
        echo "✓ Found existing checkpoint: $LATEST_CHECKPOINT (step $STEP_NUM)"
        echo "  Will automatically resume from this checkpoint"
        echo "  Estimated progress: $STEP_NUM / 800,000 steps ($(awk "BEGIN {printf \"%.2f\", $STEP_NUM/800000*100}")%)"
    else
        echo "✓ No existing checkpoints found, starting fresh training"
    fi
else
    echo "✓ No checkpoint directory found, starting fresh training"
fi

echo ""
echo "Training Info:"
echo "  Dataset: DataComp-Small (12.8M samples)"
echo "  Total steps: ~800,000"
echo "  Checkpoint frequency: Every 250 steps (~12.5 min)"
echo "  Expected per 3-hour run: ~14 checkpoints (3,500 steps)"
echo "  Estimated runs to complete: ~230 × 3 hours"
echo ""

# Calculate time until job limit (if running in SLURM)
if [ -n "$SLURM_JOB_ID" ]; then
    echo "SLURM Job Info:"
    echo "  Job ID: $SLURM_JOB_ID"
    echo "  Time limit: 3 hours"
    echo ""
fi

echo "Starting training..."
echo "Press Ctrl+C to stop (latest checkpoint will be preserved)"
echo ""
echo "=========================================="
echo ""

# Run training with automatic resume
# The trainer will automatically find and resume from the latest checkpoint
uv run python scripts/train_hf.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "⚠ Training exited with code $EXIT_CODE"
fi
echo "=========================================="
echo ""

# Show final status
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -td "$CHECKPOINT_DIR"/checkpoint-* 2>/dev/null | head -1 || echo "")
    if [ -n "$LATEST_CHECKPOINT" ]; then
        STEP_NUM=$(basename "$LATEST_CHECKPOINT" | sed 's/checkpoint-//')
        echo "Final checkpoint: $LATEST_CHECKPOINT"
        echo "Progress: $STEP_NUM / 800,000 steps ($(awk "BEGIN {printf \"%.2f\", $STEP_NUM/800000*100}")%)"
    fi
fi

echo "Checkpoints: $CHECKPOINT_DIR"
echo "Hub model: https://huggingface.co/kasohrab/theworld-datacomp-projection"
echo "Training log: $LOG_FILE"
echo ""

# Suggest next action
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training is complete! 🎉"
    echo "Next steps:"
    echo "  1. Evaluate model: python scripts/evaluate_blink.py"
    echo "  2. Test inference: python examples/inference.py"
else
    echo "To resume training, simply run this script again:"
    echo "  bash scripts/launch_datacomp_3h_window.sh"
fi
