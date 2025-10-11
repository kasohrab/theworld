#!/bin/bash
# Launch script for full DataComp-1B production training
#
# This script trains TheWorld on the full DataComp-1B dataset with:
# - Projection layer only training (1.72% of parameters)
# - Frequent checkpointing (every 500 steps)
# - Keep last 3 checkpoints only
# - Automatic HuggingFace Hub upload
# - Wandb logging for monitoring
#
# Usage:
#   bash scripts/launch_datacomp_production.sh

set -e  # Exit on error

echo "=========================================="
echo "TheWorld - DataComp Production Training"
echo "=========================================="
echo ""

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable not set!"
    echo "Please set it with: export HF_TOKEN=hf_your_token_here"
    exit 1
fi

echo "✓ HF_TOKEN is set"
echo ""

# Configuration
CONFIG_FILE="configs/datacomp_production.json"
LOG_FILE="logs/datacomp_production_$(date +%Y%m%d_%H%M%S).log"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Log file: $LOG_FILE"
echo "  Hub model: kasohrab/theworld-datacomp-projection"
echo "  Checkpoint strategy: Every 500 steps, keep last 3"
echo ""

# Check if resuming from checkpoint
if [ -d "./checkpoints/datacomp_production" ]; then
    echo "⚠ Found existing checkpoints in ./checkpoints/datacomp_production"
    echo "Do you want to resume from the latest checkpoint? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "✓ Will resume from latest checkpoint"
        RESUME_FLAG="--resume_from ./checkpoints/datacomp_production"
    else
        echo "✓ Starting fresh training (existing checkpoints will be overwritten)"
        RESUME_FLAG=""
    fi
else
    echo "✓ No existing checkpoints found, starting fresh"
    RESUME_FLAG=""
fi

echo ""
echo "Starting training..."
echo "Press Ctrl+C to stop (checkpoints will be saved)"
echo ""
echo "=========================================="
echo ""

# Run training with timeout and logging
# Note: Remove timeout if you want unlimited training time
uv run python scripts/train_hf.py \
    --config "$CONFIG_FILE" \
    $RESUME_FLAG \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo ""
echo "Checkpoints saved to: ./checkpoints/datacomp_production"
echo "Uploaded to Hub: https://huggingface.co/kasohrab/theworld-datacomp-projection"
echo "Training log: $LOG_FILE"
