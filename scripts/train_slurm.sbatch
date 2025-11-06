#!/bin/bash
#SBATCH -J theworld-training               # Job name
#SBATCH -N1 --ntasks-per-node=1             # 1 task per node (Accelerate handles GPUs)
#SBATCH --gres=gpu:H200:2                  # 4 H200 GPUs
#SBATCH --cpus-per-gpu=4                   # 2 CPUs per GPU for data loading
#SBATCH --mem=256G                         # total RAM
#SBATCH -t 4:00:00                         # time limit
#SBATCH -o logs/slurm-%j.out               # Output file with job ID
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
#SBATCH --mail-user=ksohrab3@gatech.edu    # Email address

echo "============================================================"
echo "TheWorld Training - SLURM Job ${SLURM_JOB_ID}"
echo "Started at: $(date)"
echo "============================================================"

# Change to submission directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Load required modules
echo ""
echo "============================================================"
echo "Environment Setup"
echo "============================================================"
echo "Loading anaconda3 module..."
module load anaconda3

# Activate conda environment if it exists
if [ -d "./env" ]; then
    echo "Activating conda environment at ./env..."
    conda activate ./env
else
    echo "No conda environment found at ./env (skipping)"
fi

# Activate Python virtual environment
if [ -d "./.venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
else
    echo "ERROR: .venv not found!"
    exit 1
fi

# Verify environment
echo ""
echo "Environment verification:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
echo "  Accelerate: $(python -c 'import accelerate; print(accelerate.__version__)' 2>/dev/null || echo 'Not found')"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo "============================================================"
echo ""

# Check for HF_TOKEN (required for Hub uploads)
echo "============================================================"
echo "HuggingFace Token Setup"
echo "============================================================"
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set!"
    echo ""
    echo "Recommended: Store token in secure file"
    echo "  Setup once:"
    echo "    echo 'hf_your_token_here' > ~/.hf_token"
    echo "    chmod 600 ~/.hf_token"
    echo ""
    echo "  Then before submitting job:"
    echo "    export HF_TOKEN=\$(cat ~/.hf_token)"
    echo ""
    echo "  Or use: sbatch scripts/train_slurm.sbatch <config.json> [accelerate_config.yaml]"
    echo "          (script auto-loads token from ~/.hf_token)"
    echo ""

    # Try to read from ~/.hf_token if it exists
    if [ -f "$HOME/.hf_token" ]; then
        echo "Found ~/.hf_token, loading token..."
        export HF_TOKEN=$(cat "$HOME/.hf_token")
        echo "✓ HF_TOKEN loaded from ~/.hf_token"
    else
        echo "⚠ Hub uploads will fail without HF_TOKEN"
        echo "  (Create ~/.hf_token or set HF_TOKEN environment variable)"
    fi
else
    echo "✓ HF_TOKEN is set"
fi
echo "============================================================"
echo ""

# Parse arguments
if [ -z "$1" ]; then
    echo "ERROR: No config file provided!"
    echo "Usage: sbatch scripts/train_slurm.sbatch <config.json> [accelerate_config.yaml]"
    echo ""
    echo "Arguments:"
    echo "  config.json             - Training configuration (required)"
    echo "  accelerate_config.yaml  - Accelerate config (optional, default: configs/accelerate/multi_gpu_fsdp.yaml)"
    exit 1
fi

CONFIG_FILE="$1"

# Use provided accelerate config or default to DDP
if [ -n "$2" ]; then
    ACCELERATE_CONFIG="$2"
else
    ACCELERATE_CONFIG="configs/accelerate/multi_gpu_ddp.yaml"
fi

# Validate files exist
echo ""
echo "============================================================"
echo "Configuration Validation"
echo "============================================================"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✓ Training config found: $CONFIG_FILE"

if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "ERROR: Accelerate config not found: $ACCELERATE_CONFIG"
    exit 1
fi
echo "✓ Accelerate config found: $ACCELERATE_CONFIG"

# Verify training script exists
if [ ! -f "scripts/train_hf.py" ]; then
    echo "ERROR: scripts/train_hf.py not found!"
    exit 1
fi
echo "✓ Training script found: scripts/train_hf.py"

echo ""

# Parse configuration and display summary
echo "============================================================"
echo "Training Configuration"
echo "============================================================"

python3 << 'EOF'
import json
import os

config_file = '$CONFIG_FILE'
with open(config_file) as f:
    config = json.load(f)

# Display key configuration items
print(f"Model: {config.get('model_name', 'Not specified')}")
print(f"Cosmos Model: {config.get('cosmos_model_name', 'Not specified')}")
print(f"Dataset: {config.get('dataset_name', 'Not specified')}")
print(f"Batch Size: {config.get('batch_size', 'Not specified')}")
print(f"Learning Rate: {config.get('learning_rate', 'Not specified')}")
print(f"Num Epochs: {config.get('num_epochs', 'Not specified')}")
print(f"Mixed Precision: {config.get('mixed_precision', 'Not specified')}")
print(f"World Steps: {config.get('num_world_steps', 0)}")

# Freezing configuration
print(f"\nTrainable Components:")
print(f"  Gemma Vision: {'TRAINABLE' if not config.get('freeze_gemma_vision', True) else 'frozen'}")
print(f"  Gemma Language: {'TRAINABLE' if not config.get('freeze_gemma_language', True) else 'frozen'}")
print(f"  Cosmos VAE: {'TRAINABLE' if not config.get('freeze_cosmos_vae', True) else 'frozen'}")
print(f"  Projection Layers: ALWAYS TRAINABLE")

output_dir = config.get('output_dir', './checkpoints')
print(f"\nOutput Directory: {output_dir}")
print(f"Save Steps: {config.get('save_steps', 'Not specified')}")
print(f"Hub Upload: {'Yes' if config.get('push_to_hub', False) else 'No'}")
if config.get('push_to_hub'):
    print(f"Hub Model ID: {config.get('hub_model_id', 'Not specified')}")
EOF

OUTPUT_DIR=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
print(config.get('output_dir', './checkpoints'))
")

echo ""
echo "============================================================"

# Find latest checkpoint if exists (automatically resume from most recent)
RESUME_FROM=""
if [ -d "$OUTPUT_DIR" ]; then
    echo "Checking for existing checkpoints in $OUTPUT_DIR..."

    # Find checkpoint directories (format: checkpoint-XXXX)
    LATEST_CHECKPOINT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | \
                        sed 's/.*checkpoint-//' | \
                        sort -n | \
                        tail -1)

    if [ -n "$LATEST_CHECKPOINT" ]; then
        RESUME_FROM="$OUTPUT_DIR/checkpoint-$LATEST_CHECKPOINT"
        echo ""
        echo "✓ Found latest checkpoint: checkpoint-$LATEST_CHECKPOINT"
        echo "  Full path: $RESUME_FROM"
        echo ""
        echo "Verifying checkpoint has training state files..."
        if [ -f "$RESUME_FROM/trainer_state.json" ] && [ -f "$RESUME_FROM/training_args.bin" ]; then
            echo "✓ Training state files found"
            echo "✓ Will resume training from step where it left off"
        else
            echo "⚠ WARNING: Missing training state files. May start from scratch."
            RESUME_FROM=""
        fi
    else
        echo "No checkpoints found. Starting training from scratch."
    fi
else
    echo "Output directory does not exist. Starting training from scratch."
fi

echo ""

# Print GPU information
echo ""
echo "============================================================"
echo "GPU Information:"
echo "============================================================"
nvidia-smi

# Set HuggingFace token if available
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "WARNING: HF_TOKEN not set. Hub uploads may fail."
    echo "Set with: export HF_TOKEN=hf_your_token"
fi

# Set HuggingFace cache directory to use cached models
# This prevents trying to download models on compute nodes
export HF_HOME="$HOME/.cache/huggingface"
echo "HF_HOME: $HF_HOME"

# Enable NCCL debugging for distributed training diagnostics
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
echo "NCCL debugging: enabled (NCCL_DEBUG=INFO)"

# Record training start time
TRAIN_START=$(date +%s)

# Run training with Accelerate
echo ""
echo "============================================================"
echo "Starting Training (4x H200)"
echo "============================================================"
echo "Start time: $(date)"

if [ -n "$RESUME_FROM" ]; then
    echo "Mode: RESUMING from checkpoint"
    echo "Checkpoint: $RESUME_FROM"
    echo ""
    # Resume from checkpoint
    uv run accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        scripts/train_hf.py \
        --config "$CONFIG_FILE" \
        --resume_from "$RESUME_FROM"
else
    echo "Mode: STARTING from scratch"
    echo ""
    # Start from scratch
    uv run accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        scripts/train_hf.py \
        --config "$CONFIG_FILE"
fi

EXIT_CODE=$?
TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
TRAIN_MINUTES=$((TRAIN_DURATION / 60))
TRAIN_HOURS=$((TRAIN_MINUTES / 60))

echo ""
echo "============================================================"
echo "Training Summary"
echo "============================================================"
echo "Status: $(if [ $EXIT_CODE -eq 0 ]; then echo '✓ SUCCESS'; else echo '✗ FAILED'; fi)"
echo "Exit code: $EXIT_CODE"
echo "Start time: $(date -d @$TRAIN_START)"
echo "End time: $(date -d @$TRAIN_END)"
echo "Duration: ${TRAIN_HOURS}h ${TRAIN_MINUTES}m ${TRAIN_DURATION}s"

# Print checkpoint statistics
if [ -d "$OUTPUT_DIR" ]; then
    CHECKPOINT_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | wc -l)
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        echo ""
        echo "Checkpoints saved:"
        find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort | tail -5 | while read checkpoint; do
            CKPT_SIZE=$(du -sh "$checkpoint" | cut -f1)
            CKPT_NAME=$(basename "$checkpoint")
            echo "  • $CKPT_NAME ($CKPT_SIZE)"
        done
    fi

    # Check for training logs
    if [ -f "$OUTPUT_DIR/trainer_state.json" ]; then
        echo ""
        echo "Training state: ✓ Found (can resume)"
    fi
fi

# Warning if failed
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "⚠ Training failed! Check logs for details:"
    echo "  Logs: logs/slurm-${SLURM_JOB_ID}.out"
fi

echo "============================================================"

exit $EXIT_CODE
