#!/bin/bash
#SBATCH -J theworld-training               # Job name
#SBATCH -N1 --ntasks-per-node=4            # 1 node, 4 tasks (one per GPU)
#SBATCH --gres=gpu:H200:4                  # 4 H200 GPUs
#SBATCH --cpus-per-gpu=2                   # 2 CPUs per GPU for data loading
#SBATCH --mem=128G                         # 128GB total RAM
#SBATCH -t 3:00:00                         # 3 hour time limit
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
    echo "  Or use: sbatch --export=HF_TOKEN=\$(cat ~/.hf_token) scripts/train_slurm.sbatch ..."
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

# Use provided accelerate config or default to FSDP
if [ -n "$2" ]; then
    ACCELERATE_CONFIG="$2"
else
    ACCELERATE_CONFIG="configs/accelerate/multi_gpu_fsdp.yaml"
fi

# Validate files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "ERROR: Accelerate config not found: $ACCELERATE_CONFIG"
    exit 1
fi

echo "Training config: $CONFIG_FILE"
echo "Accelerate config: $ACCELERATE_CONFIG"

# Parse output_dir from config using Python
OUTPUT_DIR=$(python3 -c "
import json
with open('$CONFIG_FILE') as f:
    config = json.load(f)
print(config.get('output_dir', './checkpoints'))
")

echo "Output directory: $OUTPUT_DIR"

# Find latest checkpoint if exists
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
        echo "Found checkpoint: $RESUME_FROM"
        echo "Resuming training from checkpoint-$LATEST_CHECKPOINT"
    else
        echo "No checkpoints found. Starting training from scratch."
    fi
else
    echo "Output directory does not exist. Starting training from scratch."
fi

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

# Run training with Accelerate
echo ""
echo "============================================================"
echo "Starting training (4x H200)"
echo "============================================================"

if [ -n "$RESUME_FROM" ]; then
    # Resume from checkpoint
    uv run accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        scripts/train_hf.py \
        --config "$CONFIG_FILE" \
        --resume_from "$RESUME_FROM"
else
    # Start from scratch
    uv run accelerate launch \
        --config_file "$ACCELERATE_CONFIG" \
        scripts/train_hf.py \
        --config "$CONFIG_FILE"
fi

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Training completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
