#!/bin/bash
#
# TheWorld Training - Configurable SLURM Launcher
#
# Usage:
#   sbatch scripts/train_slurm.sh --gpu-type H100 --gpu-count 2 configs/my_config.json [configs/accelerate/multi_gpu_ddp.yaml]
#
# Arguments:
#   --gpu-type TYPE      GPU type (H100, H200, A100, etc.) - REQUIRED
#   --gpu-count N        Number of GPUs (default: 2)
#   --time HH:MM:SS      Time limit (default: 4:00:00)
#   --mem SIZE           Memory allocation (default: 256G)
#   --email ADDRESS      Email for notifications (default: ksohrab3@gatech.edu)
#   <config.json>        Training configuration file - REQUIRED
#   [accelerate.yaml]    Accelerate config (optional, default: configs/accelerate/multi_gpu_ddp.yaml)
#

# Parse command-line arguments
GPU_TYPE=""
GPU_COUNT=2
TIME_LIMIT="4:00:00"
MEMORY="256G"
EMAIL="ksohrab3@gatech.edu"
CONFIG_FILE=""
ACCELERATE_CONFIG=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --email)
            EMAIL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: sbatch scripts/train_slurm.sh --gpu-type H100 [OPTIONS] <config.json> [accelerate.yaml]"
            echo ""
            echo "Required Arguments:"
            echo "  --gpu-type TYPE      GPU type (H100, H200, A100, etc.)"
            echo "  <config.json>        Training configuration file"
            echo ""
            echo "Optional Arguments:"
            echo "  --gpu-count N        Number of GPUs (default: 2)"
            echo "  --time HH:MM:SS      Time limit (default: 4:00:00)"
            echo "  --mem SIZE           Memory allocation (default: 256G)"
            echo "  --email ADDRESS      Email for notifications (default: ksohrab3@gatech.edu)"
            echo "  [accelerate.yaml]    Accelerate config (default: configs/accelerate/multi_gpu_ddp.yaml)"
            echo ""
            echo "Examples:"
            echo "  # H100 with 2 GPUs (most common)"
            echo "  sbatch scripts/train_slurm.sh --gpu-type H100 configs/spatial_rgpt_training_channel.json"
            echo ""
            echo "  # H200 with 4 GPUs, longer time"
            echo "  sbatch scripts/train_slurm.sh --gpu-type H200 --gpu-count 4 --time 8:00:00 configs/spatial_rgpt_training.json"
            echo ""
            echo "  # Custom accelerate config"
            echo "  sbatch scripts/train_slurm.sh --gpu-type H100 configs/my_config.json configs/accelerate/multi_gpu_fsdp.yaml"
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
        *)
            # Positional arguments
            if [ -z "$CONFIG_FILE" ]; then
                CONFIG_FILE="$1"
            elif [ -z "$ACCELERATE_CONFIG" ]; then
                ACCELERATE_CONFIG="$1"
            else
                echo "ERROR: Too many positional arguments: $1"
                echo "Expected: <config.json> [accelerate.yaml]"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$GPU_TYPE" ]; then
    echo "ERROR: --gpu-type is required"
    echo "Run with --help for usage information"
    exit 1
fi

if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: config.json is required"
    echo "Run with --help for usage information"
    exit 1
fi

# Default accelerate config if not provided
if [ -z "$ACCELERATE_CONFIG" ]; then
    ACCELERATE_CONFIG="configs/accelerate/multi_gpu_ddp.yaml"
fi

# Validate config files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "ERROR: Accelerate config not found: $ACCELERATE_CONFIG"
    exit 1
fi

# Auto-derive job name from config file and GPU settings
# Example: configs/spatial_rgpt_training_channel.json + H100x2 → theworld-spatial_rgpt_training_channel-h100x2
CONFIG_BASENAME=$(basename "$CONFIG_FILE" .json)
GPU_TYPE_LOWER=$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')
JOB_NAME="theworld-${CONFIG_BASENAME}-${GPU_TYPE_LOWER}x${GPU_COUNT}"

# Display configuration summary
echo "============================================================"
echo "TheWorld Training - SLURM Job Submission"
echo "============================================================"
echo "Job Name: $JOB_NAME"
echo "GPU Configuration: ${GPU_COUNT}x ${GPU_TYPE}"
echo "Memory: $MEMORY"
echo "Time Limit: $TIME_LIMIT"
echo "Training Config: $CONFIG_FILE"
echo "Accelerate Config: $ACCELERATE_CONFIG"
echo "Email: $EMAIL"
echo "============================================================"
echo ""

# Export config paths for the worker script
export CONFIG_FILE
export ACCELERATE_CONFIG

# Submit job with dynamically generated SLURM parameters
sbatch \
    --job-name="$JOB_NAME" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres="gpu:${GPU_TYPE}:${GPU_COUNT}" \
    --cpus-per-gpu=4 \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="logs/slurm-%j.out" \
    --mail-type=BEGIN,END,FAIL \
    --mail-user="$EMAIL" \
    --export=ALL \
    <<'SBATCH_SCRIPT_EOF'
#!/bin/bash

# ============================================================
# TheWorld Training - SLURM Worker Script
# ============================================================
# This script is dynamically generated by train_slurm.sh
# All configuration is passed via environment variables
# ============================================================

echo "============================================================"
echo "TheWorld Training - SLURM Job ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
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

# Validate files exist
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

config_file = os.environ['CONFIG_FILE']
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
print(f"Projection Mode: {config.get('world_projection_mode', 'spatial')}")

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
import os
with open(os.environ['CONFIG_FILE']) as f:
    config = json.load(f)
print(config.get('output_dir', './checkpoints'))
")

echo ""
echo "============================================================"

# Pre-download SpatialRGPT Dataset JSON (for multi-GPU training)
echo ""
echo "============================================================"
echo "Pre-download SpatialRGPT Dataset JSON"
echo "============================================================"
python scripts/spatial/download_spatial_json.py --skip-if-exists
echo "✓ Dataset JSON ready at /tmp/openspatial/result_10_depth_convs.json"
echo ""

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

# Set HuggingFace cache directory to use cached models
# This prevents trying to download models on compute nodes
export HF_HOME="$HOME/.cache/huggingface"
echo ""
echo "HF_HOME: $HF_HOME"

# Record training start time
TRAIN_START=$(date +%s)

# Run training with Accelerate
echo ""
echo "============================================================"
echo "Starting Training (${SLURM_JOB_GPUS} GPUs)"
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
SBATCH_SCRIPT_EOF
