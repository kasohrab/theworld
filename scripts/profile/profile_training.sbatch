#!/bin/bash
#SBATCH -J theworld-profile               # Job name
#SBATCH -N1 --ntasks-per-node=1           # 1 task (single GPU profiling)
#SBATCH --gres=gpu:H200:1                 # Single H200 GPU
#SBATCH --cpus-per-gpu=4                  # 4 CPUs for data loading
#SBATCH --mem=128G                        # 128GB RAM
#SBATCH -t 0:30:00                        # 30 minute time limit
#SBATCH -o logs/profile-%j.out            # Output file with job ID
#SBATCH --mail-type=BEGIN,END,FAIL        # Email notifications
#SBATCH --mail-user=ksohrab3@gatech.edu   # Email address

echo "============================================================"
echo "TheWorld Training Profiler - SLURM Job ${SLURM_JOB_ID}"
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
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not found')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo "============================================================"
echo ""

# Check for HF_TOKEN
echo "============================================================"
echo "HuggingFace Token Setup"
echo "============================================================"
if [ -z "$HF_TOKEN" ]; then
    # Try to read from ~/.hf_token if it exists
    if [ -f "$HOME/.hf_token" ]; then
        echo "Found ~/.hf_token, loading token..."
        export HF_TOKEN=$(cat "$HOME/.hf_token")
        echo "✓ HF_TOKEN loaded from ~/.hf_token"
    else
        echo "⚠ HF_TOKEN not found (set manually or create ~/.hf_token)"
    fi
else
    echo "✓ HF_TOKEN is set"
fi
echo "============================================================"
echo ""

# Parse arguments
CONFIG_FILE="${1:-configs/profile.json}"
PROFILE_STEPS="${2:-3}"

# Validate config file exists
echo ""
echo "============================================================"
echo "Configuration Validation"
echo "============================================================"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Usage: sbatch scripts/profile_training.sbatch [config.json] [profile_steps]"
    echo ""
    echo "Arguments:"
    echo "  config.json     - Profiling configuration (default: configs/profile.json)"
    echo "  profile_steps   - Number of steps to profile (default: 3)"
    exit 1
fi
echo "✓ Config found: $CONFIG_FILE"

# Verify profiling script exists
if [ ! -f "scripts/profile/profile_training.py" ]; then
    echo "ERROR: scripts/profile/profile_training.py not found!"
    exit 1
fi
echo "✓ Profiling script found: scripts/profile/profile_training.py"

echo ""

# Parse configuration and display summary
echo "============================================================"
echo "Profiling Configuration"
echo "============================================================"

python3 << EOF
import json

config_file = '$CONFIG_FILE'
with open(config_file) as f:
    config = json.load(f)

# Display key configuration items
print(f"Model: {config.get('model_name', 'Not specified')}")
print(f"Cosmos Model: {config.get('cosmos_model_name', 'Not specified')}")
print(f"Dataset: {config.get('dataset_name', 'Not specified')}")
print(f"Num Samples: {config.get('num_samples', 'Not specified')}")
print(f"Batch Size: {config.get('batch_size', 'Not specified')}")
print(f"Mixed Precision: {config.get('mixed_precision', 'Not specified')}")
print(f"Gradient Checkpointing: {config.get('use_gradient_checkpointing', False)}")
print(f"World Steps: {config.get('num_world_steps', 0)}")

# Freezing configuration
print(f"\nTrainable Components:")
print(f"  Gemma Vision: {'TRAINABLE' if not config.get('freeze_gemma_vision', True) else 'frozen'}")
print(f"  Gemma Language: {'TRAINABLE' if not config.get('freeze_gemma_language', True) else 'frozen'}")
print(f"  Cosmos VAE: {'TRAINABLE' if not config.get('freeze_cosmos_vae', True) else 'frozen'}")
print(f"  Projection Layers: ALWAYS TRAINABLE")

output_dir = config.get('output_dir', './checkpoints/profiling')
print(f"\nOutput Directory: {output_dir}")
print(f"Profile Steps: $PROFILE_STEPS")
EOF

echo ""
echo "============================================================"

# Print GPU information
echo ""
echo "============================================================"
echo "GPU Information:"
echo "============================================================"
nvidia-smi

# Set HuggingFace cache directory
export HF_HOME="$HOME/.cache/huggingface"
echo ""
echo "HF_HOME: $HF_HOME"

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints/profiling

# Record profiling start time
PROFILE_START=$(date +%s)

# Run profiling
echo ""
echo "============================================================"
echo "Starting Profiling (Single H200)"
echo "============================================================"
echo "Start time: $(date)"
echo "Profile steps: $PROFILE_STEPS"
echo ""

PYTHONPATH=/storage/ice1/7/7/ksohrab3/theworld/python:$PYTHONPATH \
uv run python scripts/profile/profile_training.py \
    --config "$CONFIG_FILE" \
    --profile-steps "$PROFILE_STEPS"

EXIT_CODE=$?
PROFILE_END=$(date +%s)
PROFILE_DURATION=$((PROFILE_END - PROFILE_START))
PROFILE_MINUTES=$((PROFILE_DURATION / 60))

echo ""
echo "============================================================"
echo "Profiling Summary"
echo "============================================================"
echo "Status: $(if [ $EXIT_CODE -eq 0 ]; then echo '✓ SUCCESS'; else echo '✗ FAILED'; fi)"
echo "Exit code: $EXIT_CODE"
echo "Duration: ${PROFILE_MINUTES}m ${PROFILE_DURATION}s"

# Check for profiling output (find the timestamped directory for this job)
if [ -d "checkpoints/profiling" ]; then
    # Find the most recent directory matching this job ID
    OUTPUT_DIR=$(find checkpoints/profiling -maxdepth 1 -type d -name "*_${SLURM_JOB_ID}" 2>/dev/null | head -n 1)

    # If no job-specific directory found, try finding the most recent one
    if [ -z "$OUTPUT_DIR" ] || [ ! -d "$OUTPUT_DIR" ]; then
        OUTPUT_DIR=$(ls -td checkpoints/profiling/*/ 2>/dev/null | head -n 1 | sed 's:/$::')
    fi

    if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR/traces" ]; then
        echo ""
        echo "Profiling output:"
        echo "  ✓ Traces directory: $OUTPUT_DIR/traces/"

        # Count trace files
        TRACE_COUNT=$(find "$OUTPUT_DIR/traces" -type f -name "*.pt.trace.json" 2>/dev/null | wc -l)
        if [ $TRACE_COUNT -gt 0 ]; then
            echo "  ✓ Found $TRACE_COUNT trace file(s)"
            echo ""
            echo "View results:"
            echo "  TensorBoard: tensorboard --logdir $OUTPUT_DIR/traces"
            echo "  Chrome trace: Open $OUTPUT_DIR/traces/*.pt.trace.json in chrome://tracing"
        fi
    fi
fi

# Warning if failed
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "⚠ Profiling failed! Check logs for details:"
    echo "  Logs: logs/profile-${SLURM_JOB_ID}.out"
fi

echo "============================================================"

exit $EXIT_CODE
