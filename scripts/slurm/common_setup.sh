#!/bin/bash
#
# Common setup for SLURM worker scripts
# Source this file at the beginning of SLURM jobs
#
# Usage (in SLURM worker script):
#   source scripts/slurm/common_setup.sh
#

echo "============================================================"
echo "Environment Setup"
echo "============================================================"

# Change to submission directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Load required modules
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

# HuggingFace Token Setup
echo "============================================================"
echo "HuggingFace Token Setup"
echo "============================================================"
if [ -z "$HF_TOKEN" ]; then
    # Try to read from ~/.hf_token if it exists
    if [ -f "$HOME/.hf_token" ]; then
        echo "Found ~/.hf_token, loading token..."
        export HF_TOKEN=$(cat "$HOME/.hf_token")
        echo "HF_TOKEN loaded from ~/.hf_token"
    else
        echo "WARNING: HF_TOKEN not set and ~/.hf_token not found"
        echo "  Model downloads may fail"
    fi
else
    echo "HF_TOKEN is set"
fi
echo "============================================================"
echo ""

# Set HuggingFace cache directory
export HF_HOME="$HOME/.cache/huggingface"
echo "HF_HOME: $HF_HOME"

# Print GPU information
echo ""
echo "============================================================"
echo "GPU Information:"
echo "============================================================"
nvidia-smi
echo "============================================================"
echo ""
