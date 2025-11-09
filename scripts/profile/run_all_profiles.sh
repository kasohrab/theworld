#!/bin/bash
# Run all three profiling configurations sequentially
# Usage: bash scripts/profile/run_all_profiles.sh

set -e  # Exit on error

echo "=================================="
echo "Running All Profiling Configurations"
echo "=================================="
echo ""

# Read HF token from ~/.hf_token
if [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN=$(cat "$HOME/.hf_token")
    echo "✓ Loaded HF_TOKEN from ~/.hf_token"
else
    echo "⚠ Warning: ~/.hf_token not found, HF_TOKEN may not be set"
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "✗ Error: HF_TOKEN is not set"
    echo "  Please create ~/.hf_token with your HuggingFace token"
    exit 1
fi

# Pre-download SpatialRGPT Dataset JSON
echo ""
echo "=================================="
echo "Pre-download SpatialRGPT Dataset JSON"
echo "=================================="
python scripts/spatial/download_spatial_json.py --skip-if-exists
echo "✓ Dataset JSON ready at /tmp/openspatial/result_10_depth_convs.json"
echo ""

# List of configs to profile
CONFIGS=(
    "configs/profile/profile_gemma.json:Gemma Baseline"
    "configs/profile/profile_theworld_projection.json:TheWorld (Projection Only)"
    "configs/profile/profile_theworld_full.json:TheWorld (Full Fine-tuning)"
)

# Run profiling for each config
for i in "${!CONFIGS[@]}"; do
    IFS=':' read -r config_path label <<< "${CONFIGS[$i]}"

    echo ""
    echo "=================================="
    echo "$((i+1))/${#CONFIGS[@]}: Profiling $label"
    echo "=================================="
    echo ""

    uv run python scripts/profile/profile_training.py --config "$config_path"
done

echo ""
echo "=================================="
echo "All Profiling Complete!"
echo "=================================="
echo ""
echo "Finding profiling directories..."
GEMMA_DIR=$(ls -td checkpoints/profiling/*_gemma 2>/dev/null | head -1)
THEWORLD_PROJ_DIR=$(ls -td checkpoints/profiling/*_theworld 2>/dev/null | grep -v full | head -1)
THEWORLD_FULL_DIR=$(ls -td checkpoints/profiling/*_theworld 2>/dev/null | grep full | head -1)

if [ -n "$GEMMA_DIR" ] && [ -n "$THEWORLD_PROJ_DIR" ] && [ -n "$THEWORLD_FULL_DIR" ]; then
    echo ""
    echo "=================================="
    echo "Generating Comparison Report"
    echo "=================================="
    echo ""
    uv run python scripts/profile/compare_profiles.py \
        "$GEMMA_DIR" \
        "$THEWORLD_PROJ_DIR" \
        "$THEWORLD_FULL_DIR"
else
    echo "Could not find all profiling directories for comparison"
    echo "  Gemma: $GEMMA_DIR"
    echo "  TheWorld (projection): $THEWORLD_PROJ_DIR"
    echo "  TheWorld (full): $THEWORLD_FULL_DIR"
fi

echo ""
echo "✓ Done!"
