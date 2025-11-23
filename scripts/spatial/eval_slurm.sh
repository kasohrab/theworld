#!/bin/bash
#
# SpatialRGPT Batch Evaluation - SLURM Launcher
#
# Usage:
#   sbatch scripts/spatial/eval_slurm.sh models.txt
#   sbatch scripts/spatial/eval_slurm.sh --time 8:00:00 models.txt
#
# Arguments:
#   --time HH:MM:SS      Time limit (default: 4:00:00)
#   --mem SIZE           Memory allocation (default: 128G)
#   --max-samples N      Max samples per model (default: 0 = all)
#   --min-new-tokens N   Minimum tokens to generate (default: 0 = no minimum)
#   --output-dir PATH    Output directory (default: outputs/spatial_results)
#   <models.txt>         Text file with HuggingFace model IDs (one per line) - REQUIRED
#

# Defaults
TIME_LIMIT="4:00:00"
MEMORY="128G"
GPU_TYPE="H100"
EMAIL="ksohrab3@gatech.edu"
MODELS_FILE=""
MAX_SAMPLES=0
MIN_NEW_TOKENS=0
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --min-new-tokens)
            MIN_NEW_TOKENS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: sbatch scripts/spatial/eval_slurm.sh [OPTIONS] <models.txt>"
            echo ""
            echo "Required Arguments:"
            echo "  <models.txt>         Text file with HuggingFace model IDs"
            echo ""
            echo "Optional Arguments:"
            echo "  --time HH:MM:SS      Time limit (default: 4:00:00)"
            echo "  --mem SIZE           Memory allocation (default: 128G)"
            echo "  --max-samples N      Max samples per model (default: 0 = all)"
            echo "  --min-new-tokens N   Minimum tokens to generate (default: 0 = no minimum)"
            echo "  --output-dir PATH    Output directory (default: outputs/spatial_results)"
            echo ""
            echo "Examples:"
            echo "  sbatch scripts/spatial/eval_slurm.sh models.txt"
            echo "  sbatch scripts/spatial/eval_slurm.sh --time 8:00:00 --max-samples 100 models.txt"
            echo "  sbatch scripts/spatial/eval_slurm.sh --min-new-tokens 10 models.txt"
            echo "  sbatch scripts/spatial/eval_slurm.sh --output-dir results/experiment1 models.txt"
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
        *)
            MODELS_FILE="$1"
            shift
            ;;
    esac
done

# Validate
if [ -z "$MODELS_FILE" ]; then
    echo "ERROR: models.txt file is required"
    echo "Run with --help for usage"
    exit 1
fi

if [ ! -f "$MODELS_FILE" ]; then
    echo "ERROR: Models file not found: $MODELS_FILE"
    exit 1
fi

# Job name from models file
MODELS_BASENAME=$(basename "$MODELS_FILE" .txt)
JOB_NAME="spatial-eval-${MODELS_BASENAME}"

# Create log directory
LOG_DATE=$(date +%Y-%m-%d)
LOG_HOUR=$(date +%H)
LOG_DIR="logs/${LOG_DATE}/${LOG_HOUR}"
mkdir -p "$LOG_DIR"

# Export for worker
export MODELS_FILE
export MAX_SAMPLES
export MIN_NEW_TOKENS
export OUTPUT_DIR

echo "============================================================"
echo "SpatialRGPT Batch Evaluation - SLURM Job Submission"
echo "============================================================"
echo "Job Name: $JOB_NAME"
echo "GPU: 1x $GPU_TYPE"
echo "Memory: $MEMORY"
echo "Time Limit: $TIME_LIMIT"
echo "Models File: $MODELS_FILE"
echo "Max Samples: $MAX_SAMPLES (0 = all)"
echo "Min New Tokens: $MIN_NEW_TOKENS (0 = no minimum)"
echo "Output Dir: ${OUTPUT_DIR:-outputs/spatial_results (default)}"
echo "Log: ${LOG_DIR}/${JOB_NAME}-<jobid>.out"
echo "============================================================"
echo ""

# Submit job
sbatch \
    --job-name="$JOB_NAME" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gres="gpu:${GPU_TYPE}:1" \
    --cpus-per-gpu=4 \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="${LOG_DIR}/${JOB_NAME}-%j.out" \
    --mail-type=BEGIN,END,FAIL \
    --mail-user="$EMAIL" \
    --export=ALL \
    <<'SBATCH_SCRIPT_EOF'
#!/bin/bash

echo "============================================================"
echo "SpatialRGPT Batch Evaluation - SLURM Job ${SLURM_JOB_ID}"
echo "Started at: $(date)"
echo "============================================================"

# Source common setup
source scripts/slurm/common_setup.sh

# Validate models file
if [ ! -f "$MODELS_FILE" ]; then
    echo "ERROR: Models file not found: $MODELS_FILE"
    exit 1
fi
echo "Models file: $MODELS_FILE"
echo "Models to evaluate:"
cat "$MODELS_FILE" | grep -v '^#' | grep -v '^$' | while read model; do
    echo "  - $model"
done
echo ""

# Build command
CMD="python scripts/spatial/batch_eval.py $MODELS_FILE"
if [ "$MAX_SAMPLES" -gt 0 ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi
if [ "$MIN_NEW_TOKENS" -gt 0 ]; then
    CMD="$CMD --min-new-tokens $MIN_NEW_TOKENS"
fi
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

echo "Running: $CMD"
echo "============================================================"
echo ""

# Run batch evaluation
START_TIME=$(date +%s)
$CMD
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Batch Evaluation Complete"
echo "============================================================"
echo "Status: $(if [ $EXIT_CODE -eq 0 ]; then echo 'SUCCESS'; else echo 'FAILED'; fi)"
echo "Duration: $((DURATION / 60))m $((DURATION % 60))s"
echo "============================================================"

exit $EXIT_CODE
SBATCH_SCRIPT_EOF
