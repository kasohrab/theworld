#!/bin/bash
#
# SpatialRGPT Batch Judging - SLURM Launcher
#
# Usage:
#   sbatch scripts/spatial/judge_slurm.sh
#   sbatch scripts/spatial/judge_slurm.sh --judge gemma
#   sbatch scripts/spatial/judge_slurm.sh --time 8:00:00
#
# Arguments:
#   --time HH:MM:SS      Time limit (default: 4:00:00)
#   --mem SIZE           Memory allocation (default: 128G)
#   --judge TYPE         Judge type: gpt-oss, gemma, gpt4 (default: gpt-oss)
#   --judge-model ID     Judge model ID (default: openai/gpt-oss-20b)
#   --batch-size N       Batch size for judging (default: 56)
#

# Defaults
TIME_LIMIT="4:00:00"
MEMORY="128G"
GPU_TYPE="H100"
EMAIL="ksohrab3@gatech.edu"
JUDGE="gpt-oss"
JUDGE_MODEL="openai/gpt-oss-20b"
BATCH_SIZE=56

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
        --judge)
            JUDGE="$2"
            shift 2
            ;;
        --judge-model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: sbatch scripts/spatial/judge_slurm.sh [OPTIONS]"
            echo ""
            echo "Optional Arguments:"
            echo "  --time HH:MM:SS      Time limit (default: 4:00:00)"
            echo "  --mem SIZE           Memory allocation (default: 128G)"
            echo "  --judge TYPE         Judge type: gpt-oss, gemma, gpt4 (default: gpt-oss)"
            echo "  --judge-model ID     Judge model ID (default: openai/gpt-oss-20b)"
            echo "  --batch-size N       Batch size for judging (default: 56)"
            echo ""
            echo "Examples:"
            echo "  sbatch scripts/spatial/judge_slurm.sh"
            echo "  sbatch scripts/spatial/judge_slurm.sh --judge gemma"
            echo "  sbatch scripts/spatial/judge_slurm.sh --time 8:00:00 --batch-size 32"
            exit 0
            ;;
        -*)
            echo "ERROR: Unknown option: $1"
            exit 1
            ;;
        *)
            echo "ERROR: Unexpected argument: $1"
            exit 1
            ;;
    esac
done

# Job name
JOB_NAME="spatial-judge-${JUDGE}"

# Create log directory
LOG_DATE=$(date +%Y-%m-%d)
LOG_HOUR=$(date +%H)
LOG_DIR="logs/${LOG_DATE}/${LOG_HOUR}"
mkdir -p "$LOG_DIR"

# Export for worker
export JUDGE
export JUDGE_MODEL
export BATCH_SIZE

echo "============================================================"
echo "SpatialRGPT Batch Judging - SLURM Job Submission"
echo "============================================================"
echo "Job Name: $JOB_NAME"
echo "GPU: 1x $GPU_TYPE"
echo "Memory: $MEMORY"
echo "Time Limit: $TIME_LIMIT"
echo "Judge: $JUDGE ($JUDGE_MODEL)"
echo "Batch Size: $BATCH_SIZE"
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
echo "SpatialRGPT Batch Judging - SLURM Job ${SLURM_JOB_ID}"
echo "Started at: $(date)"
echo "============================================================"

# Source common setup
source scripts/slurm/common_setup.sh

# Show what will be judged
PREDICTIONS_DIR="outputs/spatial_results/predictions"
if [ -d "$PREDICTIONS_DIR" ]; then
    echo "Predictions directory: $PREDICTIONS_DIR"
    echo "Prediction files found:"
    ls -1 "$PREDICTIONS_DIR"/*.jsonl 2>/dev/null | while read f; do
        echo "  - $(basename $f)"
    done
else
    echo "WARNING: Predictions directory not found: $PREDICTIONS_DIR"
fi
echo ""

# Build command
CMD="python scripts/spatial/batch_judge.py --judge $JUDGE --judge-model $JUDGE_MODEL --batch-size $BATCH_SIZE"

echo "Running: $CMD"
echo "============================================================"
echo ""

# Run batch judging
START_TIME=$(date +%s)
$CMD
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "Batch Judging Complete"
echo "============================================================"
echo "Status: $(if [ $EXIT_CODE -eq 0 ]; then echo 'SUCCESS'; else echo 'FAILED'; fi)"
echo "Duration: $((DURATION / 60))m $((DURATION % 60))s"
echo "============================================================"

exit $EXIT_CODE
SBATCH_SCRIPT_EOF
