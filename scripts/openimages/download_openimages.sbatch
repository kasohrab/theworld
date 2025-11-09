#!/bin/bash
#SBATCH -J openimages-download            # Job name
#SBATCH -N1 --ntasks=1                    # 1 node, 1 task
#SBATCH --cpus-per-task=24                # 24 CPUs for parallel downloads
#SBATCH --mem=128G                         # 32GB RAM
#SBATCH -t 8:00:00                        # 8 hour time limit
#SBATCH -o logs/download-%j.out           # Output file with job ID
#SBATCH --mail-type=BEGIN,END,FAIL        # Email notifications
#SBATCH --mail-user=ksohrab3@gatech.edu   # Email address

echo "============================================================"
echo "OpenImages Download - SLURM Job ${SLURM_JOB_ID}"
echo "Started at: $(date)"
echo "============================================================"

# Change to submission directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Verify required files exist
if [ ! -f "downloader.py" ]; then
    echo "ERROR: downloader.py not found!"
    echo "Run: wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py"
    exit 1
fi

if [ ! -f "data/required_images.txt" ]; then
    echo "ERROR: data/required_images.txt not found!"
    echo "Run: python scripts/openimages/download_openimages.py --output data/required_images.txt"
    exit 1
fi

# Auto-detect resume point based on already downloaded images
echo ""
echo "============================================================"
echo "Detecting Resume Point"
echo "============================================================"
python scripts/openimages/find_download_resume_point.py

# Use resume file if it exists, otherwise use full list
if [ -f "data/resume_images.txt" ]; then
    IMAGE_LIST="data/resume_images.txt"
    echo "Using resume file: $IMAGE_LIST"
else
    IMAGE_LIST="data/required_images.txt"
    echo "Using full image list: $IMAGE_LIST"
fi

# Check Python and dependencies
echo ""
echo "============================================================"
echo "Environment Setup"
echo "============================================================"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Verify boto3 is installed
python -c "import boto3" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: boto3 not installed!"
    echo "Run: uv pip install boto3 botocore"
    exit 1
fi
echo "✓ boto3 is installed"

# Print download configuration
echo ""
echo "============================================================"
echo "Download Configuration"
echo "============================================================"
echo "Image list: data/required_images.txt"
echo "Output folder: /home/hice1/ksohrab3/scratch/theworld/data/openimages"
echo "Parallel workers: 24"
echo "Time limit: 8 hours"

# Count total images to download
TOTAL_IMAGES=$(wc -l < data/required_images.txt)
echo "Total images in list: $TOTAL_IMAGES"

# Count already downloaded images
DOWNLOADED_DIR="/home/hice1/ksohrab3/scratch/theworld/data/openimages"
if [ -d "$DOWNLOADED_DIR" ]; then
    DOWNLOADED_COUNT=$(ls "$DOWNLOADED_DIR"/*.jpg 2>/dev/null | wc -l)
    REMAINING=$((TOTAL_IMAGES - DOWNLOADED_COUNT))
    echo "Already downloaded: $DOWNLOADED_COUNT"
    echo "Remaining to download: $REMAINING"
else
    echo "Output directory does not exist yet (will be created)"
    REMAINING=$TOTAL_IMAGES
fi

echo "============================================================"

# Run the downloader
echo ""
echo "============================================================"
echo "Starting OpenImages download (24 workers)"
echo "============================================================"

python downloader.py data/required_images.txt \
    --download_folder=/home/hice1/ksohrab3/scratch/theworld/data/openimages \
    --num_processes=24

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Download completed at: $(date)"
echo "Exit code: $EXIT_CODE"

# Print final statistics
if [ -d "$DOWNLOADED_DIR" ]; then
    FINAL_COUNT=$(ls "$DOWNLOADED_DIR"/*.jpg 2>/dev/null | wc -l)
    echo "Total images downloaded: $FINAL_COUNT / $TOTAL_IMAGES"

    if [ $FINAL_COUNT -eq $TOTAL_IMAGES ]; then
        echo "✓ All images downloaded successfully!"
    else
        STILL_REMAINING=$((TOTAL_IMAGES - FINAL_COUNT))
        echo "⚠ $STILL_REMAINING images still remaining"
        echo "Re-run this script to continue downloading"
    fi
fi

echo "============================================================"

exit $EXIT_CODE
