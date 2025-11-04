# SpatialRGPT Training - End-to-End Guide

Complete guide to training TheWorld on the OpenSpatialDataset (~900K spatial reasoning examples).

## Quick Start

```bash
# 1. Download training data (30GB JSON)
huggingface-cli download a8cheng/OpenSpatialDataset --repo-type dataset --local-dir data/openspatial

# 2. Extract image IDs
python scripts/download_openimages.py --output data/required_images.txt

# 3. Download images (background, 1-3 days)
sbatch scripts/download_openimages.sbatch

# 4. Train (loads only samples with available images)
export HF_TOKEN=hf_your_token_here
uv run accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \
    scripts/train_hf.py --config configs/spatial_rgpt_training.json
```

---

## Step 1: Obtain Training Data

The OpenSpatialDataset contains ~900K spatial reasoning QA pairs. You need two components:

1. **JSON metadata** (~30GB): Conversations, bounding boxes, segmentation masks
2. **OpenImages** (~273GB): Actual image files

### Download Training JSON

```bash
# Method 1: HuggingFace CLI (recommended)
huggingface-cli download a8cheng/OpenSpatialDataset \
    --repo-type dataset \
    --local-dir /home/hice1/ksohrab3/scratch/theworld/data/openspatial

# Finds result_10_depth_convs.json in the downloaded files
# Move to expected location:
cp data/openspatial/result_10_depth_convs.json /home/hice1/ksohrab3/scratch/theworld/data/
```

```bash
# Method 2: Python (auto-caches to ~/.cache/huggingface/datasets/)
python -c "
from datasets import load_dataset
ds = load_dataset('a8cheng/OpenSpatialDataset', split='train')
print(f'Loaded {len(ds)} samples')
"
# Note: This caches but doesn't create result_10_depth_convs.json directly
```

**Storage:** Use scratch space (not home directory) - requires ~30GB for JSON + 273GB for images.

**Expected location:** `/home/hice1/ksohrab3/scratch/theworld/data/result_10_depth_convs.json`

---

## Step 2: Extract Image IDs

Extract the 909K unique image IDs referenced in the training data.

```bash
# Extract all image IDs (~30 seconds)
python scripts/download_openimages.py --output data/required_images.txt

# Output: data/required_images.txt (909,419 lines)
# Format: train/da0f505346a22fa7 (one per line)
```

**Test with subset:**
```bash
# Extract only 1,000 images for testing
python scripts/download_openimages.py --num-images 1000 --output data/test_images.txt
```

---

## Step 3: Download Images

Download OpenImagesV7 images using the official downloader.

### One-Time Setup

```bash
# Get official OpenImages downloader
wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py

# Install dependencies
uv pip install boto3 botocore
```

### Submit Download Job

```bash
# Submit SLURM job (auto-resumes if interrupted)
sbatch scripts/download_openimages.sbatch

# Downloads to: /home/hice1/ksohrab3/scratch/theworld/data/openimages/
# Time: 1-3 days for all 909K images (~273GB)
```

**What the script does:**
- Downloads images directly to folder root (no `/train` subdirectory)
- Uses 10 parallel workers
- Automatically resumes from interruptions
- Skips already-downloaded images

### Monitor Progress

```bash
# Check SLURM job status
squeue -u $USER

# Count downloaded images
ls /home/hice1/ksohrab3/scratch/theworld/data/openimages/*.jpg 2>/dev/null | wc -l

# Check resume point (creates data/resume_images.txt)
python scripts/find_download_resume_point.py

# View download logs
tail -f logs/download-*.out
```

---

## Step 4: Train

**Images-first loading:** Dataset scans image folder at initialization, then loads only JSON entries for available images. No wasted iterations, no missing image errors.

### Configure Training

Edit `configs/spatial_rgpt_training.json`:

```json
{
  "dataset_name": "spatial_rgpt",
  "train_dataset_path": "/home/hice1/ksohrab3/scratch/theworld/data/result_10_depth_convs.json",
  "image_folder": "/home/hice1/ksohrab3/scratch/theworld/data/openimages",
  "num_samples": null,  // null = use all available images

  "batch_size": 2,
  "learning_rate": 0.0001,
  "num_epochs": 1,
  "save_steps": 1000,

  "push_to_hub": true,
  "hub_model_id": "your-username/theworld-spatial-rgpt"
}
```

**Key settings:**
- `train_dataset_path`: Points to 30GB JSON file
- `image_folder`: Points to OpenImages directory (no `/train` suffix!)
- `num_samples: null`: Uses all available images (auto-detects count)

### Run Training

```bash
# Set HuggingFace token (for model uploads)
export HF_TOKEN=hf_your_token_here

# Single GPU
python scripts/train_hf.py --config configs/spatial_rgpt_training.json

# Multi-GPU (recommended)
uv run accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \
    scripts/train_hf.py --config configs/spatial_rgpt_training.json
```

### Progressive Training Strategy

Train while images download in background:

```bash
# Step 1: Start download (background)
sbatch scripts/download_openimages.sbatch

# Step 2: Wait for ~10K images (~30 minutes)
watch -n 60 'ls data/openimages/*.jpg 2>/dev/null | wc -l'

# Step 3: Start training batch 1
python scripts/train_hf.py --config configs/spatial_rgpt_training.json
# Trains on available images (e.g., 10K samples)

# Step 4: More images downloaded? Re-initialize for batch 2
# Dataset re-scans folder, picks up new images automatically
python scripts/train_hf.py \
    --config configs/spatial_rgpt_training.json \
    --resume_from_checkpoint checkpoints/theworld-spatial-rgpt/checkpoint-1000
# Now trains on all downloaded images (e.g., 50K samples)
```

**Key:** Re-initialization scans folder fresh, detects new images, continues training seamlessly.

---

## Resume & Monitor

### Resume Downloads

If download job stops or you need to restart:

```bash
# Check progress and create resume file
python scripts/find_download_resume_point.py
# Creates: data/resume_images.txt

# Resume downloading (auto-uses resume file)
sbatch scripts/download_openimages.sbatch
```

The sbatch script automatically:
- Checks for `data/resume_images.txt`
- Skips already-downloaded images
- Continues from where it left off

### Resume Training

```bash
# From local checkpoint
python scripts/train_hf.py \
    --config configs/spatial_rgpt_training.json \
    --resume_from_checkpoint checkpoints/theworld-spatial-rgpt/checkpoint-1000

# From HuggingFace Hub
python scripts/train_hf.py \
    --config configs/spatial_rgpt_training.json \
    --resume_from_checkpoint your-username/theworld-spatial-rgpt
```

### Monitor Training

```bash
# Training logs
tail -f logs/slurm-*.out

# Checkpoints
ls -lht checkpoints/theworld-spatial-rgpt/checkpoint-*

# TensorBoard
tensorboard --logdir checkpoints/theworld-spatial-rgpt/logs

# Disk usage
pace-quota
du -sh /home/hice1/ksohrab3/scratch/theworld/data/openimages
```

---

## How Data Loading Works

**Images-First Architecture** (`python/theworld/datasets/spatial_rgpt.py`):

### Initialization (~90-180 seconds)

```python
# Step 1: Scan image folder (30-60s)
available_images = {img.stem for img in Path(image_folder).glob("*.jpg")}
# Result: Set of 909K image IDs (or however many downloaded)

# Step 2: Parse JSON and filter (60-120s)
samples = []
for entry in ijson.items(json_path, "item"):
    if entry["filename"] in available_images:
        samples.append(entry)  # Keep in memory
# Result: List of sample dicts for available images only

# Step 3: Dataset ready
len(dataset)  # Returns actual available count (e.g., 50K)
```

### Benefits

- **No wasted iterations:** Only loads samples with images
- **True dataset length:** `len(dataset)` = actual available samples
- **No `None` returns:** `__getitem__` always succeeds
- **Memory efficient:** Hybrid approach (~2-4GB for 200K samples)
- **Fast access:** O(1) sample retrieval (no JSON seeks)

### Progressive Training

**Batch 1:** Download 10K images → initialize → train
**Batch 2:** Download 50K more → re-initialize (picks up new images) → resume training
**Batch 3:** Download remaining → re-initialize → finish training

Each re-initialization scans folder fresh, detects new images automatically.

### Config

```json
{
  "train_dataset_path": "path/to/result_10_depth_convs.json",
  "image_folder": "path/to/openimages",
  "num_samples": null  // null = all available, or limit (e.g., 1000)
}
```

---

## Troubleshooting

### Missing Training JSON

**Error:** `FileNotFoundError: result_10_depth_convs.json`

**Fix:** Download from HuggingFace (see Step 1)

### Images Not Found During Training

**Error:** `FileNotFoundError: da0f505346a22fa7.jpg`

**Should not happen** with images-first loading. If it does:

```bash
# Check image count
ls data/openimages/*.jpg | wc -l

# Check image_folder config points to correct location
# Should be folder root, not /train subdirectory
grep image_folder configs/spatial_rgpt_training.json

# Verify image exists
ls data/openimages/da0f505346a22fa7.jpg
```

### Download Job Killed

**Cause:** SLURM time limit or node failure

**Fix:** Resubmit (auto-resumes)
```bash
sbatch scripts/download_openimages.sbatch
```

### OOM Crash During Dataset Loading

**Cause:** Loading too many samples into memory

**Fix:** Images-first approach uses ~2-4GB for 200K samples. If OOM:
```bash
# Limit samples in config
{
  "num_samples": 10000  // Limit to 10K samples for testing
}

# Or verify ijson is installed for efficient JSON parsing
uv pip list | grep ijson  # Should show: ijson>=3.3.0
```

### Wrong Image Folder Path

**Error:** Training runs but finds 0 images

**Cause:** `image_folder` config points to wrong location

**Fix:** Official downloader saves to folder root (no subdirectory)
```json
{
  "image_folder": "/path/to/openimages"  // ✓ Correct
  "image_folder": "/path/to/openimages/train"  // ✗ Wrong
}
```

---

## Summary

**Data requirements:**
- JSON: 30GB (`result_10_depth_convs.json`)
- Images: 273GB (909K OpenImagesV7 files)
- Total: ~303GB in scratch space

**Time estimates:**
- Step 1 (JSON download): 10-30 minutes
- Step 2 (extract IDs): 30 seconds
- Step 3 (image download): 1-3 days
- Step 4 (training): Start immediately

**Key features:**
- Images-first loading (only loads samples with available images)
- Progressive training (re-initialize to pick up new images)
- Auto-resume (downloads and checkpoints)
- Efficient memory (~2-4GB for 200K samples)

**Files created:**
- `data/required_images.txt` - List of 909K image IDs
- `data/resume_images.txt` - Auto-generated resume point
- `data/openimages/*.jpg` - Downloaded images
- `checkpoints/theworld-spatial-rgpt/` - Training checkpoints

---

## Exact Command Used

```bash
# Full command for reference
  export HF_TOKEN="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" && \
  uv run accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \
      scripts/train_hf.py --config configs/spatial_rgpt_training.json 2>&1 | tee training_debug_v2.log
```
