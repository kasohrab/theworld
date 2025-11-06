# SpatialRGPT Training - End-to-End Guide

Complete guide to training TheWorld on the OpenSpatialDataset (~900K spatial reasoning examples).

---

## Quick Start

**Two components needed for training:**
1. **JSON metadata** (30GB) - Auto-downloads to $TMPDIR during training
2. **OpenImages** (273GB) - You must download separately

**Setup (one-time):**
```bash
# Download OpenImages
sbatch scripts/download_openimages.sbatch
```

**Train:**
```bash
export HF_TOKEN=hf_your_token_here
uv run accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \
    scripts/train_hf.py --config configs/spatial_rgpt_training.json
```

**Training automatically:**
- Downloads JSON to $TMPDIR on first run
- Uses only images you've downloaded so far
- Picks up new images on subsequent runs

---

## Understanding the Two Components

### Component 1: JSON Metadata (30GB)

**What it contains:**
- Conversations (Q&A pairs for spatial reasoning)
- Bounding boxes
- Image filenames (references to OpenImages)

**Example:**
```json
{
  "filename": "00002f4ff380c64c",
  "conversations": [
    {"from": "human", "value": "What is in Region [0]?"},
    {"from": "gpt", "value": "A red car"}
  ],
  "bbox": [[100, 200, 300, 400]]
}
```

**Where it comes from:** HuggingFace Hub (`a8cheng/OpenSpatialDataset`)

**Storage:** Auto-downloads to `$TMPDIR` (SLURM job's temporary storage)
- Each SLURM job downloads fresh
- Stored on fast local SSD

**Setup required:** None - handled automatically by training script

---

### Component 2: OpenImages (273GB)

**What it contains:**
- Actual .jpg image files (909,419 total)
- Visual data referenced by JSON

**Example:**
- JSON says `"filename": "00002f4ff380c64c"`
- Image at: `data/openimages/00002f4ff380c64c.jpg`

**Where it comes from:** OpenImagesV7 (Google's dataset, separate from HuggingFace)

**Storage:** Permanent storage at `data/openimages/`
- Slow to download so put it in persistent storage

**Setup required:** Download via sbatch script (see Step 1 below)

---

### How They Connect

```
Training loop:
  ↓
1. Scan image folder → find available .jpg files
  ↓
2. Parse JSON → filter entries by available images
   "filename": "00002f4ff380c64c" → Check: 00002f4ff380c64c.jpg exists?
  ↓
3. For each sample:
   - Read conversation from JSON
   - Load image from {image_folder}/{filename}.jpg
   - Feed to model
```
---

## Step 1: Download OpenImages (~273GB)

### Automated Download (Recommended)

```bash
sbatch scripts/download_openimages.sbatch
```

**What this does:**
- Downloads 909K images from OpenImagesV7
- Stores in `data/openimages/`
- Automatically resumes if interrupted
- Uses 10 parallel workers

### Monitor Progress

```bash
# Check job status
squeue -u $USER

# Count downloaded images
find data/openimages -name "*.jpg" | wc -l

# View download logs
tail -f logs/download-*.out
```

### Manual Download (Alternative)

```bash
# Get OpenImages official downloader
wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py

# Download using provided image list
python downloader.py data/required_images.txt \
    --download_folder=data/openimages \
    --num_processes=10
```

**Note:** `data/required_images.txt` is already provided (909K image IDs, 20MB)

---

## Step 2: Train

### Basic Training

```bash
# Set HuggingFace token (for model uploads)
export HF_TOKEN=hf_your_token_here

# Single GPU
python scripts/train_hf.py --config configs/spatial_rgpt_training.json

# Multi-GPU (recommended)
uv run accelerate launch --config_file configs/accelerate/multi_gpu_ddp.yaml \
    scripts/train_hf.py --config configs/spatial_rgpt_training.json
```

### What Happens During Training

**On first run:**
1. Checks if JSON exists in `$TMPDIR`
2. If not: Downloads from HuggingFace (~5-10 min, 30GB)
   ```bash
   Downloading OpenSpatialDataset JSON to $TMPDIR...
   Download complete: $TMPDIR/openspatial/result_10_depth_convs.json
   ```
3. Scans `data/openimages/` for available images
   ```
   Scanning image folder: data/openimages
   ✓ Found 401,890 images in 45.2s
   ```
4. Parses JSON and filters by available images
   ```
   Parsing JSON and filtering by available images...
   ✓ Loaded 401,890 samples in 120.5s
   ```
5. Trains on available data

**On subsequent runs in same job:**
- Reuses JSON from `$TMPDIR` (no re-download)
- Re-scans image folder (picks up new images)

**On new job:**
- Downloads JSON fresh to new `$TMPDIR`
- Uses latest available images

---

### Progressive Training (Train While Downloading)

You can start training before all images download!

```bash
# Step 1: Start download (background)
sbatch scripts/download_openimages.sbatch

# Step 2: Wait for some images
watch -n 60 'find data/openimages -name "*.jpg" | wc -l'

# Step 3: Start training
python scripts/train_hf.py --config configs/spatial_rgpt_training.json
# Trains on available images (e.g., 10K samples)

# Step 4: More images downloaded? Continue training
python scripts/train_hf.py \
    --config configs/spatial_rgpt_training.json \
    --resume_from_checkpoint checkpoints/theworld-spatial-bbox/checkpoint-1000
# Re-scans folder, picks up new images (e.g., 50K samples)
```

---

## Configuration

### Training Config (`configs/spatial_rgpt_training.json`)

```json
{
  "dataset_name": "spatial_rgpt",
  "train_dataset_path": "${TMPDIR}/openspatial/result_10_depth_convs.json",
  "image_folder": "/home/hice1/ksohrab3/scratch/theworld/data/openimages",
  "draw_bboxes": true,
  "num_samples": null,

  "batch_size": 2,
  "learning_rate": 0.0001,
  "num_epochs": 1,
  "save_steps": 1000,

  "push_to_hub": true,
  "hub_model_id": "your-username/theworld-spatial"
}
```

**Key settings:**
- `train_dataset_path`: Points to $TMPDIR (auto-download)
- `image_folder`: Points to your persistent OpenImages directory
- `draw_bboxes`: Overlays bounding boxes on images for visualization
- `num_samples`: `null` = all available, or set limit (e.g., `1000` for testing)

---

## Resume & Monitor

### Resume Downloads

If download job stops:

```bash
# Check progress and create resume file
python scripts/find_download_resume_point.py

# Resume downloading (auto-uses resume file)
sbatch scripts/download_openimages.sbatch
```

### Resume Training

```bash
# From local checkpoint
python scripts/train_hf.py \
    --config configs/spatial_rgpt_training.json \
    --resume_from_checkpoint checkpoints/theworld-spatial-bbox/checkpoint-1000

# From HuggingFace Hub
python scripts/train_hf.py \
    --config configs/spatial_rgpt_training.json \
    --resume_from_checkpoint your-username/theworld-spatial
```

### Monitor Training

```bash
# Training logs
tail -f logs/slurm-*.out

# Checkpoints
ls -lht checkpoints/theworld-spatial-bbox/checkpoint-*

# TensorBoard
tensorboard --logdir checkpoints/theworld-spatial-bbox/logs

# Storage usage
pace-quota
du -sh data/openimages
```

---

## Troubleshooting

### JSON Download Fails

**Error:** `Failed to download OpenSpatialDataset JSON`

**Fix:** Check HuggingFace token and network:
```bash
# Test huggingface-cli
huggingface-cli download a8cheng/OpenSpatialDataset result_10_depth_convs.json --repo-type dataset --local-dir /tmp/test
```

### Training Finds 0 Images

**Error:** `✓ Found 0 images`

**Cause:** Wrong `image_folder` path in config

**Fix:** Verify path points to folder containing .jpg files (no subdirectory):
```bash
# Check config
grep image_folder configs/spatial_rgpt_training.json

# Check images exist
ls data/openimages/*.jpg | head -5

# Should see: 00002f4ff380c64c.jpg, 00003bfccf5f36c2.jpg, ...
```

### Download Job Killed

**Cause:** SLURM time limit or node failure

**Fix:** Resubmit (auto-resumes from last position):
```bash
sbatch scripts/download_openimages.sbatch
```

### Out of Memory During Training

**Cause:** Too many samples or large batch size

**Fix:** Reduce batch size or limit samples:
```json
{
  "batch_size": 1,           // Reduce from 2
  "num_samples": 10000       // Test with 10K samples first
}
```

---

## Summary

**Storage requirements:**
- JSON: 30GB (ephemeral, in $TMPDIR)
- Images: 273GB (persistent, in scratch space)
- Total scratch space needed: ~273GB

**Time estimates:**
- JSON download (first run): 5-10 minutes
- Image download: 1-3 days
- Training: Start anytime (uses available images)

**Key features:**
- **Auto-download JSON**: No manual setup needed
- **Progressive training**: Train while downloading images
- **Images-first loading**: Only uses available images
- **Auto-resume**: Downloads and training both resume automatically

**Files in your setup:**
- `data/required_images.txt` - List of 909K image IDs (20MB, pre-generated)
- `data/openimages/*.jpg` - Downloaded images (401,890 so far, 124GB)
- `${TMPDIR}/openspatial/result_10_depth_convs.json` - Auto-downloaded during training

**Your current status:**
- ✓ `required_images.txt` ready (909K image IDs)
- ✓ 401,890 images downloaded (44% complete, 124GB)
- ✓ Configs updated to use $TMPDIR for JSON
- ✓ Ready to train!

---

## Advanced: How Data Loading Works

**Images-First Architecture** ensures no wasted iterations:

### Dataset Initialization

```python
# Step 1: Scan image folder (30-60s)
available_images = {img.stem for img in Path(image_folder).glob("*.jpg")}
# Result: Set of 401,890 image IDs

# Step 2: Parse JSON and filter (60-120s)
samples = []
for entry in ijson.items(json_path, "item"):
    if entry["filename"] in available_images:
        samples.append(entry)  # Keep in memory

# Step 3: Dataset ready
len(dataset)  # Returns 401,890 (actual available count)
```

### Benefits

- **No wasted iterations**: Only loads samples with images
- **True dataset length**: `len(dataset)` reflects actual data
- **No `None` returns**: `__getitem__` always succeeds
- **Memory efficient**: ~2-4GB for 200K samples
- **Fast access**: O(1) sample retrieval

### Progressive Training Pattern

**Batch 1:** 10K images → initialize → train
**Batch 2:** 50K images → re-initialize → resume training
**Batch 3:** 401K images → re-initialize → continue training

Each re-initialization automatically picks up new images.
