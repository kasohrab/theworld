# HuggingFace Hub Upload Guide

This guide explains how to automatically upload your TheWorld model checkpoints to the HuggingFace Hub during training.

## Overview

The HuggingFace Hub integration provides:
- **Automatic checkpoint uploads** during training
- **Model card generation** with training details
- **Version control** for your checkpoints
- **Easy sharing** with the community
- **Private or public** repositories

## Quick Start

### 1. Get Your HuggingFace Token

First, you need a HuggingFace account and access token:

1. Create an account at https://huggingface.co
2. Go to Settings → Access Tokens: https://huggingface.co/settings/tokens
3. Create a new token with **write** permissions
4. Copy the token (starts with `hf_...`)

### 2. Configure Hub Upload

Add these settings to your training config (e.g., `configs/datacomp_production.json`):

```json
{
  "hf_token": null,
  "push_to_hub": true,
  "hub_model_id": "your-username/theworld-datacomp",
  "hub_strategy": "every_save",
  "hub_private_repo": false
}
```

### 3. Provide Your Token

There are three ways to provide your HuggingFace token:

**Option A: Environment Variable (Recommended)**
```bash
export HF_TOKEN="hf_your_token_here"
python scripts/train_hf.py --config configs/datacomp_production.json
```

**Option B: Command Line Argument**
```bash
python scripts/train_hf.py \
  --config configs/datacomp_production.json \
  --hf_token hf_your_token_here
```

**Option C: Config File**
```json
{
  "hf_token": "hf_your_token_here"
}
```

⚠️ **Security Note**: Never commit your token to git! Use environment variables or `.env` files (add to `.gitignore`).

## Configuration Options

### `push_to_hub` (bool)

Whether to upload checkpoints to the Hub.

- `true`: Enable automatic uploads
- `false`: Disable uploads (default)

```json
"push_to_hub": true
```

### `hub_model_id` (string)

The repository name on HuggingFace Hub in the format `username/model-name`.

- If the repository doesn't exist, it will be created automatically
- Must be unique on the Hub
- Use descriptive names (e.g., `theworld-datacomp`, `theworld-custom-vision`)

```json
"hub_model_id": "your-username/theworld-datacomp"
```

### `hub_strategy` (string)

When to upload checkpoints:

- `"every_save"`: Upload every checkpoint (default, recommended)
- `"checkpoint"`: Upload only named checkpoints
- `"end"`: Upload only at the end of training

```json
"hub_strategy": "every_save"
```

For long training runs, `"every_save"` is recommended so you have backups even if training is interrupted.

### `hub_private_repo` (bool)

Whether to create a private repository:

- `true`: Private (only you can see it)
- `false`: Public (anyone can see and download)

```json
"hub_private_repo": false
```

### `hf_token` (string)

Your HuggingFace API token. As mentioned above, prefer using environment variables instead of storing in config files.

```json
"hf_token": null
```

## What Gets Uploaded

When you enable Hub uploads, the following are automatically uploaded:

### 1. Model Checkpoints

**Important**: TheWorld uses a **trainable-parameters-only** checkpoint strategy for efficiency:

- **Only trainable parameters** are saved (~146MB for projection-only training)
- **Frozen pretrained models** (Gemma, Cosmos) are NOT saved - they reload from HuggingFace
- **Benefits**:
  - Much smaller checkpoints (146MB vs ~17GB full model)
  - Faster uploads and downloads
  - Always uses the latest pretrained base models
  - No duplicate parameter storage

**What's included:**
- Projection layers (always trainable)
- Newly added special tokens (`<the_world_start>`, `<the_world_end>`)
- Optionally: Gemma vision encoder, Gemma language model, Cosmos VAE (if unfrozen)

### 2. Model Card (README.md)

A detailed model card is automatically generated with:
- Model architecture description
- Training configuration (learning rate, batch size, etc.)
- Dataset information
- Trainable components (what was frozen/unfrozen)
- Usage examples
- Citations

You can customize the model card after upload by editing the README.md on the Hub.

### 3. Configuration Files

- `config.json`: Model configuration
- `training_args.bin`: HuggingFace Trainer arguments

## Example Workflows

### Production Training with Hub Upload

```json
{
  "_comment": "Production config with Hub upload",

  "model_name": "google/gemma-3-4b-it",
  "cosmos_model_name": "nvidia/Cosmos-Predict2-2B-Video2World",
  "dataset_name": "datacomp",
  "streaming": true,

  "push_to_hub": true,
  "hub_model_id": "your-username/theworld-datacomp-projection",
  "hub_strategy": "every_save",
  "hub_private_repo": false,

  "log_to_wandb": true,
  "wandb_project": "theworld-datacomp",
  "wandb_run_name": "projection-only-1b"
}
```

```bash
export HF_TOKEN="hf_your_token_here"
python scripts/train_hf.py --config configs/datacomp_production.json
```

### Private Experiment

```json
{
  "push_to_hub": true,
  "hub_model_id": "your-username/theworld-experiment-v1",
  "hub_strategy": "every_save",
  "hub_private_repo": true
}
```

### Test Run (No Upload)

```json
{
  "push_to_hub": false,
  "hub_model_id": null
}
```

## Loading Models from Hub

After uploading, you can load your trained model from anywhere. Since checkpoints contain only trainable parameters, the loading process:

1. Creates a fresh TheWorld instance (loads Gemma + Cosmos from HuggingFace)
2. Downloads your trained checkpoint from Hub
3. Loads only the trainable parameters into the model

### Basic Usage

```python
from theworld import TheWorld
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Step 1: Create base model (loads pretrained Gemma + Cosmos)
model = TheWorld(
    "google/gemma-3-4b-it",
    freeze_gemma_vision=True,
    freeze_gemma_language=True,
    freeze_cosmos_vae=True,
)

# Step 2: Download checkpoint from Hub
checkpoint_path = hf_hub_download(
    repo_id="your-username/theworld-datacomp",
    filename="model.safetensors",  # Main checkpoint at root
    token="hf_your_token_here",  # For private repos
)

# Step 3: Load trainable parameters
state_dict = load_file(checkpoint_path)
model.load_state_dict(state_dict, strict=False)

# Ready to use!
model.eval()
```

### Loading from Different Checkpoints

Checkpoints are stored in the repository root with these filenames:

```python
# Latest checkpoint (root)
filename="model.safetensors"

# Final checkpoint (after training completes)
filename="final/model.safetensors"

# Note: Intermediate checkpoints are also uploaded but may be cleaned up
```

### Loading Private Models

For private repositories, provide your HuggingFace token:

```python
# Via environment variable (recommended)
import os
os.environ["HF_TOKEN"] = "hf_your_token_here"

checkpoint_path = hf_hub_download(
    repo_id="your-username/private-model",
    filename="model.safetensors",
)

# Or via parameter
checkpoint_path = hf_hub_download(
    repo_id="your-username/private-model",
    filename="model.safetensors",
    token="hf_your_token_here",
)
```

### What Gets Downloaded

When you load a checkpoint from Hub:
1. **Checkpoint file** (~146MB for projection-only) is downloaded and cached locally
2. **Only trainable parameters** are in the checkpoint (projection layer, special tokens)
3. **Frozen models** (Gemma, Cosmos) are automatically downloaded fresh from HuggingFace
4. Model is ready for inference or continued training

**Important**: The first time you load a model, it will download:
- Your checkpoint (~146MB)
- Base Gemma model (~8GB, cached)
- Base Cosmos model (~4GB, cached)

Subsequent loads are much faster as base models are cached!

## Resuming Training from Hub

You can resume training from a checkpoint uploaded to the Hub. The training script will:
1. Download your checkpoint from Hub
2. Load the trainable parameters
3. Continue training from that point
4. Upload new checkpoints back to Hub

### Method 1: Command Line Argument

```bash
export HF_TOKEN="hf_your_token_here"
python scripts/train_hf.py \
  --config configs/datacomp_production.json \
  --resume_from checkpoint_path_on_hub
```

**Note**: Currently HuggingFace Trainer expects a local path or Hub repo structure. For TheWorld's trainable-only checkpoints, you need to:

1. Download the checkpoint manually first:
```bash
# Download checkpoint to local directory
python -c "
from huggingface_hub import hf_hub_download
import shutil
import os

checkpoint = hf_hub_download(
    'username/theworld-datacomp',
    'model.safetensors',
    token='hf_your_token_here'
)
os.makedirs('./checkpoints/from_hub', exist_ok=True)
shutil.copy(checkpoint, './checkpoints/from_hub/model.safetensors')
print('✓ Checkpoint downloaded to ./checkpoints/from_hub')
"
```

2. Then resume from local path:
```bash
python scripts/train_hf.py \
  --config configs/datacomp_production.json \
  --resume_from ./checkpoints/from_hub
```

### Method 2: Config File

Specify the local checkpoint path in your config:

```json
{
  "resume_from_checkpoint": "./checkpoints/from_hub",
  "push_to_hub": true,
  "hub_model_id": "username/theworld-datacomp"
}
```

Then run:
```bash
python scripts/train_hf.py --config configs/my_config.json
```

### Use Cases

Resuming from Hub is useful for:
- **Continuing training across machines**: Train on one GPU, resume on another
- **Collaborative training**: Multiple team members can resume from shared checkpoints
- **Cloud training**: Start training locally, resume on cloud GPUs
- **Fault tolerance**: If training stops, resume from the last uploaded checkpoint
- **Multi-stage training**: Train projection first, then unfreeze and continue training

## Troubleshooting

### "Repository not found" Error

Make sure:
1. Your token has **write** permissions
2. The `hub_model_id` follows the format `username/repo-name`
3. You're logged in with the correct account

### "Authentication Failed" Error

Check:
1. Your token is valid (not expired)
2. The token is correctly set (environment variable or CLI arg)
3. You've accepted the model licenses on HuggingFace (Gemma, Cosmos)

### Uploads Are Slow

- Hub uploads happen in the background and don't block training
- Large checkpoints (>1GB) may take time depending on your internet speed
- Consider using `hub_strategy: "end"` if uploads are too frequent

### How to Delete Uploaded Checkpoints

1. Go to your repository on HuggingFace: `https://huggingface.co/your-username/your-model`
2. Click on "Files and versions"
3. Delete unwanted checkpoint folders

Or use the Hub CLI:
```bash
huggingface-cli delete your-username/your-model checkpoint-1000
```

## Best Practices

1. **Use descriptive repository names**: Include dataset and training strategy
   - Good: `theworld-datacomp-projection`, `theworld-custom-full`
   - Bad: `model1`, `test`, `checkpoint`

2. **Set push_to_hub=true for production runs**: Always have backups on the Hub

3. **Use private repos for experiments**: Make public only when ready to share

4. **Update the model card**: After training, edit the README.md to add:
   - Final metrics and performance
   - Example outputs
   - Known limitations
   - Intended use cases

5. **Tag releases**: Create git tags on the Hub for important checkpoints:
   ```bash
   cd /path/to/cloned/repo
   git tag v1.0-final
   git push --tags
   ```

## Advanced: Custom Model Cards

You can customize the generated model card by modifying `theworld/hub_utils.py`:

```python
from theworld.hub_utils import generate_model_card

# Generate custom model card
model_card = generate_model_card(
    gemma_model_name="google/gemma-3-4b-it",
    cosmos_model_name="nvidia/Cosmos-Predict2-2B-Video2World",
    dataset_name="custom-dataset",
    num_samples=1000000,
    trainable_params=50000,
    total_params=6000000000,
    learning_rate=1e-4,
    batch_size=16,
    num_epochs=3,
)

# Save to your Hub repo
with open("README.md", "w") as f:
    f.write(model_card)
```

## Resources

- HuggingFace Hub Documentation: https://huggingface.co/docs/hub
- Model Cards Guide: https://huggingface.co/docs/hub/model-cards
- Trainer Hub Integration: https://huggingface.co/docs/transformers/main_classes/trainer#push-to-hub
- API Tokens: https://huggingface.co/settings/tokens
