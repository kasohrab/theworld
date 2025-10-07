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

All trainable parameters are saved in each checkpoint:
- Projection layers (always)
- Optionally: Gemma vision encoder, Gemma language model, Cosmos VAE
- Optimizer state
- Training step/epoch information

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

After uploading, you can load your trained model from anywhere using `from_pretrained()`.

### Basic Usage

```python
from theworld import TheWorld

# Load your trained model
model = TheWorld.from_pretrained("your-username/theworld-datacomp")

# Use it for inference
generated_text = model.generate(image, question="What is in this image?")
```

### Loading Specific Checkpoints

If your repository contains multiple checkpoints, you can specify which one to load:

```python
# Load a specific checkpoint
model = TheWorld.from_pretrained(
    "your-username/theworld-datacomp",
    checkpoint_name="checkpoint-1000/pytorch_model.bin"
)

# Load from final model
model = TheWorld.from_pretrained(
    "your-username/theworld-datacomp",
    checkpoint_name="pytorch_model.bin"  # This is the default
)
```

### Loading Private Models

For private models, provide your HuggingFace token:

```python
import os

# Option 1: Via environment variable
os.environ["HF_TOKEN"] = "hf_your_token_here"
model = TheWorld.from_pretrained("your-username/private-model")

# Option 2: Via parameter
model = TheWorld.from_pretrained(
    "your-username/private-model",
    hf_token="hf_your_token_here"
)
```

### Command-Line Example

Use the provided example script to load and test a model:

```bash
# Load from public repository
python examples/load_from_hub.py --model_id username/theworld-datacomp

# Load from private repository
export HF_TOKEN="hf_your_token_here"
python examples/load_from_hub.py --model_id username/private-model

# Load specific checkpoint
python examples/load_from_hub.py \
  --model_id username/theworld-datacomp \
  --checkpoint checkpoint-1000/pytorch_model.bin
```

### What Gets Downloaded

When you call `from_pretrained()`:
1. The checkpoint file is downloaded from Hub (cached locally)
2. Model configuration is extracted (model names, freeze settings, etc.)
3. A new TheWorld instance is initialized with that configuration
4. Trainable parameters are loaded from the checkpoint
5. The model is ready for inference or continued training

## Resuming Training from Hub

You can resume training from a checkpoint uploaded to the Hub:

```bash
# Resume from Hub checkpoint
export HF_TOKEN="hf_your_token_here"
python scripts/train_hf.py \
  --config configs/datacomp_production.json \
  --resume_from username/theworld-datacomp

# The script will:
# 1. Download the latest checkpoint from Hub
# 2. Resume training from that checkpoint
# 3. Continue uploading new checkpoints to Hub
```

This is useful for:
- **Continuing training across different machines**: Train on one GPU, resume on another
- **Collaborative training**: Multiple team members can resume from shared checkpoints
- **Cloud training**: Start training locally, resume on cloud GPUs
- **Fault tolerance**: If training stops, resume from the last uploaded checkpoint

### Resume Configuration

You can also specify resume checkpoint in your config file:

```json
{
  "resume_from_checkpoint": "username/theworld-datacomp",
  "push_to_hub": true,
  "hub_model_id": "username/theworld-datacomp"
}
```

Then simply run:
```bash
python scripts/train_hf.py --config configs/my_config.json
```

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
