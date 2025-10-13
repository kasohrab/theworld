# SpatialRGPT Training Dataset

TheWorld includes built-in support for the **OpenSpatialDataset** (~900K spatial reasoning examples) for training on spatial understanding tasks.

## Overview

- **Dataset**: [a8cheng/OpenSpatialDataset](https://huggingface.co/datasets/a8cheng/OpenSpatialDataset)
- **Content**: Spatial reasoning QA pairs with region references
- **Images**: OpenImagesV7 (requires separate download)
- **Size**: ~900K training examples
- **Purpose**: Teaches models about 3D spatial relationships, object positions, and visual grounding

## Dataset Format

The dataset contains questions and answers about spatial relationships:

**Example**:
- **Question**: "Is Region [0] behind Region [1]?"
- **Answer**: "No."

**Quantitative examples**:
- **Question**: "What is the height of Region [0]?"
- **Answer**: "6.91 feet"

Regions are referenced by ID in the text, and bounding box coordinates are provided in the metadata.

## Setup

### 1. Download OpenImagesV7 Images

The dataset metadata is on HuggingFace, but images must be downloaded separately from OpenImagesV7:

```bash
mkdir -p data/openimages

# Download from: https://storage.googleapis.com/openimages/web/download_v7.html
# Follow their instructions to download train/validation splits
```

### 2. Install Dependencies

```bash
# Already included in pyproject.toml
uv sync
```

### 3. Quick Start

```python
from datasets import load_dataset
from theworld.datasets import SpatialRGPTDataset

# Load dataset metadata from HuggingFace
hf_dataset = load_dataset("a8cheng/OpenSpatialDataset")

# Wrap with SpatialRGPT loader
train_dataset = SpatialRGPTDataset(
    hf_dataset["train"],
    image_folder="data/openimages/train",  # Point to your local OpenImages directory
    draw_bboxes=False,  # Training data doesn't need bbox overlay
)
```

## Training Configuration

Use the provided configuration file:

```bash
# Edit config to set your image folder path
vim configs/spatial_rgpt_training.json

# Set "image_folder": "data/openimages/train"

# Run training
python scripts/train_hf.py --config configs/spatial_rgpt_training.json
```

## Integration with TheWorld

The SpatialRGPT dataset loader automatically:
- Handles both training and evaluation formats
- Loads images from local OpenImagesV7 directory
- Formats questions/answers for Gemma 3 chat template
- Compatible with standard TheWorld training pipeline

**No need to draw bounding boxes** for training (regions are referenced in text).

## Usage in Training Script

```python
from datasets import load_dataset
from theworld.datasets import SpatialRGPTDataset

def load_datasets(config):
    # Load dataset from HuggingFace
    hf_dataset = load_dataset("a8cheng/OpenSpatialDataset")

    # Training dataset
    train_dataset = SpatialRGPTDataset(
        hf_dataset["train"],
        image_folder=config.get("image_folder", "data/openimages/train"),
        draw_bboxes=False,
    )

    # Validation dataset
    eval_dataset = SpatialRGPTDataset(
        hf_dataset["validation"],
        image_folder=config.get("image_folder_val", "data/openimages/validation"),
        draw_bboxes=False,
    )

    return train_dataset, eval_dataset
```

## Dataset Statistics

- **Training samples**: ~900,000
- **Validation samples**: ~50,000
- **Question types**: Spatial relationships, distances, positions
- **Region format**: Bounding boxes with [x1, y1, x2, y2] coordinates

## Related Documentation

- [Evaluation on SpatialRGPT-Bench](../../evaluation/benchmarks/spatial-rgpt.md) - Evaluating trained models
- [Training Infrastructure](../infrastructure.md) - General training setup
- [Multi-Stage Training](../multi-stage.md) - Progressive unfreezing strategy

## References

- **Dataset**: https://huggingface.co/datasets/a8cheng/OpenSpatialDataset
- **OpenImagesV7**: https://storage.googleapis.com/openimages/web/download_v7.html
- **SpatialRGPT Paper**: Original spatial reasoning with region grounding work
- **Integration**: SpatialRGPT repo included as submodule at `external/SpatialRGPT`
