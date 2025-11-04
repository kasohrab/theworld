# Training Infrastructure Design Document

## Executive Summary

This document outlines the design for adding production-ready training infrastructure to TheWorld model, including gradient checkpointing for memory efficiency and HuggingFace Trainer integration for distributed training, automatic checkpointing, and ecosystem compatibility.

## 1. Current State Analysis

### 1.1 Model Architecture
- **TheWorld**: Fused vision-language-world model (~6B parameters)
  - Gemma 3 (4B): Vision encoder (SigLIP) + Language model
  - Cosmos (2B): VAE encoder for world dynamics
  - Projection layers: 50K trainable parameters (0.07% of total)

### 1.2 Existing Training Setup
**File:** `python/train.py`
- ✅ Model initialization with freeze configurations
- ✅ Forward pass with loss computation
- ✅ Backward pass (gradient computation)
- ❌ No optimizer or training loop
- ❌ No checkpoint saving/loading
- ❌ No gradient checkpointing (memory bottleneck)
- ❌ No data loading pipeline
- ❌ No distributed training support
- ❌ No logging/monitoring

### 1.3 Memory Constraints
**Problem:** Training the full 6B model requires ~24GB GPU memory minimum
- Model weights (bf16): 6B × 2 bytes = 12GB
- Activations (no checkpointing): ~8-12GB
- Optimizer state (AdamW): ~24GB (2× model size for momentum + variance)
- **Total without checkpointing:** ~44-48GB (exceeds single A100 40GB)

**Solution:** Gradient checkpointing reduces activation memory by ~4-8x at cost of 30-40% slower training

## 2. Problem Statement

**Primary Goal:** Enable training TheWorld on single or multi-GPU setups with:
1. Memory efficiency via gradient checkpointing
2. Automatic checkpoint saving/resuming
3. Integration with HuggingFace ecosystem (datasets, accelerate, trainer)
4. Support for partial training (projection only vs. full model)

**Non-Goals:**
- Changing model architecture
- Dataset creation (assumes user provides their own)
- Hyperparameter tuning or AutoML

## 3. Proposed Solution Architecture

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Training Pipeline                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Dataset    │───▶│   Collator   │───▶│  Trainer  │ │
│  │  (HF/Custom) │    │  (PIL→Batch) │    │  (HF API) │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
│                                                  │        │
│                                                  ▼        │
│  ┌───────────────────────────────────────────────────┐  │
│  │              TheWorld Model                       │  │
│  ├───────────────────────────────────────────────────┤  │
│  │  • Gradient Checkpointing (optional)              │  │
│  │  • Mixed Precision (bf16)                         │  │
│  │  • Device Mapping (auto/manual)                   │  │
│  └───────────────────────────────────────────────────┘  │
│                          │                               │
│                          ▼                               │
│  ┌───────────────────────────────────────────────────┐  │
│  │          Checkpoint Manager                       │  │
│  ├───────────────────────────────────────────────────┤  │
│  │  • Save: trainable params + optimizer state       │  │
│  │  • Load: resume from checkpoint                   │  │
│  │  • Format: PyTorch .pt or SafeTensors             │  │
│  └───────────────────────────────────────────────────┘  │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### 3.2 File Structure

```
theworld/
├── python/
│   ├── model.py                    # [MODIFIED] Add gradient checkpointing + save/load
│   ├── train.py                    # [KEEP] Simple demo
│   ├── train_hf.py                 # [NEW] Full HF Trainer-based training
│   ├── training_config.py          # [NEW] Dataclass for training configuration
│   └── data_utils.py               # [NEW] Dataset collator + example dataset class
├── docs/
│   └── training_infrastructure_design.md  # [NEW] This document
├── pyproject.toml                  # [MODIFIED] Add dependencies
└── Makefile                        # [MODIFIED] Add train-hf command
```

## 4. Detailed Design

### 4.1 Gradient Checkpointing

**Implementation Location:** `python/model.py`

```python
class TheWorld(nn.Module):
    def __init__(self, ..., use_gradient_checkpointing=False):
        # ... existing init ...
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        # For Gemma language model
        if hasattr(self.gemma, 'gradient_checkpointing_enable'):
            self.gemma.gradient_checkpointing_enable()

        # For Cosmos VAE encoder (if unfrozen)
        if not self.freeze_cosmos_vae:
            if hasattr(self.cosmos_vae_encoder, 'gradient_checkpointing_enable'):
                self.cosmos_vae_encoder.gradient_checkpointing_enable()
            # Fallback: manual checkpointing via torch.utils.checkpoint
```

**Trade-offs:**
- ✅ Reduces activation memory by 4-8x
- ✅ Enables training on smaller GPUs (24GB instead of 48GB)
- ❌ Increases training time by 30-40%
- ❌ Not beneficial for inference (only training)

**When to use:**
- Training full model (all components unfrozen)
- GPU memory < 40GB
- Not needed for projection-only training (only 50K params)

### 4.2 Checkpoint Saving/Loading

**Implementation Location:** `python/model.py`

```python
class TheWorld(nn.Module):
    def save_checkpoint(self, path, optimizer=None, epoch=None, **kwargs):
        """Save trainable components and optimizer state.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer to save state
            epoch: Current epoch number
            **kwargs: Additional metadata to save
        """
        checkpoint = {
            'model_state_dict': {},  # Only trainable params
            'freeze_config': {
                'freeze_gemma_vision': self.freeze_gemma_vision,
                'freeze_gemma_language': self.freeze_gemma_language,
                'freeze_cosmos_vae': self.freeze_cosmos_vae,
            },
            'model_config': {
                'gemma_model_name': self.gemma_model_name,
                'num_world_steps': self.num_world_steps,
                'max_world_steps': self.max_world_steps,
            },
            'epoch': epoch,
            **kwargs
        }

        # Save only trainable parameters
        for name, param in self.named_parameters():
            if param.requires_grad:
                checkpoint['model_state_dict'][name] = param.data

        # Save optimizer state
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path, optimizer=None):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load trainable parameters
        missing, unexpected = self.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False  # Only load trainable params
        )

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint.get('epoch', 0)
```

**Design Decisions:**

1. **Save only trainable parameters** (not full model)
   - ✅ Small checkpoint size (~200KB for projection-only vs. 12GB full model)
   - ✅ Fast save/load times
   - ❌ Requires re-downloading base models (Gemma, Cosmos) on new machine
   - **Alternative:** Save full model (adds 12GB per checkpoint)

2. **Checkpoint format: PyTorch .pt**
   - ✅ Native PyTorch support, simple
   - ✅ Compatible with `torch.load()`
   - ❌ Not secure (pickle-based)
   - **Alternative:** SafeTensors format (more secure, requires dependency)

3. **Save optimizer state**
   - ✅ Required for proper resuming (momentum, variance)
   - ❌ Adds checkpoint size (~2x trainable params for AdamW)
   - **Optional:** User can choose whether to save optimizer

### 4.3 Training Configuration

**Implementation Location:** `python/training_config.py`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "google/gemma-3-4b-it"
    num_world_steps: int = 0
    freeze_gemma_vision: bool = True
    freeze_gemma_language: bool = True
    freeze_cosmos_vae: bool = True

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Memory optimization
    use_gradient_checkpointing: bool = False
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_steps: int = 500
    save_total_limit: int = 3  # Keep only last 3 checkpoints

    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    log_to_wandb: bool = False
    wandb_project: Optional[str] = None

    # Data
    max_seq_length: int = 2048
    num_workers: int = 4
```

**Why dataclass:**
- ✅ Type hints for IDE support
- ✅ Default values
- ✅ Easy serialization (dataclasses.asdict)
- ✅ Compatible with HF TrainingArguments conversion

### 4.4 Data Pipeline

**Implementation Location:** `python/data_utils.py`

```python
from torch.utils.data import Dataset
from typing import List, Dict, Any

class TheWorldDataset(Dataset):
    """Example dataset class for TheWorld training.

    Expected format:
    - image: PIL Image or path to image
    - text: Text prompt/question
    - label: Expected response text
    """
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def theworld_collate_fn(batch: List[Dict], processor):
    """Collate function for batching TheWorld inputs.

    Handles:
    - PIL images → batched pixel tensors
    - Text prompts → tokenized input_ids
    - Labels → tokenized label_ids
    """
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    labels = [item.get('label', None) for item in batch]

    # Process through model's forward (handles chat template)
    # Returns: dict with pixel_values, input_ids, attention_mask, labels
    # ... implementation details ...
```

**Design Decisions:**

1. **Use HuggingFace datasets format**
   - ✅ Compatible with `datasets.load_dataset()`
   - ✅ Supports streaming, caching, sharding
   - ❌ Requires learning HF datasets API
   - **Alternative:** Simple PyTorch Dataset (easier but less flexible)

2. **Collate function handles PIL → tensor conversion**
   - ✅ Keeps dataset storage efficient (PIL is lighter)
   - ✅ Lazy conversion (only during training)
   - ❌ Slight overhead per batch
   - **Alternative:** Pre-convert all images to tensors (faster but more disk space)

### 4.5 HuggingFace Trainer Integration

**Implementation Location:** `python/train_hf.py`

```python
from transformers import Trainer, TrainingArguments
from training_config import TrainingConfig
from data_utils import TheWorldDataset, theworld_collate_fn
from model import TheWorld

def train():
    config = TrainingConfig()

    # Initialize model
    model = TheWorld(
        config.model_name,
        num_world_steps=config.num_world_steps,
        freeze_gemma_vision=config.freeze_gemma_vision,
        freeze_gemma_language=config.freeze_gemma_language,
        freeze_cosmos_vae=config.freeze_cosmos_vae,
    )

    if config.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Load dataset
    train_dataset = TheWorldDataset(...)  # User implements
    eval_dataset = TheWorldDataset(...)

    # Configure HF Trainer
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        bf16=True,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        # ... more args ...
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda x: theworld_collate_fn(x, model.processor),
    )

    trainer.train()
```

**Why HuggingFace Trainer:**
- ✅ Automatic distributed training (DDP, FSDP)
- ✅ Mixed precision support
- ✅ Gradient accumulation
- ✅ Checkpoint saving/loading
- ✅ TensorBoard/WandB logging
- ✅ Evaluation loop
- ❌ More complex than manual training loop
- ❌ Less flexibility for custom training logic

**Alternative: Manual training loop**
```python
# Pros: Full control, easier debugging
# Cons: Must implement checkpointing, distributed training, logging manually
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## 5. Implementation Phases

### Phase 1: Core Infrastructure (Essential)
1. Add gradient checkpointing to `model.py`
2. Add checkpoint save/load methods to `model.py`
3. Create `training_config.py` with TrainingConfig dataclass
4. Update `pyproject.toml` dependencies

**Deliverable:** Can train projection layers with memory efficiency

### Phase 2: Data Pipeline (Required for real training)
1. Create `data_utils.py` with collate function
2. Add example dataset class
3. Document expected data format

**Deliverable:** Can load custom datasets for training

### Phase 3: HuggingFace Integration (Production-ready)
1. Create `train_hf.py` with HF Trainer
2. Add WandB logging support
3. Add Makefile command `make train-hf`
4. Update CLAUDE.md with training instructions

**Deliverable:** Production-ready training pipeline

### Phase 4: Advanced Features (Optional)
1. Add PEFT/LoRA support for parameter-efficient fine-tuning
2. Add Accelerate multi-node training configuration
3. Add evaluation metrics (BLEU, accuracy, custom)
4. Add data augmentation

**Deliverable:** Advanced training capabilities

## 6. Key Design Trade-offs

### 6.1 Checkpoint Size vs. Portability

| Option | Size (projection-only) | Size (full model) | Portability |
|--------|------------------------|-------------------|-------------|
| **Trainable params only** | ~200KB | ~200KB - 12GB* | Requires base models |
| **Full model weights** | 12GB | 12GB | Fully portable |
| **SafeTensors format** | +10% overhead | +10% overhead | More secure |

*Depends on which components unfrozen

**Recommendation:** Save trainable params only (default use case is projection-only training)

### 6.2 Training Framework

| Framework | Pros | Cons | Best For |
|-----------|------|------|----------|
| **HF Trainer** | Auto checkpointing, distributed, logging | Less control, learning curve | Production |
| **Manual loop** | Full control, simple | Manual implementation | Research, debugging |
| **PyTorch Lightning** | Modular, clean | Another framework | Complex experiments |

**Recommendation:** HF Trainer (aligns with HF ecosystem already in use)

### 6.3 Memory Optimization Strategy

| Scenario | Gradient Checkpointing | Batch Size | GPU Memory |
|----------|------------------------|------------|------------|
| **Projection-only** | No | 16-32 | 16GB |
| **+ Vision encoder** | Yes | 4-8 | 24GB |
| **+ Language model** | Yes | 2-4 | 40GB |
| **Full model** | Yes + FSDP | 1 | 24GB (multi-GPU) |

**Recommendation:** Enable gradient checkpointing only when needed (controlled by TrainingConfig)

## 7. Dependencies

### New Dependencies (add to pyproject.toml)
```toml
[project.dependencies]
# ... existing ...
datasets = ">=3.0.0"      # HF datasets
accelerate = ">=1.0.0"    # Distributed training
evaluate = ">=0.4.0"      # Metrics

[dependency-groups.dev]
# ... existing ...
wandb = ">=0.18.0"        # Optional logging
tensorboard = ">=2.18.0"  # Optional logging
```

### Existing Dependencies (no change)
- transformers >= 4.56.0
- diffusers >= 0.34.0
- torch, torchvision
- peft >= 0.15.0

## 8. Migration Path

### For existing users:
1. **No breaking changes** to `model.py` API
2. Existing `train.py` continues to work as demo
3. New training infrastructure is opt-in via `train_hf.py`

### Upgrade path:
```bash
# 1. Update dependencies
uv sync --dev

# 2. (Optional) Try gradient checkpointing
model = TheWorld(..., use_gradient_checkpointing=True)

# 3. Use new training script
make train-hf
```

## 9. Testing Strategy

### Unit Tests
- [ ] Gradient checkpointing enables/disables correctly
- [ ] Checkpoint save/load preserves trainable params
- [ ] Checkpoint save/load preserves optimizer state
- [ ] Collate function handles PIL/tensor/numpy inputs

### Integration Tests
- [ ] Full training loop runs end-to-end
- [ ] Resume from checkpoint continues training correctly
- [ ] Distributed training (2+ GPUs) works
- [ ] WandB logging reports correct metrics

### Manual Testing
- [ ] Train projection-only on single GPU
- [ ] Train full model with gradient checkpointing
- [ ] Resume training from checkpoint
- [ ] Verify checkpoint size is reasonable

## 10. Documentation Updates

### CLAUDE.md
- Add "Training" section with examples
- Document TrainingConfig options
- Document checkpoint format
- Add troubleshooting guide

### New docs
- `docs/training_guide.md`: Comprehensive training tutorial
- `docs/checkpoint_format.md`: Checkpoint structure documentation

## 11. Open Questions

1. **Dataset format:** Should we provide a reference dataset, or only document expected format?
   - **Recommendation:** Document format, let users bring their own data

2. **SafeTensors vs. PyTorch .pt:** Which checkpoint format?
   - **Recommendation:** Start with .pt (simpler), add SafeTensors as option later

3. **Default gradient checkpointing:** Should it be enabled by default?
   - **Recommendation:** No, enable only when needed (users can set in config)

4. **PEFT/LoRA integration:** Should we add LoRA support for even more efficient fine-tuning?
   - **Recommendation:** Phase 4 (advanced features), not essential for v1

## 12. Success Metrics

### Must Have (Phase 1-3)
- [ ] Can train projection layers on single GPU (16GB+)
- [ ] Can resume from checkpoint without loss
- [ ] Checkpoint save/load time < 10s for projection-only
- [ ] Documentation covers basic training workflow

### Nice to Have (Phase 4)
- [ ] Can train full model on 2x A100 (40GB)
- [ ] Training speed within 10% of manual loop
- [ ] WandB dashboard shows all relevant metrics
- [ ] Community adoption (issues, PRs, questions)

## 13. Conclusion

This design provides a **production-ready training infrastructure** for TheWorld while maintaining:
- **Simplicity:** Start with projection-only training (16GB GPU)
- **Scalability:** Expand to full model training (multi-GPU)
- **Ecosystem:** Leverage HuggingFace tools (Trainer, datasets, accelerate)
- **Flexibility:** Users can customize via TrainingConfig

**Recommended implementation order:** Phase 1 → Phase 2 → Phase 3 (Phase 4 optional)

**Estimated effort:**
- Phase 1: ~4 hours (core infrastructure)
- Phase 2: ~2 hours (data pipeline)
- Phase 3: ~4 hours (HF integration + docs)
- **Total:** ~10 hours for production-ready system
