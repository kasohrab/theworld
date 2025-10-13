# Training TheWorld on Visual Spatial Reasoning (VSR)

This guide covers training TheWorld with world embeddings enabled on the Visual Spatial Reasoning (VSR) benchmark.

## Dataset Overview

**Visual Spatial Reasoning (VSR)** is a binary visual entailment benchmark that evaluates spatial understanding in vision-language models.

- **Task**: Binary classification (True/False) for spatial relation statements
- **Format**: Each sample contains an image and a caption describing spatial relations (e.g., "A dog is to the left of a cat")
- **Splits**:
  - Train: ~8,777 samples
  - Validation: ~1,098 samples
  - Test: ~1,097 samples
- **Variants**: `random` (random object pairs) and `zeroshot` (unseen spatial relations)

**Paper**: [Visual Spatial Reasoning (TACL 2023)](https://arxiv.org/abs/2205.00363)
**Dataset**: [cambridgeltl/vsr_random](https://huggingface.co/datasets/cambridgeltl/vsr_random)

## Setup

### 1. Download VSR Images

Images are already available at:
```bash
/home/hice1/ksohrab3/scratch/theworld/data/images/
```

To download them yourself:
```bash
bash scripts/vsr/get_vsr_images.sh
```

### 2. Configure Training

The training configuration is at `configs/vsr_training.json`:

```json
{
  "model_name": "google/gemma-3-4b-it",
  "cosmos_model_name": "nvidia/Cosmos-Predict2-2B-Video2World",
  "num_world_steps": 0,  // World embeddings enabled, no temporal prediction

  "dataset_name": "vsr",
  "question_template": "Statement: {caption}\nAnswer (only '0' or '1'):",
  "image_folder": "/home/hice1/ksohrab3/scratch/theworld/data/images",

  "freeze_gemma_vision": true,
  "freeze_gemma_language": true,
  "freeze_cosmos_vae": true,  // Only train projection layers

  "batch_size": 8,
  "num_epochs": 3,
  "learning_rate": 0.0001
}
```

**Key settings**:
- `num_world_steps: 0` - Encodes current frame only (no future prediction)
- Only projection layers are trainable (~0.07% of parameters)
- Binary output format: Model learns to output "0" or "1"
- Loss computed **only on the label token** (not the prompt)
- Uses `TheWorld.from_pretrained()` for proper initialization (dtype, device_map, buffer handling)

## Training

### Start Training

```bash
# Basic training
python scripts/train_hf.py --config configs/vsr_training.json

# With WandB logging (set your token)
export WANDB_API_KEY=your_wandb_key
python scripts/train_hf.py --config configs/vsr_training.json
```

### Monitor Training

**TensorBoard**:
```bash
tensorboard --logdir checkpoints/theworld-vsr/logs
```

**Weights & Biases**:
- Set `log_to_wandb: true` in config
- Project: `theworld-vsr`
- Run name: `vsr-projection-world-enabled`

### Expected Training Time

- **~8,777 training samples** × 3 epochs = ~26,331 forward passes
- Batch size 8, gradient accumulation 2 → effective batch size 16
- **~1,645 gradient updates** per epoch
- **~5 hours on A100** (single GPU)

### Memory Requirements

- **Projection-only training**: ~20-24GB VRAM
- **Mixed precision (bf16)**: Enabled by default
- **Gradient checkpointing**: Not needed for projection-only

## Loss Computation

The collator (`python/theworld/data.py`) implements proper causal LM training:

1. **Format input**: `[USER: <image> + question] [ASSISTANT: 0 or 1]`
2. **Tokenize full sequence**: Get input_ids for entire conversation
3. **Mask prompt tokens**: Set prompt portion to `-100` in labels
4. **Compute loss**: Cross-entropy loss **only on the single label token** (0 or 1)

This ensures the model learns spatial reasoning (not just memorizing prompts).

## Evaluation

After training, evaluate on the test set:

```bash
python scripts/vsr/evaluate_vsr.py \
    --model_name checkpoints/theworld-vsr/checkpoint-1500 \
    --image-dir /home/hice1/ksohrab3/scratch/theworld/data/images \
    --split test \
    --output vsr_test_results.jsonl
```

## Visualizing Projection Alignment

After training, visualize how well the Cosmos world embeddings align with Gemma's embedding space:

```bash
python scripts/visualize_projection_alignment.py \
    --model checkpoints/theworld-vsr/checkpoint-1500 \
    --dataset vsr \
    --num_samples 100 \
    --output visualizations/vsr_projection_alignment.png
```

This generates:
- **PCA plot**: 2D projection of embedding spaces
- **t-SNE plot**: Nonlinear dimensionality reduction
- **Cosine similarity distribution**: How aligned are the embeddings?
- **Metrics**:
  - Mean cosine similarity (higher = better alignment)
  - L2 distance (lower = better alignment)
  - Embedding magnitudes

### Interpreting Results

**Good alignment** (projection layer learned well):
- Mean cosine similarity > 0.5
- Projected and Gemma embeddings overlap in PCA/t-SNE
- Similar embedding magnitudes

**Poor alignment** (needs more training or different hyperparameters):
- Mean cosine similarity < 0.3
- Embeddings form separate clusters
- Large difference in magnitudes

## Checkpoints

Checkpoints are saved to `checkpoints/theworld-vsr/`:
- `checkpoint-500/` - Every 500 steps
- `checkpoint-1000/`
- `checkpoint-1500/`
- ...

**Resume training**:
```bash
python scripts/train_hf.py \
    --config configs/vsr_training.json \
    --resume_from checkpoints/theworld-vsr/checkpoint-1000
```

## HuggingFace Hub Upload

To upload checkpoints during training:

1. Edit `configs/vsr_training.json`:
```json
{
  "push_to_hub": true,
  "hub_model_id": "your-username/theworld-vsr",
  "hub_strategy": "every_save"
}
```

2. Set your token:
```bash
export HF_TOKEN=hf_your_token_here
```

3. Train (uploads automatically):
```bash
python scripts/train_hf.py --config configs/vsr_training.json
```

## Troubleshooting

### Images not loading

Check that images are in the correct location:
```bash
ls /home/hice1/ksohrab3/scratch/theworld/data/images/ | head
# Should show: 000000000142.jpg, 000000000370.jpg, ...
```

### Out of memory

Reduce batch size in config:
```json
{
  "batch_size": 4,  // Reduce from 8
  "gradient_accumulation_steps": 4  // Increase to maintain effective batch size
}
```

### Loss not decreasing

Check that labels are being computed correctly:
```python
# In training logs, you should see:
# loss: ~0.69 (initial, random 50/50 binary classification)
# loss: ~0.3-0.4 (after training, model learning spatial patterns)
```

If loss stays at ~0.69, the model isn't learning. Try:
- Increase learning rate to `2e-4`
- Train for more epochs
- Check that world embeddings are being used (not accidentally disabled)

## Advanced: Multi-Stage Training

For better performance, consider multi-stage training:

1. **Stage 1**: Train projection only (current setup)
2. **Stage 2**: Unfreeze Gemma vision encoder
3. **Stage 3**: Unfreeze Gemma language model (if needed)

See [docs/multi_stage_training.md](../multi_stage_training.md) for details.

## Expected Results

**Baseline (Gemma-3-4B without world embeddings)**:
- VSR random test accuracy: ~65-70%

**TheWorld (with world embeddings)**:
- Expected accuracy: **70-75%** (5-10% improvement from spatial world model)

The world model provides additional geometric understanding that complements Gemma's vision encoder.

## References

- [VSR Paper (TACL 2023)](https://arxiv.org/abs/2205.00363)
- [VSR Dataset](https://huggingface.co/datasets/cambridgeltl/vsr_random)
- [VSR GitHub](https://github.com/cambridgeltl/visual-spatial-reasoning)
- [TheWorld Project](../../README.md)
