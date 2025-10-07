# Evaluation Guide

Quick reference for evaluating TheWorld model against baselines.

## Training Objective

TheWorld uses **causal language modeling** (next-token prediction):
- **Loss**: Cross-entropy on text tokens only
- **Masked**: Vision and world embeddings (label = -100, excluded from loss)
- **Gradient flow**: Loss → Gemma (frozen) → **Projection (trained)** → Cosmos (frozen)

The projection layer learns to map Cosmos's 16-dim world space to Gemma's 2304-dim embedding space.

## Baseline Comparisons

Test if world model helps by comparing against:

1. **Gemma3 Baseline** - Standard vision-language model (no world tokens)
2. **Random Projection** - TheWorld with random projection init (tests if pretrained Cosmos helps)
3. **World Token Ablation** - TheWorld with `skip_world_tokens=True` (tests if world tokens contribute)

## Quick Evaluation

### BLINK Benchmark

Tests spatial/depth perception on Relative_Depth and Spatial_Relation tasks.

```bash
# Evaluate TheWorld
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp \
  --num_world_steps 0,4 \
  --output results/theworld_blink.json

# Evaluate Gemma baseline
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model gemma3-baseline \
  --output results/gemma_blink.json

# Compare results
python scripts/compare_results.py \
  --theworld results/theworld_blink.json \
  --baseline results/gemma_blink.json \
  --output results/comparison.md \
  --print
```

**Or use Makefile:**
```bash
make eval-blink MODEL=username/theworld-datacomp
make eval-gemma
make compare-results
```

### Expected Results

**Good performance:**
- Relative_Depth: 70-85% accuracy
- Spatial_Relation: 60-75% accuracy
- TheWorld outperforms Gemma baseline by 5-15%
- world_steps=4 improves over world_steps=0

**Poor performance (needs investigation):**
- Random-level accuracy (<55% for Relative_Depth)
- No improvement with world tokens
- TheWorld ≈ Gemma baseline

## Metrics

- **Accuracy**: Primary metric (% correct)
- **F1 Macro**: Unweighted average across choices
- **F1 Weighted**: Weighted by class frequency
- **Confusion Matrix**: Shows prediction patterns

## Ablation Studies

```python
from theworld import TheWorld

model = TheWorld.from_pretrained("username/theworld-datacomp")

# 1. With world tokens (default)
response_full = model.generate(image, question, num_world_steps=4)

# 2. Minimal world tokens
response_min = model.generate(image, question, num_world_steps=0)

# 3. No world tokens (ablation)
response_ablation = model.generate(image, question, skip_world_tokens=True)

# 4. Random projection (train with random_projection_init=True)
model_random = TheWorld("google/gemma-3-4b-it", random_projection_init=True)
```

## Troubleshooting

**World model not helping:**
- Check projection layer gradients during training
- Increase projection learning rate (1e-4 → 5e-4)
- Try unfreezing more components (Stage 2/3 training)
- Verify dataset has temporal/spatial tasks

**Random-level accuracy:**
- Check prompt formatting
- Verify model trained on QA data
- Review training data balance

See [Multi-Stage Training](multi_stage_training.md) for progressive unfreezing strategy.
