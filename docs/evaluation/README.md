# Evaluation Documentation

Guide to evaluating TheWorld and measuring improvements over baselines.

## Quick Links

- **[Overview](overview.md)** - Training objective, metrics, and evaluation strategy
- **[BLINK Benchmark](benchmarks/blink.md)** - Spatial and depth perception evaluation
- **[SpatialRGPT Benchmark](benchmarks/spatial-rgpt.md)** - Spatial reasoning evaluation
- **[Baseline Comparisons](baselines.md)** - Setting up and comparing against baselines

## Quick Start

**Evaluate on BLINK:**

```bash
# Using Makefile
make eval-blink MODEL=username/theworld-datacomp
make eval-gemma
make compare-results

# Manual
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp \
  --output results/theworld.json
```

## Training Objective

TheWorld uses **causal language modeling** (next-token prediction):
- **Loss**: Cross-entropy on text tokens only
- **Masked**: Vision and world tokens (label = -100)
- **Learns**: Projection layer maps Cosmos → Gemma space

## Key Metrics

### Accuracy (Primary)

Percentage of correct predictions. Simple and interpretable.

**Expected performance:**
- BLINK Relative_Depth: 70-85% (good)
- BLINK Spatial_Relation: 60-75% (good)
- Improvement over baseline: +5-15%

### Perplexity

Exponential of cross-entropy loss. Lower is better.

**Use for:**
- Monitoring during training
- Comparing model configurations
- Sanity checking (should be < 100)

### F1 Score

Macro and weighted F1 for multi-class tasks.

**Use when:**
- Dataset has imbalanced classes
- Need detailed per-class performance

## Baseline Comparisons

To prove world model helps, compare against:

1. **Gemma3 Baseline** - Standard VLM (no world model)
2. **Random Projection** - TheWorld with untrained projection
3. **World Token Ablation** - Same model, with/without world tokens

See [Baseline Comparisons](baselines.md) for setup.

## Evaluation Workflow

### 1. Train Model

```bash
python scripts/train_hf.py --config configs/datacomp_production.json
```

### 2. Evaluate on Benchmark

```bash
# TheWorld
python scripts/evaluate_blink.py \
  --model username/theworld-datacomp \
  --output results/theworld.json

# Gemma baseline
python scripts/evaluate_blink.py \
  --model google/gemma-3-4b-it \
  --output results/gemma.json
```

### 3. Compare Results

```bash
python scripts/compare_results.py \
  --theworld results/theworld.json \
  --baseline results/gemma.json \
  --output results/comparison.md \
  --print
```

### 4. Interpret

**Strong evidence world model helps:**
- TheWorld > Gemma by 10-15%
- world_steps=4 > world_steps=0
- Consistent across tasks

**No evidence:**
- TheWorld ≈ Gemma (±2%)
- No improvement with world tokens

## Benchmarks

### BLINK

Tests spatial and depth perception:
- **Relative_Depth**: Which object is closer?
- **Spatial_Relation**: How are objects positioned?

See [BLINK Benchmark](benchmarks/blink.md)

### SpatialRGPT

Tests spatial reasoning with region references:
- Qualitative questions ("Is Region [0] behind Region [1]?")
- Quantitative questions ("What is the height of Region [0]?")

See [SpatialRGPT Benchmark](benchmarks/spatial-rgpt.md)

## Ablation Studies

### World Steps

```python
# Compare different temporal rollouts
for num_steps in [0, 2, 4, 8]:
    outputs = model.generate(**inputs, num_world_steps=num_steps)
    # Evaluate...
```

### Component Unfreezing

Compare projection-only vs vision-unfrozen vs language-unfrozen.

### Dataset Size

Test on 1K, 10K, 100K, 1M samples to measure scaling.

## Troubleshooting

**World model not helping:**
- Task may not need temporal/spatial info
- Need more training data
- Try unfreezing more components

**Random-level accuracy:**
- Check prompt formatting
- Verify model trained on QA data
- Review training data balance

**High perplexity but good accuracy:**
- Normal for multiple-choice tasks
- Focus on accuracy as primary metric

See [Troubleshooting Guide](../guides/troubleshooting.md) for more.

## Related Documentation

- [Training Guide](../training/README.md) - Train models to evaluate
- [Inference Guide](../guides/inference.md) - Run inference
- [Architecture Overview](../architecture/overview.md) - Understand the model
