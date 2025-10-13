# BLINK Benchmark Evaluation

BLINK is a benchmark for evaluating vision-language models on spatial and depth perception tasks.

## Overview

BLINK tests models on understanding:
- **Relative_Depth**: Which object is closer/farther?
- **Spatial_Relation**: How are objects positioned relative to each other?

## Quick Start

**Using Makefile (Recommended):**
```bash
# Evaluate TheWorld
make eval-blink MODEL=username/theworld-datacomp

# Evaluate Gemma baseline
make eval-gemma

# Compare results
make compare-results
```

**Manual Evaluation:**
```bash
# Evaluate on Relative_Depth task
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp \
  --num_world_steps 0,4 \
  --output results/theworld_blink.json
```

## Tasks

### Relative_Depth

Tests understanding of which object is closer to the camera.

**Example:**
- Image: Person in foreground, tree in background
- Question: "Is the person closer than the tree?"
- Expected answer: "Yes"

**Metrics:**
- Accuracy: % correct predictions
- Expected performance: 70-85% (good), <55% (poor)

### Spatial_Relation

Tests understanding of object spatial relationships (left/right, above/below, etc.).

**Example:**
- Image: Cup to the left of laptop
- Question: "Is the cup to the left of the laptop?"
- Expected answer: "Yes"

**Metrics:**
- Accuracy: % correct predictions
- Expected performance: 60-75% (good), <50% (poor)

## Configuration

```python
# Evaluate with different world steps
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp \
  --num_world_steps 0    # Single frame only

python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp \
  --num_world_steps 4    # With temporal rollout
```

## Expected Results

**Good Performance:**
- TheWorld outperforms Gemma baseline by 5-15%
- world_steps=4 improves over world_steps=0
- Relative_Depth: 70-85% accuracy
- Spatial_Relation: 60-75% accuracy

**Poor Performance (needs investigation):**
- Random-level accuracy
- No improvement with world tokens
- TheWorld ≈ Gemma baseline

See [Evaluation Overview](../overview.md#troubleshooting) for troubleshooting tips.

## Output Format

Results are saved as JSON:

```json
{
  "task": "Relative_Depth",
  "model": "username/theworld-datacomp",
  "num_world_steps": 4,
  "accuracy": 0.78,
  "f1_macro": 0.76,
  "f1_weighted": 0.77,
  "confusion_matrix": [[...], [...]]
}
```

## Comparison Against Baselines

```bash
# Compare TheWorld vs Gemma baseline
python scripts/compare_results.py \
  --theworld results/theworld_blink.json \
  --baseline results/gemma_blink.json \
  --output results/comparison.md \
  --print
```

## Related Documentation

- [Evaluation Overview](../overview.md) - Metrics and baselines
- [Baseline Comparisons](../baselines.md) - Setting up baseline models
- [SpatialRGPT Benchmark](spatial-rgpt.md) - Alternative spatial reasoning benchmark
