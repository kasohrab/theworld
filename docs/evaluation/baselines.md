# Baseline Comparisons

This guide explains how to set up and evaluate baseline models to measure TheWorld's improvements.

## Why Baselines Matter

To prove that TheWorld's world model provides value, we need to compare against:

1. **Gemma3 Baseline** - Standard vision-language model (no world)
2. **Random Projection** - TheWorld with untrained projection
3. **World Token Ablation** - TheWorld without world tokens at inference

## 1. Gemma3 Baseline

### Overview

Standard Gemma 3 with vision encoder, **without** world model components.

**Purpose:** Measures whether adding world model provides any benefit over standard VLM.

### Setup

```python
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

# Load standard Gemma3 with vision
model = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-4b-it",
    dtype=torch.bfloat16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")
```

### Training (Optional)

If you want a fair comparison on the same data:

```bash
# Train Gemma baseline on your dataset
python scripts/train_baseline_gemma.py --config configs/baseline.json
```

### Evaluation

```bash
# Evaluate on BLINK
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model google/gemma-3-4b-it \
  --output results/gemma_baseline.json
```

### Expected Results

- **If TheWorld helps**: TheWorld accuracy > Gemma by 5-15%
- **If TheWorld doesn't help**: Similar accuracy, world model is redundant

## 2. Random Projection Baseline

### Overview

Full TheWorld architecture but projection layer initialized randomly (not trained).

**Purpose:** Tests whether pretrained Cosmos world knowledge is useful.

### Setup

```python
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    random_projection_init=True,  # ← Random projection
    dtype=torch.bfloat16,
    device_map="auto"
)
```

### Training

```bash
# Train with random projection
python scripts/train_hf.py --config configs/ablation_random_projection.json
```

**Config:**
```json
{
  "model_name": "google/gemma-3-4b-it",
  "enable_world": true,
  "random_projection_init": true,
  "learning_rate": 0.0001,
  "batch_size": 4,
  "num_epochs": 3
}
```

### Expected Results

- **If Cosmos pretrain helps**: Pretrained converges faster and performs better
- **If pretrain doesn't help**: Random init performs similarly

## 3. World Token Ablation

### Overview

Inference-time comparison: same model, with/without world tokens.

**Purpose:** Measures direct contribution of world embeddings.

### Setup

```python
model = TheWorld.from_pretrained("username/theworld-datacomp")

# Test same input with different configurations
image = Image.open("test.jpg")
question = "What is in this image?"

# 1. With world tokens (default)
response_with = model.generate(
    image,
    question,
    num_world_steps=4
)

# 2. Minimal world tokens
response_min = model.generate(
    image,
    question,
    num_world_steps=0
)

# 3. No world tokens (ablation)
response_without = model.generate(
    image,
    question,
    skip_world_tokens=True
)
```

### Expected Results

- **If world tokens help**: Performance drops without them
- **If world tokens don't help**: Similar performance either way

## Comparison Workflow

### 1. Train/Prepare Models

```bash
# Train TheWorld
python scripts/train_hf.py --config configs/datacomp_production.json

# Train Gemma baseline (optional)
python scripts/train_baseline_gemma.py --config configs/baseline.json

# Train random projection
python scripts/train_hf.py --config configs/ablation_random_projection.json
```

### 2. Evaluate All Models

```bash
# TheWorld
python scripts/evaluate_blink.py \
  --model username/theworld-datacomp \
  --output results/theworld.json

# Gemma baseline
python scripts/evaluate_blink.py \
  --model google/gemma-3-4b-it \
  --output results/gemma.json

# Random projection
python scripts/evaluate_blink.py \
  --model username/theworld-random-proj \
  --output results/random.json
```

### 3. Compare Results

```bash
# Generate comparison report
python scripts/compare_results.py \
  --theworld results/theworld.json \
  --gemma results/gemma.json \
  --random results/random.json \
  --output results/comparison.md \
  --print
```

## Interpretation Guide

### Strong Evidence World Model Helps

- TheWorld > Gemma baseline by 10-15%
- TheWorld > Random projection by 5-10%
- With world tokens > Without world tokens by 5%+
- world_steps=4 > world_steps=0

### Weak Evidence

- TheWorld > Gemma baseline by 2-5%
- Inconsistent improvements across tasks
- High variance in results

### No Evidence

- TheWorld ≈ Gemma baseline (±2%)
- Random projection ≈ Pretrained projection
- With world tokens ≈ Without world tokens

## Statistical Significance

When comparing models, use:

```python
from scipy.stats import ttest_ind

# Collect per-sample accuracies
theworld_scores = [...]  # Per-sample accuracy
baseline_scores = [...]

# Run t-test
t_stat, p_value = ttest_ind(theworld_scores, baseline_scores)

if p_value < 0.05:
    print("Difference is statistically significant!")
else:
    print("Difference may be due to random variation")
```

## Common Pitfalls

1. **Different training data** - Baselines must train on same data
2. **Different hyperparameters** - Use same learning rate, batch size, etc.
3. **Different evaluation setup** - Same prompts, same preprocessing
4. **Too few samples** - Use at least 200+ test samples for reliable results
5. **Imbalanced data** - Check class distribution in training/eval

## Related Documentation

- [Evaluation Overview](overview.md) - Metrics and methodology
- [BLINK Benchmark](benchmarks/blink.md) - BLINK evaluation guide
- [Multi-Stage Training](../training/multi-stage.md) - Improving performance
