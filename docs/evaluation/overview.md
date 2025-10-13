# Evaluation Overview

This guide explains TheWorld's training objective and how to evaluate whether the world model fusion provides benefits.

## Table of Contents

1. [Training Objective](#training-objective)
2. [Baseline Comparisons](#baseline-comparisons)
3. [Evaluation Metrics](#evaluation-metrics)
4. [Quick Evaluation](#quick-evaluation)
5. [Ablation Studies](#ablation-studies)

---

## Training Objective

### Causal Language Modeling

TheWorld uses **next-token prediction** as its training objective, identical to standard causal language models.

**Cross-Entropy Loss:**
```
L = - (1/N) Σ log P(y_i | x_{<i}, vision, world)
```

Where:
- `y_i` = target token at position i
- `x_{<i}` = all previous text tokens
- `vision` = Gemma SigLIP visual features
- `world` = Projected Cosmos world embeddings
- `N` = number of tokens in sequence

### Token Sequence Structure

```
[<start>] [text] <the_world_start> [world_tokens×784] <the_world_end> [image_tokens×256] [prompt] [answer] [<end>]
```

**Token Counts (typical):**
- Chat template: ~10 tokens
- World tokens: 784 tokens (28×28 spatial grid)
- Image tokens: ~256 tokens (SigLIP features)
- Prompt: Variable (~6 tokens)
- Answer: Variable (~7 tokens)

**Total:** ~1,071 tokens per example

### Label Masking (Critical!)

The model uses **selective masking** to compute loss only on text tokens:

```python
labels_before = input_ids[:, :num_before_start]     # Template tokens
labels_world = torch.full((b, num_world), -100)     # -100 = IGNORE
labels_after = input_ids[:, end_pos:]                # Prompt + answer tokens

combined_labels = torch.cat([labels_before, labels_world, labels_after], dim=1)
```

**Why -100?**
- `-100` is the **ignore index** in PyTorch CrossEntropyLoss
- Tokens with label -100 are excluded from loss and gradients
- Prevents model from trying to "predict" embeddings

**What gets loss computed:**
- ❌ **World tokens**: Ignored (label = -100)
- ❌ **Image tokens**: Ignored (label = -100)
- ❌ **Special tokens**: Ignored (label = -100)
- ✅ **Text tokens**: Loss computed

### Gradient Flow

With default configuration (projection-only training):

**Forward Pass:**
1. Image → Cosmos VAE (frozen) → 16-dim latent
2. 16-dim latent → **Projection layer (trainable)** → 2304-dim embedding
3. Combined with Gemma vision (frozen) + text embeddings
4. Through Gemma language model (frozen) → logits
5. Compute cross-entropy loss on text tokens

**Backward Pass:**
```
Loss (text tokens only)
   ↓
Gemma LM (frozen)
   ↓
Projection layer (UPDATE!)
   ↓
Cosmos VAE (frozen)
```

**What the projection layer learns:**
- Map Cosmos's world understanding to Gemma-compatible representations
- Encode temporal/physical dynamics to improve text predictions
- Bridge the modality gap between world model and language model

---

## Baseline Comparisons

Test if world model helps by comparing against:

### 1. Gemma3 Baseline (Vision-Language Only)

**What:** Standard Gemma 3 with vision encoder, **without** world model.

**Purpose:** Measures whether adding world model provides any benefit over standard vision-language model.

**Expected Results:**
- If TheWorld helps: TheWorld accuracy > Gemma baseline (5-15% improvement expected)
- If TheWorld doesn't help: Similar performance, world model is redundant

**How to evaluate:**
```bash
# Evaluate Gemma baseline
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model gemma3-baseline \
  --output results/gemma_blink.json
```

### 2. Random Projection Baseline

**What:** TheWorld architecture but with **random initialization** of projection layer.

**Purpose:** Measures whether pretrained Cosmos world knowledge is useful.

**Expected Results:**
- If Cosmos pretrain helps: Pretrained projection converges faster and performs better
- If pretrain doesn't help: Random init performs similarly

**How to train:**
```bash
# TheWorld with random projection init
python scripts/train_hf.py --config configs/ablation_random_projection.json
```

### 3. World Tokens Ablation

**What:** Inference-time comparison with and without world tokens.

**Purpose:** Measures contribution of world embeddings to final predictions.

**Expected Results:**
- If world tokens help: Performance drops without them
- If world tokens don't help: Similar performance either way

**How to evaluate:**
```python
model = TheWorld.from_pretrained("username/theworld-datacomp")

# With world tokens (default)
response_with = model.generate(image, question, num_world_steps=4)

# Minimal world tokens
response_min = model.generate(image, question, num_world_steps=0)

# Without world tokens (ablation)
response_without = model.generate(image, question, skip_world_tokens=True)
```

---

## Evaluation Metrics

### Accuracy (Primary Metric)

**Definition:** Percentage of correct predictions

**Usage:**
- Primary metric for question-answering tasks
- Clear interpretation: 70% = 70% correct
- Easy to compare across models

**Expected Results:**
- BLINK Relative_Depth: 70-85% (good), <55% (poor)
- BLINK Spatial_Relation: 60-75% (good), <50% (poor)
- Improvement over baseline: +5-15% (good)

### F1 Score

**F1 Macro:** Unweighted average across choices (treats all options equally)
**F1 Weighted:** Weighted by class frequency (accounts for imbalanced data)

**When to use:**
- Imbalanced datasets (some choices more common)
- Multi-class classification problems

### Perplexity

**Definition:** Exponential of cross-entropy loss

**Formula:**
```python
perplexity = torch.exp(loss)
```

**Interpretation:**
- Lower is better (model is more confident/accurate)
- Perplexity of 1.0 = perfect predictions
- Perplexity of 100 = model is very uncertain

**Usage:**
- Measure during training (not just evaluation)
- Compare different model configurations
- Track improvement over time

---

## Quick Evaluation

### BLINK Benchmark

Tests spatial/depth perception on Relative_Depth and Spatial_Relation tasks.

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

---

## Ablation Studies

### 1. World Steps Comparison

Test different temporal rollout lengths:

```python
# Single frame (t=0 only)
response_0 = model.generate(image, question, num_world_steps=0)

# Future prediction (t=0, t=1, t=2, t=3, t=4)
response_4 = model.generate(image, question, num_world_steps=4)
```

**Expected:** More world steps help for temporal/dynamic tasks, minimal impact for static tasks.

### 2. Component Unfreezing

Compare different training configurations:

```python
# Projection only (default)
model_proj = TheWorld.from_pretrained(
    "username/theworld-projection-only"
)

# Projection + Vision
model_vision = TheWorld.from_pretrained(
    "username/theworld-vision-unfrozen"
)

# Projection + Language
model_lang = TheWorld.from_pretrained(
    "username/theworld-language-unfrozen"
)
```

**Expected:** More trainable parameters help for domain-specific tasks, but increase training cost.

### 3. Dataset Size Impact

Compare training on different dataset sizes:

- 1K samples (quick test)
- 10K samples (small experiment)
- 100K samples (production)
- 1M+ samples (full scale)

**Expected:** Larger datasets improve generalization, especially for projection layer.

---

## Troubleshooting

### World model not helping

**Possible causes:**
- Projection layer not learning (check gradients)
- Learning rate too low (try 1e-4 → 5e-4)
- Dataset doesn't benefit from temporal/spatial info
- Need to unfreeze more components (Stage 2/3 training)

**Solutions:**
- Check projection layer gradients during training
- Increase projection learning rate
- Try unfreezing more components
- Verify dataset has temporal/spatial tasks

### Random-level accuracy

**Possible causes:**
- Prompt formatting incorrect
- Model not trained on QA data
- Training data imbalanced
- Evaluation setup incorrect

**Solutions:**
- Check prompt formatting matches training
- Verify model trained on similar QA tasks
- Review training data distribution
- Test with known-good examples

### High perplexity but good accuracy

**Explanation:** Model is correct but not confident. Common with multiple-choice tasks.

**Solutions:**
- Focus on accuracy as primary metric
- Check if answer distribution is imbalanced
- Consider calibration techniques

---

## Related Documentation

- [BLINK Benchmark](benchmarks/blink.md) - Detailed BLINK evaluation guide
- [SpatialRGPT Benchmark](benchmarks/spatial-rgpt.md) - Spatial reasoning evaluation
- [Baseline Comparisons](baselines.md) - Setting up baseline models
- [Multi-Stage Training](../training/multi-stage.md) - Progressive unfreezing for better performance
