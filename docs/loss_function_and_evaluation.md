# Loss Function and Evaluation Guide

This guide explains the training objective for TheWorld model and how to measure whether the world model fusion provides benefits compared to baseline vision-language models.

## Table of Contents

1. [Loss Function Explanation](#loss-function-explanation)
2. [Baseline Comparisons](#baseline-comparisons)
3. [Evaluation Metrics](#evaluation-metrics)
4. [BLINK Benchmark Evaluation](#blink-benchmark-evaluation)
5. [Experimental Setup](#experimental-setup)
6. [Interpretation Guide](#interpretation-guide)

---

## Loss Function Explanation

### Training Objective: Causal Language Modeling

TheWorld uses **next-token prediction** as its training objective, identical to standard causal language models (GPT, Llama, Gemma). The model learns to predict the next token given all previous context, including visual and world model information.

### Mathematical Formulation

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

**Implementation:**
```python
# theworld/modeling.py lines 577-580
loss_fct = nn.CrossEntropyLoss()  # Automatically ignores -100 labels
shift_logits = logits[..., :-1, :].view(-1, vocab_size)
shift_labels = labels[..., 1:].view(-1)
loss = loss_fct(shift_logits, shift_labels)
```

### Token Sequence Structure

Each training example is processed as a combined sequence:

```
[<start>] [text_before] <the_world_start> [world_tokens] <the_world_end> [image_tokens] [prompt] [answer] [<end>]
     ↓           ↓              ↓               ↓               ↓          ↓         ↓         ↓
  Chat      Template    World bracket    784 projected    Vision     User     Model      EOS
  token     tokens         token          embeddings      features   question  response   token
```

**Token Counts (typical):**
- Chat template: ~10 tokens
- World tokens: 784 tokens (28×28 spatial, 1 temporal for single-step)
- Image tokens: ~264 tokens (SigLIP features)
- Prompt: Variable (e.g., "What is in this image?" = ~6 tokens)
- Answer: Variable (e.g., "A cat sitting on a couch." = ~7 tokens)

**Total sequence length:** ~1,071 tokens per example (single-step world model)

### Label Masking (Critical!)

The model uses **selective masking** to compute loss only on text tokens:

```python
# theworld/modeling.py lines 538-543
labels_before = input_ids[:, :num_before_start]     # Template tokens
labels_world = torch.full((b, num_world), -100)     # -100 = IGNORE
labels_after = input_ids[:, end_pos:]                # Prompt + answer tokens

combined_labels = torch.cat([labels_before, labels_world, labels_after], dim=1)
```

**Why -100?**
- `-100` is the **ignore index** in PyTorch CrossEntropyLoss
- Tokens with label -100 are excluded from loss computation and gradients
- This prevents the model from trying to "predict" embeddings

**What gets loss computed:**
- ❌ **World embedding tokens**: Ignored (label = -100)
- ❌ **Image embedding tokens**: Ignored (label = -100)
- ❌ **Special tokens** (`<the_world_start>`, etc.): Ignored (label = -100)
- ✅ **Text tokens** (prompt + answer): Loss computed
- ✅ **Chat template tokens**: Loss computed

**Result:** Model learns to generate text conditioned on visual and world context, without trying to predict the embeddings themselves.

### Shifted Prediction (Autoregressive)

Like all causal language models, predictions are shifted by 1 position:

```python
# theworld/modeling.py lines 573-575
shift_logits = logits_float[..., :-1, :].contiguous()  # Predict positions 1 to N
shift_labels = combined_labels[..., 1:].contiguous()   # Using inputs 0 to N-1
```

**Example:**
```
Position:     0      1     2      3       4      5    6     7       8
Input:     [What] [is]  [in]  [this]  [image] [?]  [A]  [cat]  [sitting]
Target:     [is]  [in]  [this] [image]  [?]   [A]  [cat] [sitting] [<eos>]
             ↑     ↑      ↑      ↑        ↑     ↑    ↑      ↑        ↑
          Token at position i predicts token at position i+1
```

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
Gemma LM (frozen, no gradient update)
   ↓
Projection layer (trainable, UPDATE weights!)
   ↓
Cosmos VAE (frozen, no gradient update)
```

**What the projection layer learns:**
- Map Cosmos's world understanding to Gemma-compatible representations
- Encode temporal/physical dynamics in a way that improves text predictions
- Bridge the modality gap between world model and language model

### Example Walkthrough

**Training Example:**
```python
{
    "image": <cat on couch>,
    "text": "What is in this image?",
    "label": "A cat sitting on a couch."
}
```

**Step-by-step:**

1. **Process image through Cosmos:**
   ```
   Image → Cosmos VAE → 16-dim latent (1, 16, 1, 28, 28)
   Add temporal embedding for t=0
   ```

2. **Project to Gemma space:**
   ```
   16-dim latent → Projection layer → 2304-dim embedding (1, 784, 2304)
   ```

3. **Process image through Gemma vision:**
   ```
   Image → SigLIP → Vision features (1, 264, 2304)
   ```

4. **Create combined sequence:**
   ```
   [template] <world_start> [784 world tokens] <world_end> [264 image tokens]
   "What is in this image?" "A cat sitting on a couch."
   ```

5. **Create labels with masking:**
   ```
   [template tokens] [-100 × 784] [-100 × 264]
   [What] [is] [in] [this] [image] [?] [A] [cat] [sitting] [on] [a] [couch] [.]
   ```

6. **Forward through language model:**
   ```
   Combined embeddings → Gemma LM → Logits (1, seq_len, 256,000)
   ```

7. **Compute loss (only on text):**
   ```
   CrossEntropy(
       predicted: logits for "is", "in", "this", ..., "couch", "."
       target:    "is", "in", "this", ..., "couch", ".", <eos>
   )
   ```

8. **Backprop gradients:**
   ```
   Loss → Gemma LM (frozen) → Projection (UPDATE!) → Cosmos (frozen)
   ```

**Result:** Projection layer learns to make world information useful for predicting "A cat sitting on a couch."

---

## Baseline Comparisons

To measure if the world model fusion actually helps, compare against these baselines:

### 1. Gemma 3 Baseline (Vision-Language Only)

**What:** Standard Gemma 3 with vision encoder, **without** world model.

**Setup:**
- Same architecture: Gemma 3 + SigLIP
- Same training data: DataComp-1B or your dataset
- Same hyperparameters: Learning rate, batch size, etc.
- Remove: Cosmos pipeline, world tokens, projection layer

**Purpose:** Measures whether adding world model provides any benefit over standard vision-language model.

**Expected Results:**
- If TheWorld helps: TheWorld perplexity < Gemma baseline (10-20% lower expected)
- If TheWorld doesn't help: Similar perplexity, world model is redundant

**How to train:**
```bash
# Train Gemma baseline (without world model)
# Note: Requires implementing a baseline training script
python scripts/train_baseline_gemma.py --config configs/baseline.json
```

### 2. Random Projection Baseline

**What:** TheWorld architecture but with **random initialization** of projection layer (no pretrained Cosmos knowledge).

**Setup:**
- Full TheWorld architecture
- Cosmos pipeline still frozen
- Projection layer: Random initialization instead of being trained
- Train projection from scratch

**Purpose:** Measures whether pretrained Cosmos world knowledge is useful, or if it's just about having more parameters.

**Expected Results:**
- If Cosmos pretrain helps: Pretrained projection converges faster and better
- If pretrain doesn't help: Random init performs similarly

**How to train:**
```bash
# TheWorld with random projection init
python scripts/train_hf.py --config configs/ablation_random_projection.json
```

### 3. World Tokens Ablation

**What:** Inference-time comparison with and without world tokens.

**Setup:**
- Same trained model
- Run inference twice:
  - With world tokens: `num_world_steps=0` (or 4)
  - Without world tokens: Remove world tokens entirely from sequence

**Purpose:** Measures contribution of world embeddings to final predictions.

**Expected Results:**
- If world tokens help: Performance drops without them
- If world tokens don't help: Similar performance either way

**How to evaluate:**
```python
# Compare with/without world tokens
model = TheWorld.from_pretrained("username/theworld-datacomp")

# With world tokens (default)
response_with = model.generate(image, question, num_world_steps=0)

# Without world tokens (ablation)
# This requires a special mode to skip world token insertion
response_without = model.generate(image, question, skip_world_tokens=True)
```

---

## Evaluation Metrics

### Quantitative Metrics

#### 1. Perplexity (Primary Metric)

**Definition:** Exponential of cross-entropy loss. Measures model's prediction confidence.

**Formula:**
```python
perplexity = torch.exp(loss)
```

**Interpretation:**
- Lower is better (model is more confident/accurate)
- Perplexity of 1.0 = perfect predictions
- Perplexity of 100 = model is very uncertain

**Usage:**
```bash
# Compute perplexity on test set
python scripts/evaluate.py \
  --model username/theworld-datacomp \
  --test_data data/test.json \
  --metric perplexity
```

**Expected Results:**
- TheWorld: 15-25 (depending on dataset)
- Gemma baseline: 18-30
- **Goal:** TheWorld < Gemma by 10-20%

#### 2. Generation Quality Metrics

**BLEU (Bilingual Evaluation Understudy):**
- Measures n-gram overlap between generated and reference text
- Range: 0-100 (higher is better)
- Good for factual descriptions

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
- Similar to BLEU but focuses on recall
- Variants: ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence)

**METEOR (Metric for Evaluation of Translation with Explicit ORdering):**
- Considers synonyms and stemming
- Better correlation with human judgment than BLEU

**Usage:**
```python
from evaluate import load

bleu = load("bleu")
rouge = load("rouge")
meteor = load("meteor")

predictions = model.generate_batch(images, questions)
references = [[ref] for ref in ground_truth]

bleu_score = bleu.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)
meteor_score = meteor.compute(predictions=predictions, references=references)
```

**Expected Results:**
- BLEU-4: 30-50 (higher = better overlap with references)
- ROUGE-L: 40-60 (higher = better recall)
- METEOR: 35-55 (higher = better semantic match)

#### 3. Task-Specific Accuracy

**Visual Question Answering (VQA):**
```python
# Exact match accuracy
correct = sum(pred.strip().lower() == ref.strip().lower()
              for pred, ref in zip(predictions, references))
accuracy = correct / len(predictions) * 100
```

**Image Captioning:**
- CIDEr (Consensus-based Image Description Evaluation)
- SPICE (Semantic Propositional Image Caption Evaluation)

**Expected Results:**
- VQA accuracy: 60-75%
- CIDEr: 80-120
- SPICE: 15-25

### Qualitative Metrics

#### 4. Temporal Understanding

**Test:** Questions about what will happen next in a scene.

**Examples:**
```python
test_cases = [
    {
        "image": "ball_rolling_downhill.jpg",
        "question": "What will happen to the ball?",
        "expected": "The ball will roll down the hill due to gravity."
    },
    {
        "image": "ice_cube_in_sun.jpg",
        "question": "What will happen to the ice cube?",
        "expected": "The ice cube will melt in the sunlight."
    }
]
```

**Evaluation:**
- Human evaluation: Does answer show temporal reasoning?
- Comparison: TheWorld vs. Gemma baseline
- Expected: TheWorld should better understand dynamics

#### 5. Physical Reasoning

**Test:** Questions requiring understanding of physics.

**Examples:**
```python
physics_tests = [
    {
        "image": "tower_of_blocks.jpg",
        "question": "Is this tower stable?",
        "expected": "No, it will likely fall because..."
    },
    {
        "image": "water_pour.jpg",
        "question": "Where will the water go?",
        "expected": "The water will flow downward due to gravity..."
    }
]
```

**Evaluation:**
- Score responses on physical plausibility (1-5 scale)
- Compare TheWorld vs. baseline
- Expected: TheWorld shows better physical understanding

#### 6. World Model Ablation Study

**Test:** Same model, with and without world tokens at inference.

```python
# Evaluate contribution of world tokens
for test_case in test_set:
    # With world tokens (full model)
    response_full = model.generate(
        test_case["image"],
        test_case["question"],
        num_world_steps=4  # Use world model
    )

    # Without world tokens (ablation)
    response_ablation = model.generate(
        test_case["image"],
        test_case["question"],
        num_world_steps=0  # Minimal world model
    )

    # Compare quality
    score_full = evaluate_response(response_full, test_case["reference"])
    score_ablation = evaluate_response(response_ablation, test_case["reference"])
```

**Expected Results:**
- World tokens should improve temporal/physical reasoning questions
- May not help on static description tasks
- Useful for understanding where world model adds value

---

## BLINK Benchmark Evaluation

### Overview

The **BLINK (Benchmarking Language-Image-and-Knowledge) benchmark** is a comprehensive multimodal perception benchmark designed to test visual understanding capabilities across 14 different tasks. TheWorld includes evaluation scripts for BLINK to measure visual perception, especially spatial and depth understanding.

**Currently Supported Tasks:**
- **Relative_Depth**: Tests depth perception and spatial understanding (binary choice)
- **Spatial_Relation**: Tests spatial reasoning between objects (4-choice)

**Future Support:** The architecture supports all 14 BLINK tasks - additional tasks can be added by extending the evaluation script.

### Why BLINK for TheWorld?

BLINK is particularly relevant for evaluating TheWorld because:

1. **Tests Visual Perception**: Measures if the world model improves spatial/depth understanding
2. **Ablation-Friendly**: Single-image tasks make it easy to compare with/without world tokens
3. **Standardized Benchmark**: Public leaderboard for comparison with other models
4. **Multiple-Choice Format**: Objective evaluation without ambiguous free-form text

### Evaluation Scripts

#### Batch Evaluation (`scripts/evaluate_blink.py`)

Evaluates the model on full BLINK dataset(s) and computes metrics.

**Basic Usage:**
```bash
# Evaluate on Relative_Depth
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp

# Evaluate on both tasks
python scripts/evaluate_blink.py \
  --tasks Relative_Depth,Spatial_Relation \
  --model username/theworld-datacomp \
  --output results/blink_full.json
```

**Ablation Study (Compare with/without world tokens):**
```bash
# Test num_world_steps=0 (minimal) and num_world_steps=4 (with prediction)
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp \
  --num_world_steps 0,4 \
  --output results/blink_ablation.json
```

**Quick Test (100 samples):**
```bash
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp \
  --num_samples 100
```

**Using Config File:**
```bash
# Edit configs/eval_blink.json with your settings
python scripts/evaluate_blink.py --config configs/eval_blink.json
```

**Output Format:**
```json
{
  "model": "username/theworld-datacomp",
  "config": {
    "split": "test",
    "num_samples": null,
    "max_new_tokens": 10,
    "temperature": 0.0
  },
  "results": {
    "Relative_Depth": {
      "world_steps_0": {
        "accuracy": 72.5,
        "correct": 725,
        "total": 1000,
        "f1_macro": 71.8,
        "f1_weighted": 72.3,
        "confusion_matrix": [[380, 120], [155, 345]]
      },
      "world_steps_4": {
        "accuracy": 78.2,
        "correct": 782,
        "total": 1000,
        ...
      }
    }
  },
  "summary": {
    "mean_accuracy": 75.35,
    "tasks_evaluated": 1,
    "configurations": 2
  }
}
```

#### Interactive Demo (`scripts/inference_demo.py`)

Explore BLINK examples interactively with real-time inference.

**Usage:**
```bash
# Start interactive demo
python scripts/inference_demo.py \
  --model username/theworld-datacomp \
  --task Relative_Depth

# Commands available in demo:
#   next / n        - Show next example
#   prev / p        - Show previous example
#   jump N / j N    - Jump to example N
#   steps N / s N   - Change num_world_steps to N
#   custom / c      - Test custom image/question
#   help / h        - Show commands
#   quit / q        - Exit
```

**Example Session:**
```
> next
==================================================
Example 5 / 1000
==================================================

📷 Image: (512, 512) pixels (RGB)

❓ Question: Which object is closer to the camera?

Choices:
  A) The tree
  B) The building

🤖 Generating answer (num_world_steps=0)...

✅ Results:
  Generated text: "A"
  Parsed answer:  A
  Ground truth:   A
  Status:         CORRECT ✓

> steps 4
✓ Set num_world_steps = 4
[Re-evaluates current example with world_steps=4]

> custom
Enter image path: my_image.jpg
Enter question: What is closer to the viewer?
Enter choices:
  A) Left object
  B) Right object
🤖 Generating answer...
```

### Metrics Explained

**Accuracy:**
- Primary metric: Percentage of correct predictions
- Example: 72.5% means 725/1000 examples correct

**F1 Scores:**
- **F1 Macro**: Unweighted average across classes (A, B, C, D)
  - Treats all classes equally
  - Good for balanced evaluation
- **F1 Weighted**: Weighted by class frequency
  - Accounts for class imbalance
  - Better reflects overall performance

**Confusion Matrix:**
- Shows prediction patterns
- Example for binary choice (Relative_Depth):
  ```
  [[380, 120],   # Ground truth A: 380 correct, 120 wrong
   [155, 345]]   # Ground truth B: 155 wrong, 345 correct
  ```
- Diagonal = correct predictions
- Off-diagonal = errors

### Interpretation

**What Good Results Look Like:**

✅ **High Accuracy (70%+)**
- Relative_Depth: 70-85% expected for good models
- Spatial_Relation: 60-75% expected (4-choice harder)

✅ **Improvement with World Tokens**
- Example: world_steps=0 → 72%, world_steps=4 → 78%
- Shows world model helps spatial understanding

✅ **Balanced Confusion Matrix**
- No strong bias toward one choice
- Errors distributed evenly

**What Bad Results Look Like:**

❌ **Random-Level Accuracy**
- Relative_Depth < 55% (only slightly better than 50% random)
- Spatial_Relation < 30% (only slightly better than 25% random)

❌ **No World Token Benefit**
- world_steps=0 and world_steps=4 perform identically
- Suggests world embeddings not being used

❌ **Biased Predictions**
- Model always predicts "A" regardless of input
- Check confusion matrix for this pattern

### Baseline Comparison

**Compare TheWorld vs. Gemma Baseline:**

```bash
# Evaluate TheWorld
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp \
  --output results/theworld_blink.json

# Evaluate Gemma baseline (without world model)
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/gemma-baseline \
  --output results/baseline_blink.json

# Compare results
python scripts/compare_results.py \
  results/theworld_blink.json \
  results/baseline_blink.json
```

**Expected Results:**
- TheWorld should outperform baseline on spatial/depth tasks by 5-15%
- Example: TheWorld = 78%, Gemma = 70%

### Ablation Study Guide

**Purpose:** Measure contribution of world tokens to spatial understanding.

**Setup:**
1. Train TheWorld normally (projection layer only)
2. Evaluate with different num_world_steps settings
3. Compare accuracy across configurations

**Recommended Configurations:**
```bash
# Test multiple world step settings
python scripts/evaluate_blink.py \
  --task Relative_Depth \
  --model username/theworld-datacomp \
  --num_world_steps 0,1,4,8 \
  --output results/ablation_world_steps.json
```

**Analysis Questions:**
1. Does accuracy improve with more world steps?
2. Is there a sweet spot (e.g., 4 steps optimal)?
3. Do world tokens help more on certain question types?

**Expected Patterns:**
- **Spatial tasks**: World tokens should help (accuracy increases)
- **Static description**: World tokens may not help
- **Diminishing returns**: 0→4 steps helps more than 4→8

### Common Issues

**Problem: Accuracy stuck at random chance**
- **Cause**: Model not understanding question format
- **Fix**: Check prompt formatting in `format_question()`
- **Fix**: Verify model was trained on QA data

**Problem: Model always predicts "A"**
- **Cause**: Biased training data or poor parsing
- **Fix**: Check `parse_choice()` function
- **Fix**: Review training data balance

**Problem: World tokens don't improve accuracy**
- **Cause**: Projection layer not learning useful representations
- **Fix**: Check projection layer gradients during training
- **Fix**: Try unfreezing more components (Stage 2 training)

**Problem: Errors loading BLINK dataset**
- **Cause**: Missing `trust_remote_code=True` flag
- **Fix**: Update script to include flag in `load_dataset()`

### Integration with Training

**Use BLINK as validation metric during training:**

```python
# In training script
from scripts.evaluate_blink import evaluate_task

# After each epoch
if epoch % 5 == 0:
    metrics = evaluate_task(
        task="Relative_Depth",
        model=model,
        split="val",
        num_samples=100,  # Quick validation
        num_world_steps=0,
    )
    print(f"BLINK Accuracy: {metrics['accuracy']:.2f}%")
    wandb.log({"blink_accuracy": metrics['accuracy']})
```

**Benefits:**
- Track spatial understanding over training
- Early stopping based on BLINK performance
- Correlate with perplexity improvements

---

## Experimental Setup

### Training Baselines

**1. TheWorld (Full Model):**
```bash
# Train with world model fusion
export HF_TOKEN="hf_your_token_here"
python scripts/train_hf.py --config configs/datacomp_production.json
```

**2. Gemma Baseline (No World Model):**
```bash
# Train Gemma-only baseline (requires implementation)
python scripts/train_baseline_gemma.py --config configs/baseline.json
```

**3. Random Projection (Ablation):**
```bash
# Train with random projection initialization
python scripts/train_hf.py --config configs/ablation_random.json
```

### Evaluation Setup

**Prepare Test Set:**
```python
# Create balanced test set
test_set = {
    "static_descriptions": 100,   # "Describe this image"
    "temporal_reasoning": 100,    # "What will happen next?"
    "physical_reasoning": 100,    # "Is this physically plausible?"
    "vqa": 100,                   # Visual question answering
}
```

**Run Evaluation:**
```bash
# Evaluate all models on same test set
python scripts/evaluate.py \
  --models username/theworld-datacomp,username/gemma-baseline,username/ablation-random \
  --test_data data/test_balanced.json \
  --metrics perplexity,bleu,rouge,vqa_accuracy \
  --output results/comparison.json
```

**Analyze Results:**
```python
import json
import pandas as pd

results = json.load(open("results/comparison.json"))
df = pd.DataFrame(results)

# Compare metrics
print(df.groupby("model")["perplexity"].mean())
print(df.groupby(["model", "category"])["accuracy"].mean())
```

---

## Interpretation Guide

### What Good Results Look Like

#### Quantitative Signals

✅ **TheWorld perplexity < Gemma baseline by 10-20%**
- Example: TheWorld = 18.5, Gemma = 22.3
- Indicates world model improves predictions

✅ **Better on temporal/physical reasoning tasks**
- Example: Temporal accuracy: TheWorld = 75%, Gemma = 55%
- Shows world model adds value for dynamics

✅ **BLEU/ROUGE scores improve on dynamic scenes**
- Example: Dynamic scenes BLEU: TheWorld = 45, Gemma = 38
- World model helps describe motion/changes

✅ **Ablation shows world tokens contribute**
- Example: With world tokens = 72% accuracy, Without = 65%
- World embeddings are being used by the model

#### Qualitative Signals

✅ **Generated text shows temporal understanding**
```
Question: "What will happen to the rolling ball?"
TheWorld: "The ball will continue rolling down the slope due to gravity and momentum."
Gemma: "The ball is moving."  # Static description
```

✅ **Physical reasoning is more accurate**
```
Question: "Will this tower of blocks fall?"
TheWorld: "Yes, the tower is unstable because the top block extends too far beyond the base."
Gemma: "This is a tower of blocks."  # No reasoning
```

### What Bad Results Look Like

#### Quantitative Signals

❌ **TheWorld perplexity ≈ Gemma baseline**
- Example: TheWorld = 20.1, Gemma = 20.3
- World model is not helping

❌ **Random projection performs similarly to pretrained**
- Example: Both achieve perplexity ~19
- Pretrained Cosmos knowledge not being utilized

❌ **No difference with/without world tokens**
- Example: Both achieve 68% accuracy
- World embeddings are being ignored

❌ **Worse on static description tasks**
- Example: Static BLEU: TheWorld = 35, Gemma = 42
- World model may be adding noise

#### Qualitative Signals

❌ **No temporal understanding in responses**
```
Question: "What will happen next?"
TheWorld: "This is an image of a ball."  # Same as Gemma
```

❌ **Physical reasoning fails**
```
Question: "Is this physically possible?"
TheWorld: "Yes."  # Incorrect, no reasoning
```

### Troubleshooting

#### If World Model Doesn't Help

**Potential Causes:**
1. **Projection layer not learning**
   - Check: Is loss decreasing? Are gradients flowing?
   - Solution: Increase projection learning rate (1e-4 → 5e-4)

2. **World embeddings too weak**
   - Check: Are world token norms much smaller than text tokens?
   - Solution: Add layer normalization after projection

3. **Training data doesn't need temporal info**
   - Check: Is dataset mostly static images?
   - Solution: Use dataset with dynamic scenes, videos, or temporal questions

4. **Projection layer too small**
   - Check: Is 16→2304 bottleneck too severe?
   - Solution: Add hidden layer (16→512→2304)

#### If Temporal Tasks Fail Specifically

**Potential Causes:**
1. **Single-step world model (num_world_steps=0)**
   - Check: Are you using autoregressive rollout?
   - Solution: Train with num_world_steps>0 for temporal tasks

2. **Temporal embeddings not learned**
   - Check: Are temporal embedding gradients near zero?
   - Solution: Increase temporal embedding learning rate

#### If Static Tasks Regress

**Potential Causes:**
1. **World tokens adding noise**
   - Check: Do world tokens dominate attention?
   - Solution: Reduce world token count or add weighting

2. **Vision encoder needs fine-tuning**
   - Check: Is vision encoder frozen?
   - Solution: Unfreeze vision encoder (Stage 2 training)

### Decision Framework

```
Start: TheWorld trained, ready to evaluate
  ↓
Run evaluation metrics
  ↓
Is perplexity better than baseline?
  ├─ Yes → World model is helping! ✅
  │   ↓
  │   Is improvement on temporal tasks?
  │   ├─ Yes → Design validated ✅
  │   └─ No → May need higher num_world_steps
  │
  └─ No → Investigate further
      ↓
      Does random projection perform same?
      ├─ Yes → Pretrained Cosmos not useful
      │   └─ Consider: Different world model or dataset
      │
      └─ No → Projection needs better training
          └─ Try: Higher LR, more epochs, unfreezing components
```

---

## Summary

**Training Objective:**
- Next-token prediction (causal LM)
- Cross-entropy loss on text tokens
- World/vision embeddings provide context but aren't predicted

**Key Evaluation Questions:**
1. Does TheWorld outperform Gemma baseline? (Perplexity, BLEU, accuracy)
2. Does pretrained Cosmos help? (Compare to random projection)
3. Do world tokens contribute? (Ablation study)
4. Is improvement on temporal/physical tasks? (Qualitative evaluation)

**Expected Results:**
- 10-20% perplexity improvement
- Better temporal and physical reasoning
- Meaningful contribution from world tokens
- May not help on purely static tasks

**Next Steps:**
- Train baselines for comparison
- Run comprehensive evaluation
- Analyze where world model helps vs. doesn't
- Iterate on architecture/training if needed

For implementation of evaluation scripts and baseline training, see `scripts/evaluate.py` (to be implemented).
