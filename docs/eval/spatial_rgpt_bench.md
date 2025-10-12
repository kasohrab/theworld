# Evaluating on SpatialRGPT-Bench

This guide shows you how to evaluate TheWorld (or Gemma baseline) on **SpatialRGPT-Bench**, a benchmark for spatial reasoning with vision-language models.

## Overview

SpatialRGPT-Bench tests models on spatial understanding tasks like:
- **Qualitative**: "Is Region [0] behind Region [1]?" → "No."
- **Quantitative**: "What is the height of Region [0]?" → "6.91 feet"

Our evaluation approach uses **Gemma-as-judge**: instead of expensive GPT-4 API calls, we use Gemma itself to evaluate free-form answers. This is:
- ✅ **Free** (no API costs)
- ✅ **Fast** (same model already loaded)
- ✅ **Self-contained** (no external dependencies)

## Quick Start

### 1. Run Evaluation (10 samples)

```bash
python scripts/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model google/gemma-3-4b-it \
    --output outputs/spatial_rgpt_baseline_10.jsonl \
    --max-samples 10 \
    --draw-bboxes
```

**Expected output:**
```
Loading dataset from: a8cheng/SpatialRGPT-Bench
✓ Loaded 10 samples
Loading model (Gemma-only baseline): google/gemma-3-4b-it
✓ Model loaded in evaluation mode
Evaluating: 100%|██████████| 10/10 [00:48<00:00]
✓ Saved results to outputs/spatial_rgpt_baseline_10.jsonl

============================================================
EVALUATION RESULTS
============================================================

Overall:
  Total: 10
  Correct: 6
  Accuracy: 0.6000 (60.00%)

By Question Type:
  qualitative: 0.6000 (60.00%)
  quantitative: 0.6000 (60.00%)

By Category:
  small_predicate: 1.0000 (100.00%)
  left_choice: 1.0000 (100.00%)
  right_choice: 1.0000 (100.00%)
  width_data: 0.6667 (66.67%)
  height_data: 0.5000 (50.00%)
  behind_predicate: 0.0000 (0.00%)
  tall_predicate: 0.0000 (0.00%)
============================================================
```

### 2. Run Full Evaluation (1,406 samples)

```bash
# Remove --max-samples to evaluate all samples
python scripts/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model google/gemma-3-4b-it \
    --output outputs/spatial_rgpt_baseline_full.jsonl \
    --draw-bboxes
```

**Time estimate:** ~2 hours on single GPU (4.8s per sample)

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data-path` | Dataset path (HuggingFace ID or local JSONL) | `a8cheng/SpatialRGPT-Bench` |
| `--image-folder` | Base folder for images (use `""` for HF datasets) | `""` or `/path/to/images` |

### Model Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `google/gemma-3-4b-it` | Model to evaluate (HF model ID or local path) |
| `--device` | `cuda` | Device (`cuda`, `cpu`) |
| `--load-cosmos` | `False` | Load Cosmos world model (TheWorld mode) |
| `--num-world-steps` | `0` | World prediction steps (only if `--load-cosmos`) |

### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `outputs/spatial_rgpt_results.jsonl` | Output path for results |
| `--max-samples` | `0` | Max samples to evaluate (0 = all) |
| `--draw-bboxes` | `True` | Draw bounding boxes on images |
| `--max-new-tokens` | `128` | Max tokens per answer |
| `--temperature` | `0.0` | Sampling temperature (0.0 = greedy) |

## Evaluation Modes

### Mode 1: Gemma Baseline (Default)

Evaluates Gemma 3 without the world model:

```bash
python scripts/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model google/gemma-3-4b-it \
    --output outputs/baseline.jsonl
```

### Mode 2: TheWorld (with Cosmos)

Evaluates TheWorld with temporal world model:

```bash
python scripts/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model username/theworld-model \
    --load-cosmos \
    --num-world-steps 4 \
    --output outputs/theworld.jsonl
```

### Mode 3: Local JSONL File

Evaluate on a local dataset file:

```bash
python scripts/eval_spatial_rgpt.py \
    --data-path /path/to/val_SpatialRGPT-Bench.jsonl \
    --image-folder /path/to/images \
    --model google/gemma-3-4b-it \
    --output outputs/local_eval.jsonl
```

## Visual Grounding with Bounding Boxes

The evaluation automatically draws bounding boxes on images to help the model understand spatial regions:

**Original Dataset Format:**
```json
{
  "text_q": "Is Region [0] smaller than Region [1]?",
  "bbox": [[0, 30, 668, 767], [6, 292, 233, 432]]
}
```

**Our Processing:**
1. Draw colored bounding boxes on the image
2. Label them as "Region [0]", "Region [1]", etc.
3. Pass the annotated image to the model

To disable this feature:
```bash
python scripts/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --no-draw-bboxes  # Disable bounding box overlay
    ...
```

## How Gemma-as-Judge Works

Instead of using GPT-4 to evaluate answers (expensive, requires API), we use **Gemma to judge its own outputs**:

**Step 1: Generate Answer**
```
Question: "Is Region [0] behind Region [1]?"
Model Response: "Yes, Region [0] is behind Region [1]."
```

**Step 2: Create Judge Prompt**
```
You are evaluating spatial reasoning answers. Compare the prediction to the ground truth.

Question: Is Region [0] behind Region [1]?

Ground Truth Answer: No.

Predicted Answer: Yes, Region [0] is behind Region [1].

Does the prediction correctly answer the question based on the ground truth?
Consider semantic equivalence - the prediction doesn't need exact wording.

Respond with ONLY "Yes" or "No" (one word).
```

**Step 3: Get Judgment**
```
Judge Response: "No"
→ Score: 0.0 (incorrect)
```

**Advantages:**
- No API costs
- Fast (uses already-loaded model)
- Good baseline for quick evaluation
- Self-contained (no external services)

**Limitations:**
- Not as accurate as GPT-4 evaluation
- Model may be lenient judging its own outputs
- Best used for baseline comparisons

## Understanding the Results

### Output Format

Each line in the output JSONL contains:

```json
{
  "id": "qualitative_3TrgWgKqnf",
  "question": "Can you confirm if Region [0] is smaller than Region [1]?",
  "ground_truth": "Incorrect, Region [0] is not smaller in size than Region [1].",
  "prediction": "Yes, based on the image, Region [0] is smaller than Region [1]...",
  "qa_type": "qualitative",
  "qa_category": "small_predicate",
  "score": 1.0,
  "correct": true,
  "judge_response": "\nYes\n"
}
```

### Metrics Breakdown

The evaluation prints metrics broken down by:

**Question Type:**
- `qualitative`: Yes/No or descriptive questions
- `quantitative`: Numeric measurements (height, width, distance)

**Category:**
- `small_predicate`: Size comparisons
- `tall_predicate`: Height comparisons
- `behind_predicate`: Depth/occlusion reasoning
- `left_choice`, `right_choice`: Horizontal positioning
- `height_data`, `width_data`: Measurement questions

### Interpreting Accuracy

**Typical baseline results:**
- **Overall**: 50-70% (varies by model)
- **Spatial choices** (left/right): 80-100% (easier)
- **Size comparisons**: 60-80%
- **Depth reasoning**: 20-50% (harder)
- **Quantitative measurements**: 40-60%

Low accuracy on certain categories (e.g., `behind_predicate`) indicates areas where world model understanding could help.

## Advanced Usage

### Quick Test (3 samples)

```bash
python scripts/test_spatial_eval.py
```

This runs a quick 3-sample test to verify the pipeline works.

### Custom Evaluation Script

```python
from theworld import TheWorld
from theworld.datasets.spatial_rgpt import SpatialRGPTDataset
from theworld.evaluation import evaluate_with_gemma, calculate_spatial_accuracy
from datasets import load_dataset

# Load dataset
hf_dataset = load_dataset("a8cheng/SpatialRGPT-Bench", split="val")
ds = SpatialRGPTDataset(hf_dataset, num_samples=10, draw_bboxes=True)

# Load model
model = TheWorld("google/gemma-3-4b-it", load_cosmos=False)
model.eval()

# Evaluate
results = []
for ex in ds:
    prediction = model.generate(
        image=ex["image"],
        prompt=ex["question"],
        skip_world_tokens=True,
    )

    eval_result = evaluate_with_gemma(
        model,
        question=ex["question"],
        prediction=prediction,
        ground_truth=ex["answer"],
    )

    results.append({
        "qa_type": ex["qa_type"],
        "score": eval_result["score"],
    })

# Calculate metrics
metrics = calculate_spatial_accuracy(results)
print(f"Accuracy: {metrics['accuracy']:.2%}")
```

### Comparing Models

```bash
# Baseline
python scripts/eval_spatial_rgpt.py \
    --model google/gemma-3-4b-it \
    --output outputs/baseline.jsonl \
    --max-samples 100

# TheWorld
python scripts/eval_spatial_rgpt.py \
    --model username/theworld-model \
    --load-cosmos \
    --output outputs/theworld.jsonl \
    --max-samples 100

# Compare
python -c "
from theworld.evaluation import calculate_spatial_accuracy
import json

baseline = [json.loads(l) for l in open('outputs/baseline.jsonl')]
theworld = [json.loads(l) for l in open('outputs/theworld.jsonl')]

b_acc = calculate_spatial_accuracy(baseline)['accuracy']
t_acc = calculate_spatial_accuracy(theworld)['accuracy']

print(f'Baseline: {b_acc:.2%}')
print(f'TheWorld: {t_acc:.2%}')
print(f'Improvement: {(t_acc - b_acc)*100:.2f}pp')
"
```

## Troubleshooting

### Issue: "FileNotFoundError: a8cheng/SpatialRGPT-Bench"

**Solution:** The script tries to load as a local file first. Make sure you're using `--image-folder ""` for HuggingFace datasets:

```bash
--data-path a8cheng/SpatialRGPT-Bench --image-folder ""
```

### Issue: All predictions are errors

**Problem:** Model generation failing due to parameter mismatch.

**Solution:** Make sure you're using the latest version of the script (fixed in commit after initial implementation).

### Issue: 0% accuracy

**Problem:** Judge is rejecting all answers.

**Solution:** Check if predictions are actually being generated:
```bash
head -1 outputs/spatial_rgpt_baseline_10.jsonl | python -m json.tool
```

Look at the `prediction` field. If it's an error, check model loading.

### Issue: Out of memory

**Solution:** Use smaller batch size or reduce max samples:
```bash
--max-samples 50  # Evaluate in batches
```

Or use CPU (slower but no memory issues):
```bash
--device cpu
```

## Dataset Information

**SpatialRGPT-Bench** is available at: `a8cheng/SpatialRGPT-Bench`

- **Size**: 1,406 validation samples
- **Source**: Omni3D dataset with spatial reasoning annotations
- **Task Types**:
  - Qualitative spatial reasoning (Yes/No, comparisons)
  - Quantitative measurements (heights, widths, distances)
- **Categories**: 7 spatial reasoning categories
- **Images**: Included directly in HuggingFace dataset

## Citation

If you use this evaluation in your research, please cite:

```bibtex
@article{cheng2024spatialrgpt,
  title={SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models},
  author={Cheng, An-Chieh and Li, Hongxu and Yang, Yifei and Yan, Xiaolong and Li, Siyuan and Lin, Tsung-Yi and Ramanan, Deva and Gao, Ruijia},
  journal={arXiv preprint arXiv:2410.XXXXX},
  year={2024}
}
```

## Related Documentation

- [Main Evaluation Guide](../evaluation.md) - Overview of all evaluation benchmarks
- [BLINK Benchmark](blink.md) - Another spatial reasoning benchmark
- [Training Guide](../training_infrastructure_design.md) - How to train TheWorld

## Support

For issues or questions:
- Check the [main README](../../README.md)
- Open an issue on GitHub
- See examples in `examples/` and `scripts/test_spatial_eval.py`
