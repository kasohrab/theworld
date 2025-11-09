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

## Setup

This project uses **uv** for fast, reliable Python package management.

### Install uv (if not already installed)

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
# Install project dependencies
uv sync

# Or with dev tools (optional)
uv sync --dev
```

**What is uv?**
- **Fast**: 10-100x faster than pip
- **Reliable**: Lockfile ensures reproducible installs
- **Simple**: `uv run` automatically manages virtual environments
- **Modern**: Built in Rust, handles dependency resolution efficiently

## Quick Start

**Note:** All commands use `uv run` for dependency management. If you haven't set up `uv`, see the [Setup section](#setup) below.

### 1. Run Evaluation (10 samples)

```bash
uv run python scripts/spatial/eval_spatial_rgpt.py \
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

### 2. Run Full Evaluation (1,406 samples) - WITHOUT Batching

```bash
# Evaluate all samples without batching (slow, ~2 hours)
uv run python scripts/spatial/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model google/gemma-3-4b-it \
    --output outputs/spatial_rgpt_baseline_full.jsonl \
    --draw-bboxes
```

**Time estimate:** ~2 hours on single GPU (4.8s per sample without batching)

**⚠️ Warning:** This is the slow approach. Use batching instead (see below).

### 3. Run Full Evaluation with Batching (Recommended - **30x Faster!**)

```bash
# Use batching for ~30x speedup (RECOMMENDED)
uv run python scripts/spatial/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model google/gemma-3-4b-it \
    --output outputs/spatial_rgpt_baseline_full_batched.jsonl \
    --batch-size 64 \
    --draw-bboxes
```

**Time estimate:** ~3-4 minutes on single GPU (with batch size 64)

**Batching benefits:**
- **30x faster** evaluation (1.9 hours → 3-4 minutes!)
- Processes multiple samples in parallel through the model
- Uses manual batching: processes each image individually through Gemma3 processor, then concatenates tensors
- Adjust `--batch-size` based on GPU memory:
  - 64: For 80GB GPUs (recommended)
  - 32: For 40GB GPUs
  - 16: For 24GB GPUs
  - 8: For 16GB GPUs

**How batching works:**
1. Gemma3 processor automatically resizes all images to 896x896 (fixed resolution)
2. Each image-question pair is processed individually to get uniform tensors
3. Tensors are manually concatenated into batches
4. Model processes entire batch at once for maximum efficiency

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
| `--batch-size` | `1` | Batch size for evaluation (higher = faster, more memory) |

## Evaluation Modes

### Mode 1: Gemma Baseline (Default)

Evaluates Gemma 3 without the world model:

```bash
uv run python scripts/spatial/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model google/gemma-3-4b-it \
    --output outputs/baseline.jsonl \
    --batch-size 64  # Use batching for speed!
```

### Mode 2: TheWorld (with Cosmos)

Evaluates TheWorld with temporal world model:

```bash
uv run python scripts/spatial/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model username/theworld-model \
    --load-cosmos \
    --num-world-steps 4 \
    --batch-size 64 \
    --output outputs/theworld.jsonl
```

### Mode 3: Local JSONL File

Evaluate on a local dataset file:

```bash
uv run python scripts/spatial/eval_spatial_rgpt.py \
    --data-path /path/to/val_SpatialRGPT-Bench.jsonl \
    --image-folder /path/to/images \
    --model google/gemma-3-4b-it \
    --batch-size 64 \
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
python scripts/spatial/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --no-draw-bboxes  # Disable bounding box overlay
    ...
```

## How Gemma-as-Judge Works

We provide multiple judge modes for evaluation. The **official judge** mode replicates the SpatialRGPT-Bench paper's evaluation methodology using Gemma instead of GPT-4.

### Judge Modes

**1. Official Judge (`--official-judge`)**
Replicates the official SpatialRGPT-Bench evaluation (Tables 12 & 13 from paper):

**Qualitative Questions:**
- Judge outputs: 0 (incorrect) or 1 (correct)
- Binary correctness assessment

**Quantitative Questions:**
- Judge converts both answers to meters
- Calculates relative error: `|pred - gt| / gt`
- Reports success rates at multiple thresholds:
  - ±10% (stricter)
  - ±25% (official metric from paper)
  - ±50% (more lenient)
- Calculates absolute relative error (average)

**Example Flow:**
```
Question: "What is the height of Region [0]?"
Ground Truth: "6.91 feet"
Prediction: "2.1 meters"

Judge converts both to meters:
- Ground truth: 6.91 ft = 2.106 m
- Prediction: 2.1 m

Relative error = |2.1 - 2.106| / 2.106 = 0.0028 (0.28%)
→ Within ±10%, ±25%, ±50% → Correct at all thresholds
```

**2. Lenient Judge (`--lenient-judge`)**
Semantic equivalence check (Yes/No response)

**3. Strict Judge (default)**
Prediction must contain ground truth information

### Advantages of Official Judge
- Matches paper's evaluation methodology
- Multi-threshold insights (see performance across tolerances)
- Tracks unparseable responses
- Quantifies error magnitude

### Limitations
- Gemma less accurate than GPT-4 at unit conversion
- May produce formatting artifacts
- Use GPT-4 judge for publication-quality results (see Standalone Judging below)

### Implementation Details: Paper Compliance

Our implementation **exactly matches** the official SpatialRGPT-Bench evaluation methodology from the paper (Tables 12 & 13):

**Qualitative Evaluation (Table 12):**
- System prompt asks for JSON output with 0 or 1
- 1 = "response perfectly matches the answer"
- 0 = "response is completely different from the answer"
- Implementation: `python/theworld/evaluation/judges.py:70-80` and `spatial_metrics.py:44-55`

**Quantitative Evaluation (Table 13):**
- System prompt asks judge to convert distances to meters
- Conversion factors (exact match):
  - 1 inch = 0.0254 meters ✓
  - 1 foot = 0.3048 meters ✓
  - 1 centimeter = 0.01 meters ✓
- Judge outputs two floats: [ground_truth_meters, prediction_meters]
- Relative error calculation: `|pred - gt| / gt`
- Official threshold: ±25% (with additional ±10% and ±50% for analysis)
- Implementation: `python/theworld/evaluation/judges.py:86-98` and `spatial_metrics.py:58-69`

**Verification:** The `--official-judge` flag routes to these exact prompts, ensuring paper-compliant evaluation. Both Gemma (free) and GPT-4 (paid) judges use identical prompt templates.

## Understanding the Results

### Output Format

Each line in the output JSONL contains:

**Qualitative Question:**
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
  "judge_response": "1"
}
```

**Quantitative Question (with official judge):**
```json
{
  "id": "quantitative_L1Nt6oLYo9",
  "question": "Determine the vertical dimensions of Region [0].",
  "ground_truth": "Region [0] is 30.05 inches in height.",
  "prediction": "The height is approximately 0.75 meters",
  "qa_type": "quantitative",
  "qa_category": "height_data",
  "score": 1.0,
  "correct": true,
  "judge_response": "[0.763, 0.75]",
  "relative_error": 0.017,
  "gt_meters": 0.763,
  "pred_meters": 0.75
}
```

**Fields:**
- `relative_error`: Only for quantitative with official judge. `|pred - gt| / gt`
- `gt_meters`, `pred_meters`: Converted values in meters
- `correct`: True if within ±25% threshold (official metric)

### Metrics Breakdown

**Console Output (with official judge):**
```
============================================================
EVALUATION RESULTS
============================================================

Overall:
  Total: 1406
  Correct: 748
  Accuracy: 0.5320 (53.20%)

By Question Type:
  qualitative: 0.6543 (65.43%)
  quantitative: 0.4121 (41.21%)

Quantitative Metrics (Multi-Threshold):
  Total quantitative: 703
  Success rates:
    ±10%: 0.3215 (32.15%)
    ±25%: 0.4121 (41.21%) ← official
    ±50%: 0.5873 (58.73%)
  Absolute Relative Error: 0.33
  Unparseable responses: 5 (0.7%)

By Category:
  small_predicate: 0.8542 (85.42%)
  left_choice: 0.7821 (78.21%)
  behind_predicate: 0.3254 (32.54%)
  ...
============================================================
```

**Breakdown:**

**Question Type:**
- `qualitative`: Yes/No or descriptive questions → Binary accuracy
- `quantitative`: Numeric measurements → Multi-threshold success rates

**Quantitative Metrics:**
- **Success rates**: Percentage within each tolerance threshold
  - ±10%: Stricter threshold for high precision
  - ±25%: Official metric from SpatialRGPT-Bench paper
  - ±50%: More lenient threshold
- **Absolute Relative Error**: Average error magnitude (lower is better)
- **Unparseable**: Responses judge couldn't convert to numbers

**Category:**
- `small_predicate`: Size comparisons
- `tall_predicate`: Height comparisons
- `behind_predicate`: Depth/occlusion reasoning
- `left_choice`, `right_choice`: Horizontal positioning
- `height_data`, `width_data`: Measurement questions

### Interpreting Results

**Typical baseline results:**
- **Overall**: 50-70% (varies by model)
- **Qualitative**: 60-75%
- **Quantitative @ ±25%**: 35-45%
- **Spatial choices** (left/right): 80-100% (easier)
- **Size comparisons**: 60-80%
- **Depth reasoning**: 20-50% (harder)
- **Quantitative measurements**: 40-60%

Low accuracy on certain categories (e.g., `behind_predicate`) indicates areas where world model understanding could help.

## Standalone Judging

After running evaluation with `--skip-judging`, you can re-judge predictions using different judges without re-running inference. This is useful for comparing judge accuracy or using more expensive judges like GPT-4.

### Available Judges

**1. Gemma (Free, Fast, Less Accurate)**
- Uses the same Gemma model for judging
- No additional costs
- Good for quick iterations
- May have unit conversion errors on quantitative questions

**2. GPT-4 (Paid, Accurate, API-based)**
- Uses OpenAI GPT-4 API
- Requires API key and credits
- Most accurate for unit conversions
- Official SpatialRGPT-Bench uses GPT-4

**3. GPT-OSS (Free, Accurate, Large Model)**
- Uses OpenAI's open-source GPT model (120B params)
- No API costs
- More accurate than Gemma for unit conversions
- Requires large GPU (~120-240GB VRAM)

### Judge with Gemma

```bash
# First, run evaluation without judging
python scripts/spatial/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model google/gemma-3-4b-it \
    --output outputs/predictions.jsonl \
    --skip-judging

# Then judge with Gemma
python scripts/spatial/judge_predictions.py \
    --predictions outputs/predictions.jsonl \
    --judge gemma \
    --model google/gemma-3-4b-it \
    --output outputs/results_judged_gemma.jsonl
```

### Judge with GPT-4

```bash
# Set API key
export OPENAI_API_KEY=sk-...

# Judge with GPT-4
python scripts/spatial/judge_predictions.py \
    --predictions outputs/predictions.jsonl \
    --judge gpt4 \
    --gpt-model gpt-4-turbo \
    --output outputs/results_judged_gpt4.jsonl
```

### Judge with GPT-OSS

```bash
# Judge with GPT-OSS (requires large GPU)
python scripts/spatial/judge_predictions.py \
    --predictions outputs/predictions.jsonl \
    --judge gpt-oss \
    --gpt-oss-model openai/gpt-oss-120b \
    --output outputs/results_judged_gptoss.jsonl
```

**GPT-OSS Options:**
- `--gpt-oss-model`: HuggingFace model ID (default: openai/gpt-oss-120b)
- `--gpt-oss-device-map`: Device placement strategy (default: auto)
- `--gpt-oss-dtype`: Torch dtype (default: auto)

**Memory Requirements:**
- Full precision (fp32): ~480GB VRAM
- Half precision (fp16): ~240GB VRAM
- 8-bit quantization (int8): ~120GB VRAM

For 8-bit quantization:
```bash
python scripts/spatial/judge_predictions.py \
    --predictions outputs/predictions.jsonl \
    --judge gpt-oss \
    --gpt-oss-dtype int8 \
    --output outputs/results_judged_gptoss.jsonl
```

## Advanced Usage

### Quick Test (3 samples)

```bash
python scripts/spatial/test_spatial_eval.py
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
# Baseline (with batching)
uv run python scripts/spatial/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model google/gemma-3-4b-it \
    --output outputs/baseline.jsonl \
    --batch-size 64 \
    --max-samples 100

# TheWorld (with batching)
uv run python scripts/spatial/eval_spatial_rgpt.py \
    --data-path a8cheng/SpatialRGPT-Bench \
    --image-folder "" \
    --model username/theworld-model \
    --load-cosmos \
    --batch-size 64 \
    --output outputs/theworld.jsonl \
    --max-samples 100

# Compare results
uv run python -c "
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

**Problem:** GPU runs out of memory during batched evaluation.

**Solution:** Use smaller batch size based on your GPU memory:
```bash
# For 80GB GPUs
--batch-size 64

# For 40GB GPUs
--batch-size 32

# For 24GB GPUs
--batch-size 16

# For 16GB GPUs
--batch-size 8
```

Or use CPU (slower but no memory issues):
```bash
--device cpu --batch-size 4
```

### Issue: Right-padding warning

**Warning:** `A decoder-only architecture is being used, but right-padding was detected!`

**Status:** This warning has been fixed in the latest version. The code now uses left-padding for decoder-only models (Gemma). If you see this warning, update to the latest version of the code.

**Technical details:** Decoder-only models require left-padding (prepending padding tokens) rather than right-padding (appending padding tokens) for correct generation behavior.

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
- See examples in `examples/` and `scripts/spatial/test_spatial_eval.py`
