# Interactive Inference Guide

Quick guide to using `examples/interactive_inference.py` for multi-turn conversations and SpatialRGPT dataset exploration.

## Quick Start

```bash
# Start interactive session
python examples/interactive_inference.py

# Or specify custom model
python examples/interactive_inference.py --model username/my-theworld
```

## Multi-Turn Conversations

The script maintains conversation history, allowing natural follow-up questions:

```bash
> /image cat.jpg
✓ Loaded image
✓ Conversation reset

> What is in this image?
Turn 1
🤖 A cat sitting on a couch.

> What color is it?
Turn 2 (continuing conversation)
🤖 The cat is orange and white.

> Is it sleeping?
Turn 3 (continuing conversation)
🤖 No, the cat appears to be awake and alert.
```

**Key Points:**
- Each question continues the conversation with full context
- Model sees all previous turns (question + its own answers, NOT ground truth)
- Use `/clear` to reset conversation while keeping image
- Loading a new image automatically resets conversation

## SpatialRGPT Dataset Evaluation

Iterate through ground truth questions and compare model answers to ground truth:

```bash
# Load dataset
> /spatial load 10
  Found 401890 images
  ✓ Loaded 10 samples

# Load a sample (with bboxes drawn automatically!)
> /spatial 0
✓ Loaded sample 0
  10 ground truth questions available
  Use /iterate to go through them with model answers
  Use /replay to see ground truth conversation

# Iterate through all questions (model generates answers)
> /iterate
============================================================
ITERATING THROUGH 10 QUESTIONS
============================================================

Question 1/10: Does Region [0] have a greater width?
🤖 Generating response...
Model: No, Region [0] appears narrower than Region [1].

Question 2/10: Which stands taller?
🤖 Generating response...
Model: Region [0] stands taller.

[... continues for all 10 questions ...]

============================================================
✓ Completed 10 questions
  Use /compare to see side-by-side comparison with ground truth
============================================================

# Compare model answers to ground truth
> /compare
============================================================
COMPARISON: GROUND TRUTH vs MODEL
============================================================

Exact/Partial Matches: 7/10 (70.0%)

============================================================

Turn 1:
Q: Does Region [0] have a greater width?
GT:    No, Region [0] might be narrower.
Model: No, Region [0] appears narrower than Region [1].
✓ Match

Turn 2:
Q: Which stands taller?
GT:    Region [0] stands taller.
Model: Region [0] stands taller.
✓ Match

[... continues for all turns ...]

# View ground truth conversation only
> /replay
============================================================
GROUND TRUTH CONVERSATION REPLAY
============================================================

HUMAN (Turn 0):
  Does Region [0] have a greater width?

GPT (Turn 1):
  No, Region [0] might be narrower.

[... full ground truth conversation ...]
```

## Essential Commands

| Command | Description |
|---------|-------------|
| `/image <path>` | Load image (resets conversation) |
| `/clear` | Reset conversation (keep image) |
| `/spatial load [N]` | Load N samples from dataset (default: 100) |
| `/spatial <index>` | Load sample by index (with bboxes!) |
| `/spatial random` | Load random sample |
| `/replay` | Show ground truth conversation |
| `/iterate` | Iterate through questions with model answers |
| `/compare` | Compare model vs ground truth (after /iterate) |
| `/help` | Show all commands |

## Tips

1. **Multi-turn works everywhere**: After loading any image (via `/image` or `/spatial`), ask follow-up questions freely

2. **True conversation semantics**: Model sees only its own answers, NOT ground truth - this ensures fair evaluation

3. **Bounding boxes**: When using `/spatial`, bboxes are drawn automatically using the same rendering as training

4. **Reset strategically**: Use `/clear` to start fresh analysis of the same image without reloading

5. **Dataset exploration**: Use `/spatial next/prev` to browse samples sequentially

6. **Evaluation workflow**:
   - `/spatial 0` - Load sample
   - `/iterate` - Model answers all questions
   - `/compare` - See side-by-side comparison with metrics
   - `/replay` - View ground truth conversation

## Dataset Paths

Default paths point to scratch space:
- JSON: `/home/hice1/ksohrab3/scratch/theworld/data/result_10_depth_convs.json`
- Images: `/home/hice1/ksohrab3/scratch/theworld/data/openimages`

Override with flags:
```bash
python examples/interactive_inference.py \
  --spatial-json /path/to/data.json \
  --spatial-images /path/to/images
```

## Generation Settings

Control model output with flags:
```bash
python examples/interactive_inference.py \
  --max_new_tokens 200 \
  --temperature 0.7 \
  --do_sample
```

All `model.generate()` parameters are supported (temperature, top_p, top_k, etc.).
