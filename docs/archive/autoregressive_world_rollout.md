# Autoregressive World Model Rollout

## Overview

This document explains how we use **autoregressive rollout** to predict future world states with the Cosmos world model, giving Gemma rich temporal context for reasoning about dynamics, motion, and physics.

## The Problem: Static vs. Dynamic World Understanding

### Current Approach (Single Frame)
```
Input image → Encode to latent → Project → Gemma
```
- **Pro**: Fast, simple, works for static reasoning
- **Con**: No information about future states, motion, or temporal dynamics

### Autoregressive Rollout (Multi-Frame Prediction)
```
Input image → Cosmos predicts future frames → Encode all to latents → Project → Gemma
```
- **Pro**: Gemma sees current + predicted future states
- **Pro**: Enables reasoning about motion, causality, physics
- **Con**: Slower (requires running diffusion/transformer)

## What is Autoregressive Rollout?

**Autoregressive** means predicting the next state based on previous states, step by step:

```
Frame t=0 (input) → Predict t=1 → Predict t=2 → ... → Predict t=C
```

Each prediction uses all previous frames as context, building a trajectory through latent space.

## How Cosmos Predicts Future Frames

The `nvidia/Cosmos-Predict2-2B-Video2World` model is specifically designed for world prediction:

1. **Input**: Single frame or short video
2. **Process**:
   - Encode to latent space (VAE)
   - Use transformer to predict next latent state
   - Repeat autoregressively for C steps
   - Optionally decode back to pixels (not needed for us)
3. **Output**: Sequence of latent representations for t=0, t=1, ..., t=C

### Pipeline Parameters

```python
output = cosmos_pipe(
    image=input_pixels,           # Initial frame(s)
    num_frames=1 + num_world_steps,  # Total frames to generate
    num_inference_steps=10,       # Denoising steps (speed vs quality)
    output_type="latent",         # Return latents, not decoded pixels
    return_dict=True
)
```

- `num_frames`: Total frames = 1 (input) + C (predicted)
- `num_inference_steps`: Diffusion denoising steps (10-35 typical, lower = faster)
- `output_type="latent"`: Return latent representations, skip decoding

## Implementation in TheWorld Model

### Initialization Parameters (model.py:7-13)

```python
model = TheWorld(
    gemma_model_name="google/gemma-2-2b-it",
    device="cuda",
    num_world_steps=4,      # Predict 4 future frames
    max_world_steps=16      # Maximum for embeddings (can increase at inference)
)
```

**Parameters:**
- `num_world_steps`: Number of future frames to predict (default at init)
- `max_world_steps`: Maximum timesteps for temporal embedding table

### Forward Pass Options

#### Option 1: Use Instance Default
```python
outputs = model(input_pixels, input_ids, attention_mask)
# Uses num_world_steps from initialization
```

#### Option 2: Override at Forward Time
```python
outputs = model(input_pixels, input_ids, attention_mask, num_world_steps=8)
# Predicts 8 future frames, overriding instance default
```

### Architecture Flow

```
Input: (B, 3, H, W) image

1. Add temporal dimension
   ↓ (B, 3, 1, H, W)

2a. If num_world_steps == 0 (single-step):
    VAE encode → latent_mean → (B, 16, 1, H, W)

2b. If num_world_steps > 0 (multi-step):
    Cosmos pipeline → predict C future frames → (B, 16, T, H, W)
    where T = 1 + num_world_steps

3. Reshape and add temporal embeddings
   (B, 16, T, H, W) → (B, T, H, W, 16)
   Add temporal_embedding[0..T-1] to each timestep
   ↓ (B, T, H, W, 16)

4. Flatten to sequence
   ↓ (B, T×H×W, 16)

5. Project to Gemma dimension
   ↓ (B, T×H×W, gemma_dim)

6. Concatenate with text embeddings
   [world_embeds, text_embeds] → (B, T×H×W + seq_len, gemma_dim)

7. Forward through Gemma
```

### Example Shapes

For 224×224 input with num_world_steps=4:

| Stage | Shape | Description |
|-------|-------|-------------|
| Input pixels | `(1, 3, 224, 224)` | RGB image |
| After unsqueeze | `(1, 3, 1, 224, 224)` | Add temporal dim |
| After Cosmos | `(1, 16, 5, 28, 28)` | 5 frames (1 input + 4 predicted), 8× spatial compression |
| After temporal embed | `(1, 5, 28, 28, 16)` | Permuted with temporal embeddings |
| Flattened sequence | `(1, 3920, 16)` | 5×28×28 = 3920 tokens |
| Projected | `(1, 3920, 2304)` | Ready for Gemma |
| + Text (7 tokens) | `(1, 3927, 2304)` | Combined context |

## Temporal Embeddings

**Purpose**: Help the model distinguish between current frame (t=0) and future predictions (t=1, 2, ...).

**Implementation** (model.py:49):
```python
self.temporal_embedding = nn.Embedding(max_world_steps + 1, cosmos_dim)
```

**Usage** (model.py:103-106):
```python
temporal_ids = torch.arange(t, device=self.device)  # [0, 1, ..., t-1]
temporal_embeds = self.temporal_embedding(temporal_ids)  # (T, 16)
latent_img_embeds = latent_img_embeds + temporal_embeds.view(1, t, 1, 1, c)
```

Each timestep gets a unique learned embedding added to all spatial locations, similar to positional encodings in transformers.

## Comparison: All Four Options

### Option A: Autoregressive Rollout ⭐ **IMPLEMENTED**
```python
model = TheWorld(..., num_world_steps=4)
# Predicts 4 future frames using Cosmos transformer
```

**What you get:**
- Current frame latent: represents present state
- 4 predicted future frame latents: represent predicted evolution
- Total: 5 timesteps of world model understanding

**Pros:**
- ✅ True temporal dynamics (motion, physics)
- ✅ Enables reasoning about future states
- ✅ Uses Cosmos's core strength (world prediction)
- ✅ Configurable at training/inference time

**Cons:**
- ❌ Slower (requires running full diffusion pipeline)
- ❌ More tokens for Gemma (T×H×W vs H×W)
- ❌ Requires tuning num_inference_steps (speed/quality tradeoff)

**Use cases:**
- Reasoning about motion ("Where will the ball land?")
- Physics understanding ("What happens if I push this?")
- Causal reasoning ("What will happen next?")

---

### Option B: Multi-Scale Spatial Sampling

Encode same frame at different resolutions or crop different regions.

**Pros:**
- ✅ Richer spatial representation
- ✅ No temporal modeling needed

**Cons:**
- ❌ No temporal information
- ❌ May be redundant with existing spatial tokens
- ❌ Not using Cosmos's world modeling capabilities

**Use cases:**
- Fine-grained spatial reasoning
- Multi-resolution feature extraction

---

### Option C: Stochastic Sampling (Multiple VAE Samples)

Sample multiple times from latent distribution: `z = mean + std * ε`.

**Pros:**
- ✅ Captures uncertainty
- ✅ Very simple to implement

**Cons:**
- ❌ Adds noise without new information
- ❌ Same input = similar samples
- ❌ Doesn't leverage world modeling

**Use cases:**
- Uncertainty quantification
- Ensemble methods

---

### Option D: Multi-Frame Input

Provide K actual frames from a video sequence.

**Pros:**
- ✅ Real temporal information (not predicted)
- ✅ Simple: just encode each frame
- ✅ No autoregressive generation needed

**Cons:**
- ❌ Requires multi-frame input data
- ❌ Can't reason about "unseen" futures
- ❌ Not using Cosmos's predictive capabilities

**Use cases:**
- Video understanding tasks
- Action recognition
- When you have actual video data

---

## Configuration Guide

### Training Configuration

**Fast prototyping** (single frame):
```python
model = TheWorld(
    gemma_model_name="google/gemma-2-2b-it",
    num_world_steps=0  # No rollout, just current frame
)
```

**Moderate temporal context**:
```python
model = TheWorld(
    gemma_model_name="google/gemma-2-2b-it",
    num_world_steps=4,        # Predict 4 future frames
    max_world_steps=16        # Allow up to 16 at inference
)
```

**Rich temporal context** (slower):
```python
model = TheWorld(
    gemma_model_name="google/gemma-2-2b-it",
    num_world_steps=8,        # Predict 8 future frames
    max_world_steps=16
)
```

### Inference Configuration

**Override at forward time**:
```python
# Fast inference (no rollout)
outputs = model(pixels, ids, mask, num_world_steps=0)

# Moderate rollout
outputs = model(pixels, ids, mask, num_world_steps=4)

# Heavy rollout for complex reasoning
outputs = model(pixels, ids, mask, num_world_steps=12)
```

## Computational Considerations

### Token Budget

Gemma context length is limited. With rollout:

| num_world_steps | Spatial tokens | Total tokens* | Notes |
|-----------------|----------------|---------------|-------|
| 0 | 784 (28×28) | ~800 | Fast, single frame |
| 2 | 2,352 (3×28×28) | ~2,400 | Moderate |
| 4 | 3,920 (5×28×28) | ~4,000 | Good balance |
| 8 | 7,056 (9×28×28) | ~7,100 | Rich temporal context |
| 16 | 13,328 (17×28×28) | ~13,400 | Very detailed |

*Including ~20 text tokens

**Gemma context limits:**
- Gemma-2B: typically 8192 tokens
- num_world_steps=8 uses ~7,100 tokens → safe
- num_world_steps=16 uses ~13,400 tokens → may exceed limit

### Speed Considerations

**Cosmos pipeline speed** (approximate, on GPU):
- `num_inference_steps=10`: ~2-3 seconds per batch
- `num_inference_steps=35`: ~5-7 seconds per batch

**Recommendation**: Use `num_inference_steps=10` during training for speed.

### Memory Usage

Each rollout step adds ~16×H×W latent memory. For 224×224 images:
- Single frame: ~100KB
- 4-step rollout: ~500KB
- 8-step rollout: ~900KB

Memory is mostly dominated by Gemma (2B params = ~8GB).

## When to Use Autoregressive Rollout

### ✅ Good Use Cases

1. **Motion reasoning**: "Where is the object moving?"
2. **Physics prediction**: "What happens if gravity acts on this?"
3. **Causal reasoning**: "What will happen next?"
4. **Temporal planning**: "How do I reach the goal?"
5. **Dynamic scenes**: Videos, robotics, simulations

### ❌ When Single Frame is Enough

1. **Static image understanding**: "What objects are in the image?"
2. **Classification**: "Is this a cat or dog?"
3. **Simple VQA**: "What color is the car?"
4. **Fast inference**: Real-time applications
5. **Limited compute**: Embedded systems

## Future Extensions

### 1. Adaptive Rollout Length
Dynamically choose num_world_steps based on question complexity:
```python
if requires_temporal_reasoning(question):
    num_world_steps = 8
else:
    num_world_steps = 0
```

### 2. Hierarchical Temporal Pooling
Reduce token count by pooling across time:
```python
# Pool every 2 timesteps → half the tokens
pooled_embeds = temporal_pool(world_embeds, stride=2)
```

### 3. Conditional Rollout
Condition future predictions on actions or interventions:
```python
output = cosmos_pipe(
    image=input_pixels,
    prompt="robot pushes the box",  # Action conditioning
    num_frames=5
)
```

## References

- Cosmos pipeline: `cosmos_pipe` (model.py:17-22)
- Temporal embeddings: (model.py:49)
- Rollout implementation: (model.py:76-94)
- World model latent space: `docs/world_model_latent_space.md`
