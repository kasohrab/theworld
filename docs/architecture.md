# TheWorld Architecture Guide

## Overview

**TheWorld** fuses two powerful pretrained models to create a vision-language-world model that understands both static visual content and temporal dynamics:

- **Gemma 3** (4B params) - Vision-language model for static understanding
- **Cosmos World Model** (2B params) - Temporal prediction model for dynamics

This document explains how images flow through both encoders and how the representations are combined.

---

## Table of Contents

1. [Vision Processing Pipeline](#vision-processing-pipeline)
2. [Image Patching and Tokenization](#image-patching-and-tokenization)
3. [World Model Latent Space](#world-model-latent-space)
4. [Autoregressive World Rollout](#autoregressive-world-rollout)
5. [Single-Pass Architecture](#single-pass-architecture)
6. [Token Flow and Attention](#token-flow-and-attention)

---

## Vision Processing Pipeline

TheWorld processes each input image **twice** through different encoders with different purposes:

### High-Level Flow

```
Input Image (PIL/tensor)
        │
        ├─────────────────────────┬─────────────────────────┐
        │                         │                         │
        v                         v                         v
[Gemma Processor]        [Cosmos VAE]          [Cosmos Pipeline (optional)]
    (SigLIP)                                    (Autoregressive Rollout)
        │                         │                         │
        │                    Single Frame              Multi-Frame
        │                   (num_world_steps=0)      (num_world_steps>0)
        │                         │                         │
        v                         v                         v
~264 vision tokens      784 world tokens          784×T world tokens
  (static visual)       (current state)         (current + future)
        │                         │                         │
        └─────────────────────────┴─────────────────────────┘
                                  │
                                  v
                    Combined Token Sequence
         [vision tokens | world tokens | text tokens]
                                  │
                                  v
                       Gemma Language Model
                        (All Transformer Layers)
                                  │
                                  v
                            Output Logits
```

### Dual Encoder Architecture

| Component | Purpose | Input Size | Patch Size | Output Tokens | Compression |
|-----------|---------|------------|------------|---------------|-------------|
| **SigLIP** | Static visual understanding (objects, scenes, text) | 224×224 or 896×896 | 14×14 | ~264 tokens | 16× or 64× spatial |
| **Cosmos VAE** | Temporal dynamics (motion, physics, future states) | 512×512 typical | 8×8 | 784 tokens/frame | 8× spatial |

---

## Image Patching and Tokenization

### SigLIP Vision Encoder (Gemma 3)

**Purpose:** Extract static visual features for object recognition, scene understanding, OCR, etc.

#### Processing Flow

```
Input Image
    │
    v
Gemma Processor (AutoProcessor)
    │
    ├─ Resize to target resolution (224×224 or 896×896)
    ├─ Normalize (ImageNet stats)
    └─ Add to chat template with image tokens
    │
    v
SigLIP Encoder
    │
    ├─ Divide into 14×14 patches
    ├─ Linear embedding per patch
    ├─ Add positional encodings
    └─ Vision transformer (12-24 layers)
    │
    v
Vision Features
    │
    └─ Multi-modal projector (vision dim → Gemma dim)
    │
    v
~264 vision tokens @ 2304-dim
```

#### Resolution and Token Count

For typical Gemma 3 models:

| Resolution | Patch Size | Grid Size | Token Count | Use Case |
|------------|------------|-----------|-------------|----------|
| 224×224 | 14×14 | 16×16 | 256 tokens | Fast inference |
| 896×896 | 14×14 | 64×64 | 4096 tokens | High detail |
| 448×448 | 14×14 | 32×32 | 1024 tokens | Balanced |

**Note:** Token count formula: `(height ÷ patch_size) × (width ÷ patch_size)`

#### Visual Representation

```
Original Image (224×224)                SigLIP Patches (14×14)
┌───────────────────┐                   ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│                   │                   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│                   │    Divide into    ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│                   │    14×14 patches  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│    Input Image    │    ────────────>  │ │ │ │ ... 16×16 grid ... │ │ │
│                   │                   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│                   │                   │ │ │ │                   │ │ │
│                   │                   └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
└───────────────────┘
                                        Each patch → 1 token
                                        Total: 16×16 = 256 tokens
```

---

### Cosmos VAE Encoder (World Model)

**Purpose:** Capture spatial structure and temporal dynamics in a compact latent space.

#### Processing Flow

```
Input Image
    │
    v
Convert to tensor (if PIL)
    │
    ├─ Normalize to [0, 1] or [-1, 1]
    └─ Add batch and temporal dims: (B, C, T, H, W)
    │
    v
Cosmos VAE Encoder
    │
    ├─ 3D Causal Convolutions (WanCausalConv3d)
    ├─ Spatial downsampling: 8× (512→64)
    ├─ Temporal downsampling: 4× (if video)
    ├─ Progressive channels: 96 → 192 → 384
    └─ Self-attention in mid-block
    │
    v
Encoder Output (32 channels)
    │
    ├─ First 16 channels: Mean of latent distribution
    └─ Last 16 channels: Log-variance
    │
    v
Latent Distribution
    │
    └─ Extract mean (deterministic, normalized)
    │
    v
784 world tokens/frame @ 16-dim
    │
    v
Temporal Embeddings (if multi-frame)
    │
    ├─ Add learned embedding for t=0, t=1, t=2, ...
    └─ Distinguishes current vs. predicted frames
    │
    v
Projection Layer (16-dim → 2304-dim)
    │
    v
784 world tokens/frame @ 2304-dim
```

#### Spatial Compression

```
Input Image (512×512)                   Cosmos Latent Grid (64×64)
┌───────────────────────────┐          ┌─┬─┬─┬─┬─┬─┬─┬─┐ ... 64 columns
│                           │          ├─┼─┼─┼─┼─┼─┼─┼─┤
│                           │  8×8     ├─┼─┼─┼─┼─┼─┼─┼─┤
│                           │  spatial ├─┼─┼─┼─┼─┼─┼─┼─┤
│       Input Image         │  patches │ │ │ │ 64×64  │
│       512×512 pixels      │  ──────> │ │ │ │ grid   │
│                           │          ├─┼─┼─┼─┼─┼─┼─┼─┤
│                           │          │ │ │ │        │
│                           │          │ ... 64 rows  │
└───────────────────────────┘          └─┴─┴─┴─┴─┴─┴─┴─┘

Each 8×8 pixel region → 1 latent token (16 channels)
Total: 64×64 = 4096 spatial locations

For 224×224 input (typical):
224÷8 = 28, so 28×28 = 784 tokens
```

#### Latent Dimensions

The Cosmos VAE outputs **32 channels** which are split into:
- **First 16 channels:** Mean of the latent distribution (µ)
- **Last 16 channels:** Log-variance of the latent distribution (log σ²)

We use the **mean** (16 channels) because:

| Reason | Benefit |
|--------|---------|
| **Deterministic** | Same input → same output (reproducible) |
| **Semantic** | The 16-dim space is where the world model "lives" |
| **Normalized** | Uses pre-computed `latents_mean` and `latents_std` |
| **Efficient** | 50% fewer parameters in projection (16→2304 vs 32→2304) |
| **Decoder-aligned** | VAE decoder trained to reconstruct from 16-dim latents |

**Alternative options:**
- **Raw encoder output (32-dim)**: Faster but unnormalized, not the "true" latent space
- **Sampled latent (16-dim)**: z = µ + σ·ε, stochastic, adds noise

---

## World Model Latent Space

### Three Extraction Options

#### Option 1: Raw Encoder Output (32 dims)

```python
latent_img_embeds = self.cosmos_vae_encoder(input_pixels)  # (B, 32, T, H, W)
```

**Pros:** Fastest, preserves both mean and variance
**Cons:** Not the "true" latent space, no normalization, higher dimensionality

---

#### Option 2: Sampled Latent (16 dims, stochastic)

```python
latent_dist = self.cosmos_pipe.vae.encode(input_pixels).latent_dist
latent_img_embeds = latent_dist.sample()  # (B, 16, T, H, W)
```

Sample from Gaussian: z = µ + σ·ε

**Pros:** Proper VAE sampling, uncertainty representation
**Cons:** **Non-deterministic** - different results each forward pass

---

#### Option 3: Latent Mean (16 dims, deterministic) ⭐ **CHOSEN**

```python
latent_dist = self.cosmos_pipe.vae.encode(input_pixels).latent_dist
latent_img_embeds = latent_dist.mean  # (B, 16, T, H, W)
```

**Pros:**
✅ Deterministic (same input → same output)
✅ Uses true 16-dim latent space (semantically meaningful)
✅ Normalized using learned statistics
✅ More efficient: 16→2304 projection vs 32→2304
✅ Best for inference and training consistency

**Cons:** Ignores variance information

---

### Latent Normalization

The latent distribution uses pre-computed statistics from Cosmos training:

```python
latents_mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, ...]  # 16 values
latents_std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, ...]      # 16 values
```

These normalize the latent space to have consistent scale, making it easier for the projection layer to map to Gemma's embedding space.

---

## Autoregressive World Rollout

### Single Frame vs. Multi-Frame

#### Single Frame (num_world_steps=0)

```python
model = TheWorld("google/gemma-3-4b-it", num_world_steps=0)
```

```
Input Image (t=0)
        │
        v
    VAE Encode
        │
        v
Latent Mean (B, 16, 1, 28, 28)
        │
        v
784 world tokens (current state only)
```

**Use case:** Fast inference, static image understanding, simple VQA

---

#### Multi-Frame (num_world_steps>0)

```python
model = TheWorld("google/gemma-3-4b-it", num_world_steps=4)
```

```
Input Image (t=0)
        │
        v
Cosmos Pipeline (Autoregressive Rollout)
        │
        ├─ VAE encode input
        ├─ Transformer predicts latent at t=1
        ├─ Use t=0,1 to predict t=2
        ├─ Use t=0,1,2 to predict t=3
        └─ Use t=0,1,2,3 to predict t=4
        │
        v
Latent Sequence (B, 16, 5, 28, 28)
        │
        ├─ t=0: Current state (encoded)
        ├─ t=1: Predicted 1 step ahead
        ├─ t=2: Predicted 2 steps ahead
        ├─ t=3: Predicted 3 steps ahead
        └─ t=4: Predicted 4 steps ahead
        │
        v
Add Temporal Embeddings
        │
        ├─ temporal_embedding[0] → t=0 tokens
        ├─ temporal_embedding[1] → t=1 tokens
        ├─ temporal_embedding[2] → t=2 tokens
        ├─ temporal_embedding[3] → t=3 tokens
        └─ temporal_embedding[4] → t=4 tokens
        │
        v
3,920 world tokens (784 × 5 frames)
```

**Use case:** Motion reasoning, physics prediction, temporal dynamics, "what happens next?"

---

### Temporal Embeddings

**Purpose:** Help the model distinguish between current frame (t=0) and future predictions (t=1, 2, ...).

**Implementation:**
```python
self.temporal_embedding = nn.Embedding(max_world_steps + 1, cosmos_dim)  # (17, 16)

temporal_ids = torch.arange(t, device=self.device)  # [0, 1, 2, 3, 4]
temporal_embeds = self.temporal_embedding(temporal_ids)  # (T, 16)
latent_img_embeds = latent_img_embeds + temporal_embeds.view(1, t, 1, 1, c)
```

Each timestep gets a unique learned embedding added to all spatial locations, similar to positional encodings in transformers.

---

### Configuration Examples

```python
# Fast: Single frame only
model = TheWorld("google/gemma-3-4b-it", num_world_steps=0)
# → 784 world tokens

# Moderate: Predict 4 future frames
model = TheWorld("google/gemma-3-4b-it", num_world_steps=4)
# → 3,920 world tokens (784 × 5)

# Rich temporal context: Predict 8 future frames
model = TheWorld("google/gemma-3-4b-it", num_world_steps=8)
# → 7,056 world tokens (784 × 9)

# Override at inference
outputs = model.forward(image, text, num_world_steps=12)
# → 10,192 world tokens (784 × 13)
```

---

### Token Budget Considerations

| num_world_steps | Frames | World Tokens | + Vision (~264) | + Text (~20) | Total |
|-----------------|--------|--------------|-----------------|--------------|-------|
| 0 | 1 | 784 | 1,048 | 1,068 | **~1K** |
| 2 | 3 | 2,352 | 2,616 | 2,636 | **~3K** |
| 4 | 5 | 3,920 | 4,184 | 4,204 | **~4K** |
| 8 | 9 | 7,056 | 7,320 | 7,340 | **~7K** |
| 16 | 17 | 13,328 | 13,592 | 13,612 | **~14K** |

**Gemma context limits:**
- Gemma 2B/4B: typically 8192 tokens
- Safe maximum: `num_world_steps ≤ 8` (~7K tokens)
- `num_world_steps=16` may exceed context length

---

## Single-Pass Architecture

### The Core Principle

World embeddings flow through **all transformer layers** alongside vision and text embeddings in a **single forward pass**.

### Complete Execution Flow

```
1. COSMOS WORLD MODEL
   ┌──────────────────────────────────────────┐
   │ Input Image → VAE → Latent Mean         │
   │ Optional: Autoregressive Rollout        │
   │ Add Temporal Embeddings                 │
   │ Project: 16-dim → 2304-dim              │
   └──────────────────────────────────────────┘
                    ↓
   projected_world_embeds: [B, num_world_tokens, 2304]


2. CHAT TEMPLATE
   ┌──────────────────────────────────────────┐
   │ Format: "<the_world_start> <end>"       │
   │ Add image and text                       │
   │ Apply processor.apply_chat_template()    │
   └──────────────────────────────────────────┘
                    ↓
   input_ids, pixel_values, attention_mask


3. MANUAL EMBEDDING CONSTRUCTION
   ┌──────────────────────────────────────────┐
   │ text_embeds = embed_tokens(input_ids)    │
   │ vision_features = get_image_features()   │ ← Runs SigLIP!
   │ inputs_embeds = masked_scatter()         │
   └──────────────────────────────────────────┘
                    ↓
   inputs_embeds: [B, seq_len, 2304]
   (Contains: text tokens + SigLIP vision features)


4. INSERT WORLD EMBEDDINGS
   ┌──────────────────────────────────────────┐
   │ Find bracket positions in input_ids      │
   │ Slice: [before <start>] + [WORLD] +     │
   │        [<end> onwards]                   │
   └──────────────────────────────────────────┘
                    ↓
   combined_embeds: [text] [<start>] [WORLD] [<end>] [IMAGE] [text]
                    [B, combined_seq_len, 2304]


5. SINGLE FORWARD PASS
   ┌──────────────────────────────────────────┐
   │ language_model(inputs_embeds)            │
   │                                          │
   │ ┌─────────────────────────────────────┐ │
   │ │ Transformer Layer 1                  │ │
   │ │ ├─ Self-attention (all modalities)   │ │
   │ │ └─ FFN                               │ │
   │ ├─────────────────────────────────────┤ │
   │ │ Transformer Layer 2                  │ │
   │ │  ...                                 │ │
   │ ├─────────────────────────────────────┤ │
   │ │ Transformer Layer 26 (final)         │ │
   │ └─────────────────────────────────────┘ │
   │                                          │
   │ World, Vision, Text interact at          │
   │ EVERY layer through attention            │
   └──────────────────────────────────────────┘
                    ↓
   hidden_states: [B, combined_seq_len, 2304]


6. LM HEAD
   ┌──────────────────────────────────────────┐
   │ logits = lm_head(hidden_states)          │
   └──────────────────────────────────────────┘
                    ↓
   logits: [B, combined_seq_len, vocab_size]
```

---

### Why Single-Pass Matters

#### ❌ Naive Approach (Double Processing - WRONG)

```python
# WRONG: Process ALL layers twice
gemma_outputs = self.gemma.model(input_ids, pixel_values, ...)  # Pass 1: Without world
embeddings = gemma_outputs.hidden_states[0]
combined_embeds = insert_world(embeddings)
outputs = self.gemma.language_model(combined_embeds)  # Pass 2: With world
```

**Problems:**
- ❌ Wasteful: Processes all 26 transformer layers twice
- ❌ Incomplete: First pass has no world embeddings at all
- ❌ Inconsistent: World embeddings only exist in second pass
- ❌ Wrong: Vision features in first pass are computed without world context

#### ✅ Current Approach (Single Pass - CORRECT)

```python
# CORRECT: Process ALL layers once with everything
inputs_embeds = self.gemma.model.language_model.embed_tokens(input_ids)
image_features = self.gemma.model.get_image_features(pixel_values)  # Run SigLIP
inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
combined_embeds = insert_world(inputs_embeds)  # Add world before transformer
outputs = self.gemma.language_model(combined_embeds)  # Single pass with everything
```

**Benefits:**
- ✅ Efficient: Single pass through all transformer layers
- ✅ Complete: World embeddings present from layer 1
- ✅ Proper integration: Vision, world, and text process together
- ✅ Attention: All modalities can attend to each other at every layer

---

### Implementation Details

#### Manual Embedding Construction (Step 3)

We manually replicate what `Gemma3Model.forward()` does internally:

```python
# 1. Get text token embeddings (image tokens are placeholders at this point)
inputs_embeds = self.gemma.model.language_model.embed_tokens(input_ids)
# Shape: [B, seq_len, 2304]

# 2. Process vision through SigLIP + multi-modal projector
with torch.no_grad():
    image_features = self.gemma.model.get_image_features(pixel_values)
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

    # 3. Replace image token placeholders with real SigLIP features
    special_image_mask = self.gemma.model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_features
    )
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

# Now inputs_embeds contains: [text tokens + real SigLIP features + text tokens]
```

**Why manual construction?**

We can't use `gemma.model()` directly because:
1. It processes through all transformer layers immediately
2. We need to insert world embeddings **before** the transformer processes them
3. Manual construction gives us the embeddings with vision features but **before** layer processing

---

#### World Embedding Insertion (Step 4)

```python
# Find bracket positions
start_positions = (input_ids == self.world_start_id).nonzero(as_tuple=True)[1]
end_positions = (input_ids == self.world_end_id).nonzero(as_tuple=True)[1]

start_pos = start_positions[0].item()
end_pos = end_positions[0].item()

# Slice and insert
embeddings_before = embeddings[:, : start_pos + 1, :]  # Up to and including <start>
embeddings_after = embeddings[:, end_pos:, :]  # From <end> onwards (inclusive)

# Concatenate: [before] + [world tokens] + [after]
combined_embeds = torch.cat([
    embeddings_before,
    projected_world_embeds,
    embeddings_after
], dim=1)
```

**Result sequence:**
```
[BOS] [text] [<the_world_start>] [WORLD TOKENS] [<the_world_end>] [IMAGE TOKENS] [text]
```

---

## Token Flow and Attention

### Example Token Sequence

For input: "What happens next?" with 4-step world rollout:

```
Position  | Token Type          | Count | Dimension | Source
----------|---------------------|-------|-----------|---------------------------
0         | BOS                 | 1     | 2304      | Chat template
1-5       | Text tokens         | 5     | 2304      | "What happens next?"
6         | <the_world_start>   | 1     | 2304      | Special token
7-3926    | World embeddings    | 3920  | 2304      | Cosmos (5 frames × 28×28)
3927      | <the_world_end>     | 1     | 2304      | Special token
3928-4191 | Image embeddings    | 264   | 2304      | SigLIP vision features
4192-...  | More text / answer  | ...   | 2304      | Response tokens
```

**Total context:** ~4,200 tokens (varies by prompt length)

---

### Multi-Modal Attention

With the single-pass architecture, the transformer attention enables:

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│   World    │◄────┤   Vision   │◄────┤    Text    │
│   Tokens   │────►│   Tokens   │────►│   Tokens   │
└────────────┘     └────────────┘     └────────────┘
      ▲                  ▲                  ▲
      │                  │                  │
      └──────────────────┴──────────────────┘
             Bi-directional Attention
         (at every transformer layer)
```

**Attention interactions:**

1. **World ↔ Vision**: World tokens attend to current visual state
   - Learn correlations between predicted future and present
   - Ground future predictions in observed visual features

2. **World ↔ Text**: World tokens attend to text query
   - Incorporate linguistic context (e.g., "what happens next?")
   - Condition predictions on question semantics

3. **Vision ↔ Text**: Standard vision-language interaction
   - Ground language in visual observations
   - Resolve visual references in text

4. **All ↔ All**: Full multimodal integration (causal masking)
   - Rich cross-modal reasoning
   - Holistic understanding across all three modalities

---

### Computational Characteristics

**Single-pass cost:**
```
O(L × N²)
```
where:
- L = number of transformer layers (26 for Gemma 2B/4B)
- N = total sequence length (~4K tokens with rollout)

**Previous double-pass cost:**
```
O(2 × L × N²) ≈ 2× the cost
```

**Memory:**
- Single pass stores one set of KV cache
- More efficient for generation tasks
- Better GPU memory utilization

**Quality:**
- Better integration of world embeddings
- More coherent multimodal understanding
- World features properly contextualized across all layers

---

## Summary

### Key Design Choices

| Decision | Rationale |
|----------|-----------|
| **Dual encoders** | SigLIP for static, Cosmos for dynamics |
| **Latent mean (not sample)** | Deterministic, normalized, efficient |
| **Autoregressive rollout** | True temporal prediction, not just frame encoding |
| **Temporal embeddings** | Distinguish current vs. future timesteps |
| **Single-pass architecture** | Efficient, proper integration, full attention |
| **Projection to Gemma dim** | Unified embedding space for all modalities |

---

### Token Count Summary

For a 224×224 image with 4-step rollout:

```
SigLIP vision:    ~264 tokens
Cosmos world:   3,920 tokens (784 × 5 frames)
Text prompt:      ~20 tokens
────────────────────────────
Total:         ~4,204 tokens
```

**Modality distribution:**
- World tokens: ~93% (dominant due to rollout)
- Vision tokens: ~6%
- Text tokens: ~1%

---

### Flexibility

TheWorld supports various configurations:

```python
# Fast inference: No world model
model = TheWorld(..., num_world_steps=0)
# → 1,048 tokens (264 vision + 784 world + text)

# Balanced: Moderate temporal context
model = TheWorld(..., num_world_steps=4)
# → 4,204 tokens

# Heavy: Rich temporal reasoning
model = TheWorld(..., num_world_steps=8)
# → 7,340 tokens

# Runtime override
outputs = model.forward(image, text, num_world_steps=2)
```

---

## References

- **Model implementation**: `theworld/modeling.py`
- **SigLIP documentation**: https://huggingface.co/docs/transformers/model_doc/siglip
- **Cosmos paper**: https://arxiv.org/pdf/2503.15558
- **Gemma 3 documentation**: https://huggingface.co/docs/transformers/model_doc/gemma3
- **Related docs**:
  - Training: `training_infrastructure_design.md`
  - Multi-stage training: `multi_stage_training.md`
  - Evaluation: `evaluation.md`
  - Loss function: `loss_function_and_evaluation.md`
