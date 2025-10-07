# Cosmos World Model Latent Space Extraction

## Overview

This document explains how we extract and use the **world model latent representations** from NVIDIA's Cosmos model to augment Gemma's reasoning capabilities.

## What is the Cosmos World Model?

The Cosmos **World Model** is not a separate component—it's the **VAE encoder** itself. This encoder compresses video frames into a compact latent representation that captures:

- **Spatial structure**: Object locations, scene layout, visual features
- **Temporal dynamics**: Motion, changes over time, action sequences
- **Semantic understanding**: High-level concepts about what's happening in the world

Unlike traditional image encoders, Cosmos uses **3D causal convolutions** (`WanCausalConv3d`) that process both spatial and temporal dimensions while respecting causality (can't peek into the future).

## Cosmos VAE Architecture

### Input → Latent Flow

```
Input: (B, 3, T, H, W)  # RGB video: Batch × Channels × Time × Height × Width
  ↓
Encoder (WanEncoder3d)
  - 3D Causal Convolutions
  - Spatial downsampling: 8× (scale_factor_spatial: 8)
  - Temporal downsampling: 4× (scale_factor_temporal: 4)
  - Progressive channel expansion: 96 → 192 → 384
  - Self-attention in mid-block
  ↓
conv_out: (B, 384, T', H', W') → (B, 32, T', H', W')
  ↓
quant_conv: (B, 32, T', H', W') → (B, 32, T', H', W')
  Split into mean and logvar
  ↓
Latent Distribution: mean=(B, 16, T', H', W'), logvar=(B, 16, T', H', W')
  ↓
Sample/Mean → Latent: (B, 16, T', H', W')
```

### Key Parameters

From `cosmos_pipe.vae.config`:
- `z_dim`: **16** - True latent dimension
- `base_dim`: 96 - Initial encoder channels
- `scale_factor_spatial`: 8 - Spatial compression ratio
- `scale_factor_temporal`: 4 - Temporal compression ratio
- `latents_mean` & `latents_std`: Pre-computed statistics for normalization (16 values each)

### Encoder Output

The encoder's `conv_out` produces **32 channels** (line 360 in cosmos_full_model.txt):
```python
(conv_out): WanCausalConv3d(384, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
```

These 32 channels represent:
- **First 16 channels**: Mean of the latent distribution
- **Last 16 channels**: Log-variance of the latent distribution

The `quant_conv` layer maintains these 32 channels and prepares them for the latent space.

## Three Options for Extracting World Model Features

### Option 1: Raw Encoder Output (32 dims)

**Code:**
```python
latent_img_embeds = self.cosmos_vae_encoder(input_pixels)  # (B, 32, T, H, W)
```

**What you get:**
- Direct encoder output before VAE quantization
- 32 channels: concatenated [mean, logvar]
- Raw, unnormalized features

**Pros:**
- ✅ Fastest (single forward pass through encoder)
- ✅ Preserves both mean and variance information
- ✅ Direct access to encoder features

**Cons:**
- ❌ Not the "true" latent space (pre-quantization)
- ❌ No latent normalization applied
- ❌ Higher dimensionality (32 vs 16)
- ❌ Includes log-variance which may not be useful for conditioning

**Use case:** Quick prototyping, when you want all encoder info

---

### Option 2: Sampled Latent (16 dims, stochastic)

**Code:**
```python
latent_dist = self.cosmos_pipe.vae.encode(input_pixels).latent_dist
latent_img_embeds = latent_dist.sample()  # (B, 16, T, H, W)
```

**What you get:**
- Sample from the Gaussian latent distribution: `z = mean + std * ε`, where `ε ~ N(0,1)`
- 16 channels of true latent representation
- Normalized using pre-computed `latents_mean` and `latents_std`

**Pros:**
- ✅ Proper VAE sampling procedure
- ✅ Uses true 16-dim latent space (more compact)
- ✅ Benefits from learned latent normalization
- ✅ Represents uncertainty in the encoding

**Cons:**
- ❌ **Stochastic**: Different forward passes produce different results
- ❌ Harder to reproduce results
- ❌ May add noise during inference

**Use case:** Training with VAE-style regularization, when stochasticity is desired

---

### Option 3: Latent Mean (16 dims, deterministic) ⭐ **CHOSEN**

**Code:**
```python
latent_dist = self.cosmos_pipe.vae.encode(input_pixels).latent_dist
latent_img_embeds = latent_dist.mean  # (B, 16, T, H, W)
```

**What you get:**
- Mean of the latent distribution (deterministic)
- 16 channels of true latent representation
- Normalized using pre-computed `latents_mean` and `latents_std`

**Pros:**
- ✅ **Deterministic**: Same input always produces same output
- ✅ Uses true 16-dim latent space (semantically meaningful)
- ✅ Benefits from learned latent normalization
- ✅ More efficient: 16 dims vs 32 dims → smaller projection layer
- ✅ Best for inference and training consistency

**Cons:**
- ❌ Slightly slower than Option 1 (full VAE encode path)
- ❌ Ignores variance information

**Use case:** Production inference, stable training, reasoning tasks

---

## Why Option 3 is Best for Our Use Case

### 1. Semantic Meaning
The 16-dim latent space is where Cosmos's world model truly "lives." During Cosmos training, the VAE learns to compress visual information into this 16-dim bottleneck while preserving the ability to reconstruct the world. This forces the model to learn semantically meaningful features.

### 2. Normalization
The latent distribution uses pre-computed statistics:
```python
latents_mean = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, ...]  # 16 values
latents_std = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, ...]      # 16 values
```

These normalize the latent space to have consistent scale and distribution, making it easier for downstream models (Gemma) to process.

### 3. Deterministic Inference
For reasoning tasks, we want consistency. Given the same image, we should get the same world model representation. Option 2's stochasticity would make debugging and evaluation harder.

### 4. Efficiency
16 dims → Gemma (2304 dims) requires a smaller projection layer than 32 dims → Gemma:
- Option 1/2: `nn.Linear(32, 2304)` = 73,728 parameters
- Option 3: `nn.Linear(16, 2304)` = 36,864 parameters (50% smaller)

### 5. World Model Fidelity
The decoder is trained to reconstruct from the 16-dim latents, not the 32-dim encoder output. This means the 16-dim space is optimized to contain all the world information needed for reconstruction.

## Implementation in Our Model

### Current Architecture (model.py)

```python
# 1. Extract world model latent (Option 3)
latent_dist = self.cosmos_pipe.vae.encode(input_pixels).latent_dist
latent_img_embeds = latent_dist.mean  # (B, 16, T, H, W)

# 2. Reshape for sequence processing
b, c, t, h, w = latent_img_embeds.shape
latent_img_embeds = latent_img_embeds.squeeze(2)  # Remove temporal dim for single frame
reshaped_world_embeds = latent_img_embeds.permute(0, 2, 3, 1).reshape(b, h * w, c)
# Shape: (B, H'×W', 16) where H'=H/8, W'=W/8

# 3. Project to Gemma dimension
projected_world_embeds = self.world_projection(reshaped_world_embeds)
# Shape: (B, H'×W', 2304)

# 4. Get Gemma text embeddings
text_embeds = self.gemma.get_input_embeddings()(input_ids)
# Shape: (B, seq_len, 2304)

# 5. Concatenate world model + text
combined_embeds = torch.cat([projected_world_embeds, text_embeds], dim=1)
# Shape: (B, H'×W' + seq_len, 2304)

# 6. Forward through Gemma
outputs = self.gemma(inputs_embeds=combined_embeds, attention_mask=combined_attention_mask)
```

### Input/Output Shapes

For a 224×224 image:
- Input: `(1, 3, 224, 224)`
- After adding temporal dim: `(1, 3, 1, 224, 224)`
- Latent mean: `(1, 16, 1, 28, 28)` - compressed 8×8 spatially
- Reshaped: `(1, 784, 16)` - 784 = 28×28 spatial tokens
- Projected: `(1, 784, 2304)` - ready for Gemma

## Comparison Summary

| Feature | Option 1 (Raw) | Option 2 (Sample) | Option 3 (Mean) ⭐ |
|---------|----------------|-------------------|-------------------|
| Dimensions | 32 | 16 | 16 |
| Deterministic | ✅ | ❌ | ✅ |
| Normalized | ❌ | ✅ | ✅ |
| Speed | Fastest | Slower | Slower |
| VAE-proper | ❌ | ✅ | ✅ |
| Training stability | Medium | Low | High |
| Inference quality | Good | Variable | Best |
| Projection params | 73K | 37K | 37K |

## References

- Cosmos VAE config: `cosmos_full_model.txt:365`
- Encoder architecture: `cosmos_full_model.txt:260-361`
- Quantization layer: `cosmos_full_model.txt:139`
- AutoencoderKL docs: https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl
