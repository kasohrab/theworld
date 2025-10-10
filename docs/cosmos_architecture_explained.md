# Cosmos Architecture Explained

This document explains the Cosmos world model architecture and specifically how the WanEncoder3d VAE works.

## Table of Contents
1. [Overview of Cosmos](#overview-of-cosmos)
2. [WanEncoder3d Architecture](#wanencoder3d-architecture)
3. [How We Use Cosmos in TheWorld](#how-we-use-cosmos-in-theworld)
4. [Why .mode() is the Correct API](#why-mode-is-the-correct-api)

---

## Overview of Cosmos

**Cosmos** is NVIDIA's world model for video prediction and understanding, described in their paper: https://arxiv.org/pdf/2503.15558

### Two-Stage Architecture

Cosmos consists of two main components:

```
Input Video
    ↓
[Stage 1: VAE (WanEncoder3d + WanDecoder3d)]
    → Encode: RGB frames (B, 3, T, H, W) → Latent (B, 16, T, h, w)
    → Decode: Latent → RGB frames
    ↓
[Stage 2: Diffusion Transformer]
    → Text-conditioned diffusion in latent space
    → Predicts future frames autoregressively
```

**Stage 1: Variational Autoencoder (VAE)**
- **Encoder (WanEncoder3d)**: Compresses video frames into 16-dim latent space
- **Decoder (WanDecoder3d)**: Reconstructs video frames from latents
- **Purely visual** - No text conditioning at this stage
- **Spatial compression** - 8× downsampling (512×512 → 64×64 latent)

**Stage 2: Diffusion Transformer**
- **Text conditioning** - Uses T5 text encoder for prompts
- **Future prediction** - Generates future frames via diffusion
- **Autoregressive** - Can predict multiple timesteps ahead

### TheWorld Uses Only Stage 1 (VAE)

We only need the **visual encoding** capability of Cosmos, not the future prediction. Therefore:
- ✅ Use WanEncoder3d to get 16-dim latent embeddings
- ❌ Skip diffusion transformer (slow, text-conditioned, not needed)

---

## WanEncoder3d Architecture

### What is "Wan"?

"Wan" refers to the architecture style - it's a **3D Causal VAE** designed for video data. The key components are:

```python
class WanEncoder3d(nn.Module):
    """3D encoder module for video compression."""

    def __init__(
        self,
        dim=128,                          # Base channel dimension
        z_dim=4,                          # Latent dimension (16 for mean+logvar)
        dim_mult=[1, 2, 4, 4],           # Channel multipliers per block
        num_res_blocks=2,                 # Residual blocks per stage
        attn_scales=[],                   # Attention scales
        temperal_downsample=[True, True, False],  # Temporal downsampling
        dropout=0.0,
        non_linearity="silu",
    ):
```

### Key Architectural Components

#### 1. **WanCausalConv3d** - Causal 3D Convolutions
```python
class WanCausalConv3d(nn.Conv3d):
    """Causal convolution that only looks at past frames."""
```

**Why causal?**
- For video generation, we can't look into the future
- Padding is asymmetric: `(left, right, top, bottom, past, future)` becomes `(p, p, p, p, 2p, 0)`
- Ensures temporal causality for autoregressive generation

**For TheWorld:** We only encode single frames, so causality doesn't matter (but doesn't hurt either).

#### 2. **WanRMS_norm** - RMS Normalization
```python
class WanRMS_norm(nn.Module):
    """Root Mean Square normalization."""

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias
```

**Why RMS instead of LayerNorm/BatchNorm?**
- Faster than LayerNorm (no mean subtraction)
- More stable than BatchNorm for video data
- Common in modern architectures (LLaMA, Gemma use RMSNorm too)

#### 3. **WanResidualBlock** - Residual Blocks with Causal Convs
```python
class WanResidualBlock(nn.Module):
    """Residual block with two causal conv3d layers."""

    def forward(self, x):
        h = self.conv_shortcut(x)              # Shortcut connection
        x = self.norm1(x)
        x = self.nonlinearity(x)               # SiLU activation
        x = self.conv1(x)                      # First causal conv
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv2(x)                      # Second causal conv
        return x + h                           # Residual connection
```

**Standard ResNet-style blocks** adapted for 3D video data with causal convolutions.

#### 4. **WanAttentionBlock** - Spatial Self-Attention
```python
class WanAttentionBlock(nn.Module):
    """Causal self-attention with a single head."""

    def forward(self, x):
        # Shape: (B, C, T, H, W)
        # Process each frame independently: (B*T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

        qkv = self.to_qkv(x)                   # Compute Q, K, V
        x = F.scaled_dot_product_attention(q, k, v)  # Self-attention

        return x + identity                    # Residual connection
```

**Applies attention within each frame** (not across time) - this helps capture spatial dependencies.

#### 5. **WanResample** - Spatial/Temporal Downsampling
```python
class WanResample(nn.Module):
    """Resampling for 2D and 3D data."""

    Modes:
    - "downsample2d": 2× spatial downsampling (T unchanged)
    - "downsample3d": 2× spatial + 2× temporal downsampling
    - "upsample2d": 2× spatial upsampling (decoder)
    - "upsample3d": 2× spatial + 2× temporal upsampling (decoder)
```

**For Cosmos Encoder:**
- `temperal_downsample = [True, True, False]`
- First two stages downsample time: T=8 → T=4 → T=2
- All stages downsample space: 512×512 → 256×256 → 128×128 → 64×64

### Complete Encoder Forward Pass

```python
def forward(self, x):
    # Input: (B, 3, T, H, W) - RGB video

    # 1. Initial convolution
    x = self.conv_in(x)  # (B, 3, T, H, W) → (B, 128, T, H, W)

    # 2. Downsample blocks (4 stages)
    for layer in self.down_blocks:
        x = layer(x)
        # After each stage:
        # - Channels increase: 128 → 256 → 512 → 512
        # - Spatial dims halve: H/2, W/2
        # - Temporal may halve (depends on temperal_downsample)

    # 3. Middle block (residual + attention)
    x = self.mid_block(x)

    # 4. Output head
    x = self.norm_out(x)
    x = self.nonlinearity(x)
    x = self.conv_out(x)  # (B, 512, t, h, w) → (B, 32, t, h, w)

    # Output: (B, 32, t, h, w)
    # Split into mean and logvar: (B, 16, t, h, w) each
    return x
```

### From Encoder Output to Latent Distribution

After the encoder, the `AutoencoderKLWan` class processes the output:

```python
class AutoencoderKLWan(nn.Module):
    def encode(self, x):
        # 1. Run encoder
        h = self._encode(x)  # (B, 32, T, h, w)

        # 2. Quantize (actually just a 1×1 conv)
        enc = self.quant_conv(h)  # (B, 32, T, h, w)

        # 3. Create distribution
        posterior = DiagonalGaussianDistribution(enc)
        # Splits 32 channels → 16 mean + 16 logvar

        return AutoencoderKLOutput(latent_dist=posterior)
```

---

## How We Use Cosmos in TheWorld

### Our Simplified Usage

```python
# TheWorld's CosmosEncoder
class CosmosEncoder(nn.Module):
    def forward(self, images: List[Image.Image]):
        # 1. PIL → Tensor (B, C, H, W)
        tensor_batch = preprocess_images(images)

        # 2. Add time dimension: (B, C, 1, H, W) - single frame
        cosmos_input_5d = tensor_batch.unsqueeze(2)

        # 3. VAE encode → latent distribution
        latent_dist = self.cosmos_pipe.vae.encode(cosmos_input_5d).latent_dist

        # 4. Get deterministic latents using .mode()
        latents = latent_dist.mode()  # (B, 16, 1, h, w)

        # 5. Reshape to tokens: (B, 16, 1, h, w) → (B, h×w, 16)
        latents = latents.squeeze(2).permute(0, 2, 3, 1)
        latents = latents.reshape(B, h*w, 16)

        # 6. Project to Gemma space: (B, h×w, 16) → (B, h×w, 2304)
        world_embeds = self.world_projection(latents)

        return world_embeds
```

### Why This Simplification Works

1. **Single frame only** - We don't need temporal prediction, just visual encoding
2. **No text conditioning** - VAE is purely visual; text goes to Gemma instead
3. **Direct VAE access** - Bypass the full pipeline (avoids diffusion, safety checker)
4. **Deterministic** - Using `.mode()` ensures reproducibility

### Computational Savings

| Component | Full Pipeline | TheWorld | Speedup |
|-----------|--------------|----------|---------|
| Text encoding (T5) | ✅ Used | ❌ Skipped | N/A |
| Safety checker | ✅ Used | ❌ Skipped | N/A |
| VAE encode | ✅ Used | ✅ Used | 1× |
| Diffusion steps | ✅ 10-35 steps | ❌ Skipped | **10-35×** |
| VAE decode | ✅ Used | ❌ Skipped | 2× |

**Total:** ~10-35× faster than using the full pipeline!

---

## Why .mode() is the Correct API

### Understanding DiagonalGaussianDistribution

The VAE encoder outputs a `DiagonalGaussianDistribution`:

```python
class DiagonalGaussianDistribution:
    def __init__(self, parameters):
        # Split 32 channels into mean (16) and logvar (16)
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator=None):
        # Sample from N(mean, std) using reparameterization trick
        noise = torch.randn_like(self.std, generator=generator)
        x = self.mean + self.std * noise
        return x

    def mode(self):
        # Return the mode (peak) of the distribution
        # For Gaussian, mode = mean
        return self.mean

    # NOTE: There is NO .mean() method! Only .mode()
```

### Three Ways to Extract Latents (and Why We Choose .mode())

1. **`.sample()` - Stochastic (VAE Training)**
   ```python
   latents = latent_dist.sample()
   ```
   - ❌ Random: same input → different outputs
   - ❌ Non-deterministic inference
   - ✅ **Differentiable** via reparameterization trick (gradients flow through mean and std)
   - ✅ When to use: VAE training (KL divergence regularization)

2. **Direct mean access - WRONG**
   ```python
   latents = latent_dist.mean  # Accessing internal attribute
   ```
   - ⚠️ Works, but not the public API
   - ⚠️ Bypasses any normalization/scaling that `.mode()` might do
   - ❌ Don't do this

3. **`.mode()` - Deterministic (CORRECT) ✅**
   ```python
   latents = latent_dist.mode()
   ```
   - ✅ Deterministic: same input → same output
   - ✅ **Differentiable** (returns mean tensor, gradients flow normally)
   - ✅ Public API (correct usage)
   - ✅ Returns the mode of the distribution (= mean for Gaussian)
   - ✅ This is what the official decoder uses (see AutoencoderKLWan line 1087)

### Are .mode() and .sample() Differentiable?

**Yes, both are fully differentiable!**

```python
# Backprop through .mode()
latents = latent_dist.mode()  # latents = mean
loss = some_loss(latents)
loss.backward()
# ✅ Gradients flow: loss → mean → encoder parameters

# Backprop through .sample()
latents = latent_dist.sample()  # latents = mean + std * noise
loss = some_loss(latents)
loss.backward()
# ✅ Gradients flow: loss → mean, std → encoder parameters
# (Reparameterization trick: noise has no gradient, so gradients go through mean/std)
```

**Key point:** We use `.mode()` because it's deterministic AND differentiable - perfect for inference and fine-tuning!

### Why Not Use vae.forward()?

You might wonder: "Why not just call `vae.forward(x, sample_posterior=False)`?"

**Because `vae.forward()` returns decoded pixels, not latents!**

```python
# Option A: What we do (correct) ✅
latents = vae.encode(x).latent_dist.mode()
# Returns: (B, 16, 1, H, W) - latent embeddings ✅

# Option B: Using forward() ❌
decoded_images = vae.forward(x, sample_posterior=False)
# Returns: (B, 3, 1, H, W) - reconstructed RGB pixels ❌
# This does: encode → get latents → DECODE back to pixels
# We'd need to re-encode to get latents again!
```

**We need the latent embeddings**, not reconstructed images, so we must call `vae.encode()` directly.

### Proof from Source Code

From `AutoencoderKLWan.forward()` (line 1082-1088):

```python
def forward(self, sample, sample_posterior=False, generator=None):
    x = sample
    posterior = self.encode(x).latent_dist

    if sample_posterior:
        z = posterior.sample(generator=generator)  # Stochastic VAE sampling
    else:
        z = posterior.mode()                       # Deterministic (default!)

    dec = self.decode(z, return_dict=return_dict)  # ← Decodes back to pixels!
    return dec
```

**The official implementation uses `.mode()` by default AND decodes to pixels!**

---

## What Do the Latents Represent?

### The Latents Are the World Model's Clean Understanding

```python
latents = vae.encode(image).latent_dist.mode()  # Shape: (B, 16, 1, H, W)
```

**This is the world model's representation of the visual scene:**
- ✅ **Compressed visual understanding** - 16-dim latent space encodes spatial structure
- ✅ **World state encoding** - Learned to capture information that can reconstruct the scene
- ✅ **Clean, no noise** - This is the actual understanding, not corrupted/noisy
- ✅ **Deterministic** - Same image → same latent representation

### Noise is ONLY for Diffusion (Which We Don't Use)

**Important:** Noise is added ONLY during diffusion-based generation, which we completely skip:

```python
# Full Cosmos Pipeline - Text-to-Video Generation (we DON'T do this)
def generate_future_frames(image, text_prompt):
    # 1. Get initial latent (clean, no noise)
    z0 = vae.encode(image).latent_dist.mode()

    # 2. Diffusion process - iteratively denoise (WE SKIP THIS!)
    for t in diffusion_timesteps:  # 10-35 steps
        # Add noise: z_noisy = z0 + noise * sigma(t)
        z_noisy = scheduler.add_noise(z0, noise, timestep=t)

        # Predict what the noise is (using transformer + text conditioning)
        predicted_noise = transformer(z_noisy, text_embeds, timestep=t)

        # Remove predicted noise to get cleaner latent
        z0 = scheduler.step(predicted_noise, t, z_noisy).prev_sample

    # 3. Decode final clean latent to pixels
    video = vae.decode(z0)
    return video
```

**We never add noise because we're not generating - just encoding!**

### Our Use: Clean Latents Only

```python
# TheWorld's CosmosEncoder
latents = vae.encode(image).latent_dist.mode()
# ✅ Clean latents representing the world state
# ❌ No noise added
# ❌ No diffusion process

world_embeds = projection(latents)
# ✅ Project clean world understanding to Gemma's embedding space
```

### Confusion: VAE Sampling vs Diffusion

**These are two completely different things:**

| Operation | Where | What It Does | Has Noise? |
|-----------|-------|--------------|------------|
| **VAE `.mode()`** | Encoder | Get deterministic latent (mode of distribution) | ❌ No - clean latent |
| **VAE `.sample()`** | Encoder | Sample from latent distribution: `mean + std * noise` | ⚠️ Minimal (reparameterization noise for training) |
| **Diffusion Process** | Transformer | Add noise, then iteratively denoise to generate | ✅ Yes - adds then removes noise over many steps |
| **Our Usage** | TheWorld | Extract world understanding for Gemma | ❌ No - we skip diffusion entirely |

**Key insight:** The `.sample()` method in `AutoencoderKLWan.forward()` is **VAE sampling** (stochastic latent extraction), NOT diffusion! Diffusion is a separate process that happens in the transformer.

---

## Summary

### What is WanEncoder3d?
- **3D Causal VAE encoder** for video data
- **Compresses** RGB frames (B, 3, T, H, W) → latents (B, 16, T, h, w)
- **Spatial compression:** 8× downsampling (512×512 → 64×64)
- **Architecture:** ResNet-style blocks with causal 3D convs, RMS norm, and self-attention

### How TheWorld Uses It
1. ✅ **Direct VAE encoding** - Skip diffusion pipeline
2. ✅ **Single frame** - T=1 (no temporal prediction needed)
3. ✅ **Deterministic** - Use `.mode()` not `.sample()`
4. ✅ **Project to Gemma** - 16-dim → 2304-dim for language model
5. ✅ **10-35× faster** - Bypass text encoding, diffusion, decoding

### Key Takeaways
- **VAE is purely visual** - No text conditioning at this stage
- **`.mode()` is the correct API** - Don't use `.mean` or `.sample()`
- **Causal convolutions** - Designed for autoregressive video generation (but we use single frames)
- **RMS normalization** - Faster and more stable than LayerNorm
- **Spatial compression only** - We don't use temporal compression (T=1 always)

This architecture provides high-quality visual encodings that complement Gemma's language understanding, creating a true vision-language-world model.
