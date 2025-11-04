# Cosmos Architecture Deep Dive

This document provides a comprehensive explanation of the Cosmos world model architecture, focusing on how the WanEncoder3d VAE works and how we integrate it with TheWorld.

## Table of Contents
1. [Overview of Cosmos](#overview-of-cosmos)
2. [WanEncoder3d Architecture](#wanencoder3d-architecture)
3. [VAE Latent Space: Deterministic vs Stochastic](#vae-latent-space-deterministic-vs-stochastic)
4. [Decoder Architecture (That We Skip)](#decoder-architecture-that-we-skip)
5. [Training vs Inference Behavior](#training-vs-inference-behavior)
6. [How We Use Cosmos in TheWorld](#how-we-use-cosmos-in-theworld)
7. [Why .mode() is the Correct API](#why-mode-is-the-correct-api)
8. [Gradient Flow Analysis](#gradient-flow-analysis)

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

### Spatial Compression Math

For different input sizes, here's how the compression works:

**224×224 input (typical for TheWorld):**
```
Input:   (B, 3, 1, 224, 224)      # Single frame
         ↓ 8× spatial compression
Encoded: (B, 32, 1, 28, 28)       # 28 = 224 / 8
         ↓ split channels
Mean:    (B, 16, 1, 28, 28)       # First 16 channels
LogVar:  (B, 16, 1, 28, 28)       # Last 16 channels
         ↓ .mode() → mean only
Latents: (B, 16, 1, 28, 28)       # Deterministic output
         ↓ reshape to tokens
Tokens:  (B, 784, 16)             # 784 = 28×28 spatial positions
         ↓ project
Embeddings: (B, 784, 2304)        # Ready for Gemma
```

**512×512 input (high-resolution):**
```
Input:   (B, 3, 1, 512, 512)
         ↓ 8× spatial compression
Latents: (B, 16, 1, 64, 64)       # 64 = 512 / 8
         ↓ reshape to tokens
Tokens:  (B, 4096, 16)            # 4096 = 64×64 spatial positions
```

**Key parameters from Cosmos config:**
- `z_dim`: 16 (true latent dimension)
- `base_dim`: 96-128 (initial encoder channels)
- `scale_factor_spatial`: 8 (spatial compression ratio)
- `scale_factor_temporal`: 4 (temporal compression, unused when T=1)
- `latents_mean` & `latents_std`: Pre-computed normalization statistics (16 values each)

---

## VAE Latent Space: Deterministic vs Stochastic

### The Critical Question: How Do We Extract Latents?

The VAE encoder outputs a `DiagonalGaussianDistribution` object, which provides **two main extraction methods**:

```python
class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        # Split 32 channels into mean (16) and logvar (16)
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, generator=None) -> torch.Tensor:
        """STOCHASTIC: Sample from the distribution using reparameterization trick."""
        # z = μ + σ * ε, where ε ~ N(0,1)
        noise = randn_tensor(self.mean.shape, generator=generator, ...)
        x = self.mean + self.std * noise
        return x

    def mode(self) -> torch.Tensor:
        """DETERMINISTIC: Return the mode (peak) of Gaussian = mean."""
        return self.mean
```

### Three Ways to Extract Latents

| Method | Code | Deterministic? | Differentiable? | When to Use |
|--------|------|----------------|-----------------|-------------|
| **`.sample()`** | `latents = dist.sample()` | ❌ No (adds noise) | ✅ Yes | VAE training (KL regularization) |
| **`.mode()` ⭐** | `latents = dist.mode()` | ✅ Yes | ✅ Yes | Inference with frozen VAE |
| **`.mean` attribute** | `latents = dist.mean` | ✅ Yes | ✅ Yes | ❌ Not public API, don't use |

### What TheWorld Uses

**In `python/theworld/modeling/cosmos_encoder.py:120`:**

```python
encoder_output = self.cosmos_vae.encode(cosmos_input_5d)
latent_dist = encoder_output.latent_dist
latents = latent_dist.mode()  # ✅ DETERMINISTIC - Always returns mean
```

**Why `.mode()` is correct for TheWorld:**

1. ✅ **Deterministic** - Same input always produces same output (critical for reproducibility)
2. ✅ **Fully differentiable** - Gradients flow through the mean tensor (just returns `self.mean`)
3. ✅ **Public API** - This is the official method, not accessing internal attributes
4. ✅ **No noise added** - Clean latent representation
5. ✅ **Production-ready** - Used by official Cosmos decoder (see `AutoencoderKLWan.forward()` line 1087)
6. ✅ **VAE is frozen** - Not training the encoder, just using it for inference

### Understanding Stochastic vs Deterministic

**When would you use `.sample()` (stochastic)?**

During **VAE pre-training** (which NVIDIA already did for us):

```python
# VAE training mode (not what TheWorld does)
latents = dist.sample()  # z = mean + std * noise
reconstructed = decoder(latents)

# Two loss components:
reconstruction_loss = MSE(reconstructed, original_image)
kl_loss = KL_divergence(dist, N(0,1))  # Regularize to unit Gaussian
total_loss = reconstruction_loss + β * kl_loss
```

The stochastic sampling is **essential for VAE training** because:
- The KL loss needs the distribution parameters (mean, std)
- Sampling enables the reparameterization trick for gradient flow
- Regularizes the latent space to approximate N(0,1)

**When do you use `.mode()` (deterministic)?**

During **inference with frozen VAE** (what TheWorld does):

```python
# TheWorld training/inference mode
latents = dist.mode()  # z = mean (no noise)
world_embeds = projection(latents)  # Only this is trained
combined = fuse(world_embeds, vision_embeds, text_embeds)
logits = gemma(combined)
loss = CrossEntropy(logits, labels)  # Standard LM loss
```

We use deterministic mode because:
- VAE is frozen (not being trained)
- No KL regularization needed (VAE already pre-trained)
- Want consistent embeddings for stable training
- Projection layer learns mapping from fixed latent space

---

## Decoder Architecture (That We Skip)

### What the Decoder Does

The **WanDecoder3d** mirrors the encoder architecture in reverse, reconstructing RGB video frames from 16-dimensional latent representations.

**Architecture (mirror of encoder):**
```
Latent: (B, 16, T, h, w)
    ↓
Post-quant conv: 16 → 16 channels (1×1 conv)
    ↓
Decoder (WanDecoder3d):
    ├─ conv_in: Conv3d(16 → 512)
    ├─ mid_block:
    │   ├─ WanResidualBlock (512)
    │   ├─ WanAttentionBlock (spatial self-attention)
    │   └─ WanResidualBlock (512)
    ├─ up_blocks[0]:
    │   ├─ WanResidualBlock × 3 (512 channels)
    │   └─ WanUpsample2d (512 → 512, spatial 2×)
    ├─ up_blocks[1]:
    │   ├─ WanResidualBlock × 3 (512 channels)
    │   └─ WanUpsample2d (512 → 256, spatial 2×)
    ├─ up_blocks[2]:
    │   ├─ WanResidualBlock × 3 (256 channels)
    │   └─ WanUpsample2d (256 → 128, spatial 2×)
    ├─ up_blocks[3]:
    │   └─ WanResidualBlock × 3 (128 channels)
    └─ conv_out: Conv3d(128 → 3 RGB)
    ↓
Output: (B, 3, T, H, W) - Reconstructed video frames
```

### When is the Decoder Used?

**In TheWorld: NEVER** ❌
- We only need latent embeddings, not reconstructed images
- Skipping decode saves ~50% of VAE computation time
- We project latents directly to Gemma space: 16 → 2304 dims

**In Cosmos Pipeline: YES** ✅
- During autoregressive future frame prediction
- When `output_type="video"` (default)
- We use `output_type="latent"` to skip this expensive step

**During VAE Training: YES** ✅
- Decoder is trained to reconstruct images from latents
- This forces encoder to capture all visual information in 16 dims
- Reconstruction loss: `MSE(decoded_image, original_image)`
- Combined with KL divergence loss for proper VAE training

### Why We Skip Decoding

```python
# Cosmos full pipeline (what we DON'T do)
encoded = vae.encode(image).latent_dist.mode()  # (B, 16, T, h, w)
decoded = vae.decode(encoded)                   # (B, 3, T, H, W) ← Expensive!

# TheWorld (what we DO)
latents = vae.encode(image).latent_dist.mode()  # (B, 16, T, h, w)
world_embeds = projection(latents)              # (B, h*w, 2304) ← Skip decode!
```

**Benefits of skipping decoder:**
1. ✅ ~50% faster VAE processing
2. ✅ Lower memory usage (no pixel-space tensors)
3. ✅ Direct access to semantic latent space
4. ✅ Cleaner integration (no pixel→latent round-trip)

---

## Training vs Inference Behavior

### Understanding the Two Training Contexts

There are **two completely different training contexts** to understand:

#### 1. VAE Pre-training (Done by NVIDIA)

**Goal:** Train the VAE to compress/reconstruct video frames

```python
# During VAE pre-training at NVIDIA:
latents = dist.sample()  # ← STOCHASTIC (need noise for KL loss)
reconstructed = decoder(latents)

# Two loss components:
reconstruction_loss = MSE(reconstructed, original)
kl_loss = KL(dist, N(0,1))  # Regularize latent space
total_loss = reconstruction_loss + β * kl_loss

# Both encoder and decoder weights are updated
```

**Why stochastic sampling?**
- KL loss needs distribution parameters (μ, σ)
- Reparameterization trick enables gradient flow
- Forces latent space to approximate N(0,1)

#### 2. TheWorld Training (What We Do)

**Goal:** Train projection layer to map Cosmos latents → Gemma embeddings

```python
# During TheWorld training (VAE frozen):
latents = dist.mode()  # ← DETERMINISTIC (VAE frozen)
world_embeds = projection(latents)  # Only this is trained
combined = fuse(world_embeds, vision_embeds, text_embeds)
logits = gemma(combined)
loss = CrossEntropy(logits, labels)  # Standard LM loss

# Only projection.weight and projection.bias are updated
# VAE encoder weights remain frozen
```

**Why deterministic mode?**
- ✅ VAE is frozen (not training encoder/decoder)
- ✅ No KL loss needed (VAE already trained)
- ✅ Want consistent embeddings for stable projection training
- ✅ Projection learns from fixed latent space

### Summary Table

| Component | Training Mode | VAE Trainable? | Uses `.sample()`? | Uses `.mode()`? |
|-----------|---------------|----------------|-------------------|-----------------|
| **VAE Pre-training** | Train encoder+decoder | ✅ Yes | ✅ Yes | ❌ No |
| **VAE Inference** | Frozen VAE | ❌ No | ❌ No | ✅ Yes |
| **TheWorld Training** | Train projection only | ❌ No | ❌ No | ✅ Yes |
| **TheWorld Inference** | Everything frozen | ❌ No | ❌ No | ✅ Yes |

**Key insight:** TheWorld isn't training a VAE—it's training a projection layer on top of a frozen, pre-trained VAE. This is why we use deterministic `.mode()` instead of stochastic `.sample()`.

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

## Gradient Flow Analysis

### Are `.mode()` and `.sample()` Differentiable?

**Yes, both methods are fully differentiable!** This is a common point of confusion.

```python
# Both methods return tensors with gradients:
latents_mode = latent_dist.mode()      # Returns self.mean (has grad_fn)
latents_sample = latent_dist.sample()  # Returns self.mean + self.std * noise (has grad_fn)
```

### How Gradients Flow Through `.mode()`

```python
# Forward pass with .mode()
latents = latent_dist.mode()  # latents = mean tensor
loss = some_loss(latents)

# Backward pass
loss.backward()
# ✅ Gradients flow: loss → latents (mean) → encoder parameters
```

**Why this works:**
- `.mode()` simply returns `self.mean`, which is a tensor computed from encoder output
- `self.mean` has a `grad_fn` that traces back through the encoder
- Gradients flow normally through the computational graph

### How Gradients Flow Through `.sample()`

```python
# Forward pass with .sample()
latents = latent_dist.sample()  # latents = mean + std * noise
loss = some_loss(latents)

# Backward pass
loss.backward()
# ✅ Gradients flow: loss → mean, std → encoder parameters
# ❌ No gradients to noise (random, not connected to encoder)
```

**Reparameterization trick:**
- Noise has no gradient (it's sampled randomly)
- Gradients flow through `mean` and `std` parameters
- This is how VAEs are trained despite stochastic sampling

### Gradient Flow in TheWorld

**With frozen VAE (default):**

```python
# Forward pass
with torch.no_grad():
    latents = cosmos_vae.encode(image).latent_dist.mode()  # Frozen
# OR: latents requires_grad=True but cosmos_vae params frozen

world_embeds = projection(latents)  # Trainable
combined = fuse(world_embeds, vision_embeds, text_embeds)
logits = gemma(combined)
loss = CrossEntropy(logits, labels)

# Backward pass
loss.backward()
# ✅ Gradients flow to projection.weight, projection.bias
# ❌ Gradients stop at latents (encoder frozen)
```

**With unfrozen VAE (if `freeze_cosmos_vae=False`):**

```python
# Forward pass (no torch.no_grad)
latents = cosmos_vae.encode(image).latent_dist.mode()  # Trainable
world_embeds = projection(latents)
combined = fuse(world_embeds, vision_embeds, text_embeds)
logits = gemma(combined)
loss = CrossEntropy(logits, labels)

# Backward pass
loss.backward()
# ✅ Gradients flow to projection parameters
# ✅ Gradients flow through .mode() to encoder parameters
# ✅ Both projection and VAE encoder are updated
```

### Key Points

1. **`.mode()` is differentiable** - Don't confuse "deterministic" with "non-differentiable"
2. **Frozen ≠ No gradient flow** - Gradients can flow through frozen layers, they just don't update
3. **Both methods work for training** - `.mode()` for deterministic, `.sample()` for stochastic
4. **TheWorld uses frozen VAE** - So encoder params don't update, but gradients still flow to projection

### Comparison Table

| Method | Deterministic? | Differentiable? | Gradient Flow | When to Use |
|--------|----------------|-----------------|---------------|-------------|
| **`.mode()`** | ✅ Yes | ✅ Yes | loss → mean → encoder | Frozen VAE inference |
| **`.sample()`** | ❌ No (adds noise) | ✅ Yes | loss → mean, std → encoder | VAE training |
| **Frozen VAE** | N/A | ✅ Yes | Gradients flow but params don't update | TheWorld default |
| **Unfrozen VAE** | N/A | ✅ Yes | Gradients flow AND params update | Advanced training |

**Bottom line:** `.mode()` is both deterministic AND differentiable—perfect for our use case where we want consistent embeddings but still need gradient flow to the projection layer.

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

**Architecture:**
- **3D Causal VAE** - WanEncoder3d/WanDecoder3d for video compression
- **16-dim latent space** - Semantic representation learned via reconstruction
- **8× spatial compression** - 224×224 → 28×28 (784 tokens)
- **Causal convolutions** - Designed for autoregressive video (but we use single frames)
- **RMS normalization** - Faster and more stable than LayerNorm

**Latent Extraction:**
- **`.mode()` is correct** ⭐ - Deterministic, differentiable, public API
- **`.sample()` for VAE training** - Stochastic, needed for KL regularization
- **Never use `.mean` attribute** - Not the public API

**Training Context:**
- **VAE pre-trained by NVIDIA** - We don't train it, just use it
- **TheWorld trains projection** - 16 → 2304 dims, only ~4.5M params
- **Frozen VAE is default** - Deterministic embeddings, stable training
- **Can unfreeze if needed** - For domain-specific visual features

**Integration:**
- **Skip decoder** - ~50% faster, we only need latent embeddings
- **Skip diffusion** - 10-35× faster, no text-to-video generation needed
- **Single frame mode** - T=1, no temporal prediction (yet)
- **Direct projection** - Clean path from Cosmos latents to Gemma embeddings

**Gradient Flow:**
- **`.mode()` is differentiable** - Gradients flow through mean tensor
- **Frozen ≠ non-differentiable** - Gradients flow but params don't update
- **Projection always trainable** - Learns to map Cosmos → Gemma space

This architecture provides high-quality visual encodings that complement Gemma's language understanding, creating a true vision-language-world model that can reason about both static visual understanding and temporal dynamics.
