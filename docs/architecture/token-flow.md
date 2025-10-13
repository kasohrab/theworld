# TheWorld Architecture: Complete Tensor Shape Reference

**Purpose:** This document traces the exact tensor shapes through every step of TheWorld's architecture, from input images to output logits.

**Use this document to:**
- Understand exact dimensions at each processing stage
- Debug tensor shape mismatches
- Calculate memory requirements
- Verify implementation correctness

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Input Processing](#input-processing)
3. [Gemma Vision Path (SigLIP)](#gemma-vision-path-siglip)
4. [Cosmos World Path (VAE Encoder)](#cosmos-world-path-vae-encoder)
5. [Embedding Fusion](#embedding-fusion)
6. [Language Model Processing](#language-model-processing)
7. [Complete Examples](#complete-examples)
8. [Shape Verification & Debugging](#shape-verification--debugging)

---

## Architecture Overview

TheWorld processes images through two parallel encoders, then fuses their embeddings:

```
Input: List[PIL.Image] (B images)
    │
    ├─────────────────────────┬─────────────────────────┐
    │                         │                         │
    ▼                         ▼                         ▼
Gemma Processor          Cosmos VAE           Token Injection
(SigLIP Vision)          Encoder              (SOW/EOW)
    │                         │                         │
    ▼                         ▼                         ▼
(B, ~256, 2304)        (B, 784, 2304)           (B, seq_len+2, ...)
vision tokens          world tokens             add SOW/EOW
    │                         │                         │
    └─────────────────────────┴─────────────────────────┘
                              │
                              ▼
                    Embedding Fusion
                (insert world between SOW/EOW)
                              │
                              ▼
                    Combined Embeddings
                (B, combined_len, 2304)
                              │
                              ▼
                  Gemma Language Model
                  (26 transformer layers)
                              │
                              ▼
                    Output Logits
                (B, combined_len, vocab_size)
```

**Key dimensions:**
- `B` = batch size
- `2304` = Gemma embedding dimension (hidden_size)
- `16` = Cosmos latent dimension (z_dim)
- `vocab_size` = 262272 (Gemma 3 vocabulary)

---

## Input Processing

### Step 1: Raw Images

**Input:** `List[PIL.Image]` - B PIL images

**Shape:** List of length B, each element is a PIL Image of arbitrary size

### Step 2: Gemma Processor (Chat Template)

**Code:** `python/theworld/modeling/theworld.py:693-745`

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": "What is in this image?"}
        ]
    }
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)
```

**Output shapes:**
- `input_ids`: **(B, seq_len)**
  - Contains: `[BOS, text_tokens..., IMAGE_SOFT_TOKEN, text_tokens..., EOS]`
  - `seq_len` = number of text tokens + 1 image placeholder + 2 (BOS/EOS)
  - Example: "What is this?" → seq_len ≈ 20 tokens

- `pixel_values`: **(B, 3, H_img, W_img)**
  - Preprocessed for SigLIP vision encoder
  - Typical: (B, 3, 224, 224) or (B, 3, 896, 896)
  - Normalized to ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

- `attention_mask`: **(B, seq_len)**
  - Binary mask: 1 for real tokens, 0 for padding
  - All 1s if no padding

### Step 3: Inject World Tokens (SOW/EOW)

**Code:** `python/theworld/modeling/theworld.py:693-744`

The forward pass automatically injects `<start_of_world>` and `<end_of_world>` tokens:

```python
# Before: [BOS, text..., IMAGE, text..., EOS]
# After:  [BOS, SOW, EOW, text..., IMAGE, text..., EOS]
```

**Output shapes:**
- `input_ids`: **(B, seq_len + 2)**
  - Added 2 tokens: SOW at position 1, EOW at position 2

- `attention_mask`: **(B, seq_len + 2)**
  - Extended with 1s for SOW/EOW tokens

**Example sequence:**
```
Position | Token              | ID
---------|--------------------|---------
0        | <bos>              | 2
1        | <start_of_world>   | 262145
2        | <end_of_world>     | 262146
3-18     | "What is this?"    | various
19       | <image_soft_token> | 262144
20       | <eos>              | 1
```

---

## Gemma Vision Path (SigLIP)

**Goal:** Extract static visual features using SigLIP vision encoder

**Code:** `python/theworld/modeling/theworld.py:874-881`

### Step 1: Text Token Embeddings

```python
inputs_embeds = self.model.language_model.embed_tokens(input_ids)
```

**Shape transformation:**
- Input: `input_ids` **(B, seq_len)**
- Output: `inputs_embeds` **(B, seq_len, 2304)**

At this point, `IMAGE_SOFT_TOKEN` (ID 262144) is just a placeholder embedding.

### Step 2: SigLIP Vision Encoding

```python
image_features = self.model.get_image_features(pixel_values)
```

**Shape transformation:**
- Input: `pixel_values` **(B, 3, H_img, W_img)**
- Output: `image_features` **(B, num_vision_tokens, 2304)**

**Vision token count calculation:**

| Input Resolution | Patch Size | Grid Size | num_vision_tokens | Use Case |
|------------------|------------|-----------|-------------------|----------|
| 224×224 | 14×14 | 16×16 | **256** | Fast (default) |
| 448×448 | 14×14 | 32×32 | **1024** | Balanced |
| 896×896 | 14×14 | 64×64 | **4096** | High detail |

**Formula:**
```
num_vision_tokens = (H_img ÷ patch_size) × (W_img ÷ patch_size)
```

For 224×224 with 14×14 patches: `(224÷14) × (224÷14) = 16 × 16 = 256`

### Step 3: Replace Image Placeholders

```python
special_image_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds, image_features)
inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
```

**Shape transformation:**
- Input: `inputs_embeds` **(B, seq_len, 2304)** - with placeholder
- Mask: `special_image_mask` **(B, seq_len, 2304)** - True where IMAGE_SOFT_TOKEN
- Features: `image_features` **(B, 256, 2304)** - actual vision features
- Output: `inputs_embeds` **(B, seq_len, 2304)** - placeholder replaced with real features

**Result:** `inputs_embeds` now contains text embeddings + real SigLIP vision features

---

## Cosmos World Path (VAE Encoder)

**Goal:** Extract world model latents using Cosmos VAE

**Code:** `python/theworld/modeling/cosmos_encoder.py:53-138`

### Step 1: Preprocess Images

```python
# Convert PIL to tensor
target_size = (512, 512)  # Standard size for Cosmos
img = img.resize(target_size)
tensor_img = TF.to_tensor(img) * 255.0  # [0,1] → [0,255]
tensor_batch = torch.stack(tensor_images, dim=0)
```

**Shape transformation:**
- Input: `List[PIL.Image]` - B images (arbitrary sizes)
- Resize each to: (512, 512)
- Convert to tensor: (3, 512, 512) per image
- Stack: **(B, 3, 512, 512)**
- dtype: bfloat16

### Step 2: Add Temporal Dimension

```python
cosmos_input_5d = tensor_batch.unsqueeze(2)  # Add time dimension
```

**Shape transformation:**
- Input: **(B, 3, 512, 512)**
- Output: **(B, 3, 1, 512, 512)**
  - Added dimension at position 2 for temporal frames (T=1 for single frame)

### Step 3: VAE Encode

```python
encoder_output = self.cosmos_vae.encode(cosmos_input_5d)
latent_dist = encoder_output.latent_dist
latents = latent_dist.mode()  # Deterministic: use mode, not sample
```

**Shape transformations:**

1. **VAE encoder output:** **(B, 32, 1, H_latent, W_latent)**
   - 32 channels = 16 mean + 16 logvar
   - Spatial downsampling: 8× (512÷8 = 64)
   - H_latent = W_latent = 64 (for 512×512 input)

2. **Latent distribution split:**
   - `mean`: **(B, 16, 1, 64, 64)**
   - `logvar`: **(B, 16, 1, 64, 64)**

3. **Extract mode (mean):** **(B, 16, 1, 64, 64)**
   - Mode of Gaussian distribution = mean
   - Deterministic (same input → same output)

**Spatial dimension calculation:**

| Input Size | Downsampling Factor | Latent Size | Calculation |
|------------|---------------------|-------------|-------------|
| 512×512 | 8× | 64×64 | 512÷8 = 64 |
| 224×224 | 8× | 28×28 | 224÷8 = 28 |
| 448×448 | 8× | 56×56 | 448÷8 = 56 |

**Formula:**
```
H_latent = H_input ÷ 8
W_latent = W_input ÷ 8
```

### Step 4: Reshape to Token Sequence

```python
# Remove time dimension and permute
latents = latents.squeeze(2).permute(0, 2, 3, 1)
# Reshape to 2D
num_tokens = h * w
reshaped_latents = latents.reshape(b, num_tokens, c)
```

**Shape transformations:**

1. **Remove time dimension:**
   - Input: **(B, 16, 1, 64, 64)**
   - `squeeze(2)`: **(B, 16, 64, 64)**

2. **Permute to channels-last:**
   - Input: **(B, 16, 64, 64)**
   - `permute(0, 2, 3, 1)`: **(B, 64, 64, 16)**

3. **Flatten spatial dimensions:**
   - Input: **(B, 64, 64, 16)**
   - `reshape(B, 64*64, 16)`: **(B, 4096, 16)**
   - Each spatial location becomes one token

**Token count calculation:**

| Input Resolution | Latent Grid | num_world_tokens | Calculation |
|------------------|-------------|------------------|-------------|
| 512×512 | 64×64 | **4096** | 64 × 64 |
| 224×224 | 28×28 | **784** | 28 × 28 |
| 448×448 | 56×56 | **3136** | 56 × 56 |

### Step 5: Project to Gemma Dimension

```python
projected_embeds = self.world_projection(reshaped_latents)
```

**Shape transformation:**
- Input: **(B, num_world_tokens, 16)**
- Projection: Linear(16, 2304)
- Output: **(B, num_world_tokens, 2304)**

**For 224×224 input:** (B, 784, 16) → (B, 784, 2304)
**For 512×512 input:** (B, 4096, 16) → (B, 4096, 2304)

**Final world embeddings shape:** **(B, num_world_tokens, 2304)**

---

## Embedding Fusion

**Goal:** Insert world tokens between SOW and EOW brackets

**Code:** `python/theworld/modeling/fusion.py:36-111`

### Input Shapes

- `gemma_embeds`: **(B, seq_len, 2304)**
  - Contains: text embeddings + vision features + SOW/EOW tokens

- `world_embeds`: **(B, num_world_tokens, 2304)**
  - From Cosmos encoder projection

- `input_ids`: **(B, seq_len)**
  - Used to locate SOW and EOW positions

### Step 1: Find Bracket Positions

```python
start_positions = (input_ids == self.sow_token_id).nonzero(as_tuple=True)[1]
end_positions = (input_ids == self.eow_token_id).nonzero(as_tuple=True)[1]
start_pos = start_positions[0].item()  # e.g., 1
end_pos = end_positions[0].item()      # e.g., 2
```

**Typical positions:**
- `start_pos` = 1 (SOW after BOS)
- `end_pos` = 2 (EOW after SOW)

### Step 2: Slice Embeddings

```python
embeddings_before = gemma_embeds[:, :start_pos + 1, :]    # Up to and including SOW
embeddings_after = gemma_embeds[:, end_pos:, :]           # From EOW onwards
```

**Shape breakdown:**

| Slice | Range | Shape | Contains |
|-------|-------|-------|----------|
| before | `[:start_pos+1]` | **(B, start_pos+1, 2304)** | BOS + SOW |
| world | - | **(B, num_world_tokens, 2304)** | World embeddings |
| after | `[end_pos:]` | **(B, seq_len-end_pos, 2304)** | EOW + rest |

**Example with seq_len=21, start_pos=1, end_pos=2:**
- `embeddings_before`: (B, 2, 2304) - positions [0:2] = BOS, SOW
- `world_embeds`: (B, 784, 2304) - world tokens
- `embeddings_after`: (B, 19, 2304) - positions [2:21] = EOW, text, IMAGE, text, EOS

### Step 3: Concatenate

```python
combined_embeds = torch.cat([embeddings_before, world_embeds, embeddings_after], dim=1)
```

**Shape transformation:**
- Input slices: (B, 2, 2304) + (B, 784, 2304) + (B, 19, 2304)
- Output: **(B, 805, 2304)**
  - combined_len = 2 + 784 + 19 = 805

**Combined length formula:**
```
combined_len = (start_pos + 1) + num_world_tokens + (seq_len - end_pos)
            = (1 + 1) + 784 + (21 - 2)
            = 2 + 784 + 19
            = 805
```

### Step 4: Update Attention Mask

```python
attention_mask_before = attention_mask[:, :start_pos + 1]
attention_mask_after = attention_mask[:, end_pos:]
world_attention_mask = torch.ones((batch_size, num_world_tokens), dtype=torch.long, device=device)
combined_attention_mask = torch.cat([attention_mask_before, world_attention_mask, attention_mask_after], dim=1)
```

**Shape transformation:**
- Input: (B, 2) + (B, 784) + (B, 19)
- Output: **(B, 805)**

### Final Token Sequence

```
Position  | Token Type           | Count | Source
----------|----------------------|-------|---------------------------
0         | BOS                  | 1     | Original input_ids
1         | <start_of_world>     | 1     | Original input_ids
2-785     | World embeddings     | 784   | Cosmos VAE (inserted!)
786       | <end_of_world>       | 1     | Original input_ids
787-1042  | SigLIP vision tokens | 256   | Gemma vision encoder
1043-1047 | Text tokens          | 5     | "What is this?"
1048      | EOS                  | 1     | Original input_ids
```

**Total length:** 1 + 1 + 784 + 1 + 256 + 5 + 1 = **1049 tokens**

---

## Language Model Processing

**Code:** `python/theworld/modeling/theworld.py:916-932`

### Input to Language Model

```python
outputs = super().forward(
    input_ids=None,  # Using inputs_embeds instead
    pixel_values=None,  # Already processed
    attention_mask=fusion_output.combined_attention_mask,
    inputs_embeds=fusion_output.combined_embeds,
    labels=combined_labels,
    ...
)
```

**Input shapes:**
- `inputs_embeds`: **(B, combined_len, 2304)**
  - e.g., (B, 805, 2304) for 224×224 input
- `attention_mask`: **(B, combined_len)**

### Transformer Processing

The language model processes all tokens through 26 transformer layers:

```python
# Pseudo-code for transformer processing
hidden_states = inputs_embeds  # (B, combined_len, 2304)

for layer in transformer_layers:  # 26 layers
    hidden_states = layer(hidden_states, attention_mask)
    # Self-attention allows all tokens to attend to each other
    # World tokens ↔ Vision tokens ↔ Text tokens
```

**Shape through transformer:**
- Input: **(B, combined_len, 2304)**
- Each layer: **(B, combined_len, 2304)** → **(B, combined_len, 2304)**
- Output: **(B, combined_len, 2304)**

### LM Head (Output Projection)

```python
logits = lm_head(hidden_states)
```

**Shape transformation:**
- Input: **(B, combined_len, 2304)**
- LM head: Linear(2304, vocab_size)
- Output: **(B, combined_len, vocab_size)**

**For Gemma 3:** vocab_size = 262272
**Final output:** **(B, combined_len, 262272)**

### Loss Computation

During training, only text tokens contribute to the loss:

```python
# World and vision tokens masked with -100
combined_labels = [
    ...,                    # Text tokens (actual token IDs)
    -100, -100, ..., -100,  # World tokens (784 × -100)
    -100, -100, ..., -100,  # Vision tokens (256 × -100)
    ...,                    # Text tokens
]
```

**Labels shape:** **(B, combined_len)**
- Compute cross-entropy loss only where labels ≠ -100

---

## Complete Examples

### Example 1: Small Image (224×224)

**Input:**
- Image resolution: 224×224
- Text prompt: "What is in this image?" (≈18 tokens)

**Shape trace:**

| Step | Operation | Output Shape | Notes |
|------|-----------|--------------|-------|
| 1 | PIL Image | - | 224×224 RGB |
| 2 | Gemma processor (pixel_values) | (1, 3, 224, 224) | For SigLIP |
| 3 | Gemma processor (input_ids) | (1, 21) | BOS + text + IMG + text + EOS |
| 4 | Inject SOW/EOW | (1, 23) | +2 tokens |
| 5 | embed_tokens | (1, 23, 2304) | Text embeddings |
| 6 | SigLIP encode | (1, 256, 2304) | 16×16 vision tokens |
| 7 | masked_scatter | (1, 23, 2304) | Vision features inserted |
| 8 | Resize for Cosmos | (1, 3, 224, 224) | Input to VAE |
| 9 | Add time dim | (1, 3, 1, 224, 224) | T=1 |
| 10 | VAE encode | (1, 16, 1, 28, 28) | 224÷8=28 |
| 11 | Reshape | (1, 784, 16) | 28×28=784 tokens |
| 12 | Project | (1, 784, 2304) | World embeddings |
| 13 | Fusion | (1, 1061, 2304) | 2+784+275 |
| 14 | Transformer | (1, 1061, 2304) | 26 layers |
| 15 | LM head | (1, 1061, 262272) | Output logits |

**Combined length calculation:**
```
combined_len = (BOS + SOW) + world_tokens + (EOW + rest)
             = 2 + 784 + 275
             = 1061
```

**Token breakdown:**
- World tokens: 784 (74%)
- Vision tokens: 256 (24%)
- Text tokens: ~21 (2%)

### Example 2: Large Image (512×512)

**Input:**
- Image resolution: 512×512
- Text prompt: "Describe this scene." (≈16 tokens)

**Shape trace:**

| Step | Operation | Output Shape | Notes |
|------|-----------|--------------|-------|
| 1 | PIL Image | - | 512×512 RGB |
| 2 | Gemma processor (pixel_values) | (1, 3, 224, 224) | **Resized to 224!** |
| 3 | Gemma processor (input_ids) | (1, 19) | BOS + text + IMG + text + EOS |
| 4 | Inject SOW/EOW | (1, 21) | +2 tokens |
| 5 | embed_tokens | (1, 21, 2304) | Text embeddings |
| 6 | SigLIP encode | (1, 256, 2304) | Still 256 (from 224×224) |
| 7 | masked_scatter | (1, 21, 2304) | Vision features inserted |
| 8 | Resize for Cosmos | (1, 3, 512, 512) | **Full resolution!** |
| 9 | Add time dim | (1, 3, 1, 512, 512) | T=1 |
| 10 | VAE encode | (1, 16, 1, 64, 64) | 512÷8=64 |
| 11 | Reshape | (1, 4096, 16) | 64×64=4096 tokens |
| 12 | Project | (1, 4096, 2304) | World embeddings |
| 13 | Fusion | (1, 4373, 2304) | 2+4096+275 |
| 14 | Transformer | (1, 4373, 2304) | 26 layers |
| 15 | LM head | (1, 4373, 262272) | Output logits |

**Key observation:** Cosmos uses full 512×512 resolution (4096 tokens) while Gemma resizes to 224×224 (256 tokens)

**Combined length calculation:**
```
combined_len = 2 + 4096 + 275 = 4373
```

**Token breakdown:**
- World tokens: 4096 (94%)
- Vision tokens: 256 (6%)
- Text tokens: ~21 (<1%)

---

## Shape Verification & Debugging

### Common Assertions in Code

From `cosmos_encoder.py:114-136`:

```python
# After VAE encode
b, c, t, h, w = latents.shape
assert b == batch_size, f"Batch size mismatch: expected {batch_size}, got {b}"
assert c == self.cosmos_dim, f"Latent dim mismatch: expected {self.cosmos_dim}, got {c}"
assert t == 1, f"Time dimension should be 1, got {t}"

# After projection
assert projected_embeds.dim() == 3, f"Expected 3D tensor, got {projected_embeds.dim()}D"
assert projected_embeds.size(0) == batch_size
assert projected_embeds.size(1) == num_tokens
assert projected_embeds.size(2) == self.world_projection.out_features  # 2304
```

From `fusion.py:55-106`:

```python
assert gemma_embeds.dim() == 3
assert world_embeds.dim() == 3
assert input_ids.dim() == 2
assert attention_mask.dim() == 2
assert world_embeds.size(0) == batch_size
assert world_embeds.size(2) == embed_dim
assert start_pos < end_pos

# After concatenation
expected_len = (start_pos + 1) + num_world_tokens + (seq_len - end_pos)
assert combined_embeds.size(1) == expected_len
```

### Debugging Shape Mismatches

**Problem:** "Expected tensor of shape (B, N, 2304) but got (B, N, 16)"

**Solution:** Missing projection layer
```python
# ❌ Wrong: forgot to project
world_embeds = cosmos_encoder.forward(images)  # Returns (B, N, 16)

# ✅ Correct: projection happens inside CosmosEncoder.forward()
world_embeds = cosmos_encoder(images)  # Returns (B, N, 2304)
```

**Problem:** "RuntimeError: The size of tensor a (805) must match the size of tensor b (23)"

**Solution:** Labels not aligned with combined sequence
```python
# ❌ Wrong: using original input_ids as labels
labels = input_ids  # Shape: (B, 23)

# ✅ Correct: expand labels to match combined length
# Build: [tokens_before | -100 for world | tokens_after]
combined_labels = build_aligned_labels(input_ids, num_world_tokens)  # (B, 805)
```

**Problem:** "IndexError: index out of range"

**Solution:** SOW/EOW tokens not present in input_ids
```python
# Check if world tokens are present
assert (input_ids == sow_token_id).any(), "No <start_of_world> token found"
assert (input_ids == eow_token_id).any(), "No <end_of_world> token found"
```

### Memory Requirements

**Model parameters:**
- Gemma 3 4B: ~4.3B params × 2 bytes (bf16) = **8.6 GB**
- Cosmos 2B: ~2.0B params × 2 bytes (bf16) = **4.0 GB**
- Projection: 16×2304 × 2 bytes = **0.07 MB** (negligible)
- **Total:** ~12.6 GB for model weights

**Activation memory (forward pass, batch_size=1):**

For 224×224 image:
- Combined embeddings: 1 × 1061 × 2304 × 2 bytes = **4.9 MB**
- Transformer activations: ~26 layers × 4.9 MB = **127 MB** (cached)
- Output logits: 1 × 1061 × 262272 × 2 bytes = **556 MB**
- **Total forward pass:** ~690 MB

For 512×512 image:
- Combined embeddings: 1 × 4373 × 2304 × 2 bytes = **20 MB**
- Transformer activations: ~26 layers × 20 MB = **520 MB**
- Output logits: 1 × 4373 × 262272 × 2 bytes = **2.3 GB**
- **Total forward pass:** ~2.8 GB

**Training memory (backward pass):** ~3-4× forward pass memory

---

## Summary Table

### Shape Quick Reference

| Component | Input Shape | Output Shape | Parameters |
|-----------|-------------|--------------|------------|
| **Gemma Processor** | List[PIL] | (B, 3, 224, 224) | - |
| **embed_tokens** | (B, seq_len) | (B, seq_len, 2304) | Gemma embedding table |
| **SigLIP** | (B, 3, H, W) | (B, num_vis, 2304) | Gemma vision encoder |
| **Cosmos VAE** | (B, 3, 1, 512, 512) | (B, 16, 1, 64, 64) | Cosmos encoder (~2B) |
| **Projection** | (B, N, 16) | (B, N, 2304) | Linear(16, 2304) ~37K |
| **Fusion** | (B, seq_len, 2304) + (B, N, 2304) | (B, combined, 2304) | None (concat) |
| **Transformer** | (B, combined, 2304) | (B, combined, 2304) | Gemma LM (~4B) |
| **LM Head** | (B, combined, 2304) | (B, combined, 262272) | Gemma lm_head |

### Token Count by Resolution

| Input Res | SigLIP Tokens | Cosmos Tokens | Text Tokens | Total (approx) |
|-----------|---------------|---------------|-------------|----------------|
| 224×224 | 256 | 784 | ~20 | **~1,060** |
| 448×448 | 1,024 | 3,136 | ~20 | **~4,180** |
| 512×512 | 256 | 4,096 | ~20 | **~4,370** |
| 896×896 | 4,096 | 10,816 | ~20 | **~14,930** |

**Note:** Gemma processor may resize images, so SigLIP tokens depend on processor config, not input size.

---

## References

- Implementation: `python/theworld/modeling/theworld.py`
- Cosmos encoder: `python/theworld/modeling/cosmos_encoder.py`
- Fusion module: `python/theworld/modeling/fusion.py`
- Architecture overview: `docs/architecture.md`
- Cosmos details: `docs/cosmos_architecture_explained.md`
