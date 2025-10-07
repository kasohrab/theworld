# World Model Embedding Integration Options

## Background

We need to integrate Cosmos world model embeddings with Gemma 3 vision-language model. The challenge is that Gemma 3 has a built-in vision encoder (SigLIP) that processes images, and we want to add temporal world dynamics from Cosmos.

**Current Problem:**
- Using only `get_input_embeddings()(input_ids)` gives placeholder embeddings, NOT real vision features
- The SigLIP vision encoder never runs
- Experimental proof shows placeholder embeddings are identical `[0.0077, 0.0046, ...]` while real vision embeddings are unique

**World Model Output:**
- Cosmos produces ~784-3920 tokens depending on number of frames
- For single frame: 28×28 = 784 tokens
- For 5 frames: 5×28×28 = 3920 tokens

---

## Option 1: Let Gemma Handle Everything, Then Prepend World Tokens

### Approach
1. Call `self.gemma.model()` with `input_ids` + `pixel_values` to get properly processed embeddings
2. Extract first layer embeddings (which include real vision features from SigLIP)
3. Concatenate world tokens to the left: `[world | gemma_multimodal_embeds]`
4. Forward through language model only

### Implementation
```python
# 1. Let Gemma do its thing normally (with vision encoder)
with torch.no_grad():
    gemma_inputs = processor.apply_chat_template(...)

    # Get properly processed embeddings (with real vision features)
    gemma_outputs = self.gemma.model(
        input_ids=gemma_inputs["input_ids"],
        pixel_values=gemma_inputs["pixel_values"],
        attention_mask=gemma_inputs["attention_mask"],
        token_type_ids=gemma_inputs.get("token_type_ids"),
        output_hidden_states=True,
        return_dict=True,
    )
    gemma_multimodal_embeds = gemma_outputs.hidden_states[0]  # First layer

# 2. Concatenate with world model
combined_embeds = torch.cat([projected_world_embeds, gemma_multimodal_embeds], dim=1)

# 3. Forward through language model only
outputs = self.gemma.language_model(
    inputs_embeds=combined_embeds,
    attention_mask=combined_attention_mask,
    ...
)
```

### Pros
- ✅ Gemma handles vision encoding completely its own way
- ✅ Non-intrusive - just grab embeddings after processing
- ✅ Guaranteed to use SigLIP vision encoder correctly

### Cons
- ❌ Two model calls (once for embeddings, once through language model)
- ❌ May be slightly slower
- ❌ Have to understand Gemma's internal structure (`model` vs `language_model`)

---

## Option 2: Manual Vision Encoding with masked_scatter

### Approach
Replicate what Gemma does internally:
1. Get text embeddings with placeholders: `get_input_embeddings()(input_ids)`
2. Run vision encoder manually: `self.gemma.model.get_image_features(pixel_values)`
3. Use `masked_scatter()` to replace placeholder embeddings with real vision features
4. Concatenate world tokens
5. Forward through model

### Implementation
```python
# 1. Get initial embeddings with placeholders
inputs_embeds = self.gemma.get_input_embeddings()(gemma_inputs["input_ids"])

# 2. Process actual image pixels through SigLIP vision encoder
if "pixel_values" in gemma_inputs:
    pixel_values = gemma_inputs["pixel_values"]
    image_features = self.gemma.model.get_image_features(pixel_values)
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

    # 3. Replace placeholder embeddings with real vision features
    special_image_mask = self.gemma.model.get_placeholder_mask(
        gemma_inputs["input_ids"],
        inputs_embeds=inputs_embeds,
        image_features=image_features
    )
    gemma_multimodal_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
else:
    gemma_multimodal_embeds = inputs_embeds

# 4. Concatenate world tokens
combined_embeds = torch.cat([projected_world_embeds, gemma_multimodal_embeds], dim=1)

# 5. Forward through full model
outputs = self.gemma(
    inputs_embeds=combined_embeds,
    attention_mask=combined_attention_mask,
    ...
)
```

### Pros
- ✅ More control over the process
- ✅ Single forward pass through full model
- ✅ Cleaner conceptually

### Cons
- ❌ More intrusive - reimplements Gemma's internal logic
- ❌ May break if Gemma's internals change
- ❌ Need to understand `get_image_features()` and `get_placeholder_mask()` APIs

---

## Option 3: Just Pass pixel_values Through

### Approach
Don't use `inputs_embeds` at all - let Gemma handle `input_ids` + `pixel_values` normally.

### Implementation
```python
# Just call Gemma normally
outputs = self.gemma(
    input_ids=gemma_inputs["input_ids"],
    pixel_values=gemma_inputs["pixel_values"],
    attention_mask=gemma_inputs["attention_mask"],
    ...
)
```

### Pros
- ✅ Least intrusive
- ✅ Gemma handles everything

### Cons
- ❌ **Cannot concatenate world tokens this way!**
- ❌ No way to insert world embeddings in the middle
- ❌ Doesn't solve our problem

**Verdict:** Not viable for our use case.

---

## Option 4: Special Token `<the_world>` (RECOMMENDED)

### Approach
Add a new special token `<the_world>` to represent world model embeddings. This is architecturally clean and follows how models handle special tokens like `<image>`.

### Implementation

#### Setup (in `__init__`):
```python
def __init__(self, ...):
    # ... existing code ...

    # Add special token for world model embeddings
    special_tokens = {"additional_special_tokens": ["<the_world>"]}
    self.processor.tokenizer.add_special_tokens(special_tokens)
    self.gemma.resize_token_embeddings(len(self.processor.tokenizer))

    # Get the token ID for later use
    self.world_token_id = self.processor.tokenizer.convert_tokens_to_ids("<the_world>")
```

#### Forward pass:
```python
def forward(self, input_pixels, text, ...):
    # Calculate how many <the_world> tokens we need
    # For 28×28 world features, we need 784 tokens
    num_world_tokens = projected_world_embeds.size(1)
    world_tokens_str = "<the_world> " * num_world_tokens

    # Insert world tokens before image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": world_tokens_str},  # World tokens
                {"type": "image", "image": pil_image},      # Image
                {"type": "text", "text": text_prompt},      # Text
            ],
        }
    ]

    gemma_inputs = self.processor.apply_chat_template(...)

    # Let Gemma process everything (vision encoder runs automatically)
    gemma_outputs = self.gemma.model(
        input_ids=gemma_inputs["input_ids"],
        pixel_values=gemma_inputs["pixel_values"],
        attention_mask=gemma_inputs["attention_mask"],
        token_type_ids=gemma_inputs.get("token_type_ids"),
        output_hidden_states=True,
        return_dict=True,
    )

    # Get embeddings from first layer
    embeddings = gemma_outputs.hidden_states[0]

    # Find where <the_world> tokens are and replace with world features
    world_token_mask = (gemma_inputs["input_ids"] == self.world_token_id).unsqueeze(-1)
    world_token_mask = world_token_mask.expand_as(embeddings)
    embeddings = embeddings.masked_scatter(world_token_mask, projected_world_embeds)

    # Forward through language model
    outputs = self.gemma.language_model(
        inputs_embeds=embeddings,
        attention_mask=gemma_inputs["attention_mask"],
        ...
    )

    return outputs
```

### Token Count Strategies

#### Strategy A: Multiple Repeated Tokens (Recommended)
- Insert 784 copies of `<the_world>` token: `"<the_world> <the_world> <the_world> ..."`
- Each gets replaced with one world feature vector

**Pros:**
- ✅ Simple to implement
- ✅ Flexible - works for any number of world tokens
- ✅ Model sees them as separate positions

**Cons:**
- ❌ Long sequence in chat template

#### Strategy B: Single Token with Pooling
- Use one `<the_world>` token
- Pool 784 world features into a single vector (e.g., mean pooling)

**Pros:**
- ✅ Clean in chat template
- ✅ Short sequence

**Cons:**
- ❌ Loses spatial information from world model
- ❌ Information bottleneck

#### Strategy C: Indexed Special Tokens
- Add 784 tokens to vocabulary: `<the_world_0>`, `<the_world_1>`, ..., `<the_world_783>`
- Each has its own ID

**Pros:**
- ✅ Each position is distinct
- ✅ Could learn position-specific embeddings

**Cons:**
- ❌ Inflates vocabulary by 784 tokens
- ❌ Not flexible for different world token counts
- ❌ More complex to implement

### Overall Pros (Option 4)
- ✅ **Most non-intrusive** - works with Gemma's existing architecture
- ✅ **Architecturally clean** - follows special token patterns
- ✅ **Gemma handles vision encoding** automatically
- ✅ **Natural sequence order**: `[world | image | text]`
- ✅ **Single forward pass** (after getting embeddings)
- ✅ **Training-friendly** - can train projection to match token space
- ✅ **Flexible** - can put tokens anywhere in sequence

### Overall Cons (Option 4)
- ❌ Need to add special token(s) to vocabulary
- ❌ Slightly more complex setup in `__init__`
- ❌ Token count mismatch requires strategy choice
- ❌ Still need two model calls (model → language_model)

---

## Comparison Summary

| Feature | Option 1 | Option 2 | Option 3 | Option 4 |
|---------|----------|----------|----------|----------|
| **Vision encoder used?** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Intrusiveness** | Medium | High | Low | Low |
| **Code complexity** | Medium | High | Low | Medium |
| **Architectural cleanliness** | Medium | Low | High | ✅ High |
| **Flexibility** | Medium | High | Low | ✅ High |
| **Forward passes** | 2 (model + language_model) | 1 | 1 | 2 (model + language_model) |
| **Can insert world tokens?** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **Risk of breaking** | Low | Medium | Low | Low |

---

## Option 5: Bracket Tokens `<the_world_start>` and `<the_world_end>` (BEST)

### Approach
Add two special tokens that act as **brackets** around where we'll insert world embeddings:
- `<the_world_start>`: Marks beginning of world embedding region
- `<the_world_end>`: Marks end of world embedding region

The actual world embeddings (e.g., 784 tokens from 28×28 grid) are **inserted between** these brackets.

### Key Insight: Execution Order
**Critical:** We must process Cosmos BEFORE creating the chat template, because we need to know how many world tokens we have.

```
OLD ORDER (Wrong):
1. Create chat template with image + text
2. Process Cosmos world model
3. Try to insert world tokens (but template is already made!)

NEW ORDER (Correct):
1. Process Cosmos world model → get projected_world_embeds (shape: [1, 784, 2560])
2. Create chat template with brackets: "<the_world_start> <the_world_end>"
3. Process through Gemma to get embeddings with vision encoded
4. Insert world embeddings between the bracket positions
```

### Implementation

#### Setup (in `__init__`):
```python
def __init__(self, ...):
    # ... existing code ...

    # Add bracket special tokens
    special_tokens = {
        "additional_special_tokens": ["<the_world_start>", "<the_world_end>"]
    }
    self.processor.tokenizer.add_special_tokens(special_tokens)
    self.gemma.resize_token_embeddings(len(self.processor.tokenizer))

    # Store token IDs for later use
    self.world_start_id = self.processor.tokenizer.convert_tokens_to_ids("<the_world_start>")
    self.world_end_id = self.processor.tokenizer.convert_tokens_to_ids("<the_world_end>")
```

#### Forward Pass (CORRECTED ORDER):
```python
def forward(self, input_pixels, text, labels=None, num_world_steps=None):
    # ... input processing code ...

    # ========================================
    # STEP 1: Process Cosmos FIRST
    # ========================================
    # Get Cosmos world model embeddings with autoregressive rollout
    cosmos_input_5d = tensor_image.unsqueeze(2) if tensor_image.ndim == 4 else tensor_image

    if num_world_steps == 0:
        # Single-step: just encode current frame
        with torch.no_grad():
            latent_dist = self.cosmos_pipe.vae.encode(cosmos_input_5d).latent_dist
            latent_img_embeds = latent_dist.mean
    else:
        # Multi-step: use Cosmos to predict future frames
        with torch.no_grad():
            output = self.cosmos_pipe(
                prompt=text_prompt,
                image=pil_image,
                num_frames=1 + num_world_steps,
                num_inference_steps=10,
                output_type="latent",
                return_dict=True,
            )
            latent_img_embeds = output.frames

    # Process and project world embeddings
    b, c, t, h, w = latent_img_embeds.shape
    latent_img_embeds = latent_img_embeds.permute(0, 2, 3, 4, 1)  # (B, T, H, W, 16)

    # Add temporal embeddings
    temporal_ids = torch.arange(t, device=self.device)
    temporal_embeds = self.temporal_embedding(temporal_ids)
    latent_img_embeds = latent_img_embeds + temporal_embeds.view(1, t, 1, 1, c)

    # Project to Gemma dimension
    reshaped_world_embeds = latent_img_embeds.reshape(b, t * h * w, c)
    projected_world_embeds = self.world_projection(
        reshaped_world_embeds.to(dtype=torch.bfloat16)
    )
    # Now we have: projected_world_embeds shape: [1, 784, 2560] (for single frame 28×28)

    # ========================================
    # STEP 2: Create chat template with brackets
    # ========================================
    with torch.no_grad():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<the_world_start> <the_world_end>"},  # Brackets
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        gemma_inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move to correct device
        target_device = self.gemma.get_input_embeddings().weight.device
        gemma_inputs = {k: v.to(target_device) for k, v in gemma_inputs.items()}

        # ========================================
        # STEP 3: Process through Gemma to get embeddings with vision encoded
        # ========================================
        gemma_outputs = self.gemma.model(
            input_ids=gemma_inputs["input_ids"],
            pixel_values=gemma_inputs["pixel_values"],
            attention_mask=gemma_inputs["attention_mask"],
            token_type_ids=gemma_inputs.get("token_type_ids"),
            output_hidden_states=True,
            return_dict=True,
        )

        # Get embeddings from first layer (includes real vision features from SigLIP)
        embeddings = gemma_outputs.hidden_states[0]  # [1, seq_len, 2560]

    # ========================================
    # STEP 4: Insert world embeddings between brackets
    # ========================================
    # Find bracket positions
    input_ids = gemma_inputs["input_ids"]
    start_positions = (input_ids == self.world_start_id).nonzero(as_tuple=True)[1]
    end_positions = (input_ids == self.world_end_id).nonzero(as_tuple=True)[1]

    if len(start_positions) > 0 and len(end_positions) > 0:
        start_pos = start_positions[0].item()
        end_pos = end_positions[0].item()

        # Slice embeddings: [before_start] + [<start>] + [WORLD] + [<end>] + [after_end]
        embeddings_before = embeddings[:, :start_pos+1, :]  # Up to and including <start>
        embeddings_after = embeddings[:, end_pos:, :]        # From <end> onwards

        # Move world embeddings to target device
        projected_world_embeds = projected_world_embeds.to(target_device)

        # Concatenate
        combined_embeds = torch.cat([
            embeddings_before,
            projected_world_embeds,  # Insert world tokens here
            embeddings_after
        ], dim=1)

        # ========================================
        # STEP 5: Update attention mask
        # ========================================
        attention_mask = gemma_inputs["attention_mask"]
        attention_mask_before = attention_mask[:, :start_pos+1]
        attention_mask_after = attention_mask[:, end_pos:]
        world_attention_mask = torch.ones(
            (b, projected_world_embeds.size(1)),
            dtype=torch.long,
            device=target_device
        )
        combined_attention_mask = torch.cat([
            attention_mask_before,
            world_attention_mask,
            attention_mask_after
        ], dim=1)
    else:
        # No brackets found (shouldn't happen, but safety)
        combined_embeds = embeddings
        combined_attention_mask = gemma_inputs["attention_mask"]

    # ========================================
    # STEP 6: Prepare labels if provided
    # ========================================
    if labels is not None:
        # Structure: [text | <start> | world | <end> | image | text]
        # Only compute loss on text tokens, not special tokens or world/image
        num_before_start = start_pos + 1
        num_world = projected_world_embeds.size(1)
        num_after_end = input_ids.size(1) - end_pos

        # Create labels: -100 for all non-text tokens
        labels_before = gemma_inputs["input_ids"][:, :num_before_start].to(target_device)
        labels_world = torch.full(
            (b, num_world), -100, dtype=torch.long, device=target_device
        )
        labels_after = gemma_inputs["input_ids"][:, end_pos:].to(target_device)

        combined_labels = torch.cat([
            labels_before,
            labels_world,
            labels_after
        ], dim=1)
    else:
        combined_labels = None

    # ========================================
    # STEP 7: Forward through language model
    # ========================================
    outputs = self.gemma.language_model(
        inputs_embeds=combined_embeds,
        attention_mask=combined_attention_mask,
        labels=combined_labels,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    return outputs
```

### Execution Flow Diagram

```
Input Image + Text
       |
       v
[1] Process Cosmos World Model
       ├─> Get latent embeddings (16-dim)
       ├─> Add temporal embeddings
       └─> Project to Gemma dimension (2560-dim)
       Result: projected_world_embeds [1, 784, 2560]
       |
       v
[2] Create Chat Template
       ├─> Text: "<the_world_start> <the_world_end>"
       ├─> Image: PIL image
       └─> Text: user prompt
       Result: gemma_inputs with pixel_values
       |
       v
[3] Process Through Gemma
       ├─> Gemma.model() runs vision encoder (SigLIP)
       ├─> Image placeholders replaced with vision features
       └─> Get first layer embeddings
       Result: embeddings [1, seq_len, 2560] with real vision
       |
       v
[4] Insert World Embeddings
       ├─> Find <the_world_start> position
       ├─> Find <the_world_end> position
       ├─> Slice: [before_start] + [world] + [after_end]
       └─> Update attention masks
       Result: combined_embeds [1, seq_len+782, 2560]
       (seq_len - 2 original tokens + 784 world tokens = seq_len + 782)
       |
       v
[5] Forward Through Language Model
       └─> Generate predictions
```

### Sequence Structure

**Before insertion:**
```
[BOS] [text] <the_world_start> <the_world_end> [<image_tokens>...] [text] [EOS]
 ^                ^                  ^                ^
 |                |                  |                |
Position 0    Position i        Position i+1      Position i+2...
```

**After insertion:**
```
[BOS] [text] <the_world_start> [world_0] [world_1] ... [world_783] <the_world_end> [<image_tokens>...] [text] [EOS]
 ^                ^                 ^                      ^             ^
 |                |                 |                      |             |
Position 0    Position i      Position i+1...        Position i+784  Position i+785...
```

### Advantages

✅ **Semantically clear** - Brackets explicitly mark world embedding region
✅ **Only 2 special tokens** - Minimal vocabulary expansion
✅ **Flexible** - Works for any number of world tokens (784 for 1 frame, 3920 for 5 frames)
✅ **Natural in template** - `"<the_world_start> <the_world_end>"` is readable
✅ **Model awareness** - Brackets give model explicit boundary markers
✅ **Easy debugging** - Can see bracket positions in tokenized output
✅ **Correct order** - Cosmos runs first, so we know world token count
✅ **Vision encoder runs** - SigLIP processes actual image pixels

### Disadvantages

⚠️ **Sequence length changes** - Final sequence is longer than original by (num_world_tokens - 2)
⚠️ **Slicing complexity** - Need to carefully slice and concatenate embeddings
⚠️ **Position encoding** - Position IDs may need adjustment (though Gemma uses RoPE which is relative)
⚠️ **Two model calls** - Once through `model()`, once through `language_model()`

### Comparison to Option 4

| Aspect | Option 4: Repeated `<the_world>` | Option 5: `<start>` ... `<end>` |
|--------|----------------------------------|--------------------------------|
| Special tokens added | 1 token | 2 tokens |
| Template string | 784× `"<the_world> "` | ✅ `"<the_world_start> <the_world_end>"` |
| Semantics | Implicit boundaries | ✅ **Explicit boundaries** |
| Implementation | `masked_scatter` | Slice + concat |
| Model boundary awareness | Unclear | ✅ **Very clear** |
| Debugging | Harder | ✅ **Easier** |
| Execution order | Same as Option 5 | **Cosmos → Template → Insert** |

---

## Recommendation (UPDATED)

**Use Option 5: Bracket Tokens `<the_world_start>` and `<the_world_end>`**

### Rationale:
1. ✅ **Most semantically meaningful** - Explicit boundary markers
2. ✅ **Clean in chat template** - No repetition, just two tokens
3. ✅ **Model gets clear signals** - Knows exactly where world features are
4. ✅ **Architecturally sound** - Follows special token patterns
5. ✅ **Correct execution order** - Process Cosmos first, then build template
6. ✅ **Vision encoder properly used** - SigLIP runs on actual pixels

### Implementation Priority:
1. ✅ **First**: Implement Option 5 with bracket tokens
2. 🔄 **Optimize**: Profile performance and optimize if needed
3. 🔄 **Future**: Could explore different world token counts for efficiency
