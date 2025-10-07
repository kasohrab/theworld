# Single-Pass Architecture: World Embeddings Through All Layers

## Overview

TheWorld model implements a **single-pass architecture** where world model embeddings flow through **all transformer layers** alongside vision and text embeddings. This ensures world embeddings are properly integrated into the language model's processing.

## Architecture Flow

### Complete Execution Pipeline

```
Input: Image + Text
    ↓
[1. Cosmos World Model]
    → VAE encoder
    → Autoregressive rollout (if multi-step)
    → Temporal embeddings
    → Projection to Gemma dimension
    → Result: projected_world_embeds [B, num_world_tokens, 2560]
    ↓
[2. Chat Template]
    → Format with bracket tokens: "<the_world_start> <the_world_end>"
    → Add image and text
    → Result: input_ids, pixel_values, attention_mask
    ↓
[3. Manual Embedding Construction]
    → Get text embeddings: embed_tokens(input_ids)
    → Run SigLIP vision encoder: get_image_features(pixel_values)
    → Replace image placeholders with vision features (masked_scatter)
    → Result: inputs_embeds [B, seq_len, 2560]
    ↓
[4. Insert World Embeddings]
    → Find bracket positions in input_ids
    → Slice embeddings and insert world tokens between brackets
    → Result: combined_embeds = [text + <start> + WORLD + <end> + IMAGE + text]
    ↓
[5. Single Forward Pass]
    → language_model(inputs_embeds=combined_embeds)
    → Process through ALL transformer layers
    → World embeddings interact with vision and text at every layer
    → Result: hidden_states [B, combined_seq_len, 2560]
    ↓
[6. LM Head]
    → lm_head(hidden_states)
    → Result: logits [B, combined_seq_len, vocab_size]
```

## Why Single-Pass Matters

### ❌ Previous Approach (Double Processing)

```python
# WRONG: Two passes through transformer
gemma_outputs = self.gemma.model(...)  # Pass 1: ALL layers without world
embeddings = gemma_outputs.hidden_states[0]
combined_embeds = insert_world(embeddings)
outputs = self.gemma.language_model(combined_embeds)  # Pass 2: ALL layers again
```

**Problems:**
- Wasteful: Processes all layers twice
- Incomplete: First pass has no world embeddings
- Inconsistent: World embeddings only exist in second pass

### ✓ Current Approach (Single Pass)

```python
# CORRECT: One pass through transformer
inputs_embeds = self.gemma.model.language_model.embed_tokens(input_ids)
image_features = self.gemma.model.get_image_features(pixel_values)
inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
combined_embeds = insert_world(inputs_embeds)
outputs = self.gemma.language_model(combined_embeds)  # Single pass with everything
```

**Benefits:**
- ✓ Efficient: Single pass through all layers
- ✓ Complete: World embeddings present from the start
- ✓ Proper integration: Vision, world, and text process together

## Implementation Details

### Step 3: Manual Embedding Construction

We replicate what `Gemma3Model.forward()` does internally (lines 887-903 in `modeling_gemma3.py`):

```python
# Reference: Gemma3Model.forward() line 887-888
inputs_embeds = self.gemma.model.language_model.embed_tokens(input_ids)

# Reference: Gemma3Model.forward() line 897-903
with torch.no_grad():
    # Line 898: image_features = self.get_image_features(pixel_values)
    image_features = self.gemma.model.get_image_features(pixel_values)

    # Line 899: image_features = image_features.to(...)
    image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

    # Line 900-902: special_image_mask = self.get_placeholder_mask(...)
    special_image_mask = self.gemma.model.get_placeholder_mask(
        input_ids, inputs_embeds=inputs_embeds, image_features=image_features
    )

    # Line 903: inputs_embeds = inputs_embeds.masked_scatter(...)
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
```

**Why manual construction?**

We can't use `gemma.model()` directly because:
1. It processes through all transformer layers immediately
2. We need to insert world embeddings BEFORE the transformer processes them
3. Manual construction gives us the embeddings with vision features but before layer processing

### Step 4: World Embedding Insertion

```python
# Find bracket positions
start_positions = (input_ids == self.world_start_id).nonzero(as_tuple=True)[1]
end_positions = (input_ids == self.world_end_id).nonzero(as_tuple=True)[1]

# Slice and insert
embeddings_before = embeddings[:, : start_pos + 1, :]  # Up to <start>
embeddings_after = embeddings[:, end_pos:, :]  # From <end> onwards
combined_embeds = torch.cat([embeddings_before, projected_world_embeds, embeddings_after], dim=1)
```

**Result sequence:**
```
[BOS] [text tokens] [<the_world_start>] [WORLD TOKENS] [<the_world_end>] [IMAGE TOKENS] [text tokens]
```

### Step 7: Single Forward Pass

```python
# Reference: Gemma3Model.forward() line 937-948
lm_outputs = self.gemma.language_model(
    inputs_embeds=combined_embeds,  # Vision + World + Text
    attention_mask=combined_attention_mask,
    return_dict=True,
)
```

This single call processes through:
- All transformer layers (typically 26 layers for Gemma 2B)
- World embeddings interact with vision and text at every layer
- Attention mechanisms can attend across all modalities

### Step 8: LM Head and Loss

```python
# Reference: Gemma3ForConditionalGeneration.forward() line 1094-1119
hidden_states = lm_outputs.last_hidden_state
logits = self.gemma.lm_head(hidden_states)

# Compute loss (if training)
if combined_labels is not None:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = combined_labels[..., 1:].contiguous()
    loss = CrossEntropyLoss()(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
```

## Token Sequence Example

For a typical input:

**Input:**
- Text: "What happens next?"
- Image: 896×896 photo
- World: 4 future frame predictions (28×28 latent grid each)

**Token sequence after Step 4:**
```
Position  | Token Type          | Count | Source
----------|---------------------|-------|--------
0         | BOS                 | 1     | Chat template
1-5       | Text tokens         | 5     | Tokenized "What happens next?"
6         | <the_world_start>   | 1     | Special token
7-3926    | World embeddings    | 3920  | Cosmos (5 frames × 28×28)
3927      | <the_world_end>     | 1     | Special token
3928-4183 | Image embeddings    | 256   | SigLIP vision (896÷14)²
4184-...  | More text           | ...   | Continuation
```

**Total context:** ~4200 tokens (varies by prompt length)

## Attention Mechanism

With the single-pass architecture, the transformer attention can:

1. **World ↔ Vision**: World tokens can attend to vision features
   - Learn correlations between predicted future and current visual state

2. **World ↔ Text**: World tokens can attend to text
   - Incorporate linguistic context into world understanding

3. **Vision ↔ Text**: Vision tokens can attend to text (standard multimodal)
   - Standard vision-language interaction

4. **All ↔ All**: Every token can attend to every other token (causal masking)
   - Full multimodal integration across all three modalities

## Performance Characteristics

**Computational cost:**
- Single pass: O(L × N²) where L = num_layers, N = total_sequence_length
- Previous double pass: O(2 × L × N²) ≈ 2× the cost

**Memory:**
- Single pass stores one set of KV cache
- More efficient for generation tasks

**Quality:**
- Better integration of world embeddings
- More coherent multimodal understanding
- World features properly contextualized across all layers

## Code References

All line numbers reference `transformers/models/gemma3/modeling_gemma3.py`:

- **Gemma3Model.forward()**: Lines 826-956
  - Line 887-888: `embed_tokens(input_ids)`
  - Line 897-903: Vision processing and masked_scatter
  - Line 937-948: `language_model(inputs_embeds=...)`

- **Gemma3ForConditionalGeneration.forward()**: Lines 1008-1132
  - Line 1077-1092: Calls `self.model()`
  - Line 1094-1119: LM head and loss computation

Our implementation replicates this flow but inserts world embeddings before the language_model call.
