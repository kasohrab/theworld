# Tokenization and Special Tokens in TheWorld

This document provides a comprehensive guide to how tokenization works in TheWorld, including all special tokens from Gemma 3 and our custom world model tokens.

## Table of Contents

1. [Overview](#overview)
2. [Gemma 3 Special Tokens](#gemma-3-special-tokens)
3. [TheWorld Custom Tokens](#theworld-custom-tokens)
4. [Token Sequence Structure](#token-sequence-structure)
5. [Chat Template Format](#chat-template-format)
6. [Training vs Inference](#training-vs-inference)
7. [Best Practices](#best-practices)
8. [Common Pitfalls](#common-pitfalls)
9. [Validation Checklist](#validation-checklist)
10. [Troubleshooting](#troubleshooting)

---

## Overview

TheWorld uses the Gemma 3 tokenizer with additional custom tokens for world model embeddings. Understanding the token flow is critical for:

- **Training**: Ensuring labels align with the combined embedding sequence
- **Inference**: Proper generation with world context
- **Debugging**: Identifying tokenization issues quickly

**Key Principle**: Always use `processor.apply_chat_template()` instead of manual tokenization. This ensures all special tokens are inserted correctly.

---

## Gemma 3 Special Tokens

Gemma 3 has several built-in special tokens that control sequence structure and multimodal input:

### Control Tokens

| Token | ID | Purpose | Usage |
|-------|----|---------| ------|
| `<bos>` | 2 | Beginning of sequence | Automatically added by chat template |
| `<eos>` | 1 | End of sequence | Marks completion of generation |
| `<pad>` | 0 | Padding token | Used for batch padding |

### Chat Tokens

| Token | Purpose | Example |
|-------|---------|---------|
| `<start_of_turn>` | Marks speaker turn | `<start_of_turn>user\n` |
| `<end_of_turn>` | Ends speaker turn | `...<end_of_turn>` |

Valid speakers: `user`, `model`

### Vision Tokens

| Token | ID | Purpose | Visibility |
|-------|----|---------| ----------|
| `<start_of_image>` | N/A | Marks image placeholder start | In input text |
| `<end_of_image>` | N/A | Marks image placeholder end | Internal only (not in token IDs) |
| `<image_soft_token>` | 262144 | Vision encoder output | Replaces image placeholder |

**Important**: When you pass a PIL Image in the `content` list, the chat template automatically inserts `<start_of_image>` and processes the image through SigLIP. The image tokens are then replaced with `<image_soft_token>` placeholders (~264 tokens for 896×896 images).

### Reference

- Gemma 3 Tokenizer Guide: https://ai.google.dev/gemma/docs/formatting
- Tokenizer Colab: https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/tokenizer.ipynb

---

## TheWorld Custom Tokens

TheWorld adds two custom tokens using Gemma's reserved custom token slots:

### Custom Token Slots

Gemma reserves **99 custom token slots** (`tokenizer.special_tokens.CUSTOM + 0` to `98`) for application-specific tokens. We use the first two:

| Slot | Token | Abbreviation | Purpose |
|------|-------|--------------|---------|
| 0 | `<start_of_world>` | SOW | Marks beginning of world model embeddings |
| 1 | `<end_of_world>` | EOW | Marks end of world model embeddings |

### Token Registration

Tokens are registered during `TheWorld.__init__()`:

```python
# python/theworld/modeling/theworld.py:141-163
custom_tokens = {
    0: "<start_of_world>",  # SOW
    1: "<end_of_world>",    # EOW
}

# Add to tokenizer
num_added = processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": list(custom_tokens.values())}
)

# Resize embedding layer
if num_added > 0:
    gemma.resize_token_embeddings(len(processor.tokenizer))

# Store token IDs
self.sow_token_id = processor.tokenizer.convert_tokens_to_ids("<start_of_world>")
self.eow_token_id = processor.tokenizer.convert_tokens_to_ids("<end_of_world>")
```

**Why custom slots?** This is the proper way to extend Gemma's vocabulary for application-specific needs. The tokens are:
- Part of the official vocabulary
- Have dedicated embedding vectors (trainable)
- Won't conflict with future Gemma updates

### Token IDs

The exact token IDs depend on the tokenizer vocabulary size:
- Gemma 3 base vocabulary: 256,000 tokens
- After adding custom tokens: 256,002 tokens
- Typical IDs: `sow_token_id = 256000`, `eow_token_id = 256001`

**Access via constants**:
```python
from theworld.constants import CUSTOM_TOKEN_SOW, CUSTOM_TOKEN_EOW
# "<start_of_world>", "<end_of_world>"
```

---

## Token Sequence Structure

### Training Sequence

Here's a complete training example with token annotations:

```
Input:
  Image: cat.jpg (PIL Image)
  Text: "What is in this image?"
  Label: "A fluffy cat sitting on a couch."

Formatted by collator:
  "<start_of_world> <end_of_world>" + [Image] + "What is in this image?"

After apply_chat_template():
  <bos> <start_of_turn> user
  <start_of_world> <end_of_world> <start_of_image> <image_soft_token>×264 What is in this image? <end_of_turn>
  <start_of_turn> model

Token ID sequence (showing first 20 tokens):
  [2, 106, 1645, 256000, 256001, 108, 262144, 262144, 262144, ...]
   │   │    │      │       │      │     │       │       │
   │   │    │      │       │      │     └─ Image tokens (×264)
   │   │    │      │       │      └─ <start_of_image>
   │   │    │      │       └─ <end_of_world>
   │   │    │      └─ <start_of_world>
   │   │    └─ "user"
   │   └─ <start_of_turn>
   └─ <bos>
```

### Forward Pass Transformation

During `TheWorld.forward()`, the token sequence is transformed:

```
1. Input IDs → Gemma embeddings (vision + text)
   [BOS | user | SOW | EOW | <image> | text | model]
                                ↓ SigLIP
   [BOS | user | SOW | EOW | vision_embeds×264 | text_embeds | model]

2. Cosmos encodes image → world embeddings
   PIL Image → Cosmos VAE → latents (16-dim) → projection (2304-dim)
   Result: world_embeds (batch, 784, 2304)

3. Fusion inserts world embeddings between SOW/EOW
   [BOS | user | SOW | world_embeds×784 | EOW | vision_embeds×264 | text_embeds | model]
                       ^^^^^^^^^^^^^^^^
                       World tokens inserted here

4. Combined sequence → Gemma language model → logits
```

### Label Alignment

Labels must align with the combined sequence:

```python
# theworld/modeling/theworld.py:632-646
# Build labels: [tokens_before_SOW | SOW | -100 for world | EOW | tokens_after]
labels_before = input_ids[:, :start_pos+1]  # Up to and including SOW
labels_world = torch.full((batch_size, num_world_tokens), -100, ...)  # Ignore world
labels_after = input_ids[:, end_pos:]  # From EOW onwards

combined_labels = torch.cat([labels_before, labels_world, labels_after], dim=1)
```

**Critical**: World tokens get `-100` (ignore index) so loss is only computed on text tokens.

---

## Chat Template Format

### Basic Usage

```python
from transformers import AutoProcessor
from PIL import Image

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

# Format with chat template
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "<start_of_world> <end_of_world>"},
            {"type": "image", "image": pil_image},  # PIL Image object!
            {"type": "text", "text": "What is in this image?"}
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,  # Training: no assistant prefix
    return_tensors="pt"
)
```

### Key Parameters

| Parameter | Value (Training) | Value (Inference) | Purpose |
|-----------|------------------|-------------------|---------|
| `tokenize` | `True` | `True` | Convert to token IDs |
| `add_generation_prompt` | `False` | `True` | Add `<start_of_turn>model` prefix |
| `return_tensors` | `"pt"` | `"pt"` | Return PyTorch tensors |

### Why Chat Template?

**Correct** ✅:
```python
messages = [{"role": "user", "content": [...]}]
inputs = processor.apply_chat_template(messages, ...)
```

**Incorrect** ❌:
```python
# Manual tokenization - DON'T DO THIS!
text = "<bos><start_of_turn>user\n..."
inputs = tokenizer(text)  # Wrong! Missing image processing
```

**Reason**: `apply_chat_template` handles:
- BOS token insertion
- Turn markers
- Image token placeholders
- Image preprocessing for SigLIP
- Proper token ordering

---

## Training vs Inference

### Training

```python
# Training collator (data.py:190-200)
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "<start_of_world> <end_of_world>"},
        {"type": "image", "image": pil_image},
        {"type": "text", "text": question}
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,  # No assistant prefix
    return_tensors="pt"
)

# Forward pass includes world tokens
outputs = model.forward(**inputs, labels=labels)
loss = outputs.loss  # Computed on text tokens only
```

### Inference

```python
# Inference (theworld.py:672-689)
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": prompt}
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Add assistant prefix for generation
    return_tensors="pt"
)

# Generate response
with torch.no_grad():
    outputs = model.gemma.generate(**inputs, max_new_tokens=50)

# Decode (skip special tokens)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

**Note**: Current `generate()` delegates to Gemma's `generate()`, which doesn't use world tokens during autoregressive decoding. This is a known limitation (see `theworld.py:718-736` TODO).

---

## Best Practices

### ✅ DO

1. **Always use `apply_chat_template()`**
   ```python
   inputs = processor.apply_chat_template(messages, tokenize=True, ...)
   ```

2. **Pass PIL Images in content list**
   ```python
   {"type": "image", "image": Image.open("path.jpg")}
   ```

3. **Use `skip_special_tokens=True` when decoding**
   ```python
   text = processor.decode(token_ids, skip_special_tokens=True)
   ```

4. **Check token counts during development**
   ```python
   ids = inputs["input_ids"][0].tolist()
   print(f"BOS at position 0: {ids[0] == 2}")
   print(f"Image tokens: {ids.count(262144)}")
   ```

5. **Use constants for token IDs**
   ```python
   from theworld.constants import BOS_TOKEN_ID, IMAGE_SOFT_TOKEN_ID
   ```

### ❌ DON'T

1. **Don't manually add BOS/EOS**
   ```python
   # Wrong! Chat template handles this
   text = "<bos>" + text + "<eos>"
   ```

2. **Don't use tensor images in messages**
   ```python
   # Wrong! Must be PIL Image
   {"type": "image", "image": torch.randn(3, 224, 224)}
   ```

3. **Don't hardcode token IDs**
   ```python
   # Wrong! IDs may change
   if token_id == 256000:  # Bad
   ```

4. **Don't skip chat template**
   ```python
   # Wrong! Missing image processing
   tokens = tokenizer.encode(text)
   ```

5. **Don't include special tokens in decoded output**
   ```python
   # Wrong! Will show <bos>, <eos>, etc.
   text = processor.decode(ids, skip_special_tokens=False)
   ```

---

## Common Pitfalls

### 1. BOS Token Missing

**Symptom**: Loss is NaN or model generates gibberish

**Cause**: Chat template not properly inserting BOS

**Fix**: Always use `apply_chat_template()`, never manual concatenation

**Validation**:
```python
ids = inputs["input_ids"][0].tolist()
assert ids[0] == 2, "BOS token must be at position 0"
```

### 2. Image Tokens Not Present

**Symptom**: Model treats image as text

**Cause**: Wrong image format or missing image in messages

**Fix**: Use PIL Image objects
```python
from PIL import Image
pil_image = Image.open("path.jpg").convert("RGB")
messages = [{"role": "user", "content": [{"type": "image", "image": pil_image}, ...]}]
```

**Validation**:
```python
ids = inputs["input_ids"][0].tolist()
assert ids.count(262144) > 0, "No image tokens found"
```

### 3. World Tokens Missing in Training

**Symptom**: Model fails with assertion error in fusion module

**Cause**: Collator not adding `<start_of_world> <end_of_world>` text

**Fix**: Check collator format (data.py:182)
```python
{"type": "text", "text": "<start_of_world> <end_of_world>"}
```

**Validation**:
```python
assert "<start_of_world>" in processor.tokenizer.get_vocab()
```

### 4. Label Misalignment

**Symptom**: Loss doesn't decrease during training

**Cause**: Labels don't account for world tokens

**Fix**: Use model's forward pass which handles label construction (theworld.py:632-646)

### 5. Generation Includes Special Tokens

**Symptom**: Output like "The answer is cat <eos>"

**Cause**: Not using `skip_special_tokens=True`

**Fix**:
```python
response = processor.decode(output_ids, skip_special_tokens=True)
```

---

## Validation Checklist

Use this checklist to verify tokenization is working correctly:

### During Development

- [ ] BOS token is at position 0 of input_ids
- [ ] Image soft tokens are present (count > 0)
- [ ] SOW and EOW tokens are in vocabulary
- [ ] SOW appears exactly once in input_ids
- [ ] EOW appears exactly once in input_ids
- [ ] SOW comes before EOW in sequence
- [ ] Label sequence has same length as combined embeddings
- [ ] World token positions have -100 in labels

### Testing Code

```python
from theworld.constants import BOS_TOKEN_ID, IMAGE_SOFT_TOKEN_ID

def validate_tokenization(model, inputs):
    """Validate tokenization for debugging."""
    ids = inputs["input_ids"][0].tolist()

    # 1. BOS token
    assert ids[0] == BOS_TOKEN_ID, f"BOS missing: {ids[0]} != {BOS_TOKEN_ID}"

    # 2. Image tokens
    img_count = ids.count(IMAGE_SOFT_TOKEN_ID)
    assert img_count > 0, f"No image tokens found"
    print(f"✓ Image tokens: {img_count}")

    # 3. SOW/EOW tokens
    sow_count = ids.count(model.sow_token_id)
    eow_count = ids.count(model.eow_token_id)
    assert sow_count == 1, f"SOW count: {sow_count}"
    assert eow_count == 1, f"EOW count: {eow_count}"

    sow_pos = ids.index(model.sow_token_id)
    eow_pos = ids.index(model.eow_token_id)
    assert sow_pos < eow_pos, f"SOW must come before EOW"
    print(f"✓ World tokens: SOW at {sow_pos}, EOW at {eow_pos}")

    print("✓ All validation checks passed!")
```

---

## Troubleshooting

### Error: "No <start_of_world> token found in input_ids"

**Cause**: Collator not adding world bracket tokens

**Solution**:
1. Check that collator includes world token text:
   ```python
   {"type": "text", "text": "<start_of_world> <end_of_world>"}
   ```
2. Verify tokens are in vocabulary:
   ```python
   print(model.processor.tokenizer.get_vocab().get("<start_of_world>"))
   ```

### Error: "Expected input_ids to be a Tensor"

**Cause**: Incorrect format from `apply_chat_template`

**Solution**: Ensure `return_tensors="pt"` is set

### Warning: "BOS token missing at position 0"

**Cause**: Manual tokenization or wrong chat template usage

**Solution**: Use `apply_chat_template()` with proper format

### Loss is NaN

**Possible causes**:
1. Label misalignment (check that labels have -100 for world tokens)
2. Learning rate too high
3. Mixed precision issues (try `bf16` instead of `fp16`)

**Debug**:
```python
# Check label structure
print(f"Labels shape: {labels.shape}")
print(f"Logits shape: {outputs.logits.shape}")
print(f"Ignore tokens (-100 count): {(labels == -100).sum().item()}")
```

### Model generates empty strings

**Cause**: EOS token generated immediately

**Solution**:
1. Check that `add_generation_prompt=True` for inference
2. Verify model has been trained
3. Try higher temperature: `temperature=0.7`

---

## Summary

**Key Takeaways**:

1. **Always use `processor.apply_chat_template()`** - Never manual tokenization
2. **PIL Images in messages** - Not tensors or paths
3. **BOS at position 0** - Automatically added by chat template
4. **World tokens use custom slots** - Proper Gemma vocabulary extension
5. **Validation is critical** - Check token structure during development
6. **Skip special tokens when decoding** - For clean output

**Files to reference**:
- `python/theworld/constants.py` - Token ID constants
- `python/theworld/data.py:190-243` - Collator with validation
- `python/theworld/modeling/theworld.py:141-163` - Token registration
- `python/theworld/modeling/theworld.py:614-646` - Label construction

**External resources**:
- [Gemma 3 Tokenizer Guide](https://ai.google.dev/gemma/docs/formatting)
- [Gemma Tokenizer Colab](https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/tokenizer.ipynb)
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)

---

*Last updated: 2025-01-29*
