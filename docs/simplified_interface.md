# TheWorld Simplified Interface Guide

**Last Updated**: January 2025

## Overview

TheWorld now inherits directly from `Gemma3ForConditionalGeneration` and uses the **identical interface** to Gemma3. This eliminates custom high-level APIs and maximizes delegation to the parent class.

## Key Changes

### 1. Renamed Parameter: `load_cosmos` → `enable_world`

**Old API**:
```python
model = TheWorld("google/gemma-3-4b-it", load_cosmos=True)
```

**New API**:
```python
model = TheWorld("google/gemma-3-4b-it", enable_world=True)
```

The new name better reflects that it's enabling world model features, not just loading a model.

### 2. Standard Gemma3 Interface

**Before** (custom generate method):
```python
model = TheWorld("google/gemma-3-4b-it")
response = model.generate(
    image=pil_image,
    prompt="What is in this image?",
    use_world_tokens=True,
    max_new_tokens=50
)
```

**After** (standard Gemma3 interface):
```python
model = TheWorld("google/gemma-3-4b-it", enable_world=True)
processor = model.processor

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": "What is in this image?"}
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# For world tokens, pass raw PIL image
inputs["images"] = [pil_image]

# Generate (same as Gemma3!)
outputs = model.generate(**inputs, max_new_tokens=50)
decoded = processor.decode(outputs[0], skip_special_tokens=True)
```

## Usage Examples

### Example 1: Without World Model (Pure Gemma3)

```python
from theworld import TheWorld

# Initialize without world model
model = TheWorld("google/gemma-3-4b-it", enable_world=False)
processor = model.processor

# Prepare inputs
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": "Describe this image."}
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

input_len = inputs["input_ids"].shape[1]

# Generate
outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0, input_len:], skip_special_tokens=True)
print(response)
```

**Behavior**: Identical to pure Gemma3 - no world tokens, delegates directly to parent's forward().

### Example 2: With World Model (Automatic Token Injection)

```python
from theworld import TheWorld

# Initialize with world model enabled
model = TheWorld("google/gemma-3-4b-it", enable_world=True)
processor = model.processor

# Prepare inputs (same format as Gemma3)
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": "What will happen next in this scene?"}
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# IMPORTANT: Pass raw PIL image for Cosmos processing
inputs["images"] = [pil_image]

input_len = inputs["input_ids"].shape[1]

# Generate - SOW/EOW tokens are injected automatically!
outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0, input_len:], skip_special_tokens=True)
print(response)
```

**Behavior**: SOW/EOW tokens are automatically injected after BOS token during generation. World embeddings are fused with text/image embeddings.

### Example 3: Training Mode

```python
from theworld import TheWorld
import torch

# Initialize model
model = TheWorld(
    "google/gemma-3-4b-it",
    enable_world=True,
    freeze_gemma_vision=True,
    freeze_gemma_language=True,
    freeze_cosmos_vae=True
)  # Only projection layers trainable

# Prepare training batch
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": "What is this?"}
    ]
}]

inputs = model.processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

# Add images for world processing
inputs["images"] = [pil_image]

# Add labels
labels = inputs["input_ids"].clone()
labels[:, :input_len] = -100  # Mask prompt tokens

# Forward pass
outputs = model(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    attention_mask=inputs["attention_mask"],
    images=inputs["images"],
    labels=labels
)

loss = outputs.loss
loss.backward()
```

## Automatic World Token Injection

### How It Works

When `enable_world=True`, the model automatically injects `<start_of_world>` and `<end_of_world>` tokens during generation:

1. **Tokenization**: User's prompt is tokenized normally
2. **Token Injection** (in `prepare_inputs_for_generation`): SOW/EOW tokens are inserted after BOS token
3. **Forward Pass**: Model detects SOW token and uses world-augmented path
4. **World Embeddings**: Cosmos encodes the PIL image and embeddings are fused
5. **Generation**: Standard Gemma3 generation with augmented context

**Sequence**:
```
Before injection:  [BOS, text..., <image>, ...]
After injection:   [BOS, SOW, EOW, text..., <image>, ...]
```

### Why This Approach?

1. ✅ **Standard Interface**: Same as Gemma3, no custom high-level API
2. ✅ **Maximum Delegation**: Uses parent's generate(), forward(), and all utilities
3. ✅ **Transparent**: Token injection happens automatically, user doesn't see it
4. ✅ **Flexible**: Control via init parameter (`enable_world`) instead of per-call flag

## Migration Guide

### Updating Existing Code

**If you were using the old custom generate**:

```python
# OLD
model = TheWorld("google/gemma-3-4b-it", load_cosmos=True)
response = model.generate(
    image=pil_image,
    prompt="What is this?",
    use_world_tokens=True,
    max_new_tokens=50
)

# NEW
model = TheWorld("google/gemma-3-4b-it", enable_world=True)
processor = model.processor

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": "What is this?"}
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

inputs["images"] = [pil_image]  # For world processing

input_len = inputs["input_ids"].shape[1]
outputs = model.generate(**inputs, max_new_tokens=50)
response = processor.decode(outputs[0, input_len:], skip_special_tokens=True)
```

### Breaking Changes

| Old | New | Impact |
|-----|-----|--------|
| `load_cosmos` | `enable_world` | Rename parameter in __init__ |
| `model.generate(image, prompt, use_world_tokens)` | Standard Gemma3 interface | Must update all generate calls |
| Per-call world token control | Init-time control | Decide at model creation instead |

### Non-Breaking Changes

- ✅ **Training**: All training code works unchanged
- ✅ **Forward pass**: Signature compatible (added `images` optional parameter)
- ✅ **Checkpointing**: Save/load works the same way
- ✅ **Freezing logic**: Unchanged

## Architecture Benefits

### 1. Zero Code Duplication

**Before**: Custom `_generate_gemma_only()` duplicated Gemma3's preprocessing logic (~100 lines)

**After**: Delegates to `super().generate()` directly (0 lines of duplication)

### 2. Guaranteed Equivalence

When `enable_world=False`, TheWorld's behavior is **provably identical** to Gemma3:

```python
def forward(self, input_ids, pixel_values, attention_mask, images=None, ...):
    has_world_tokens = (
        self.enable_world
        and self.sow_token_id is not None
        and input_ids is not None
        and (input_ids == self.sow_token_id).any()
    )

    if has_world_tokens and images is not None:
        # World-augmented path
        return self._forward_with_world(...)
    else:
        # Pure Gemma path - EXACTLY the same as Gemma3
        return super().forward(...)
```

### 3. Better HuggingFace Integration

By inheriting from `Gemma3ForConditionalGeneration`, we automatically get:

- ✅ All generation strategies (beam search, sampling, contrastive search, etc.)
- ✅ Caching (KV cache for fast generation)
- ✅ Gradient checkpointing support
- ✅ Device map (multi-GPU, CPU offloading)
- ✅ All utilities (resize_token_embeddings, get_input_embeddings, etc.)

### 4. Simpler Codebase

**Code reduction**:
- Removed `_prepare_image()` helper: 15 lines
- Removed custom `generate()`: 100 lines
- **Total**: ~115 lines removed

## Technical Details

### Token Injection Implementation

Located in `prepare_inputs_for_generation()` override:

```python
def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    cache_position=None,
    images=None,
    **kwargs,
):
    # If first generation step and world enabled, inject SOW/EOW
    if (
        self.enable_world
        and cache_position is not None
        and cache_position[0] == 0
        and self.sow_token_id is not None
        and input_ids is not None
    ):
        # Inject after BOS token (position 1)
        batch_size = input_ids.shape[0]
        sow_eow = torch.tensor(
            [[self.sow_token_id, self.eow_token_id]] * batch_size,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat([
            input_ids[:, :1],   # BOS
            sow_eow,            # SOW, EOW
            input_ids[:, 1:],   # Rest
        ], dim=1)

        # Update attention mask accordingly
        ...

    # Delegate to parent
    model_inputs = super().prepare_inputs_for_generation(...)

    # Pass images only on first step (not during cached decoding)
    if cache_position is not None and cache_position[0] == 0 and images is not None:
        model_inputs["images"] = images

    return model_inputs
```

**Why `cache_position[0] == 0`?**

During generation, the model generates one token at a time and caches intermediate key/value states. We only inject world tokens and pass images on the **first step** (when `cache_position[0] == 0`), not during subsequent cached decoding steps.

### Forward Pass Routing

```python
def forward(self, input_ids, pixel_values, attention_mask, images=None, ...):
    # Auto-detect world tokens
    has_world_tokens = (
        self.enable_world
        and self.sow_token_id is not None
        and input_ids is not None
        and (input_ids == self.sow_token_id).any()
    )

    if has_world_tokens and images is not None:
        # World-augmented: Get world embeddings, fuse, pass to parent
        return self._forward_with_world(...)
    else:
        # Pure Gemma: Delegate directly to parent
        return super().forward(...)
```

## FAQ

### Q: Can I still use `model.generate()` with kwargs like `do_sample`, `temperature`, etc.?

**A**: Yes! Since we inherit from `Gemma3ForConditionalGeneration`, all parent's generate kwargs work:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    num_beams=5,
    # ... any Gemma3 generation parameter
)
```

### Q: How do I disable world tokens at runtime?

**A**: You can't disable them per-call anymore. Decide at initialization:

```python
# World model enabled
model_with_world = TheWorld("google/gemma-3-4b-it", enable_world=True)

# World model disabled (pure Gemma)
model_without_world = TheWorld("google/gemma-3-4b-it", enable_world=False)
```

If you need both behaviors, load two separate models or reload with different config.

### Q: What if I don't pass `images` when `enable_world=True`?

**A**: The model will still work, but world tokens won't trigger the world-augmented path. Forward detection requires:
1. `enable_world=True`
2. SOW token present in `input_ids`
3. `images` provided

If any condition is false, it falls back to pure Gemma path.

### Q: Does this work with HuggingFace Trainer?

**A**: Yes! The interface is fully compatible:

```python
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=dataset,
    data_collator=collator,
)

trainer.train()
```

### Q: Can I use `model.from_pretrained()` to load a trained model?

**A**: Yes (once implemented). This is a TODO item:

```python
# Future API (when implemented)
model = TheWorld.from_pretrained("username/theworld-datacomp")
```

## Related Documentation

- [Inheritance Refactoring Design](inheritance_refactoring_design.md) - Design rationale
- [Refactoring Progress](refactoring_progress.md) - Implementation status
- [Training Guide](../CLAUDE.md#training-with-huggingface-trainer) - Training setup

---

**Questions or Issues?** File an issue on the GitHub repository.
