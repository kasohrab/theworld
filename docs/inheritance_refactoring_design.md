# TheWorld Inheritance Refactoring Design Doc

## Goal
Refactor `TheWorld` to inherit from `Gemma3ForConditionalGeneration` to eliminate code duplication and guarantee identical behavior when world tokens are not used.

## Current Architecture Problems

1. **Code Duplication**: `_generate_gemma_only()` duplicates preprocessing logic that Gemma3 already handles
2. **Two Wrapped Models**: We have both `self.gemma` (wrapped) and inherit from `nn.Module`, creating confusion
3. **No Guarantee**: Can't easily prove that TheWorld without world tokens === pure Gemma3
4. **Maintenance Burden**: Have to keep manual vision processing in sync with Gemma3 updates

## Proposed Architecture

```python
class TheWorld(Gemma3ForConditionalGeneration):
    # Inherit all of Gemma3's functionality
    # Add Cosmos components as extensions
```

## Key Design Decisions

### 1. Initialization Strategy

**Problem**: `Gemma3ForConditionalGeneration.__init__()` expects a `Gemma3Config` object, but we want to construct from model name string.

**Solution**:
```python
def __init__(self, gemma_model_name, cosmos_model_name=None, ...):
    # Load config from pretrained
    config = Gemma3Config.from_pretrained(gemma_model_name, ...)

    # Initialize parent (creates model structure)
    super().__init__(config)

    # Load pretrained weights
    pretrained = Gemma3ForConditionalGeneration.from_pretrained(gemma_model_name, ...)
    self.load_state_dict(pretrained.state_dict(), strict=False)
    del pretrained

    # Add Cosmos components
    if cosmos_model_name:
        self._init_cosmos(cosmos_model_name)
```

### 2. Forward Pass Strategy

**Key Insight**: Check if world tokens are present in `input_ids` to decide which path to take.

```python
def forward(self, input_ids, pixel_values, attention_mask, images=None, labels=None, **kwargs):
    # Detect if world tokens present
    has_world_tokens = (
        self.load_cosmos and
        self.sow_token_id is not None and
        input_ids is not None and
        (input_ids == self.sow_token_id).any()
    )

    if has_world_tokens and images is not None:
        # World-augmented path
        return self._forward_with_world(input_ids, pixel_values, attention_mask, images, labels, **kwargs)
    else:
        # Pure Gemma path - delegate to parent
        return super().forward(input_ids, pixel_values, attention_mask, labels=labels, **kwargs)
```

### 3. World-Augmented Forward

```python
def _forward_with_world(self, input_ids, pixel_values, attention_mask, images, labels, **kwargs):
    # 1. Get embeddings (use parent's method)
    inputs_embeds = self.get_input_embeddings()(input_ids)

    # 2. Get vision features (use parent's method)
    if pixel_values is not None:
        image_features = self.model.get_image_features(pixel_values)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask = self.model.get_placeholder_mask(input_ids, inputs_embeds, image_features)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # 3. Get world embeddings
    world_embeds = self.cosmos_encoder(images)

    # 4. Fuse embeddings
    fusion_output = self.fusion(
        gemma_embeds=inputs_embeds,
        world_embeds=world_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    # 5. Update labels to mask world tokens
    if labels is not None:
        combined_labels = self._align_labels_with_world_tokens(
            input_ids, labels, world_embeds.size(1)
        )
    else:
        combined_labels = None

    # 6. Call parent forward with fused embeddings
    return super().forward(
        inputs_embeds=fusion_output.combined_embeds,
        attention_mask=fusion_output.combined_attention_mask,
        labels=combined_labels,
        **kwargs
    )
```

### 4. Generation Strategy

**Remove entirely**: `_generate_gemma_only()`, `generate_with_world()`

**Replace with**:
```python
def generate(self, image, prompt, max_new_tokens=50, use_world_tokens=True, **kwargs):
    # Prepare input with or without world tokens
    if use_world_tokens and self.load_cosmos:
        # Add world bracket tokens to prompt
        prompt_with_world = f"<start_of_world> <end_of_world> {prompt}"
    else:
        prompt_with_world = prompt

    # Prepare inputs using processor
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": self._prepare_image(image)},
            {"type": "text", "text": prompt_with_world}
        ]
    }]

    inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(self.device)

    # If using world tokens, pass raw images for Cosmos
    if use_world_tokens and self.load_cosmos:
        inputs["images"] = [self._prepare_image(image)]

    # Call parent's generate (inherits all HF generation features)
    outputs = super().generate(**inputs, max_new_tokens=max_new_tokens, **kwargs)

    # Decode
    return self.processor.decode(outputs[0], skip_special_tokens=True)
```

## Implementation Steps

1. ✅ Change class declaration: `class TheWorld(Gemma3ForConditionalGeneration)`
2. ⏳ Rewrite `__init__` to initialize parent properly
3. ⏳ Simplify `forward()` to conditionally delegate
4. ⏳ Write `_forward_with_world()` helper
5. ⏳ Remove `_generate_gemma_only()`, `generate_with_world()`
6. ⏳ Simplify `generate()` to just prepare inputs and call `super().generate()`
7. ⏳ Update `_apply_freezing()` to work with inherited structure
8. ⏳ Write test: `test_gemma_equivalence.py`

## Testing Strategy

```python
def test_gemma_equivalence():
    """Verify TheWorld without world tokens === Gemma3"""
    # Load models
    theworld = TheWorld("google/gemma-3-4b-it", load_cosmos=False)
    gemma3 = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")

    # Same input
    inputs = processor.apply_chat_template(...)

    # Forward pass
    theworld_logits = theworld(**inputs).logits
    gemma3_logits = gemma3(**inputs).logits

    # Should be identical (or very close due to numerical precision)
    assert torch.allclose(theworld_logits, gemma3_logits, atol=1e-5)
```

## Migration Notes

### Breaking Changes
- `skip_world_tokens` parameter removed from `generate()`
- Use `use_world_tokens=False` instead
- Forward signature slightly different (adds `images` parameter)

### Non-Breaking Changes
- All training code works the same
- Checkpointing/saving unchanged
- Freezing logic unchanged

## Benefits Summary

1. **Zero duplication**: Pure Gemma path uses parent's code directly
2. **Guaranteed equivalence**: `use_world_tokens=False` === pure Gemma3
3. **Better maintenance**: Automatically inherit Gemma3 updates
4. **Cleaner code**: ~500 lines removed
5. **Better HF integration**: Inherit all generation features (beam search, sampling strategies, etc.)

## Risks & Mitigation

**Risk 1**: Initialization complexity
- *Mitigation*: Thoroughly test model loading and weight initialization

**Risk 2**: Forward pass behavior changes
- *Mitigation*: Write comprehensive tests comparing old vs new behavior

**Risk 3**: Breaking existing code
- *Mitigation*: Clear migration guide, deprecation warnings

**Risk 4**: Training compatibility
- *Mitigation*: Test with existing training scripts before merging
