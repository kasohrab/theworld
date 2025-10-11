# World-Aware Generation Implementation Plan

## Problem Statement

**Current Issue** (theworld.py:752-772):
The `generate()` method currently delegates to `self.gemma.generate()`, which:
- ❌ Doesn't know about world embeddings
- ❌ Uses only Gemma's vision tokens (SigLIP)
- ❌ World model contribution is "training-only"
- ❌ Inference doesn't leverage temporal/spatial world knowledge

## Solution: Prefix-KV Cache Approach (Option A)

### Strategy
Pre-compute world embeddings once, pass as prefix to generation via KV cache.

### How It Works

1. **Process image** → Get world embeddings (static, one-time)
2. **Format prompt** with SOW/EOW bracket tokens
3. **Run ONE forward pass** with world embeddings → cache KV states
4. **Use cached KV** as "past" for autoregressive generation
5. **Generate text tokens** autoregressively using Gemma's generate with KV cache

### Advantages
- ✅ Compatible with HuggingFace's generate() infrastructure
- ✅ Uses world embeddings for full context
- ✅ Efficient: world encoding done once, reused via KV cache
- ✅ Supports all generate() features (beam search, sampling, top-k/top-p)

### Disadvantages
- ⚠️ Requires understanding HuggingFace's past_key_values format
- ⚠️ World embeddings are static (no per-token update)

## Implementation Steps

### Step 1: Add `use_cache` parameter to forward()

**File**: `python/theworld/modeling/theworld.py`

**Changes**:
```python
def forward(
    self,
    input_ids: Tensor,
    pixel_values: Tensor,
    attention_mask: Tensor,
    images: List[Image.Image],
    labels: Optional[Tensor] = None,
    use_cache: bool = False,  # NEW
    past_key_values: Optional[Tuple] = None,  # NEW
):
    # ... existing world/vision encoding ...

    # STEP 4: Forward through language model
    lm_outputs = self.gemma.language_model(
        inputs_embeds=fusion_output.combined_embeds,
        attention_mask=fusion_output.combined_attention_mask,
        return_dict=True,
        use_cache=use_cache,  # Pass through
        past_key_values=past_key_values,  # Pass through
    )

    # Return KV cache
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=lm_outputs.past_key_values if use_cache else None,
        ...
    )
```

### Step 2: Implement `generate_with_world()`

**File**: `python/theworld/modeling/theworld.py`

```python
def generate_with_world(
    self,
    image: Union[Image.Image, np.ndarray, Tensor],
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    **kwargs,
) -> str:
    """Generate with world embeddings using KV cache."""

    # 1. Prepare inputs with world brackets
    pil_image = self._prepare_image(image)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "<start_of_world> <end_of_world>"},
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt}
        ]
    }]

    inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # 2. Move to device
    target_device = self.gemma.get_input_embeddings().weight.device
    inputs = {k: v.to(target_device) if hasattr(v, "to") else v
              for k, v in inputs.items()}

    # 3. Preprocess image
    pixel_values = self.processor.image_processor(
        images=pil_image,
        return_tensors="pt"
    )["pixel_values"].to(target_device)

    # 4. Forward pass to get KV cache with world embeddings
    with torch.no_grad():
        prompt_outputs = self.forward(
            input_ids=inputs["input_ids"],
            pixel_values=pixel_values,
            attention_mask=inputs["attention_mask"],
            images=[pil_image],
            labels=None,
            use_cache=True,  # Enable caching
        )

    # 5. Generate using cached context
    with torch.no_grad():
        generated = self.gemma.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            past_key_values=prompt_outputs.past_key_values,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **kwargs,
        )

    # 6. Decode
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_len:]
    response = self.processor.decode(new_tokens[0], skip_special_tokens=True)

    return response.strip()
```

### Step 3: Update `generate()` to use new method

```python
def generate(
    self,
    image: Union[Image.Image, np.ndarray, Tensor],
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    skip_world_tokens: bool = False,
    **kwargs,
) -> str:
    """Generate with or without world embeddings."""

    if skip_world_tokens:
        # Ablation: Gemma-only baseline
        return self._generate_gemma_only(image, prompt, max_new_tokens, temperature, **kwargs)
    else:
        # Use world embeddings
        return self.generate_with_world(image, prompt, max_new_tokens, temperature, **kwargs)
```

### Step 4: Extract Gemma-only generation to separate method

```python
def _generate_gemma_only(
    self,
    image: Union[Image.Image, np.ndarray, Tensor],
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
    **kwargs,
) -> str:
    """Generate using only Gemma (no world model) - for ablation."""

    pil_image = self._prepare_image(image)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": prompt}
        ]
    }]

    inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Move to device
    if hasattr(inputs, "to"):
        inputs = inputs.to(self.gemma.device)
    elif isinstance(inputs, dict):
        inputs = {k: v.to(self.gemma.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = self.gemma.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **kwargs,
        )

    # Decode
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, input_len:]
    response = self.processor.decode(generated_ids[0], skip_special_tokens=True)

    return response.strip()
```

## Testing Plan

### Unit Tests (`tests/test_generation_with_world.py`)

1. **Test KV cache is populated**
   - Verify `past_key_values` is not None
   - Check tuple structure

2. **Test world embeddings affect output**
   - Compare world-aware vs Gemma-only
   - Outputs should differ

3. **Test generation parameters work**
   - Temperature, top_k, top_p
   - max_new_tokens

4. **Test ablation mode**
   - `skip_world_tokens=True` works
   - Output matches Gemma baseline

### Integration Tests

1. **Memory usage** - Should not be significantly higher
2. **Speed** - Should be comparable to Gemma-only
3. **Quality** - Qualitative assessment on sample images

## Success Criteria

- ✅ World embeddings included in generation context
- ✅ Uses HuggingFace generate() infrastructure
- ✅ Supports all sampling modes
- ✅ Faster than iterative re-encoding
- ✅ Ablation mode works
- ✅ Unit tests pass

## Fallback Plan

If KV cache approach has issues:
- Fall back to iterative generation (Option B)
- Accept slower generation for correctness
- Document limitation

## References

- HuggingFace generate() docs: https://huggingface.co/docs/transformers/main_classes/text_generation
- Gemma 3 architecture: https://huggingface.co/docs/transformers/model_doc/gemma3
- KV cache explanation: https://huggingface.co/docs/transformers/main/en/kv_cache

---

*Implementation Date: 2025-01-29*
*Status: Ready to implement*
