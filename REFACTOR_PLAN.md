# Refactor Plan: Modular nn.Module Architecture

## Overview
Refactor TheWorld model into separate nn.Module components for better modularity, testability, and maintainability.

## Current Architecture (Monolithic)
```
TheWorld.forward():
  - Cosmos processing (VAE + temporal + projection)
  - Gemma vision processing (SigLIP)
  - Embedding fusion (insert world tokens)
  - Gemma language model forward
  - Loss computation
```

**Problems:**
- 300+ line forward() method
- Hard to test individual components
- Difficult to freeze/unfreeze specific parts
- All preprocessing mixed with model logic

## Proposed Architecture (Modular)

### 1. CosmosEncoder (nn.Module)
**Purpose:** Encode images using Cosmos VAE → temporal embeddings → projection to Gemma space

**Inputs:**
```python
images: List[PIL.Image]  # Raw PIL images from collator
texts: List[str]         # Text prompts for conditioning
num_world_steps: int     # 0 = single frame, >0 = autoregressive rollout
```

**Outputs:**
```python
world_embeds: torch.Tensor  # Shape: (B, num_world_tokens, 2304)
                            # num_world_tokens = 784 * (1 + num_world_steps)
                            # 784 = 28x28 spatial tokens per frame
```

**Internal Flow:**
1. Convert PIL images to tensors (B, C, H, W)
2. If num_world_steps == 0:
   - VAE encode: latent_dist = vae.encode(image)
   - Extract mean: latents = latent_dist.mean (B, 16, 1, H, W)
3. If num_world_steps > 0:
   - Autoregressive rollout via Cosmos pipeline
   - latents = cosmos_pipe(..., num_frames=1+num_world_steps)
4. Add temporal embeddings: (T, 16) → broadcast to (B, T, H, W, 16)
5. Project: (B, T*H*W, 16) → (B, T*H*W, 2304) via self.world_projection

**Parameters:**
```python
self.temporal_embedding: nn.Embedding(max_world_steps, 16)
self.world_projection: nn.Linear(16, 2304)
```

**External References:**
```python
self.cosmos_pipe: Cosmos2VideoToWorldPipeline (not a parameter, just reference)
self.device: torch.device
```

---

### 2. GemmaVisionEncoder (nn.Module)
**Purpose:** Process vision through SigLIP and combine with text token embeddings

**Inputs:**
```python
input_ids: torch.Tensor       # Shape: (B, seq_len) - preprocessed from collator
pixel_values: torch.Tensor    # Shape: (B, C, H, W) - preprocessed from collator
attention_mask: torch.Tensor  # Shape: (B, seq_len) - preprocessed from collator
```

**Outputs:**
```python
gemma_embeds: torch.Tensor       # Shape: (B, seq_len, 2304) - combined vision+text embeddings
input_ids: torch.Tensor          # Pass-through for later use
attention_mask: torch.Tensor     # Pass-through for later use
```

**Internal Flow:**
1. Get text token embeddings: `embed_tokens(input_ids)` → (B, seq_len, 2304)
2. Process vision through SigLIP: `gemma.get_image_features(pixel_values)` → (B, num_image_tokens, 2304)
3. Find image placeholder positions in input_ids
4. Replace placeholders with real vision features via masked_scatter
5. Return combined embeddings

**Parameters:**
None (uses external gemma model)

**External References:**
```python
self.gemma: Gemma3ForConditionalGeneration (reference to parent model)
```

---

### 3. EmbeddingFusion (nn.Module)
**Purpose:** Fuse Gemma vision embeddings with Cosmos world embeddings by inserting between bracket tokens

**Inputs:**
```python
gemma_embeds: torch.Tensor       # Shape: (B, seq_len, 2304) - from GemmaVisionEncoder
world_embeds: torch.Tensor       # Shape: (B, num_world_tokens, 2304) - from CosmosEncoder
input_ids: torch.Tensor          # Shape: (B, seq_len) - for finding bracket positions
attention_mask: torch.Tensor     # Shape: (B, seq_len) - to update for world tokens
```

**Outputs:**
```python
combined_embeds: torch.Tensor         # Shape: (B, combined_len, 2304)
combined_attention_mask: torch.Tensor # Shape: (B, combined_len)
combined_len = seq_len + num_world_tokens - 2  # -2 for removed bracket tokens
```

**Internal Flow:**
1. Find <the_world_start> token position: `start_pos`
2. Find <the_world_end> token position: `end_pos`
3. Split embeddings:
   - embeds_before = gemma_embeds[:, :start_pos+1, :]
   - embeds_after = gemma_embeds[:, end_pos:, :]
4. Concatenate: [embeds_before | world_embeds | embeds_after]
5. Similarly update attention_mask:
   - mask_before = attention_mask[:, :start_pos+1]
   - world_mask = torch.ones((B, num_world_tokens))
   - mask_after = attention_mask[:, end_pos:]
   - combined_mask = [mask_before | world_mask | mask_after]

**Parameters:**
```python
self.world_start_id: int  # Token ID for <the_world_start>
self.world_end_id: int    # Token ID for <the_world_end>
```

---

### 4. TheWorld (Main nn.Module - Refactored)

**Sub-modules:**
```python
self.cosmos_encoder = CosmosEncoder(
    cosmos_pipe=self.cosmos_pipe,
    temporal_embedding=self.temporal_embedding,
    world_projection=self.world_projection,
    device=self.device
)

self.gemma_vision = GemmaVisionEncoder(
    gemma_model=self.gemma
)

self.fusion = EmbeddingFusion(
    world_start_id=self.world_start_id,
    world_end_id=self.world_end_id
)

# Existing components
self.gemma.language_model  # Language model
self.gemma.lm_head         # LM head for logits
```

**Forward Method:**
```python
def forward(
    self,
    input_ids: torch.Tensor = None,           # From collator
    pixel_values: torch.Tensor = None,        # From collator (Gemma preprocessed)
    attention_mask: torch.Tensor = None,      # From collator
    images: List[PIL.Image] = None,           # From collator (raw for Cosmos)
    texts: List[str] = None,                  # From collator (raw for Cosmos)
    labels: torch.Tensor = None,              # From collator (tokenized captions)
    num_world_steps: int = None,              # Optional override
    **kwargs
) -> CausalLMOutputWithPast:

    # Default num_world_steps
    if num_world_steps is None:
        num_world_steps = self.num_world_steps

    # 1. Encode world via Cosmos
    world_embeds = self.cosmos_encoder(
        images=images,
        texts=texts,
        num_world_steps=num_world_steps
    )
    # world_embeds: (B, 784*(1+num_world_steps), 2304)

    # 2. Encode vision+text via Gemma
    gemma_embeds, input_ids, attention_mask = self.gemma_vision(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask
    )
    # gemma_embeds: (B, seq_len, 2304)

    # 3. Fuse embeddings
    combined_embeds, combined_attention_mask = self.fusion(
        gemma_embeds=gemma_embeds,
        world_embeds=world_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    # combined_embeds: (B, combined_len, 2304)

    # 4. Forward through language model
    lm_outputs = self.gemma.language_model(
        inputs_embeds=combined_embeds,
        attention_mask=combined_attention_mask,
        return_dict=True
    )

    # 5. Apply LM head
    logits = self.gemma.lm_head(lm_outputs.last_hidden_state)

    # 6. Compute loss
    loss = None
    if labels is not None:
        # Align labels with combined sequence
        combined_labels = self._align_labels(
            labels, input_ids, world_embeds.size(1)
        )
        loss = self._compute_loss(logits, combined_labels)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=lm_outputs.past_key_values,
        hidden_states=lm_outputs.hidden_states,
        attentions=lm_outputs.attentions
    )
```

---

## Type Safety & Validation

### Type Annotations
Use Python type hints throughout:
```python
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch import nn, Tensor
```

### Runtime Assertions
Add shape validation at module boundaries:
```python
def forward(self, images: List[Image.Image], ...) -> Tensor:
    assert isinstance(images, list), "images must be List[PIL.Image]"
    assert all(isinstance(img, Image.Image) for img in images)

    world_embeds = ...
    assert world_embeds.dim() == 3, f"Expected 3D tensor, got {world_embeds.dim()}D"
    assert world_embeds.size(-1) == 2304, f"Expected dim 2304, got {world_embeds.size(-1)}"

    return world_embeds
```

### Dataclasses for Complex Outputs
Instead of returning tuples, use dataclasses:
```python
from dataclasses import dataclass

@dataclass
class GemmaVisionOutput:
    embeddings: Tensor      # (B, seq_len, 2304)
    input_ids: Tensor       # (B, seq_len)
    attention_mask: Tensor  # (B, seq_len)

@dataclass
class FusionOutput:
    combined_embeds: Tensor         # (B, combined_len, 2304)
    combined_attention_mask: Tensor # (B, combined_len)
```

---

## Testing Strategy

### Unit Tests for Each Module

**Test CosmosEncoder:**
```python
def test_cosmos_encoder_single_frame():
    encoder = CosmosEncoder(...)
    images = [Image.new("RGB", (224, 224))]
    texts = ["test"]

    output = encoder(images, texts, num_world_steps=0)

    assert output.shape == (1, 784, 2304)  # 1 frame, 28x28 tokens
    assert output.dtype == torch.bfloat16

def test_cosmos_encoder_multi_frame():
    encoder = CosmosEncoder(...)
    output = encoder(images, texts, num_world_steps=4)

    assert output.shape == (1, 3920, 2304)  # 5 frames * 784
```

**Test GemmaVisionEncoder:**
```python
def test_gemma_vision_encoder():
    encoder = GemmaVisionEncoder(gemma_model)
    input_ids = torch.randint(0, 1000, (1, 100))
    pixel_values = torch.randn(1, 3, 224, 224)
    attention_mask = torch.ones(1, 100)

    embeds, ids, mask = encoder(input_ids, pixel_values, attention_mask)

    assert embeds.shape == (1, 100, 2304)
    assert torch.equal(ids, input_ids)  # Pass-through
```

**Test EmbeddingFusion:**
```python
def test_embedding_fusion():
    fusion = EmbeddingFusion(world_start_id=1000, world_end_id=1001)

    # Create test inputs with brackets at positions 10 and 11
    input_ids = torch.zeros(1, 50, dtype=torch.long)
    input_ids[0, 10] = 1000  # <start>
    input_ids[0, 11] = 1001  # <end>

    gemma_embeds = torch.randn(1, 50, 2304)
    world_embeds = torch.randn(1, 784, 2304)
    attention_mask = torch.ones(1, 50)

    combined, combined_mask = fusion(gemma_embeds, world_embeds, input_ids, attention_mask)

    # Expected: 50 - 2 (brackets) + 784 (world) = 832
    assert combined.shape == (1, 832, 2304)
    assert combined_mask.shape == (1, 832)
```

### Integration Test
```python
def test_full_forward_pass():
    model = TheWorld(...)

    # Simulate collator output
    batch = {
        "input_ids": torch.randint(0, 1000, (1, 100)),
        "pixel_values": torch.randn(1, 3, 224, 224),
        "attention_mask": torch.ones(1, 100),
        "images": [Image.new("RGB", (224, 224))],
        "texts": ["test prompt"],
        "labels": torch.randint(0, 1000, (1, 50))
    }

    output = model(**batch)

    assert output.loss is not None
    assert output.logits.shape[0] == 1  # Batch size
    assert output.logits.shape[-1] == model.vocab_size
```

---

## Migration Plan

### Phase 1: Create New Modules (No Breaking Changes)
1. Create `CosmosEncoder` class
2. Create `GemmaVisionEncoder` class
3. Create `EmbeddingFusion` class
4. Add unit tests for each

### Phase 2: Update TheWorld.__init__()
1. Instantiate sub-modules instead of direct parameters
2. Move temporal_embedding and world_projection to CosmosEncoder
3. Ensure backward compatibility in checkpoint loading

### Phase 3: Refactor TheWorld.forward()
1. Create new `forward_modular()` method using sub-modules
2. Keep old `forward()` for backward compatibility
3. Add flag to switch between old/new: `use_modular=True`

### Phase 4: Update Collator (Already Done)
Collator already returns correct format - no changes needed

### Phase 5: Testing & Validation
1. Run smoke test with both old and new forward
2. Verify outputs match
3. Check gradient flow
4. Benchmark performance

### Phase 6: Deprecate Old Forward
1. Make `forward_modular()` the default
2. Add deprecation warning to old `forward()`
3. Remove old implementation after 1-2 releases

---

## Parameter Management

### Before (Monolithic):
```python
TheWorld:
  - temporal_embedding (direct parameter)
  - world_projection (direct parameter)
  - cosmos_pipe (not a parameter)
  - gemma (sub-module with parameters)
```

### After (Modular):
```python
TheWorld:
  - cosmos_encoder (nn.Module)
    - temporal_embedding (parameter)
    - world_projection (parameter)
  - gemma_vision (nn.Module)
    - no parameters (uses parent gemma)
  - fusion (nn.Module)
    - no parameters (just logic)
  - gemma (nn.Module)
    - language_model (parameter)
    - lm_head (parameter)
```

### Freezing Strategy:
```python
# Freeze Cosmos encoder
model.cosmos_encoder.requires_grad_(False)

# Freeze Gemma vision
model.gemma.model.vision_tower.requires_grad_(False)

# Freeze Gemma language
model.gemma.language_model.requires_grad_(False)

# Only train projection
for param in model.cosmos_encoder.world_projection.parameters():
    param.requires_grad = True
```

---

## Checkpoint Compatibility

### Save Format:
```python
state_dict = {
    "cosmos_encoder.temporal_embedding.weight": ...,
    "cosmos_encoder.world_projection.weight": ...,
    "cosmos_encoder.world_projection.bias": ...,
    "gemma.language_model.layers.0.weight": ...,
    ...
}
```

### Load Old Checkpoints:
```python
def load_legacy_checkpoint(path):
    checkpoint = torch.load(path)

    # Remap old keys to new structure
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("temporal_embedding"):
            new_key = f"cosmos_encoder.{key}"
        elif key.startswith("world_projection"):
            new_key = f"cosmos_encoder.{key}"
        else:
            new_key = key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
```

---

## Documentation Updates

### CLAUDE.md Changes:
1. Add "Architecture" section explaining modular design
2. Update forward() signature documentation
3. Add examples of using individual modules
4. Document how to freeze specific components

### Code Comments:
Each module should have:
- Detailed docstring with Args, Returns, Example
- Type annotations on all methods
- Shape comments on key tensors: `# (B, T, H, W, C)`

---

## Benefits Summary

1. **Modularity**: Each component is independently testable
2. **Clarity**: Clear separation of concerns
3. **Flexibility**: Easy to swap Cosmos or Gemma for alternatives
4. **Performance**: Can cache intermediate outputs
5. **Maintainability**: Easier to understand and modify
6. **Type Safety**: Strong typing catches errors early
7. **Testing**: Comprehensive unit and integration tests
8. **Freezing**: Granular control over trainable parameters

---

## Risk Mitigation

### Risks:
1. Breaking existing checkpoints
2. Performance regression
3. Gradient flow issues
4. Device management complexity

### Mitigations:
1. Checkpoint remapping utility
2. Benchmark before/after
3. Extensive gradient tests
4. Clear device handling per module
