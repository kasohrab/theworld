# Cleanup Plan for TheWorld Codebase

**Date:** 2025-10-12
**Status:** Proposed

## Analysis Summary

After reviewing the codebase, I found that **`num_world_steps` and `max_world_steps` are DEPRECATED and completely unused** in the core model. According to the test validation docs, the architecture was simplified to use **single-frame VAE encoding only**—no temporal prediction anymore.

However, these parameters still appear in 75+ locations across configs, docs, scripts, and examples, creating confusion about what the model actually does.

---

## What to Remove/Clean Up

### 1. **Remove `num_world_steps` and `max_world_steps` (HIGH PRIORITY)**

**Status:** Deprecated, unused, misleading

**Why this matters:**
- ❌ Not used in `CosmosEncoder` (single-frame only since refactor)
- ❌ Not used in `TheWorld.forward()`
- ❌ Explicitly marked as "deprecated" in `tests/COSMOS_TEST_VALIDATION.md:9`
- ⚠️ Still lingering in 75+ locations (configs, docs, scripts, examples)
- 🚨 **Users may think temporal prediction works when it doesn't**

**Files to clean:**

#### Core Code
- `python/theworld/config.py` - Remove `num_world_steps`, `max_world_steps` fields (lines 75-76)
- `python/theworld/modeling/theworld.py` - Remove any references (currently none in implementation)

#### Configuration Files
- `configs/default.json`
- `configs/smoke_test.json`
- `configs/datacomp_test.json`
- `configs/datacomp_production.json`
- `configs/eval_blink.json`

#### Scripts
- `scripts/evaluate_blink.py` - Remove `--num_world_steps` argument
- `scripts/eval_spatial_rgpt.py` - Remove `num_world_steps` parameter
- `scripts/inference_demo.py` - Remove interactive "steps N" command
- `scripts/train_hf.py` - Remove from config loading

#### Examples
- `examples/inference.py` - Remove all `num_world_steps` overrides (lines 18, 46)
- `examples/simple_training.py` - Remove from model initialization (line 17)
- `examples/load_from_hub.py` - Remove from display (line 115)

#### Documentation
- `CLAUDE.md` - Remove all references (39+ occurrences)
- `docs/architecture.md` - Remove temporal prediction sections
- `docs/loss_function_and_evaluation.md` - Remove num_world_steps examples
- `docs/multi_stage_training.md` - Remove from config examples
- `docs/evaluation.md` - Update evaluation commands
- `docs/training_infrastructure_design.md` - Remove from TrainingConfig
- `REFACTOR_PLAN.md` - Archive or update

**Estimated impact:** ~300+ lines of documentation, ~20 config changes

---

### 2. **Remove Unused Generation Utilities (MEDIUM PRIORITY)**

**File:** `python/theworld/generation.py` (~106 lines)

**Why remove:**
- Contains: `get_next_token_prediction()`, `greedy_decode()`, `sample_with_temperature()`, `nucleus_sample()`
- Only used in `examples/inference.py` (which imports but never actually uses it meaningfully)
- Redundant: `TheWorld.generate()` already handles generation properly with KV caching
- Maintenance burden: Need to keep these functions working despite not using them

**Action:**
1. Delete `python/theworld/generation.py`
2. Remove from `examples/inference.py` import (line 4)
3. Update `examples/inference.py` to use `model.generate()` directly

**Lines removed:** ~106 lines of code

---

### 3. **Remove Unused Output Dataclass (LOW PRIORITY)**

**File:** `python/theworld/modeling/outputs.py`

**Status:** `GemmaVisionOutput` is exported but never used

**Analysis:**
- `GemmaVisionOutput` was part of the old modular design with separate `GemmaVisionEncoder` class
- That class was removed in January 2025 refactor (vision processing now inlined)
- `FusionOutput` is still used ✓
- `GemmaVisionOutput` is exported in `__init__.py` but no code imports or uses it

**Action:**
1. Remove `GemmaVisionOutput` class from `python/theworld/modeling/outputs.py`
2. Remove from `python/theworld/modeling/__init__.py` exports (line 3, line 9)

**Lines removed:** ~10 lines

---

### 4. **Clean Up Compatibility Parameters (LOW PRIORITY)**

**File:** `python/theworld/data.py:140-142`

```python
sow_token_id: Optional[int] = None,  # Unused but kept for compatibility
eow_token_id: Optional[int] = None,  # Unused but kept for compatibility
```

**Why remove:**
- Marked as "kept for compatibility" but never actually used in function body
- No external code references these parameters
- Just dead weight

**Action:**
Remove these parameters from `theworld_collate_fn()` signature

**Lines removed:** ~3 lines + docstring updates

---

### 5. **Archive/Delete Deprecated Docs (LOW PRIORITY)**

**Location:** `docs/archive/`

**Contents:**
1. `autoregressive_world_rollout.md` - Old multi-step temporal architecture (OBSOLETE)
2. `world_model_latent_space.md` - Still referenced in CLAUDE.md (KEEP or move back)
3. `world_embedding_integration_options.md` - Design doc for old architecture (OBSOLETE)
4. `siglip_vision_encoder_verification.md` - Historical verification (ARCHIVE-WORTHY)
5. `single_pass_architecture.md` - Old design doc (OBSOLETE)

**Recommendation:**
- Keep `siglip_vision_encoder_verification.md` - useful historical context
- Keep or move back `world_model_latent_space.md` - explains latent extraction choice
- Delete the other 3 files (fully obsolete)

**Lines removed:** ~500+ lines of obsolete documentation

---

## Weak Abstraction Points Needing Improvement

### 1. **Batch Processing Inconsistency (HIGH PRIORITY)**

**Location:** `python/theworld/modeling/theworld.py`

**Problem:**
- `forward()` method expects **batch tensors** (B, seq_len) with manual preprocessing
- `generate()` method accepts **single or batch** (auto-detects) with auto-preprocessing
- `_generate_gemma_only()` has complex batch detection logic (lines 916-926)
- Inconsistent APIs: training vs inference use different input formats

**Current messy code:**
```python
# Detect if batch or single input
is_batch = isinstance(image, list) or isinstance(prompt, list)

if not is_batch:
    images = [image]
    prompts = [prompt]
else:
    # Handle mixed single/batch inputs
    images = image if isinstance(image, list) else [image] * len(prompt)
    prompts = prompt if isinstance(prompt, list) else [prompt] * len(image)
```

**Recommendation:**
Create a unified `_prepare_batch()` helper method that:
1. Accepts single or batch inputs
2. Returns normalized lists of images and prompts
3. Used by both `generate_with_world()` and `_generate_gemma_only()`
4. Eliminates duplication

---

### 2. **Device Management Complexity (MEDIUM PRIORITY)**

**Location:** Multiple places in `python/theworld/modeling/theworld.py`

**Problem:**
- Model uses `device_map="auto"` for Gemma (distributed across GPUs)
- Manual device detection throughout: `target_device = self.gemma.get_input_embeddings().weight.device`
- Repeated `.to(target_device)` calls in multiple methods
- Easy to introduce device mismatch bugs

**Occurrences:**
- `forward()` line 638: `target_device = self.gemma.get_input_embeddings().weight.device`
- `generate_with_world()` line 817: `target_device = self.gemma.get_input_embeddings().weight.device`
- Manual tensor moves scattered throughout

**Recommendation:**
Create a `@property` for target device:
```python
@property
def target_device(self):
    """Primary device where Gemma embeddings live."""
    return self.gemma.get_input_embeddings().weight.device
```

Replace all `target_device = self.gemma.get_input_embeddings().weight.device` with `self.target_device`.

---

### 3. **Image Preprocessing Duplication (MEDIUM PRIORITY)**

**Location:**
- `python/theworld/modeling/theworld.py:746-760` - `_prepare_image()`
- `python/theworld/modeling/cosmos_encoder.py:72-87` - Similar logic in `forward()`

**Problem:**
- Image preprocessing (PIL/numpy/tensor conversion) duplicated in two places
- `_prepare_image()` in TheWorld converts to PIL
- CosmosEncoder also converts to PIL (lines 73-84)
- Same logic, different locations

**Recommendation:**
Extract to a shared utility module `python/theworld/utils/image.py`:
```python
def normalize_to_pil(image: Union[Image.Image, np.ndarray, Tensor]) -> Image.Image:
    """Convert any image format to PIL.Image RGB."""
    # Consolidate all conversion logic here
```

Use in both TheWorld and CosmosEncoder.

---

### 4. **Collator Function Complexity (LOW PRIORITY)**

**Location:** `python/theworld/data.py:135-250` (~115 lines)

**Problem:**
- `theworld_collate_fn()` is a monolithic function doing multiple things:
  1. Chat template formatting (with/without world tokens)
  2. Image preprocessing for SigLIP
  3. PIL image passing for Cosmos
  4. Label preparation
  5. Padding and batching
- Hard to test individual components
- Mixing concerns (preprocessing vs formatting vs batching)

**Recommendation:**
Break into smaller functions:
```python
def _format_chat_messages(image, text, include_world_tokens=True) -> list
def _preprocess_images(images, processor) -> dict
def _prepare_labels(batch, processor) -> Tensor
def theworld_collate_fn(...) -> TheWorldBatch  # Orchestrates the above
```

Each function testable in isolation.

---

### 5. **State Dict Save/Load Inconsistency (LOW PRIORITY)**

**Location:** `python/theworld/modeling/theworld.py`

**Problem:**
- `state_dict()` (line 330) returns ONLY trainable parameters
- `load_state_dict()` (line 348) expects trainable parameters (strict=False)
- `save_checkpoint()` (line 396) saves full checkpoint dict with metadata
- `load_checkpoint()` (line 445) loads checkpoint dict
- Two separate save/load APIs doing similar things

**Current API:**
```python
# Method 1: State dict (for HF Trainer)
state = model.state_dict()
model.load_state_dict(state)

# Method 2: Checkpoint (manual training)
model.save_checkpoint("path.pt", optimizer=opt)
model.load_checkpoint("path.pt", optimizer=opt)
```

**Recommendation:**
Document clearly when to use which API, or consolidate into one unified save/load system.

---

### 6. **Forward Pass Label Alignment Complexity (MEDIUM PRIORITY)**

**Location:** `python/theworld/modeling/theworld.py:696-736`

**Problem:**
40 lines of complex logic to align labels with world token insertions:
- Find bracket token positions
- Validation that brackets exist
- Manual label construction: `[tokens_before | -100 for world | tokens_after]`
- Shift for causal LM
- Multiple tensor ops and concatenations

**Why this is fragile:**
- Easy to introduce off-by-one errors
- Hard to understand for contributors
- Coupled to fusion module internals

**Recommendation:**
Move label alignment logic into `EmbeddingFusion` module:
```python
class EmbeddingFusion:
    def align_labels(self, input_ids, world_tokens_size) -> Tensor:
        """Align labels with fused sequence (inserts -100 for world tokens)."""
        # All the complex logic moves here
```

Then `forward()` just calls: `aligned_labels = self.fusion.align_labels(input_ids, world_embeds.size(1))`

---

## Estimated Total Impact

### Lines Removed
- **Code:** ~150-200 lines (generation.py + config params + unused classes)
- **Documentation:** ~800+ lines (num_world_steps references + obsolete docs)
- **Config files:** ~20 parameter removals

### Breaking Changes
- ⚠️ **Breaking:** Any external code passing `num_world_steps` will fail
- ⚠️ **Breaking:** `examples/inference.py` needs update
- ⚠️ **Breaking:** Custom training scripts using `generation.py` need update
- ✓ **Safe:** All other cleanups are internal-only

### Benefits
1. **Clarity:** Removes confusing deprecated parameters that don't do anything
2. **Maintainability:** ~1000 fewer lines of code/docs to maintain
3. **Correctness:** Prevents users from thinking temporal prediction works when it doesn't
4. **Onboarding:** New contributors won't waste time understanding unused features
5. **Abstraction:** Fixing weak points makes codebase more extensible

---

## Recommended Execution Order

### Phase 1: Critical Cleanups (High Priority)
1. Remove `num_world_steps`/`max_world_steps` from all code and configs
2. Update all documentation to reflect single-frame architecture
3. Fix/test all examples and scripts

**Estimated time:** 2-3 hours

### Phase 2: Code Cleanup (Medium Priority)
4. Remove `generation.py` and update examples
5. Remove unused `GemmaVisionOutput` dataclass
6. Clean up compatibility parameters in data.py

**Estimated time:** 1 hour

### Phase 3: Abstraction Improvements (Medium Priority)
7. Create unified `_prepare_batch()` helper
8. Add `target_device` property
9. Extract image preprocessing to shared utility
10. Refactor label alignment into fusion module

**Estimated time:** 2-3 hours

### Phase 4: Documentation (Low Priority)
11. Delete obsolete archive docs
12. Update architecture docs with cleaner design
13. Add comments explaining design decisions

**Estimated time:** 1 hour

---

## Open Questions

1. **Breaking changes policy:** Should we maintain backward compatibility for `num_world_steps` parameter (accept but ignore) or fail fast?
2. **Archive docs:** Keep for historical reference or delete entirely?
3. **`world_model_latent_space.md`:** Move out of archive (still referenced) or update CLAUDE.md references?
4. **Version bump:** Does this cleanup warrant a major version bump (breaking changes)?

---

## Conclusion

The codebase has accumulated technical debt from the temporal prediction → single-frame refactor. The `num_world_steps` parameter is the most critical cleanup (appears everywhere, does nothing, misleads users). The abstraction improvements are less urgent but will make the codebase more maintainable long-term.

**Priority ranking:**
1. 🔴 Remove `num_world_steps` (correctness issue)
2. 🟡 Abstraction improvements (maintainability)
3. 🟢 Documentation cleanup (nice-to-have)
