# Logit Validation Investigation Report

**Date**: January 12, 2025
**Issue**: TheWorld(enable_world=False) produces different logits than pure Gemma3
**Status**: Root cause identified - initialization issue

## Summary

TheWorld inherits from Gemma3ForConditionalGeneration and when `enable_world=False`, should produce identical outputs to pure Gemma3. However, logits differ by 35+ units, indicating a fundamental issue with how the model is initialized.

## Investigation Timeline

### 1. Initial Symptoms
- **Test**: Forward pass comparison between TheWorld and Gemma3
- **Result**: Logits differ by max 35.6 units
- **Both models**: Same vocabulary size (262208), same weights, same eval mode

### 2. Suspected Causes (Ruled Out)

#### ❌ Dtype Mismatch
- **Initial issue**: TheWorld was float32, Gemma3 was bfloat16
- **Fix**: Added `self.to(torch.bfloat16)` in `__init__`
- **Result**: Both now bfloat16, but logits still differ

#### ❌ Training Mode
- **Initial issue**: TheWorld was in training mode (`model.training = True`)
- **Fix**: Called `model.eval()` before testing
- **Result**: Both now in eval mode, but logits still differ

#### ❌ Processor Differences
- **Initial suspicion**: Different processors cause different preprocessing
- **Fix**: Used same processor instance for both models
- **Result**: Input IDs identical, but logits still differ

#### ❌ Vision Features
- **Test**: Compared vision tower outputs
- **Result**: Vision features **identical** (0.0 difference)
- **Conclusion**: Vision processing is not the issue

#### ❌ Weight Differences
- **Test**: Compared all parameters in state_dict
- **Result**: All weights **identical** (0.0 difference)
- **Conclusion**: Weights loaded correctly

#### ❌ Gemma3 Non-Determinism
- **Initial observation**: Gemma3 run1 vs run2 differs by 6.875
- **Fix**: Added warmup passes before comparison
- **Result**: After warmup, Gemma3 is deterministic
- **Key finding**: Two independently loaded Gemma3 models produce **identical** outputs (0.0 difference)

### 3. Root Cause Identified ✅

**Breakthrough Test**:
```python
# Load two Gemma3 models independently
g3_1 = Gemma3ForConditionalGeneration.from_pretrained(...)
g3_2 = Gemma3ForConditionalGeneration.from_pretrained(...)

# Compare outputs
diff = torch.abs(g3_1(**inputs).logits - g3_2(**inputs).logits).max()
# Result: 0.0 (identical!)
```

**But**:
```python
theworld = TheWorld(..., enable_world=False)
gemma3 = Gemma3ForConditionalGeneration.from_pretrained(...)

diff = torch.abs(theworld(**inputs).logits - gemma3(**inputs).logits).max()
# Result: 35.6 (huge difference!)
```

**Conclusion**: TheWorld's initialization is incorrect.

## Current Initialization Method (BROKEN)

```python
def __init__(self, gemma_model_name, ...):
    # 1. Load config
    config = Gemma3Config.from_pretrained(
        gemma_model_name,
        dtype=torch.bfloat16,
        local_files_only=False,
    )

    # 2. Initialize parent class with random weights
    super().__init__(config)  # ← Problem starts here!

    # 3. Load pretrained model
    pretrained = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=False,
    )

    # 4. Copy weights
    self.load_state_dict(pretrained.state_dict(), strict=False)
    del pretrained

    # 5. Convert to bfloat16
    self.to(torch.bfloat16)
```

### Why This Is Broken

1. **Step 2** (`super().__init__(config)`) initializes the model with **random weights** from the config
2. **Step 4** (`load_state_dict()`) copies parameter values but may miss:
   - Registered buffers not in state_dict
   - Internal initialization order effects
   - Device map configuration
   - RNG state
   - KV cache state
   - Other HuggingFace initialization magic

3. **Result**: TheWorld is not truly identical to a freshly loaded Gemma3

### Evidence

- All parameter **values** are identical (verified)
- But forward pass produces different outputs
- This suggests internal state or buffer differences not captured in `state_dict()`

## What We Need to Fix

The initialization must ensure TheWorld is **byte-for-byte identical** to a freshly loaded Gemma3ForConditionalGeneration, not just having the same parameter values.

## Solution: Override `from_pretrained()` ✅

After investigating HuggingFace patterns and LLaVA's implementation, the correct solution is to override `from_pretrained()` as a classmethod.

### Final Implementation

```python
class TheWorld(Gemma3ForConditionalGeneration):

    def __init__(self, config: Gemma3Config):
        """Simple structural init - called by from_pretrained()."""
        super().__init__(config)
        # Just initialize attributes, no weight loading
        self.cosmos_pipe = None
        self.cosmos_encoder = None
        # ...

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        enable_world: bool = True,
        cosmos_model_name: str = "nvidia/Cosmos-Predict2-2B-Video2World",
        **kwargs  # dtype, device_map, etc.
    ):
        # Parent's from_pretrained handles EVERYTHING correctly
        model = super(TheWorld, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Now add Cosmos components
        if enable_world:
            model.cosmos_pipe = Cosmos2VideoToWorldPipeline.from_pretrained(...)
            model.cosmos_encoder = CosmosEncoder(...)
            # ...

        return model
```

### Why This Works

1. **Parent does the heavy lifting**: `super().from_pretrained()` handles:
   - Weight loading with correct dtype
   - Buffer preservation (inv_freq stays float32)
   - Device mapping
   - All HuggingFace initialization magic

2. **No manual dtype conversion needed**: Parent already converts params to bfloat16

3. **No double-loading**: Gemma loaded once (not twice like before)

4. **Standard pattern**: How LLaVA and other multimodal models inherit from LLMs

### Validation Results

```
Max absolute diff:  0.00e+00
Mean absolute diff: 0.00e+00
✅ PASS: Logits are identical within tolerance!
```

**Conclusion**: TheWorld(enable_world=False) is **perfectly identical** to pure Gemma3.

## Test Commands for Future Validation

```bash
# Run full logit validation test
PYTHONPATH=python:$PYTHONPATH uv run python tests/test_logit_validation.py

# Quick determinism check
PYTHONPATH=python:$PYTHONPATH uv run python << 'EOF'
import torch
from theworld.modeling.theworld_refactored import TheWorld
from transformers import Gemma3ForConditionalGeneration

tw = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=False,
    dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)
g3 = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-4b-it",
    dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True
)

tw.eval()
g3.eval()

# Test with same inputs...
print("✓ Models loaded successfully with new from_pretrained() API")
EOF
```

## Key Learnings

1. **Weight equality ≠ Behavioral equality**: Same parameter values don't guarantee same outputs
2. **Initialization matters**: How you initialize matters as much as what values you load
3. **State beyond parameters**: Models have state beyond `state_dict()` (buffers, RNG, cache, etc.)
4. **Buffers not in state_dict**: Non-persistent buffers like `inv_freq` aren't in state_dict, created during init
5. **`__init__()` always creates float32**: HuggingFace's `__init__(config)` ignores `config.torch_dtype`
6. **`from_pretrained()` is the way**: Parent's `from_pretrained()` handles dtype, device_map, buffers correctly
7. **Override pattern**: Override `from_pretrained()` classmethod, not `__init__()`, for custom model loading
8. **Gemma3 determinism**: After first run, Gemma3 is deterministic across repeated calls
9. **Vision tower works**: TheWorld correctly delegates vision processing to parent
10. **Forward routing works**: TheWorld correctly uses `super().forward()` when `enable_world=False`

## References

- Test file: `tests/test_logit_validation.py`
- Refactored model: `python/theworld/modeling/theworld_refactored.py`
- Investigation: This document

---

**Status**: Awaiting fix to initialization method
