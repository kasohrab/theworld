# RetinaFace Gradient Bug and Workaround

## Problem Summary

Training fails with gradients not flowing through the model, causing `loss.requires_grad=False`. This prevents backpropagation and makes the model untrainable.

## Root Cause

The `retinaface` library (a dependency of `cosmos_guardrail`, which is required by Cosmos pipelines) **globally disables PyTorch gradients** at module import time.

**Specific code location**:
```python
# File: retinaface/inference_framework.py (line 4)
import torch
import numpy as np

torch.set_grad_enabled(False)  # ← THIS LINE BREAKS TRAINING!
```

This executes when the module is imported, affecting **all PyTorch code** in the same Python process.

## Import Chain

```
TheWorld.__init__
  → imports Cosmos2VideoToWorldPipeline
    → imports cosmos_guardrail.CosmosSafetyChecker
      → imports retinaface.data.cfg_re50
        → imports retinaface/__init__.py
          → imports retinaface.inference_framework
            → EXECUTES torch.set_grad_enabled(False)  ← BUG HERE
```

## Symptoms

- `loss.requires_grad = False` even when model is in train mode
- `loss.grad_fn = None`
- No gradients flow to parameters during `.backward()`
- Training appears to run but model weights never update
- `torch.is_grad_enabled()` returns `False` everywhere

## Workaround (Implemented)

Since we cannot modify the `retinaface` library, we **re-enable gradients** immediately after importing TheWorld components:

```python
from theworld import TheWorld

# TheWorld initialization imports retinaface, which disables gradients
model = TheWorld("google/gemma-3-4b-it")

# No workaround needed! TheWorld.__init__ already re-enables gradients at line 185:
# torch.set_grad_enabled(True)
```

**The fix is automatic** - `TheWorld.__init__` already includes this workaround.

## When You Need Manual Workarounds

If you import Cosmos components **directly** (without using `TheWorld`), you must manually re-enable gradients:

```python
import torch
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline

# CRITICAL: Re-enable gradients after import
torch.set_grad_enabled(True)

# Now training will work
cosmos_pipe = Cosmos2VideoToWorldPipeline.from_pretrained(...)
```

## Code Locations

### Fixed locations:
1. **`python/theworld/modeling/theworld.py:185`** - Re-enables gradients after component initialization
2. **`tests/test_fusion_train.py:24`** - Workaround for standalone test
3. **`tests/test_cosmos_encoder_train.py:26`** - Workaround for standalone test

### Why this works:
`torch.set_grad_enabled()` is **thread-local** state. Calling `torch.set_grad_enabled(True)` after the import re-enables gradients for the current thread, overriding the `False` setting from `retinaface`.

## Verification

To verify gradients are enabled:
```python
import torch
from theworld import TheWorld

model = TheWorld("google/gemma-3-4b-it")
print(f"Gradients enabled: {torch.is_grad_enabled()}")  # Should print True
```

## Long-term Solution

**Ideal fix**: Submit a pull request to the `retinaface` library to remove the module-level `torch.set_grad_enabled(False)` call. This line should be removed or placed inside inference-only functions with a context manager:

```python
# BAD (current code):
torch.set_grad_enabled(False)  # Affects entire process!

# GOOD (proposed fix):
def inference_method(self, ...):
    with torch.no_grad():  # Only affects this function
        ...
```

## Related Issues

- PyTorch issue: https://github.com/pytorch/pytorch/issues/60789 (discusses similar gradient disabling bugs)
- This affects **any** code that imports `retinaface` transitively
- The bug is in `retinaface` v0.0.1 through latest version as of 2025-01

## Testing

To verify the fix works, run:
```bash
# Quick test (fusion component only)
uv run python tests/test_fusion_train.py

# Full smoke test (complete training pipeline)
make smoke-test
```

Expected output: `RESULT: ✓ EmbeddingFusion CAN TRAIN!`
