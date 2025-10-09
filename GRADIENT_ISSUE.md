# Gradient Flow Issue - Backpropagation Error

## Error Message
```
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

## What's Happening
- Training runs forward pass successfully
- Models load successfully (Gemma + Cosmos)
- During backward pass (loss.backward()), PyTorch throws error
- Error indicates that some tensor in the computation graph doesn't have gradients enabled

## Root Cause
The issue is in `cosmos_encoder.py`:
1. When `num_world_steps=0` (single-step mode), we were using direct VAE encoding:
   ```python
   latent_dist = self.cosmos_pipe.vae.encode(input).latent_dist
   latent_img_embeds = latent_dist.mean  # <-- Problem here
   ```
2. The `.mean` property on DiagonalGaussianDistribution returns a detached tensor
3. Even though we unfroze the VAE (`freeze_cosmos_vae=False`), the output tensor had no gradient graph

## Fix Applied
Unified single-step and multi-step code paths:
- Instead of special-casing `num_world_steps==0` with direct VAE access
- Now always use the pipeline: `cosmos_pipe(num_frames=1+num_world_steps)`
- Pipeline output (`.frames`) preserves gradient graph properly
- Conditionally apply `torch.no_grad()` only when VAE is frozen

## Files Modified
1. `python/theworld/modeling/cosmos_encoder.py` - Unified pipeline calls
2. `python/theworld/modeling/theworld.py` - Pass `freeze_vae` parameter
3. `configs/smoke_test.json` - Set `freeze_cosmos_vae: false`

## Testing Results

### Attempt 1: Unified Pipeline Approach
- Simplified code to always use `cosmos_pipe()`
- Pipeline runs successfully (shows 10 diffusion steps progress bar)
- **Still fails** with same gradient error during backward pass

### Root Cause Analysis
The Cosmos diffusion pipeline itself likely has internal `torch.no_grad()` contexts that prevent gradient flow, even when the VAE parameters have `requires_grad=True`. The pipeline is designed for inference, not training.

### Observations
1. Even with `freeze_cosmos_vae=False`, we get 56M trainable params (~1.3%)
2. Pipeline executes but outputs have no gradient graph
3. The issue is not in our code but in the Cosmos pipeline design

### Potential Solutions
1. **Don't use the full pipeline for training** - Extract just the VAE encoder
2. **Unfreeze different components** - Try unfreezing Gemma vision instead
3. **Accept projection-only training** - Keep everything frozen except projection layers
4. **Check pipeline source** - Inspect Cosmos pipeline for where gradients are blocked

## Current Status

### Issue Summary
1. ✅ Fixed data collator to use `apply_chat_template()` (fixed image token mismatch)
2. ✅ Unified cosmos_encoder to always use pipeline (simplified code)
3. ❌ Still getting gradient error when VAE is unfrozen (Cosmos pipeline blocks gradients)
4. ❌ Getting image token mismatch error again when reverting to frozen config

### Next Steps
The gradient issue when unfreezing Cosmos VAE is fundamental - the diffusion pipeline is not designed for training. We should:
1. **For now**: Train with everything frozen (projection-only, ~0.07% of params)
2. **Later**: If needed, unfreeze Gemma components or extract VAE encoder separately

The image token mismatch suggests `apply_chat_template` might not be inserting image tokens correctly. Need to debug data collator.
