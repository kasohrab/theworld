#!/usr/bin/env python3
"""
Inspect trained checkpoint parameters to verify training worked correctly.

Usage:
    python examples/inspect_checkpoint.py [checkpoint_path]

Example:
    python examples/inspect_checkpoint.py checkpoints/llava_pretrain_test/checkpoint-54
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def inspect_checkpoint(checkpoint_path: str):
    """Load and inspect parameter values from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    # Load safetensors file
    if checkpoint_path.is_dir():
        model_file = checkpoint_path / "model.safetensors"
    else:
        model_file = checkpoint_path

    if not model_file.exists():
        print(f"❌ Checkpoint not found: {model_file}")
        return

    print(f"\nLoading checkpoint: {model_file}")
    print(f"File size: {model_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 80)

    # Load state dict
    state_dict = load_file(str(model_file))

    print(f"\nFound {len(state_dict)} parameters in checkpoint")
    print("=" * 80)

    total_params = 0
    total_zeros = 0
    has_nan = False
    has_inf = False

    # Inspect each parameter
    for name, param in state_dict.items():
        # Convert to float for statistics
        if param.dtype == torch.bfloat16:
            param_float = param.float()
        else:
            param_float = param

        # Calculate statistics
        mean = param_float.mean().item()
        std = param_float.std().item()
        min_val = param_float.min().item()
        max_val = param_float.max().item()
        num_params = param.numel()
        num_zeros = (param_float == 0).sum().item()
        pct_nonzero = 100 * (1 - num_zeros / num_params)

        # Check for NaN/Inf
        has_param_nan = torch.isnan(param_float).any().item()
        has_param_inf = torch.isinf(param_float).any().item()
        has_nan = has_nan or has_param_nan
        has_inf = has_inf or has_param_inf

        # Print parameter info
        print(f"\n{name}")
        print(f"  Shape: {tuple(param.shape)}")
        print(f"  Dtype: {param.dtype}")
        print(f"  Count: {num_params:,}")
        print(f"  Mean: {mean:+.6f} | Std: {std:.6f}")
        print(f"  Min: {min_val:+.6f} | Max: {max_val:+.6f}")
        print(f"  Non-zero: {pct_nonzero:.2f}%")

        # Show sample values (first 10)
        sample_vals = param_float.flatten()[:10].tolist()
        sample_str = ", ".join(f"{v:+.4f}" for v in sample_vals)
        print(f"  Sample values: [{sample_str}{'...' if num_params > 10 else ''}]")

        # Warnings
        if has_param_nan:
            print("  ⚠️  WARNING: Contains NaN values!")
        if has_param_inf:
            print("  ⚠️  WARNING: Contains Inf values!")
        if abs(mean) < 1e-6 and std < 1e-6:
            print("  ⚠️  WARNING: Parameters appear to be zero or near-zero!")
        if pct_nonzero < 50:
            print(f"  ⚠️  WARNING: High sparsity ({100-pct_nonzero:.1f}% zeros)")

        total_params += num_params
        total_zeros += num_zeros

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Total zeros: {total_zeros:,} ({100 * total_zeros / total_params:.2f}%)")
    print(f"Non-zero: {total_params - total_zeros:,} ({100 * (total_params - total_zeros) / total_params:.2f}%)")

    # Health check
    print("\n" + "=" * 80)
    print("HEALTH CHECK")
    print("=" * 80)
    if has_nan:
        print("❌ FAILED: Checkpoint contains NaN values")
    elif has_inf:
        print("❌ FAILED: Checkpoint contains Inf values")
    elif total_zeros == total_params:
        print("❌ FAILED: All parameters are zero (training did not update weights)")
    elif total_zeros > 0.9 * total_params:
        print("⚠️  WARNING: >90% of parameters are zero (possible training issue)")
    else:
        print("✅ PASSED: Parameters have reasonable values")
        print("   - No NaN or Inf values")
        print("   - Parameters are non-zero")
        print("   - Training appears to have updated weights successfully")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # Default to latest test checkpoint
        checkpoint_path = "checkpoints/llava_pretrain_test/checkpoint-54"

    inspect_checkpoint(checkpoint_path)
