"""Trace exactly where torch.is_grad_enabled() becomes False."""
import torch
import sys

# Monkey-patch torch.set_grad_enabled to trace calls
original_set_grad_enabled = torch.set_grad_enabled

def traced_set_grad_enabled(mode):
    import traceback
    print(f"\n{'='*80}")
    print(f"torch.set_grad_enabled({mode}) called!")
    print(f"{'='*80}")
    print("Call stack:")
    for line in traceback.format_stack()[:-1]:
        print(line.strip())
    print(f"{'='*80}\n")
    return original_set_grad_enabled(mode)

torch.set_grad_enabled = traced_set_grad_enabled

print("Starting trace...")
print(f"Initial state: torch.is_grad_enabled() = {torch.is_grad_enabled()}")

print("\nImporting Cosmos2VideoToWorldPipeline...")
from diffusers.pipelines.cosmos.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline

print(f"\nFinal state: torch.is_grad_enabled() = {torch.is_grad_enabled()}")
