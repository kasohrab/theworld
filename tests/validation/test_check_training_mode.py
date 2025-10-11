"""Check if model components are in training mode."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from theworld import TheWorld

print("Loading model...")
model = TheWorld("google/gemma-3-4b-it", load_full_cosmos_pipeline=True)

print("\n=== Checking training mode ===")
print(f"model.training: {model.training}")
print(f"gemma.training: {model.gemma.training}")
print(f"cosmos_vae.training: {model.cosmos_vae.training}")
print(f"cosmos_encoder.training: {model.cosmos_encoder.training}")
print(f"gemma_vision.training: {model.gemma_vision.training}")
print(f"fusion.training: {model.fusion.training}")

print("\n=== Setting to train mode ===")
model.train()

print(f"model.training: {model.training}")
print(f"gemma.training: {model.gemma.training}")
print(f"cosmos_vae.training: {model.cosmos_vae.training}")
print(f"cosmos_encoder.training: {model.cosmos_encoder.training}")
print(f"gemma_vision.training: {model.gemma_vision.training}")
print(f"fusion.training: {model.fusion.training}")

print("\n=== Checking if torch.is_grad_enabled() ===")
import torch
print(f"torch.is_grad_enabled(): {torch.is_grad_enabled()}")
