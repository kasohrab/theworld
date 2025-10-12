"""Minimal test: Can a Linear layer with trainable params create gradients from non-grad input?"""

import torch
import torch.nn as nn

print("=" * 60)
print("Testing: Linear layer gradient creation")
print("=" * 60)

# Create a simple linear layer with trainable parameters
projection = nn.Linear(16, 2304, dtype=torch.bfloat16).cuda()
print(f"Projection weight requires_grad: {projection.weight.requires_grad}")

# Create input WITHOUT requires_grad (simulating frozen upstream)
input_tensor = torch.randn(1, 100, 16, dtype=torch.bfloat16, device='cuda')
print(f"\nInput requires_grad: {input_tensor.requires_grad}")

# Forward pass
output = projection(input_tensor)
print(f"Output requires_grad: {output.requires_grad}")
print(f"Output grad_fn: {output.grad_fn}")

# Try to compute loss and backward
if output.requires_grad:
    loss = output.sum()
    print(f"\nLoss requires_grad: {loss.requires_grad}")
    print("Attempting backward...")
    loss.backward()
    print(f"✓ Backward successful!")
    print(f"Projection grad norm: {projection.weight.grad.norm().item()}")
else:
    print("\n✗ Output doesn't require grad - this is THE PROBLEM!")
    print("\nTrying fix: output.requires_grad_(True)")
    output_fixed = output.requires_grad_(True)
    loss = output_fixed.sum()
    print(f"Loss requires_grad: {loss.requires_grad}")
    loss.backward()
    print(f"✓ Backward successful with fix!")
    print(f"Projection grad norm: {projection.weight.grad.norm().item()}")
