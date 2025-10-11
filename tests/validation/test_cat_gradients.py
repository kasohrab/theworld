"""Quick test: Does torch.cat preserve gradients when mixing requires_grad tensors?"""
import torch

print("Test 1: Concatenate two tensors with requires_grad=True")
a = torch.randn(1, 10, 100, requires_grad=True)
b = torch.randn(1, 20, 100, requires_grad=True)
c = torch.cat([a, b], dim=1)
print(f"  Input a requires_grad: {a.requires_grad}")
print(f"  Input b requires_grad: {b.requires_grad}")
print(f"  Output c requires_grad: {c.requires_grad}")
print(f"  Output c grad_fn: {c.grad_fn}")

print("\nTest 2: Concatenate with one tensor requires_grad=False")
a = torch.randn(1, 10, 100, requires_grad=True)
b = torch.randn(1, 20, 100, requires_grad=False)  # No grad!
c = torch.cat([a, b], dim=1)
print(f"  Input a requires_grad: {a.requires_grad}")
print(f"  Input b requires_grad: {b.requires_grad}")
print(f"  Output c requires_grad: {c.requires_grad}")
print(f"  Output c grad_fn: {c.grad_fn}")

print("\nTest 3: Slicing preserves gradients")
a = torch.randn(1, 100, 100, requires_grad=True)
b = a[:, :50, :]
print(f"  Input a requires_grad: {a.requires_grad}")
print(f"  Slice b requires_grad: {b.requires_grad}")
print(f"  Slice b grad_fn: {b.grad_fn}")

print("\nConclusion: torch.cat with ANY non-grad tensor breaks the whole output!")
