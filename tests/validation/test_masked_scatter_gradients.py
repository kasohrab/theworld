"""Quick test: Does masked_scatter preserve gradients?"""
import torch

print("Test 1: masked_scatter with both tensors having gradients")
a = torch.randn(1, 10, 100, requires_grad=True)
b = torch.randn(1, 10, 100, requires_grad=True)
mask = torch.randint(0, 2, (1, 10, 100), dtype=torch.bool)

c = a.masked_scatter(mask, b)

print(f"  Input a requires_grad: {a.requires_grad}, grad_fn: {a.grad_fn}")
print(f"  Input b requires_grad: {b.requires_grad}, grad_fn: {b.grad_fn}")
print(f"  Output c requires_grad: {c.requires_grad}, grad_fn: {c.grad_fn}")

if c.requires_grad:
    loss = c.sum()
    loss.backward()
    print(f"  ✓ a.grad is not None: {a.grad is not None}")
    print(f"  ✓ b.grad is not None: {b.grad is not None}")
else:
    print(f"  ✗ Output has no gradients!")

print("\nTest 2: masked_scatter on CUDA with bfloat16")
a = torch.randn(1, 10, 100, dtype=torch.bfloat16, device="cuda", requires_grad=True)
b = torch.randn(1, 10, 100, dtype=torch.bfloat16, device="cuda", requires_grad=True)
mask = torch.randint(0, 2, (1, 10, 100), dtype=torch.bool, device="cuda")

c = a.masked_scatter(mask, b)

print(f"  Input a requires_grad: {a.requires_grad}, grad_fn: {a.grad_fn}")
print(f"  Input b requires_grad: {b.requires_grad}, grad_fn: {b.grad_fn}")
print(f"  Output c requires_grad: {c.requires_grad}, grad_fn: {c.grad_fn}")

if c.requires_grad:
    loss = c.sum()
    loss.backward()
    print(f"  ✓ a.grad is not None: {a.grad is not None}")
    print(f"  ✓ b.grad is not None: {b.grad is not None}")
else:
    print(f"  ✗ Output has no gradients!")
