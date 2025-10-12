"""Quick test: Does torch.cat work with leaf tensors vs non-leaf tensors?"""
import torch

print("Test 1: Concatenate TWO leaf tensors")
a = torch.randn(1, 10, 100, requires_grad=True)  # Leaf tensor
b = torch.randn(1, 20, 100, requires_grad=True)  # Leaf tensor
c = torch.cat([a, b], dim=1)
print(f"  Input a: requires_grad={a.requires_grad}, grad_fn={a.grad_fn}, is_leaf={a.is_leaf}")
print(f"  Input b: requires_grad={b.requires_grad}, grad_fn={b.grad_fn}, is_leaf={b.is_leaf}")
print(f"  Output c: requires_grad={c.requires_grad}, grad_fn={c.grad_fn}, is_leaf={c.is_leaf}")

print("\nTest 2: Concatenate leaf + non-leaf (with operation)")
a = torch.randn(1, 10, 100, requires_grad=True)  # Leaf
b = torch.randn(1, 20, 100, requires_grad=True) + 0  # Non-leaf (has grad_fn)
c = torch.cat([a, b], dim=1)
print(f"  Input a: requires_grad={a.requires_grad}, grad_fn={a.grad_fn}, is_leaf={a.is_leaf}")
print(f"  Input b: requires_grad={b.requires_grad}, grad_fn={b.grad_fn}, is_leaf={b.is_leaf}")
print(f"  Output c: requires_grad={c.requires_grad}, grad_fn={c.grad_fn}, is_leaf={c.is_leaf}")

print("\nTest 3: Slicing a leaf tensor")
a = torch.randn(1, 100, 100, requires_grad=True)  # Leaf
b = a[:, :50, :]  # Slice
print(f"  Input a: requires_grad={a.requires_grad}, grad_fn={a.grad_fn}, is_leaf={a.is_leaf}")
print(f"  Slice b: requires_grad={b.requires_grad}, grad_fn={b.grad_fn}, is_leaf={b.is_leaf}")

print("\nTest 4: Slicing a non-leaf tensor")
a = torch.randn(1, 100, 100, requires_grad=True) + 0  # Non-leaf
b = a[:, :50, :]  # Slice
print(f"  Input a: requires_grad={a.requires_grad}, grad_fn={a.grad_fn}, is_leaf={a.is_leaf}")
print(f"  Slice b: requires_grad={b.requires_grad}, grad_fn={b.grad_fn}, is_leaf={b.is_leaf}")

print("\n" + "="*80)
print("Hypothesis: torch.cat with ALL leaf tensors might not create grad_fn?")
print("="*80)
