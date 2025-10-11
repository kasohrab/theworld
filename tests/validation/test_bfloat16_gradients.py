"""Test: Does bfloat16 affect gradient flow?"""
import torch

print("Test 1: float32 leaf tensors")
a = torch.randn(1, 10, 100, dtype=torch.float32, requires_grad=True)
b = torch.randn(1, 20, 100, dtype=torch.float32, requires_grad=True)
c = torch.cat([a, b], dim=1)
print(f"  Output c: requires_grad={c.requires_grad}, grad_fn={c.grad_fn}")

print("\nTest 2: bfloat16 leaf tensors on CPU")
a = torch.randn(1, 10, 100, dtype=torch.bfloat16, requires_grad=True)
b = torch.randn(1, 20, 100, dtype=torch.bfloat16, requires_grad=True)
c = torch.cat([a, b], dim=1)
print(f"  Output c: requires_grad={c.requires_grad}, grad_fn={c.grad_fn}")

print("\nTest 3: bfloat16 leaf tensors on CUDA")
a = torch.randn(1, 10, 100, dtype=torch.bfloat16, device="cuda", requires_grad=True)
b = torch.randn(1, 20, 100, dtype=torch.bfloat16, device="cuda", requires_grad=True)
c = torch.cat([a, b], dim=1)
print(f"  Output c: requires_grad={c.requires_grad}, grad_fn={c.grad_fn}")

print("\nTest 4: bfloat16 slicing on CUDA")
a = torch.randn(1, 100, 100, dtype=torch.bfloat16, device="cuda", requires_grad=True)
b = a[:, :50, :]
print(f"  Input a: requires_grad={a.requires_grad}, grad_fn={a.grad_fn}, is_leaf={a.is_leaf}")
print(f"  Slice b: requires_grad={b.requires_grad}, grad_fn={b.grad_fn}, is_leaf={b.is_leaf}")

print("\nTest 5: EXACT fusion scenario - bfloat16 slicing + cat")
gemma = torch.randn(1, 100, 2304, dtype=torch.bfloat16, device="cuda", requires_grad=True)
world = torch.randn(1, 784, 2304, dtype=torch.bfloat16, device="cuda", requires_grad=True)

start_pos = 10
end_pos = 11

before = gemma[:, :start_pos+1, :]
after = gemma[:, end_pos:, :]

print(f"  before: requires_grad={before.requires_grad}, grad_fn={before.grad_fn}")
print(f"  after: requires_grad={after.requires_grad}, grad_fn={after.grad_fn}")
print(f"  world: requires_grad={world.requires_grad}, grad_fn={world.grad_fn}")

combined = torch.cat([before, world, after], dim=1)
print(f"  combined: requires_grad={combined.requires_grad}, grad_fn={combined.grad_fn}")

print("\nConclusion: Testing if bfloat16 + CUDA affects gradient tracking")
