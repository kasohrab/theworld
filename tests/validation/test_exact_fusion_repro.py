"""Exact reproduction of fusion failure."""
import torch

print("="*80)
print("EXACT FUSION REPRODUCTION TEST")
print("="*80)

# Create EXACT same inputs as test
batch_size = 1
seq_len = 100
embed_dim = 2304
num_world_tokens = 784

dummy_gemma = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
dummy_world = torch.randn(batch_size, num_world_tokens, embed_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

print(f"\n[INPUTS]")
print(f"  gemma_embeds: requires_grad={dummy_gemma.requires_grad}, grad_fn={dummy_gemma.grad_fn}, is_leaf={dummy_gemma.is_leaf}")
print(f"  world_embeds: requires_grad={dummy_world.requires_grad}, grad_fn={dummy_world.grad_fn}, is_leaf={dummy_world.is_leaf}")

# Exact same slicing as fusion.py
start_pos = 10
end_pos = 11

embeddings_before = dummy_gemma[:, : start_pos + 1, :]
embeddings_after = dummy_gemma[:, end_pos:, :]

print(f"\n[AFTER SLICING]")
print(f"  embeddings_before: requires_grad={embeddings_before.requires_grad}, grad_fn={embeddings_before.grad_fn}, is_leaf={embeddings_before.is_leaf}")
print(f"  embeddings_after: requires_grad={embeddings_after.requires_grad}, grad_fn={embeddings_after.grad_fn}, is_leaf={embeddings_after.is_leaf}")
print(f"  world_embeds: requires_grad={dummy_world.requires_grad}, grad_fn={dummy_world.grad_fn}, is_leaf={dummy_world.is_leaf}")

# Exact same concatenation
combined_embeds = torch.cat([embeddings_before, dummy_world, embeddings_after], dim=1)

print(f"\n[AFTER CONCATENATION]")
print(f"  combined_embeds: requires_grad={combined_embeds.requires_grad}, grad_fn={combined_embeds.grad_fn}, is_leaf={combined_embeds.is_leaf}")

# Test backward
if combined_embeds.requires_grad:
    loss = combined_embeds.sum()
    loss.backward()
    print(f"\n[AFTER BACKWARD]")
    print(f"  dummy_gemma.grad is not None: {dummy_gemma.grad is not None}")
    print(f"  dummy_world.grad is not None: {dummy_world.grad is not None}")
    if dummy_gemma.grad is not None:
        print(f"  Gemma grad norm: {dummy_gemma.grad.norm().item():.4f}")
    if dummy_world.grad is not None:
        print(f"  World grad norm: {dummy_world.grad.norm().item():.4f}")
    print(f"\n✓ SUCCESS: Gradients flow correctly!")
else:
    print(f"\n✗ FAILURE: combined_embeds has no gradients!")

print("="*80)
