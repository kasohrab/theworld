"""Test if wrapping in nn.Module affects gradient flow."""
import torch
import torch.nn as nn

class SimpleFusion(nn.Module):
    """Minimal fusion module."""

    def __init__(self):
        super().__init__()

    def forward(self, gemma_embeds, world_embeds):
        """Fuse embeddings."""
        print(f"[ENTRY] gemma_embeds requires_grad: {gemma_embeds.requires_grad}, grad_fn: {gemma_embeds.grad_fn}")
        print(f"[ENTRY] world_embeds requires_grad: {world_embeds.requires_grad}, grad_fn: {world_embeds.grad_fn}")

        # Slice
        start_pos = 10
        end_pos = 11
        embeddings_before = gemma_embeds[:, :start_pos + 1, :]
        embeddings_after = gemma_embeds[:, end_pos:, :]

        print(f"[AFTER SLICE] embeddings_before requires_grad: {embeddings_before.requires_grad}, grad_fn: {embeddings_before.grad_fn}")
        print(f"[AFTER SLICE] embeddings_after requires_grad: {embeddings_after.requires_grad}, grad_fn: {embeddings_after.grad_fn}")

        # Concatenate
        combined = torch.cat([embeddings_before, world_embeds, embeddings_after], dim=1)

        print(f"[AFTER CAT] combined requires_grad: {combined.requires_grad}, grad_fn: {combined.grad_fn}")

        return combined

print("="*80)
print("MODULE WRAPPER TEST")
print("="*80)

# Create module
fusion = SimpleFusion()
fusion.train()  # Important: set to train mode

# Create inputs
batch_size = 1
seq_len = 100
embed_dim = 2304
num_world_tokens = 784

dummy_gemma = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)
dummy_world = torch.randn(batch_size, num_world_tokens, embed_dim, dtype=torch.bfloat16, device="cuda", requires_grad=True)

print(f"\n[BEFORE MODULE]")
print(f"  dummy_gemma requires_grad: {dummy_gemma.requires_grad}")
print(f"  dummy_world requires_grad: {dummy_world.requires_grad}")

# Forward pass
print(f"\n[DURING FORWARD]")
output = fusion(dummy_gemma, dummy_world)

print(f"\n[AFTER MODULE]")
print(f"  output requires_grad: {output.requires_grad}, grad_fn: {output.grad_fn}")

# Test backward
if output.requires_grad:
    loss = output.sum()
    loss.backward()
    print(f"\n[AFTER BACKWARD]")
    print(f"  dummy_gemma.grad is not None: {dummy_gemma.grad is not None}")
    print(f"  dummy_world.grad is not None: {dummy_world.grad is not None}")
    print(f"\n✓ SUCCESS: Module wrapper preserves gradients!")
else:
    print(f"\n✗ FAILURE: Module wrapper breaks gradients!")

print("="*80)
