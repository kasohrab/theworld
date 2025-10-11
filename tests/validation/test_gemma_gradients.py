"""Quick test: Does Gemma embedding layer preserve gradients?"""
import torch
from transformers import Gemma3ForConditionalGeneration

print("Loading Gemma...")
gemma = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    local_files_only=False,
)
gemma.train()

# Make all params trainable
for param in gemma.parameters():
    param.requires_grad = True

print(f"Embedding layer requires_grad: {gemma.model.language_model.embed_tokens.weight.requires_grad}")

# Test embedding
input_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda:0")
embeds = gemma.model.language_model.embed_tokens(input_ids)

print(f"Embeds requires_grad: {embeds.requires_grad}")
print(f"Embeds grad_fn: {embeds.grad_fn}")

# Try backward
if embeds.requires_grad:
    loss = embeds.sum()
    loss.backward()
    print(f"✓ Embedding weight has grad: {gemma.model.language_model.embed_tokens.weight.grad is not None}")
else:
    print("✗ Embeds don't have gradients!")
