"""Test 4: Can Gemma language model + lm_head train by itself?

Tests if Gemma language model → lm_head can create and backpropagate gradients.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("TEST 4: Gemma Language Model + LM Head Training Test")
print("=" * 80)

# Load Gemma
print("\n[1/3] Loading Gemma model...")
from transformers import Gemma3ForConditionalGeneration

gemma = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    local_files_only=False,
)
gemma.train()  # Set to training mode

# Make all params trainable
for param in gemma.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in gemma.parameters() if p.requires_grad)
total = sum(p.numel() for p in gemma.parameters())
print(f"✓ Gemma loaded: {trainable:,} / {total:,} params trainable")

# Create test data
print("\n[2/3] Creating test data...")
target_device = gemma.get_input_embeddings().weight.device
hidden_size = gemma.config.text_config.hidden_size

dummy_embeds = torch.randn(1, 100, hidden_size, dtype=torch.bfloat16, device=target_device, requires_grad=True)
dummy_mask = torch.ones(1, 100, dtype=torch.long, device=target_device)

print(f"✓ Test data created")
print(f"  Embeddings shape: {dummy_embeds.shape}, requires_grad: {dummy_embeds.requires_grad}")

# Forward pass
print("\n[3/3] Testing forward + backward pass...")
gemma.zero_grad()

lm_output = gemma.language_model(inputs_embeds=dummy_embeds, attention_mask=dummy_mask, return_dict=True)
logits = gemma.lm_head(lm_output.last_hidden_state)

print(f"  LM hidden state shape: {lm_output.last_hidden_state.shape}")
print(f"  LM hidden state requires_grad: {lm_output.last_hidden_state.requires_grad}")
print(f"  Logits shape: {logits.shape}")
print(f"  Logits requires_grad: {logits.requires_grad}")
print(f"  Logits grad_fn: {logits.grad_fn}")

# Backward pass
if logits.requires_grad:
    loss = logits.sum()
    print(f"  Loss value: {loss.item():.4f}")
    loss.backward()

    # Check gradients
    lm_grads = sum(1 for p in gemma.parameters() if p.grad is not None and p.requires_grad)
    print(f"  ✓ Gemma params with grad: {lm_grads}")

    print("\n" + "=" * 80)
    print("RESULT: ✓ Gemma Language Model + LM Head CAN TRAIN!")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("RESULT: ✗ Gemma Language Model + LM Head output has NO GRADIENTS")
    print("=" * 80)
