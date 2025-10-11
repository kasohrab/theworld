"""Test 2: Can GemmaVisionEncoder train by itself?

Tests if Gemma SigLIP encoder → embedding fusion can create and backpropagate gradients.
"""

import torch
from PIL import Image
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("TEST 2: GemmaVisionEncoder Training Test")
print("=" * 80)

# Load Gemma
print("\n[1/4] Loading Gemma model...")
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

gemma = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
    local_files_only=False,
)
gemma.train()  # Set to training mode
print(f"✓ Gemma loaded")

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it", local_files_only=False)

# Create GemmaVisionEncoder
print("\n[2/4] Creating GemmaVisionEncoder...")
from theworld.modeling.gemma_vision import GemmaVisionEncoder

# CRITICAL FIX: Importing theworld.modeling triggers Cosmos import which disables gradients
torch.set_grad_enabled(True)

gemma_vision = GemmaVisionEncoder(gemma_model=gemma)
gemma_vision.train()

# Make all Gemma params trainable
for param in gemma.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in gemma.parameters() if p.requires_grad)
total = sum(p.numel() for p in gemma.parameters())
print(f"✓ GemmaVisionEncoder created: {trainable:,} / {total:,} params trainable")

# Create test data
print("\n[3/4] Creating test data...")
dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": dummy_image},
            {"type": "text", "text": "What is this?"},
        ],
    }
]

inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
input_ids = inputs["input_ids"]
pixel_values = inputs["pixel_values"]
attention_mask = inputs["attention_mask"]

target_device = gemma.get_input_embeddings().weight.device
input_ids = input_ids.to(target_device)
pixel_values = pixel_values.to(target_device)
attention_mask = attention_mask.to(target_device)

print(f"✓ Test data created")
print(f"  input_ids shape: {input_ids.shape}")
print(f"  pixel_values shape: {pixel_values.shape}")

# Forward pass
print("\n[4/4] Testing forward + backward pass...")
print(f"  torch.is_grad_enabled(): {torch.is_grad_enabled()}")
gemma.zero_grad()
output = gemma_vision(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)

print(f"  Output shape: {output.embeddings.shape}")
print(f"  Output requires_grad: {output.embeddings.requires_grad}")
print(f"  Output grad_fn: {output.embeddings.grad_fn}")

# Backward pass
if output.embeddings.requires_grad:
    loss = output.embeddings.sum()
    print(f"  Loss value: {loss.item():.4f}")
    loss.backward()

    # Check gradients
    gemma_grads = sum(1 for p in gemma.parameters() if p.grad is not None and p.requires_grad)
    print(f"  ✓ Gemma params with grad: {gemma_grads}")

    print("\n" + "=" * 80)
    print("RESULT: ✓ GemmaVisionEncoder CAN TRAIN!")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("RESULT: ✗ GemmaVisionEncoder output has NO GRADIENTS")
    print("=" * 80)
