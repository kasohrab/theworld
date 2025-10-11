"""Test 5: Full pipeline training test

Tests the complete TheWorld forward pass to identify where gradient flow breaks.
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
print("TEST 5: Full Pipeline Training Test")
print("=" * 80)

# Load full model
print("\n[1/4] Loading TheWorld model...")
from theworld import TheWorld, create_theworld_collator

model = TheWorld("google/gemma-3-4b-it", load_full_cosmos_pipeline=True)
model.train()

trainable, total, pct = model.get_trainable_parameters()
print(f"✓ TheWorld loaded: {trainable:,} / {total:,} params trainable ({pct:.2f}%)")

# Create test data
print("\n[2/4] Creating test data...")
dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
batch = [{"image": dummy_image, "text": "What is this?", "label": "A test image"}]

collator = create_theworld_collator(model)
batch_tensors = collator(batch)

print("✓ Test data created")

# Forward pass step by step
print("\n[3/4] Testing forward pass (step by step)...")

target_device = model.gemma.get_input_embeddings().weight.device
input_ids = batch_tensors["input_ids"].to(target_device)
pixel_values = batch_tensors["pixel_values"].to(target_device)
attention_mask = batch_tensors["attention_mask"].to(target_device)
images = batch_tensors["images"]

# Step 1: CosmosEncoder
print("\n  Step 1: CosmosEncoder")
world_embeds = model.cosmos_encoder(images=images)
print(f"    Output requires_grad: {world_embeds.requires_grad}")
print(f"    Output grad_fn: {world_embeds.grad_fn}")

# Step 2: GemmaVisionEncoder
print("\n  Step 2: GemmaVisionEncoder")
gemma_output = model.gemma_vision(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
print(f"    Output requires_grad: {gemma_output.embeddings.requires_grad}")
print(f"    Output grad_fn: {gemma_output.embeddings.grad_fn}")

# Step 3: EmbeddingFusion
print("\n  Step 3: EmbeddingFusion")
fusion_output = model.fusion(
    gemma_embeds=gemma_output.embeddings,
    world_embeds=world_embeds,
    input_ids=gemma_output.input_ids,
    attention_mask=gemma_output.attention_mask,
)
print(f"    Output requires_grad: {fusion_output.combined_embeds.requires_grad}")
print(f"    Output grad_fn: {fusion_output.combined_embeds.grad_fn}")

# Step 4: Language Model
print("\n  Step 4: Gemma Language Model")
lm_outputs = model.gemma.language_model(
    inputs_embeds=fusion_output.combined_embeds,
    attention_mask=fusion_output.combined_attention_mask,
    return_dict=True,
)
print(f"    Output requires_grad: {lm_outputs.last_hidden_state.requires_grad}")
print(f"    Output grad_fn: {lm_outputs.last_hidden_state.grad_fn}")

# Step 5: LM Head
print("\n  Step 5: LM Head")
logits = model.gemma.lm_head(lm_outputs.last_hidden_state)
print(f"    Output requires_grad: {logits.requires_grad}")
print(f"    Output grad_fn: {logits.grad_fn}")

# Compute loss
print("\n  Step 6: Compute Loss")
vocab_size = model.gemma.config.text_config.vocab_size
shift_logits = logits[..., :-1, :].contiguous().float()
dummy_labels = torch.zeros(shift_logits.size(0), shift_logits.size(1), dtype=torch.long, device=target_device)

loss_fct = torch.nn.CrossEntropyLoss()
loss = loss_fct(shift_logits.view(-1, vocab_size), dummy_labels.view(-1))

print(f"    Loss value: {loss.item():.4f}")
print(f"    Loss requires_grad: {loss.requires_grad}")
print(f"    Loss grad_fn: {loss.grad_fn}")

# Backward pass
print("\n[4/4] Testing backward pass...")
if loss.requires_grad:
    model.zero_grad()
    loss.backward()

    # Check which components got gradients
    proj_grad = model.cosmos_encoder.world_projection.weight.grad
    gemma_grads = sum(1 for p in model.gemma.parameters() if p.grad is not None and p.requires_grad)
    cosmos_grads = sum(1 for p in model.cosmos_pipe.vae.parameters() if p.grad is not None and p.requires_grad)

    print(f"  Projection layer grad: {proj_grad is not None}")
    if proj_grad is not None:
        print(f"    Grad norm: {proj_grad.norm().item():.4f}")
    print(f"  Gemma params with grad: {gemma_grads}")
    print(f"  Cosmos VAE params with grad: {cosmos_grads}")

    print("\n" + "=" * 80)
    print("RESULT: ✓ Full Pipeline CAN TRAIN!")
    print("=" * 80)
else:
    print("\n" + "=" * 80)
    print("RESULT: ✗ Full Pipeline loss has NO GRADIENTS")
    print("  → Check which step above shows requires_grad=False first")
    print("=" * 80)
