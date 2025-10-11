"""Detailed debugging of gradient flow."""

import torch
from PIL import Image
import numpy as np

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld, create_theworld_collator

# Create model
print("Loading model...")
model = TheWorld("google/gemma-3-4b-it", load_full_cosmos_pipeline=True)
model.train()  # IMPORTANT: Must be in training mode for gradient tracking!

# Check that model is in training mode
print(f"\n=== Training Mode Check ===")
print(f"model.training: {model.training}")
print(f"gemma.training: {model.gemma.training}")
print(f"cosmos_vae.training: {model.cosmos_vae.training}")

# Create dummy data
dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
batch = [
    {
        "image": dummy_image,
        "text": "What is this?",
        "label": "A test image"
    }
]

# Create collator and process batch
collator = create_theworld_collator(model)
batch_tensors = collator(batch)

print("\n=== Testing gradient flow at each step ===")

# Step 1: Check cosmos encoder
print("\n1. CosmosEncoder")
world_embeds = model.cosmos_encoder(images=batch_tensors["images"])
print(f"   world_embeds shape: {world_embeds.shape}")
print(f"   world_embeds requires_grad: {world_embeds.requires_grad}")
print(f"   world_embeds grad_fn: {world_embeds.grad_fn}")

# Step 2: Check gemma vision encoder
print("\n2. GemmaVisionEncoder")
target_device = model.gemma.get_input_embeddings().weight.device
input_ids = batch_tensors["input_ids"].to(target_device)
pixel_values = batch_tensors["pixel_values"].to(target_device)
attention_mask = batch_tensors["attention_mask"].to(target_device)

gemma_output = model.gemma_vision(
    input_ids=input_ids,
    pixel_values=pixel_values,
    attention_mask=attention_mask,
)
print(f"   gemma_output.embeddings shape: {gemma_output.embeddings.shape}")
print(f"   gemma_output.embeddings requires_grad: {gemma_output.embeddings.requires_grad}")
print(f"   gemma_output.embeddings grad_fn: {gemma_output.embeddings.grad_fn}")

# Step 3: Check fusion
print("\n3. EmbeddingFusion")
fusion_output = model.fusion(
    gemma_embeds=gemma_output.embeddings,
    world_embeds=world_embeds,
    input_ids=gemma_output.input_ids,
    attention_mask=gemma_output.attention_mask,
)
print(f"   fusion_output.combined_embeds shape: {fusion_output.combined_embeds.shape}")
print(f"   fusion_output.combined_embeds requires_grad: {fusion_output.combined_embeds.requires_grad}")
print(f"   fusion_output.combined_embeds grad_fn: {fusion_output.combined_embeds.grad_fn}")

# Step 4: Check language model
print("\n4. Gemma Language Model")
lm_outputs = model.gemma.language_model(
    inputs_embeds=fusion_output.combined_embeds,
    attention_mask=fusion_output.combined_attention_mask,
    return_dict=True,
)
print(f"   lm_outputs.last_hidden_state shape: {lm_outputs.last_hidden_state.shape}")
print(f"   lm_outputs.last_hidden_state requires_grad: {lm_outputs.last_hidden_state.requires_grad}")
print(f"   lm_outputs.last_hidden_state grad_fn: {lm_outputs.last_hidden_state.grad_fn}")

# Step 5: Check LM head
print("\n5. LM Head")
logits = model.gemma.lm_head(lm_outputs.last_hidden_state)
print(f"   logits shape: {logits.shape}")
print(f"   logits requires_grad: {logits.requires_grad}")
print(f"   logits grad_fn: {logits.grad_fn}")

# Step 6: Check loss computation
print("\n6. Loss Computation")
labels = batch_tensors["labels"]
print(f"   labels shape: {labels.shape if labels is not None else None}")

# Simplified loss (just to test)
vocab_size = model.gemma.config.text_config.vocab_size
shift_logits = logits[..., :-1, :].contiguous()
print(f"   shift_logits requires_grad: {shift_logits.requires_grad}")
print(f"   shift_logits grad_fn: {shift_logits.grad_fn}")

# Try computing a simple loss
if labels is not None:
    loss_fct = torch.nn.CrossEntropyLoss()
    # Create dummy labels that match logits shape
    dummy_labels = torch.zeros(shift_logits.size(0), shift_logits.size(1), dtype=torch.long, device=shift_logits.device)
    simple_loss = loss_fct(shift_logits.view(-1, vocab_size), dummy_labels.view(-1))
    print(f"   simple_loss: {simple_loss.item()}")
    print(f"   simple_loss requires_grad: {simple_loss.requires_grad}")
    print(f"   simple_loss grad_fn: {simple_loss.grad_fn}")

# Step 7: Check projection layer gradients
print("\n7. Projection Layer Parameters")
proj_weight = model.cosmos_encoder.world_projection.weight
print(f"   projection.weight requires_grad: {proj_weight.requires_grad}")
print(f"   projection.weight grad: {proj_weight.grad}")

print("\n=== Summary ===")
print("If any step shows requires_grad=False or grad_fn=None, that's where the graph breaks!")

print("\n" + "=" * 80)
print("INDIVIDUAL COMPONENT TRAINING TESTS")
print("=" * 80)

# Test 1: Can CosmosEncoder train by itself?
print("\n### TEST 1: CosmosEncoder alone ###")
try:
    model.cosmos_encoder.zero_grad()
    world_embeds_test = model.cosmos_encoder(images=images)
    print(f"  Output requires_grad: {world_embeds_test.requires_grad}")
    print(f"  Output grad_fn: {world_embeds_test.grad_fn}")

    if world_embeds_test.requires_grad:
        loss_test = world_embeds_test.sum()
        loss_test.backward()
        proj_grad = model.cosmos_encoder.world_projection.weight.grad
        if proj_grad is not None:
            print(f"  ✓ CosmosEncoder CAN train! Grad norm: {proj_grad.norm().item():.4f}")
        else:
            print(f"  ✗ CosmosEncoder forward works but backward failed")
    else:
        print(f"  ✗ CosmosEncoder output has no gradients")
except Exception as e:
    print(f"  ✗ CosmosEncoder test failed: {e}")

# Test 2: Can GemmaVisionEncoder train by itself?
print("\n### TEST 2: GemmaVisionEncoder alone ###")
try:
    model.gemma_vision.zero_grad()
    gemma_output_test = model.gemma_vision(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
    )
    print(f"  Output requires_grad: {gemma_output_test.embeddings.requires_grad}")
    print(f"  Output grad_fn: {gemma_output_test.embeddings.grad_fn}")

    if gemma_output_test.embeddings.requires_grad:
        loss_test = gemma_output_test.embeddings.sum()
        loss_test.backward()
        # Check if any Gemma params got gradients
        gemma_grads = sum(1 for p in model.gemma.parameters() if p.grad is not None and p.requires_grad)
        print(f"  ✓ GemmaVisionEncoder forward works. Gemma params with grad: {gemma_grads}")
    else:
        print(f"  ✗ GemmaVisionEncoder output has no gradients")
except Exception as e:
    print(f"  ✗ GemmaVisionEncoder test failed: {e}")

# Test 3: Can EmbeddingFusion train by itself?
print("\n### TEST 3: EmbeddingFusion alone ###")
try:
    # Create dummy inputs with requires_grad
    dummy_gemma = torch.randn_like(gemma_output.embeddings, requires_grad=True)
    dummy_world = torch.randn_like(world_embeds, requires_grad=True)

    fusion_test = model.fusion(
        gemma_embeds=dummy_gemma,
        world_embeds=dummy_world,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    print(f"  Output requires_grad: {fusion_test.combined_embeds.requires_grad}")
    print(f"  Output grad_fn: {fusion_test.combined_embeds.grad_fn}")

    if fusion_test.combined_embeds.requires_grad:
        loss_test = fusion_test.combined_embeds.sum()
        loss_test.backward()
        print(f"  ✓ EmbeddingFusion CAN train!")
        print(f"  Dummy gemma grad: {dummy_gemma.grad is not None}")
        print(f"  Dummy world grad: {dummy_world.grad is not None}")
    else:
        print(f"  ✗ EmbeddingFusion output has no gradients")
except Exception as e:
    print(f"  ✗ EmbeddingFusion test failed: {e}")

# Test 4: Can Gemma language model + lm_head train by itself?
print("\n### TEST 4: Gemma language model + lm_head alone ###")
try:
    model.gemma.zero_grad()
    # Create dummy embeddings with requires_grad
    dummy_embeds = torch.randn(1, 100, model.gemma.config.text_config.hidden_size,
                                dtype=torch.bfloat16, device=target_device, requires_grad=True)
    dummy_mask = torch.ones(1, 100, dtype=torch.long, device=target_device)

    lm_out = model.gemma.language_model(inputs_embeds=dummy_embeds, attention_mask=dummy_mask, return_dict=True)
    logits_test = model.gemma.lm_head(lm_out.last_hidden_state)

    print(f"  LM hidden state requires_grad: {lm_out.last_hidden_state.requires_grad}")
    print(f"  Logits requires_grad: {logits_test.requires_grad}")
    print(f"  Logits grad_fn: {logits_test.grad_fn}")

    if logits_test.requires_grad:
        loss_test = logits_test.sum()
        loss_test.backward()
        lm_grads = sum(1 for p in model.gemma.parameters() if p.grad is not None and p.requires_grad)
        print(f"  ✓ Gemma LM+head CAN train! Params with grad: {lm_grads}")
    else:
        print(f"  ✗ Gemma LM+head output has no gradients")
except Exception as e:
    print(f"  ✗ Gemma LM+head test failed: {e}")

print("\n" + "=" * 80)
print("CONCLUSION: Check which components can train individually vs combined")
print("=" * 80)
