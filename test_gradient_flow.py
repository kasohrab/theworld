"""Test that gradients flow through the projection layer during training."""

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
model = TheWorld("google/gemma-3-4b-it")
model.eval()  # Disable dropout for deterministic test

# Get initial projection weights
initial_weights = model.cosmos_encoder.world_projection.weight.clone()
print(f"Initial projection weights sum: {initial_weights.sum().item()}")

# Check which parameters require grad
trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
print(f"\nTrainable parameters ({len(trainable_params)}):")
for name in trainable_params[:10]:  # Show first 10
    print(f"  - {name}")

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

print(f"\nBatch keys: {batch_tensors.keys()}")
print(f"Images type: {type(batch_tensors['images'][0])}")
print(f"Labels shape: {batch_tensors['labels'].shape if batch_tensors['labels'] is not None else None}")

# Forward pass
print("\nRunning forward pass...")
outputs = model(
    input_ids=batch_tensors["input_ids"],
    pixel_values=batch_tensors["pixel_values"],
    attention_mask=batch_tensors["attention_mask"],
    images=batch_tensors["images"],
    labels=batch_tensors["labels"],
)

print(f"Loss: {outputs.loss}")
print(f"Loss requires_grad: {outputs.loss.requires_grad if outputs.loss is not None else None}")
print(f"Loss grad_fn: {outputs.loss.grad_fn if outputs.loss is not None else None}")

# Try backward pass
if outputs.loss is not None and outputs.loss.requires_grad:
    print("\nRunning backward pass...")
    outputs.loss.backward()

    # Check if projection layer received gradients
    proj_grad = model.cosmos_encoder.world_projection.weight.grad
    if proj_grad is not None:
        print(f"✓ Projection layer received gradients!")
        print(f"  Gradient norm: {proj_grad.norm().item()}")
    else:
        print(f"✗ Projection layer did NOT receive gradients")
else:
    print(f"\n✗ Loss does not require grad - cannot backprop")
