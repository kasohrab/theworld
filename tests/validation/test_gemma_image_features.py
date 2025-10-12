"""Test if Gemma get_image_features preserves gradients."""
import torch
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
from PIL import Image
import numpy as np

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

processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it", local_files_only=False)

# Create test image
test_image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": test_image},
            {"type": "text", "text": "What is this?"},
        ],
    }
]

inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
pixel_values = inputs["pixel_values"]

target_device = gemma.get_input_embeddings().weight.device
pixel_values = pixel_values.to(target_device)

print(f"Pixel values requires_grad: {pixel_values.requires_grad}")

# Get image features
image_features = gemma.model.get_image_features(pixel_values)

print(f"Image features requires_grad: {image_features.requires_grad}")
print(f"Image features grad_fn: {image_features.grad_fn}")

if image_features.requires_grad:
    loss = image_features.sum()
    loss.backward()
    print(f"✓ Backward passed!")
else:
    print(f"✗ Image features have no gradients!")
