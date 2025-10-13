# Inference Guide

Learn how to run inference with TheWorld to generate responses to image-based questions.

## Quick Example

```python
from theworld import TheWorld
from transformers import AutoProcessor
from PIL import Image

# 1. Load model
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Load processor
processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

# 3. Prepare input
image = Image.open("cat.jpg")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "What is in this image?"}
    ]
}]

# 4. Run inference
inputs = processor.apply_chat_template(messages, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print(response)
```

## Loading Models

### From HuggingFace Hub

```python
# Load pretrained TheWorld from Hub
model = TheWorld.from_pretrained(
    "username/theworld-datacomp",
    dtype=torch.bfloat16,
    device_map="auto"
)
```

### From Local Checkpoint

```python
# Load from local checkpoint
model = TheWorld.from_pretrained(
    "./checkpoints/checkpoint-1000",
    dtype=torch.bfloat16,
    device_map="auto"
)
```

### Base Model (No Training)

```python
# Start from base Gemma + Cosmos
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

## Controlling World Model

### Temporal Rollout

```python
# No future prediction (single frame)
outputs = model.generate(**inputs, num_world_steps=0)

# Predict 4 future frames
outputs = model.generate(**inputs, num_world_steps=4)
```

**When to use more world steps:**
- Temporal reasoning tasks
- Motion understanding
- Physical dynamics questions

**When to use fewer:**
- Static image understanding
- Fast inference needed
- Limited VRAM

### Disabling World Model

```python
# Completely skip world model (Gemma-only mode)
model = TheWorld.from_pretrained(
    "username/theworld-datacomp",
    enable_world=False,  # ← Disable world
    dtype=torch.bfloat16,
    device_map="auto"
)
```

## Generation Parameters

### Basic Parameters

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,      # Maximum length
    temperature=0.7,          # Sampling temperature (lower = more deterministic)
    top_p=0.9,               # Nucleus sampling
    do_sample=True,          # Enable sampling
)
```

### Deterministic Generation

```python
# Greedy decoding (most likely token at each step)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,         # Disable sampling
)
```

### Beam Search

```python
# Better quality, slower
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    num_beams=5,             # Number of beams
    do_sample=False,
)
```

## Batch Inference

```python
# Process multiple images at once
images = [Image.open(f"img{i}.jpg") for i in range(4)]
prompts = ["Describe this image."] * 4

messages_list = [
    [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}
        ]
    }]
    for img, prompt in zip(images, prompts)
]

# Process batch
inputs = processor.apply_chat_template(
    messages_list,
    return_tensors="pt",
    padding=True
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=100)
responses = processor.batch_decode(outputs, skip_special_tokens=True)
```

## Memory Management

### Reducing Memory Usage

```python
# Use bfloat16 (half precision)
model = TheWorld.from_pretrained(
    "username/theworld-datacomp",
    dtype=torch.bfloat16,  # ← Saves ~50% memory
    device_map="auto"
)

# Clear cache between generations
import torch
torch.cuda.empty_cache()
```

### Multi-GPU Inference

```python
# Automatic device mapping across GPUs
model = TheWorld.from_pretrained(
    "username/theworld-datacomp",
    dtype=torch.bfloat16,
    device_map="auto"  # ← Distributes across GPUs
)
```

## Example Use Cases

### Spatial Reasoning

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Is the cup to the left of the laptop?"}
    ]
}]

inputs = processor.apply_chat_template(messages, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=10)
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
# Expected: "Yes" or "No"
```

### Depth Perception

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Which object is closer: the person or the tree?"}
    ]
}]

# Use world model for better depth understanding
inputs = processor.apply_chat_template(messages, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=20, num_world_steps=0)
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

### Visual Question Answering

```python
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "How many people are in this image?"}
    ]
}]

inputs = processor.apply_chat_template(messages, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

outputs = model.generate(**inputs, max_new_tokens=10)
response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

## Performance Tips

1. **Use bfloat16**: Saves memory with minimal accuracy loss
2. **Batch inference**: Process multiple images together
3. **Fewer world steps**: Start with 0, increase if needed
4. **Cache models**: Models download once, then cached locally
5. **device_map="auto"**: Automatic multi-GPU distribution

## Common Patterns

### Loading from HuggingFace Hub

```python
from theworld import TheWorld

model = TheWorld.from_checkpoint_hub(
    "username/theworld-datacomp",
    checkpoint_name="checkpoint-1000/pytorch_model.bin"  # Optional
)
```

### Comparing World Steps

```python
# Compare different temporal rollouts
for num_steps in [0, 2, 4, 8]:
    outputs = model.generate(**inputs, num_world_steps=num_steps)
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Steps={num_steps}: {response}")
```

## Next Steps

- [Evaluation Guide](../evaluation/overview.md) - Evaluate model performance
- [Training Guide](../training/README.md) - Train your own model
- [Architecture Overview](../architecture/overview.md) - Understand how it works

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues:
- CUDA out of memory
- Slow inference
- Incorrect outputs
- Model loading errors
