"""Simple inference example using TheWorld with AutoConfig."""

import torch
from PIL import Image
from theworld import TheWorld


def main():
    """Demonstrate basic inference with TheWorld model."""

    # 1. Load model using AutoConfig pattern
    print("Loading TheWorld model...")
    model = TheWorld.from_pretrained(
        "google/gemma-3-4b-it",
        enable_world=True,  # Enable Cosmos world model
        device="cuda",
        freeze_gemma_vision=True,
        freeze_gemma_language=True,
        freeze_cosmos_vae=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"✓ Model loaded")

    # 2. Prepare input
    print("\nPreparing input...")
    image = Image.new("RGB", (512, 512), color=(100, 150, 200))  # Dummy image

    # Use Gemma's chat template format
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": "What is in this image?"}],
        }
    ]

    # Process with Gemma processor
    inputs = model.processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")

    # Move to device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Add PIL image for Cosmos
    inputs["images"] = [image]

    # 3. Generate response
    print("\nGenerating response...")
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,  # Greedy decoding
    )

    # 4. Decode and display
    # Skip the prompt tokens (only show generated text)
    prompt_length = inputs["input_ids"].shape[1]
    generated_text = model.processor.tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True)

    print(f"✓ Generated {len(generated_ids[0]) - prompt_length} tokens")
    print(f"\nGenerated response:\n  {generated_text}")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
