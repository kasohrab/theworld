import torch
from model import TheWorld
import os

def main():
    # Set device - configurable via environment variable
    device = os.getenv("DEVICE", "cpu")
    print(f"Using device: {device}")

    # Initialize model (tokenizer is loaded inside TheWorld)
    model = TheWorld("google/gemma-2-2b-it", device=device)

    # Dump models to files
    with open("cosmos_full_model.txt", "w") as f:
        f.write("=== FULL COSMOS PIPELINE ===\n\n")
        f.write(str(model.cosmos_pipe))
        f.write("\n\n=== COSMOS VAE ===\n\n")
        f.write(str(model.cosmos_pipe.vae))
        f.write("\n\n=== COSMOS VAE ENCODER ===\n\n")
        f.write(str(model.cosmos_vae_encoder))
        f.write("\n\n=== COSMOS VAE CONFIG ===\n\n")
        f.write(str(model.cosmos_pipe.vae.config))

    with open("gemma_full_model.txt", "w") as f:
        f.write("=== FULL GEMMA MODEL ===\n\n")
        f.write(str(model.gemma))
        f.write("\n\n=== GEMMA CONFIG ===\n\n")
        f.write(str(model.gemma.config))

    print("Models dumped to cosmos_full_model.txt and gemma_full_model.txt")

    # Create sample inputs
    # Image input: random tensor simulating an image (1, 3, 224, 224)
    input_pixels = torch.randn(1, 3, 224, 224)

    # Text input: sample question using Gemma's tokenizer
    text = "What is in this image?"
    text_inputs = model.tokenizer(text, return_tensors="pt", padding=True)
    input_ids = text_inputs.input_ids
    text_attention_mask = text_inputs.attention_mask

    # Run forward pass
    print("Running forward pass...")
    outputs = model.forward(input_pixels, input_ids, text_attention_mask)

    print(f"Output logits shape: {outputs.logits.shape}")
    print("Forward pass completed successfully!")


if __name__ == "__main__":
    main()
