"""
Example: Loading TheWorld models from HuggingFace Hub.

This demonstrates how to:
1. Load a trained model from the Hub for inference
2. Use the model for text generation
3. Load specific checkpoints
4. Handle private models with authentication

Usage:
    # Load from public repository
    python examples/load_from_hub.py --model_id username/theworld-datacomp

    # Load from private repository
    export HF_TOKEN="hf_your_token_here"
    python examples/load_from_hub.py --model_id username/private-model

    # Load specific checkpoint
    python examples/load_from_hub.py --model_id username/theworld-datacomp --checkpoint checkpoint-1000/pytorch_model.bin
"""

import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld


def download_example_image(url: str) -> Image.Image:
    """Download an example image from URL."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Load TheWorld model from HuggingFace Hub")
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace Hub model ID (e.g., username/theworld-datacomp)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="pytorch_model.bin",
        help="Checkpoint file to load (default: pytorch_model.bin)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token for private models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to load model on (default: cuda)",
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
        help="URL of example image to test with",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe this image in detail.",
        help="Question to ask about the image",
    )

    args = parser.parse_args()

    # Get HF token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    print("=" * 60)
    print("Loading TheWorld Model from HuggingFace Hub")
    print("=" * 60)

    # Load model from Hub
    print(f"\nModel ID: {args.model_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    model = TheWorld.from_pretrained(
        model_id=args.model_id,
        checkpoint_name=args.checkpoint,
        device=args.device,
        hf_token=hf_token,
    )

    print("\n" + "=" * 60)
    print("Model Loaded Successfully!")
    print("=" * 60)

    # Print model info
    trainable, total, percentage = model.get_trainable_parameters()
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,} ({percentage:.4f}%)")
    print(f"  Gemma model: {model.gemma_model_name}")
    print(f"  Cosmos model: {model.cosmos_model_name}")
    print(f"  Default world steps: {model.num_world_steps}")

    # Download example image
    print("\n" + "=" * 60)
    print("Testing Inference")
    print("=" * 60)
    print(f"\nDownloading example image from: {args.image_url}")

    try:
        image = download_example_image(args.image_url)
        print(f"✓ Image loaded: {image.size}")

        # Generate response
        print(f"\nQuestion: {args.question}")
        print("Generating response...")

        response = model.generate(image, args.question, max_new_tokens=100)

        print("\n" + "-" * 60)
        print("Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)

    except Exception as e:
        print(f"⚠ Error during inference: {e}")
        print("Model loaded successfully, but inference failed.")
        print("You can still use the model programmatically.")

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    print("\nTo use this model in your code:")
    print(f">>> from theworld import TheWorld")
    print(f">>> model = TheWorld.from_pretrained('{args.model_id}')")
    print(f">>> response = model.generate(image, 'Your question here')")


if __name__ == "__main__":
    main()
