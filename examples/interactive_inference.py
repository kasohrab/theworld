#!/usr/bin/env python3
"""
Interactive inference script for TheWorld using HuggingFace pipeline API.

Loads model once and allows multi-modal interaction via REPL.
Maximally reuses HuggingFace infrastructure.

Usage:
    # Default model (kasohrab/theworld-spatial)
    python examples/interactive_inference.py

    # Custom model
    python examples/interactive_inference.py --model username/my-theworld

    # Custom generation parameters
    python examples/interactive_inference.py --max_new_tokens 200 --temperature 0.7

Interactive commands:
    /image <path>    - Load new image
    /help           - Show available commands
    /quit, /exit    - Exit script
    <text>          - Ask question about current image
"""

import argparse
from typing import Any, Optional, cast

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, PreTrainedModel, ProcessorMixin
from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline


def load_image(image_path: str) -> Optional[Image.Image]:
    """
    Load image from local file path.

    Args:
        image_path: Path to image file

    Returns:
        PIL Image or None if loading failed
    """
    try:
        img = Image.open(image_path).convert("RGB")
        print(f"✓ Loaded image: {image_path}")
        return img
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return None


def print_help():
    """Print available commands."""
    print("\n" + "=" * 60)
    print("Available commands:")
    print("  /image <path>  - Load new image from file path")
    print("  /help          - Show this help message")
    print("  /quit, /exit   - Exit the script")
    print("  <text>         - Ask question about current image")
    print("=" * 60 + "\n")


def interactive_loop(pipe: ImageTextToTextPipeline, gen_kwargs: dict[str, Any]) -> None:
    """
    Interactive REPL loop for multi-modal inference.

    Args:
        pipe: HuggingFace pipeline for image-text-to-text
        gen_kwargs: Generation parameters dict
    """
    print("\n" + "=" * 60)
    print("TheWorld Interactive Inference")
    print("=" * 60)
    print_help()

    current_image: Optional[Image.Image] = None

    while True:
        try:
            # Get user input
            prompt = input("\n> ").strip()

            if not prompt:
                continue

            # Handle commands
            if prompt.startswith("/"):
                cmd_parts = prompt.split(maxsplit=1)
                cmd = cmd_parts[0].lower()

                if cmd in ["/quit", "/exit"]:
                    print("Goodbye!")
                    break

                elif cmd == "/help":
                    print_help()

                elif cmd == "/image":
                    if len(cmd_parts) < 2:
                        print("✗ Usage: /image <path>")
                        continue

                    image_path = cmd_parts[1]
                    img = load_image(image_path)
                    if img is not None:
                        current_image = img

                else:
                    print(f"✗ Unknown command: {cmd}")
                    print("  Type /help for available commands")

            else:
                # User asked a question
                if current_image is None:
                    print("✗ No image loaded. Use /image <path> to load an image first.")
                    continue

                # Generate response using pipeline
                print("🤖 Generating response...")
                result = pipe(images=current_image, text=prompt, **gen_kwargs)

                # Extract generated text
                response = result[0]["generated_text"]
                print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
            continue
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive inference for TheWorld model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="kasohrab/theworld-spatial",
        help="HuggingFace model ID (default: kasohrab/theworld-spatial)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run model on (default: cuda)")

    # Generation parameters (all HuggingFace generate() args supported)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling threshold (default: 1.0)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter (default: 50)")
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable sampling (required for temperature/top_p/top_k)",
    )
    parser.add_argument(
        "--return_full_text",
        action="store_true",
        help="Return full text including prompt (default: only new tokens)",
    )

    args = parser.parse_args()

    # Load model and processor
    print(f"Loading model: {args.model}...")
    model: PreTrainedModel = cast(
        PreTrainedModel, AutoModelForImageTextToText.from_pretrained(args.model)  # type: ignore[arg-type]
    )
    processor: ProcessorMixin = cast(
        ProcessorMixin, AutoProcessor.from_pretrained(args.model)  # type: ignore[arg-type]
    )

    # Create pipeline
    pipe: ImageTextToTextPipeline = ImageTextToTextPipeline(
        model=model,
        processor=processor,
        device=args.device,
    )
    print("✓ Pipeline loaded successfully!")

    # Prepare generation kwargs
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample or args.temperature > 0,
        "return_full_text": args.return_full_text,
    }

    print(f"\nGeneration settings:")
    for k, v in gen_kwargs.items():
        print(f"  {k}: {v}")

    # Run interactive loop
    interactive_loop(pipe, gen_kwargs)


if __name__ == "__main__":
    main()
