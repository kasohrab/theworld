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
import random
from pathlib import Path
from typing import Any, Optional, cast, List, Dict

from PIL import Image
from transformers import AutoProcessor, ProcessorMixin
from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline

from theworld import TheWorld
from theworld.datasets import SpatialRGPTDataset


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


class SpatialRGPTSession:
    """Manages SpatialRGPT dataset browsing session."""

    def __init__(self, json_path: str, image_folder: str):
        self.json_path = json_path
        self.image_folder = image_folder
        self.dataset: Optional[SpatialRGPTDataset] = None
        self.current_index: Optional[int] = None
        self.draw_bboxes = True
        self.ground_truth_conversation: Optional[List[Dict[str, str]]] = None  # Store GT separately

    def load_dataset(self, num_samples: Optional[int] = 100) -> bool:
        """Load SpatialRGPT dataset from configured paths."""
        # Check if JSON exists
        if not Path(self.json_path).exists():
            print(f"✗ JSON file not found: {self.json_path}")
            print("  See docs/training/spatial-rgpt.md for download instructions")
            return False

        # Check if image folder exists
        if not Path(self.image_folder).exists():
            print(f"✗ Image folder not found: {self.image_folder}")
            print("  See docs/training/spatial-rgpt.md for download instructions")
            return False

        # Count available images
        image_files = list(Path(self.image_folder).glob("*.jpg"))
        if len(image_files) == 0:
            print(f"✗ No images found in {self.image_folder}")
            print("  See docs/training/spatial-rgpt.md for download instructions")
            return False

        print(f"  Found {len(image_files)} images in {self.image_folder}")

        # Load dataset
        try:
            print(f"  Loading dataset (this may take a minute)...")
            self.dataset = SpatialRGPTDataset(
                data_source=self.json_path,
                image_folder=self.image_folder,
                draw_bboxes=self.draw_bboxes,
                num_samples=num_samples,
            )
            return True
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return False

    def load_sample(self, index: int) -> Optional[dict]:
        """Load sample by index and store ground truth separately."""
        if self.dataset is None:
            print("✗ Dataset not loaded. Use /spatial load first.")
            return None

        if not 0 <= index < len(self.dataset):
            print(f"✗ Index {index} out of range [0, {len(self.dataset)})")
            return None

        self.current_index = index
        sample = self.dataset[index]

        # Store ground truth conversation separately (for /replay, /compare, /iterate)
        # Check both 'messages' (dataset format) and 'conversations' (raw format)
        if "messages" in sample and sample["messages"]:
            # Convert messages format to conversations format
            conversations = []
            for msg in sample["messages"]:
                conversations.append({"from": msg.get("role"), "value": msg.get("content", "")})
            self.ground_truth_conversation = conversations
        else:
            # Try raw format
            raw_sample = self.dataset.items[index]
            self.ground_truth_conversation = raw_sample.get("conversations", [])

        return sample

    def get_ground_truth_conversation(self) -> Optional[List[Dict[str, str]]]:
        """Get ground truth conversation for current sample."""
        return self.ground_truth_conversation

    def get_ground_truth_questions(self) -> List[str]:
        """Extract just the questions from ground truth conversation."""
        if not self.ground_truth_conversation:
            return []

        # Extract user messages (questions)
        # Handle both "role" and "from" fields for compatibility
        questions = []
        for msg in self.ground_truth_conversation:
            role = msg.get("role") or msg.get("from")
            content = msg.get("content") or msg.get("value")
            if role in ["user", "human"]:
                questions.append(content.replace("<image>\n", "").replace("<image>", ""))
        return questions

    def toggle_bboxes(self, enabled: bool):
        """Toggle bounding box drawing."""
        self.draw_bboxes = enabled
        if self.dataset:
            self.dataset.draw_bboxes = enabled
            print(f"✓ Bounding boxes: {'enabled' if enabled else 'disabled'}")
            print("  Note: Reload current sample to see changes")


def print_help():
    """Print available commands."""
    print("\n" + "=" * 60)
    print("Available commands:")
    print("  /image <path>       - Load new image from file path (resets conversation)")
    print("  /spatial load [N]   - Load SpatialRGPT dataset (N samples, default: 100)")
    print("  /spatial <index>    - Load specific sample by index (resets conversation)")
    print("  /spatial random     - Load random sample (resets conversation)")
    print("  /spatial next/prev  - Navigate to next/previous sample")
    print("  /replay             - Show ground truth conversation for current sample")
    print("  /iterate            - Iterate through ground truth questions with model answers")
    print("  /compare            - Compare model answers to ground truth (after /iterate)")
    print("  /info               - Show current sample metadata")
    print("  /bbox on|off        - Toggle bounding box overlay")
    print("  /clear              - Reset conversation (keep current image)")
    print("  /help               - Show this help message")
    print("  /quit, /exit        - Exit the script")
    print("  <text>              - Ask question (multi-turn: continues conversation)")
    print("=" * 60 + "\n")


def interactive_loop(
    pipe: ImageTextToTextPipeline,
    gen_kwargs: dict[str, Any],
    spatial_session: Optional[SpatialRGPTSession] = None,
) -> None:
    """
    Interactive REPL loop for multi-modal inference.

    Args:
        pipe: HuggingFace pipeline for image-text-to-text
        gen_kwargs: Generation parameters dict
        spatial_session: Optional SpatialRGPT dataset session
    """
    print("\n" + "=" * 60)
    print("TheWorld Interactive Inference")
    print("=" * 60)
    print_help()

    current_image: Optional[Image.Image] = None
    conversation_history: list[dict[str, str]] = []  # Track conversation turns

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
                        conversation_history = []  # Reset conversation for new image
                        print("✓ Conversation reset")

                elif cmd == "/clear":
                    conversation_history = []
                    print("✓ Conversation reset (image kept)")

                elif cmd == "/spatial":
                    if spatial_session is None:
                        print("✗ Spatial session not initialized (missing --spatial-json or --spatial-images)")
                        continue

                    if len(cmd_parts) < 2:
                        print("✗ Usage: /spatial load [N] | <index> | random | next | prev | info")
                        continue

                    subcmd = cmd_parts[1].split()[0]  # First word after /spatial

                    if subcmd == "load":
                        # Parse optional number of samples
                        parts = cmd_parts[1].split()
                        num_samples = int(parts[1]) if len(parts) > 1 else 100
                        print(f"Loading SpatialRGPT dataset ({num_samples} samples)...")
                        if spatial_session.load_dataset(num_samples):
                            print(f"✓ Loaded {len(spatial_session.dataset)} samples")

                    elif subcmd.isdigit():
                        # Load sample by index
                        index = int(subcmd)
                        sample = spatial_session.load_sample(index)
                        if sample:
                            current_image = sample["image"]
                            # Reset conversation (don't copy ground truth!)
                            conversation_history = []
                            num_questions = len(spatial_session.get_ground_truth_questions())
                            print(f"✓ Loaded sample {index}")
                            if num_questions > 0:
                                print(f"  {num_questions} ground truth questions available")
                                print(f"  Use /iterate to go through them with model answers")
                                print(f"  Use /replay to see ground truth conversation")
                            else:
                                print(f"  No ground truth conversation data")

                    elif subcmd == "random":
                        if spatial_session.dataset is None:
                            print("✗ Dataset not loaded. Use /spatial load first.")
                            continue
                        index = random.randint(0, len(spatial_session.dataset) - 1)
                        sample = spatial_session.load_sample(index)
                        if sample:
                            current_image = sample["image"]
                            # Reset conversation (don't copy ground truth!)
                            conversation_history = []
                            num_questions = len(spatial_session.get_ground_truth_questions())
                            print(f"✓ Loaded random sample {index}")
                            if num_questions > 0:
                                print(f"  {num_questions} ground truth questions available")
                                print(f"  Use /iterate to go through them with model answers")
                            else:
                                print(f"  No ground truth conversation data")

                    elif subcmd == "next":
                        if spatial_session.current_index is None:
                            print("✗ No sample loaded. Use /spatial <index> first.")
                            continue
                        if spatial_session.dataset is None:
                            print("✗ Dataset not loaded.")
                            continue
                        next_idx = spatial_session.current_index + 1
                        if next_idx >= len(spatial_session.dataset):
                            print("✗ Already at last sample")
                            continue
                        sample = spatial_session.load_sample(next_idx)
                        if sample:
                            current_image = sample["image"]
                            print(f"✓ Loaded sample {next_idx}")
                            print(f"  Question: {sample['text']}")

                    elif subcmd == "prev":
                        if spatial_session.current_index is None:
                            print("✗ No sample loaded. Use /spatial <index> first.")
                            continue
                        prev_idx = spatial_session.current_index - 1
                        if prev_idx < 0:
                            print("✗ Already at first sample")
                            continue
                        sample = spatial_session.load_sample(prev_idx)
                        if sample:
                            current_image = sample["image"]
                            print(f"✓ Loaded sample {prev_idx}")
                            print(f"  Question: {sample['text']}")

                    elif subcmd == "info":
                        if spatial_session.dataset:
                            print(f"\nDataset: {len(spatial_session.dataset)} samples")
                            if spatial_session.current_index is not None:
                                print(f"Current index: {spatial_session.current_index}")
                                sample = spatial_session.dataset[spatial_session.current_index]
                                print(f"  ID: {sample.get('id', 'N/A')}")
                                print(f"  Question: {sample['text']}")
                                print(f"  Answer: {sample['label']}")
                        else:
                            print("No dataset loaded. Use /spatial load first.")

                    else:
                        print(f"✗ Unknown spatial subcommand: {subcmd}")
                        print("  Usage: /spatial load [N] | <index> | random | next | prev | info")

                elif cmd == "/replay":
                    if spatial_session is None or spatial_session.current_index is None:
                        print("✗ No sample loaded. Use /spatial <index> first.")
                        continue

                    conversations = spatial_session.get_ground_truth_conversation()
                    if conversations:
                        print("\n" + "=" * 60)
                        print("GROUND TRUTH CONVERSATION REPLAY")
                        print("=" * 60)
                        for i, turn in enumerate(conversations):
                            role = turn["from"].upper()
                            text = turn["value"].replace("<image>\n", "").replace("<image>", "")
                            print(f"\n{role} (Turn {i}):")
                            print(f"  {text}")
                        print("=" * 60)
                    else:
                        print("✗ No conversation data available")

                elif cmd == "/iterate":
                    if spatial_session is None or spatial_session.current_index is None:
                        print("✗ No sample loaded. Use /spatial <index> first.")
                        continue

                    if current_image is None:
                        print("✗ No image loaded.")
                        continue

                    questions = spatial_session.get_ground_truth_questions()
                    if not questions:
                        print("✗ No ground truth questions available for this sample.")
                        continue

                    print(f"\n{'='*60}")
                    print(f"ITERATING THROUGH {len(questions)} QUESTIONS")
                    print(f"{'='*60}\n")

                    # Reset conversation for fresh start
                    conversation_history = []

                    # Iterate through each question
                    for i, question in enumerate(questions, 1):
                        print(f"Question {i}/{len(questions)}: {question}")
                        print("🤖 Generating response...")

                        # Add question to conversation history
                        conversation_history.append({"role": "user", "content": question})

                        # Build messages for pipeline
                        messages = []
                        for j, msg in enumerate(conversation_history):
                            if j == 0:
                                # First message includes image
                                messages.append(
                                    {
                                        "role": msg["role"],
                                        "content": [
                                            {"type": "image", "image": current_image},
                                            {"type": "text", "text": msg["content"]},
                                        ],
                                    }
                                )
                            else:
                                # Subsequent messages are text-only
                                messages.append(
                                    {"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]}
                                )

                        # Generate model answer
                        result = pipe(text=messages, **gen_kwargs)
                        response = result[0]["generated_text"]

                        # Add model answer to conversation history
                        conversation_history.append({"role": "assistant", "content": response})

                        # Print response
                        print(f"Model: {response}\n")

                    print(f"{'='*60}")
                    print(f"✓ Completed {len(questions)} questions")
                    print(f"  Use /compare to see side-by-side comparison with ground truth")
                    print(f"{'='*60}\n")

                elif cmd == "/compare":
                    if spatial_session is None or spatial_session.current_index is None:
                        print("✗ No sample loaded. Use /spatial <index> first.")
                        continue

                    # Check if /iterate was run (conversation_history has model answers)
                    if not conversation_history or len(conversation_history) < 2:
                        print("✗ No model answers to compare. Run /iterate first.")
                        continue

                    gt_conv = spatial_session.get_ground_truth_conversation()
                    if not gt_conv:
                        print("✗ No ground truth conversation available.")
                        continue

                    # Extract ground truth Q&A pairs
                    gt_pairs = []
                    for i in range(0, len(gt_conv), 2):
                        if i + 1 < len(gt_conv):
                            question = gt_conv[i].get("value", "").replace("<image>\n", "").replace("<image>", "")
                            answer = gt_conv[i + 1].get("value", "")
                            gt_pairs.append((question, answer))

                    # Extract model Q&A pairs from conversation_history
                    model_pairs = []
                    for i in range(0, len(conversation_history), 2):
                        if i + 1 < len(conversation_history):
                            question = conversation_history[i].get("content", "")
                            answer = conversation_history[i + 1].get("content", "")
                            model_pairs.append((question, answer))

                    # Calculate metrics
                    exact_matches = 0
                    for i, (gt_pair, model_pair) in enumerate(zip(gt_pairs, model_pairs)):
                        gt_answer = gt_pair[1].lower().strip()
                        model_answer = model_pair[1].lower().strip()
                        if gt_answer in model_answer or model_answer in gt_answer:
                            exact_matches += 1

                    match_percentage = (exact_matches / len(gt_pairs) * 100) if gt_pairs else 0

                    # Print comparison
                    print(f"\n{'='*60}")
                    print(f"COMPARISON: GROUND TRUTH vs MODEL")
                    print(f"{'='*60}")
                    print(f"\nExact/Partial Matches: {exact_matches}/{len(gt_pairs)} ({match_percentage:.1f}%)\n")
                    print(f"{'='*60}\n")

                    # Print each turn side-by-side
                    for i, (gt_pair, model_pair) in enumerate(zip(gt_pairs, model_pairs), 1):
                        gt_q, gt_a = gt_pair
                        model_q, model_a = model_pair

                        print(f"Turn {i}:")
                        print(f"Q: {gt_q}")
                        print(f"GT:    {gt_a}")
                        print(f"Model: {model_a}")

                        # Check if it's a match
                        gt_answer_lower = gt_a.lower().strip()
                        model_answer_lower = model_a.lower().strip()
                        if gt_answer_lower in model_answer_lower or model_answer_lower in gt_answer_lower:
                            print("✓ Match")
                        else:
                            print("✗ Different")
                        print()

                    print(f"{'='*60}\n")

                elif cmd == "/bbox":
                    if spatial_session is None:
                        print("✗ Spatial session not initialized")
                        continue

                    if len(cmd_parts) < 2:
                        print("✗ Usage: /bbox on|off")
                        continue

                    toggle = cmd_parts[1].lower() == "on"
                    spatial_session.toggle_bboxes(toggle)

                elif cmd == "/info":
                    if spatial_session and spatial_session.current_index is not None:
                        if spatial_session.dataset:
                            sample = spatial_session.dataset[spatial_session.current_index]
                            print(f"\nSample {spatial_session.current_index}:")
                            print(f"  ID: {sample.get('id', 'N/A')}")
                            print(f"  Question: {sample['text']}")
                            print(f"  Answer: {sample['label']}")
                    else:
                        print("No sample loaded")

                else:
                    print(f"✗ Unknown command: {cmd}")
                    print("  Type /help for available commands")

            else:
                # User asked a question
                if current_image is None:
                    print("✗ No image loaded. Use /image <path> to load an image first.")
                    continue

                # Add user question to conversation history
                conversation_history.append({"role": "user", "content": prompt})
                turn_num = len([m for m in conversation_history if m["role"] == "user"])

                if turn_num == 1:
                    print(f"Turn {turn_num}")
                else:
                    print(f"Turn {turn_num} (continuing conversation)")

                # Generate response using pipeline with full conversation context
                print("🤖 Generating response...")

                # Build messages for pipeline chat format
                # First message includes image, subsequent are text-only
                messages = []
                for i, msg in enumerate(conversation_history):
                    if i == 0:
                        # First message includes image
                        messages.append(
                            {
                                "role": msg["role"],
                                "content": [
                                    {"type": "image", "image": current_image},
                                    {"type": "text", "text": msg["content"]},
                                ],
                            }
                        )
                    else:
                        # Subsequent messages are text-only (wrap in list for consistency)
                        messages.append({"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]})

                # Use pipeline directly with chat format (handles tokenization + generation + decoding)
                result = pipe(text=messages, **gen_kwargs)
                response = result[0]["generated_text"]

                # Add assistant response to history
                conversation_history.append({"role": "assistant", "content": response})

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

    # SpatialRGPT dataset configuration
    parser.add_argument(
        "--spatial-json",
        type=str,
        default="/home/hice1/ksohrab3/scratch/theworld/data/result_10_depth_convs.json",
        help="Path to SpatialRGPT JSON file",
    )
    parser.add_argument(
        "--spatial-images",
        type=str,
        default="/home/hice1/ksohrab3/scratch/theworld/data/openimages",
        help="Path to OpenImages folder",
    )

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

    # Load model (includes processor)
    print(f"Loading model: {args.model}...")
    import torch

    model = TheWorld.from_pretrained(args.model, device=args.device, torch_dtype=torch.bfloat16)
    processor: ProcessorMixin = cast(ProcessorMixin, model.processor)

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

    # Initialize spatial session if paths are provided
    spatial_session = None
    if args.spatial_json and args.spatial_images:
        spatial_session = SpatialRGPTSession(args.spatial_json, args.spatial_images)
        print(f"\n✓ SpatialRGPT session initialized")
        print(f"  JSON: {args.spatial_json}")
        print(f"  Images: {args.spatial_images}")
        print(f"  Use /spatial load to load dataset")

    # Run interactive loop
    interactive_loop(pipe, gen_kwargs, spatial_session)


if __name__ == "__main__":
    main()
