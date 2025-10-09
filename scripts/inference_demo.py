"""
Interactive inference demo for BLINK benchmark.

Allows exploring BLINK examples interactively:
- Browse through dataset examples
- See model predictions vs ground truth
- Test custom images and questions
- Compare different num_world_steps settings

Example usage:
    # Start interactive demo
    python scripts/inference_demo.py \
        --model username/theworld-datacomp \
        --task Relative_Depth

    # Commands in demo:
    # - next: Show next example
    # - prev: Show previous example
    # - jump N: Jump to example N
    # - custom: Test custom image/question
    # - steps N: Change num_world_steps
    # - quit: Exit demo
"""

import argparse
import sys
from pathlib import Path
from typing import Any, cast

from PIL import Image
from datasets import load_dataset, Dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld


class BLINKDemo:
    """Interactive demo for BLINK evaluation."""

    def __init__(
        self,
        model: TheWorld,
        task: str = "Relative_Depth",
        split: str = "test",
        num_world_steps: int = 0,
        max_new_tokens: int = 10,
        temperature: float = 0.0,
    ):
        """
        Initialize BLINK demo.

        Args:
            model: TheWorld model instance
            task: BLINK task name
            split: Dataset split (test/val)
            num_world_steps: Number of world model steps
            max_new_tokens: Max tokens to generate
            temperature: Generation temperature
        """
        self.model = model
        self.task = task
        self.split = split
        self.num_world_steps = num_world_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Load dataset
        print(f"Loading BLINK/{task} ({split} split)...")
        dataset_raw = load_dataset("BLINK-Benchmark/BLINK", task, split=split, trust_remote_code=True)
        self.dataset = cast(Dataset, dataset_raw)
        print(f"Loaded {len(self.dataset)} examples\n")

        self.current_idx = 0

    def format_question(self, example) -> str:
        """Format question with choices."""
        prompt = f"Question: {example['question']}\n"
        for i, choice in enumerate(example["choices"]):
            letter = chr(ord("A") + i)
            prompt += f"{letter}) {choice}\n"
        prompt += "Answer:"
        return prompt

    def parse_choice(self, generated_text: str, choices: list) -> str:
        """Parse choice letter from generated text."""
        import re

        text = generated_text.strip()

        # Single letter
        if text.upper() in ["A", "B", "C", "D"]:
            return text.upper()

        # Letter at start
        match = re.match(r"^([A-Da-d])[):\.\s]", text)
        if match:
            return match.group(1).upper()

        # "answer is X"
        match = re.search(r"answer is ([A-Da-d])", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Match choice text
        text_lower = text.lower()
        for i, choice in enumerate(choices):
            if choice.lower() in text_lower:
                return chr(ord("A") + i)

        # Extract first A-D letter
        for char in text.upper():
            if char in ["A", "B", "C", "D"]:
                return char

        return "A"  # Fallback

    def show_example(self, idx: int) -> None:
        """Display an example with prediction."""
        if idx < 0 or idx >= len(self.dataset):
            print(f"❌ Invalid index: {idx} (dataset has {len(self.dataset)} examples)")
            return

        self.current_idx = idx
        example = cast(dict[str, Any], self.dataset[idx])

        # Print header
        print("\n" + "=" * 80)
        print(f"Example {idx + 1} / {len(self.dataset)}")
        print("=" * 80)

        # Show image info
        image = cast(Image.Image, example["image_1"])
        print(f"\n📷 Image: {image.size} pixels ({image.mode})")

        # Show question
        print(f"\n❓ Question: {example['question']}")
        print("\nChoices:")
        for i, choice in enumerate(example["choices"]):
            letter = chr(ord("A") + i)
            print(f"  {letter}) {choice}")

        # Generate prediction
        print(f"\n🤖 Generating answer (num_world_steps={self.num_world_steps})...")
        prompt = self.format_question(example)

        try:
            response = self.model.generate(
                image,
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                num_world_steps=self.num_world_steps,
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            response = "A"

        predicted = self.parse_choice(response, example["choices"])
        ground_truth = example["answer"]
        is_correct = predicted == ground_truth

        # Show results
        print(f"\n{'✅' if is_correct else '❌'} Results:")
        print(f'  Generated text: "{response}"')
        print(f"  Parsed answer:  {predicted}")
        print(f"  Ground truth:   {ground_truth}")
        print(f"  Status:         {'CORRECT ✓' if is_correct else 'INCORRECT ✗'}")

        # Show navigation help
        print("\n" + "-" * 80)
        print("Commands: next | prev | jump N | steps N | custom | quit")
        print("-" * 80)

    def test_custom(self) -> None:
        """Test custom image and question."""
        print("\n" + "=" * 80)
        print("Custom Image Test")
        print("=" * 80)

        # Get image path
        image_path = input("\nEnter image path (or 'cancel'): ").strip()
        if image_path.lower() == "cancel":
            return

        try:
            image = Image.open(image_path).convert("RGB")
            print(f"✓ Loaded image: {image.size} pixels")
        except Exception as e:
            print(f"❌ Failed to load image: {e}")
            return

        # Get question
        question = input("Enter question: ").strip()
        if not question:
            print("❌ Empty question")
            return

        # Get choices
        print("Enter choices (one per line, empty line to finish):")
        choices = []
        while True:
            choice = input(f"  {chr(ord('A') + len(choices))}) ").strip()
            if not choice:
                break
            choices.append(choice)

        if len(choices) < 2:
            print("❌ Need at least 2 choices")
            return

        # Format prompt
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            letter = chr(ord("A") + i)
            prompt += f"{letter}) {choice}\n"
        prompt += "Answer:"

        # Generate
        print(f"\n🤖 Generating answer (num_world_steps={self.num_world_steps})...")
        try:
            response = self.model.generate(
                image,
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                num_world_steps=self.num_world_steps,
            )
            predicted = self.parse_choice(response, choices)

            print(f"\n✓ Results:")
            print(f'  Generated text: "{response}"')
            print(f"  Parsed answer:  {predicted}")
        except Exception as e:
            print(f"❌ Error: {e}")

    def run(self) -> None:
        """Run interactive demo loop."""
        print("\n" + "=" * 80)
        print(f"BLINK Interactive Demo - {self.task}")
        print("=" * 80)
        print(f"\nDataset: {len(self.dataset)} examples")
        print(f"Model: {self.model.gemma_model_name}")
        print(f"World steps: {self.num_world_steps}")
        print("\nCommands:")
        print("  next        - Show next example")
        print("  prev        - Show previous example")
        print("  jump N      - Jump to example N")
        print("  steps N     - Set num_world_steps to N")
        print("  custom      - Test custom image/question")
        print("  quit        - Exit demo")

        # Show first example
        self.show_example(0)

        # Main loop
        while True:
            try:
                command = input("\n> ").strip().lower()

                if command == "quit" or command == "q" or command == "exit":
                    print("Goodbye!")
                    break

                elif command == "next" or command == "n":
                    self.show_example(self.current_idx + 1)

                elif command == "prev" or command == "p":
                    self.show_example(self.current_idx - 1)

                elif command.startswith("jump ") or command.startswith("j "):
                    try:
                        idx = int(command.split()[1]) - 1  # 1-indexed for user
                        self.show_example(idx)
                    except (ValueError, IndexError):
                        print("❌ Usage: jump N (e.g., jump 42)")

                elif command.startswith("steps ") or command.startswith("s "):
                    try:
                        steps = int(command.split()[1])
                        self.num_world_steps = steps
                        print(f"✓ Set num_world_steps = {steps}")
                        # Re-show current example with new setting
                        self.show_example(self.current_idx)
                    except (ValueError, IndexError):
                        print("❌ Usage: steps N (e.g., steps 4)")

                elif command == "custom" or command == "c":
                    self.test_custom()

                elif command == "help" or command == "h":
                    print("\nCommands:")
                    print("  next        - Show next example")
                    print("  prev        - Show previous example")
                    print("  jump N      - Jump to example N")
                    print("  steps N     - Set num_world_steps to N")
                    print("  custom      - Test custom image/question")
                    print("  quit        - Exit demo")

                elif command == "":
                    # Empty input = next
                    self.show_example(self.current_idx + 1)

                else:
                    print(f"❌ Unknown command: {command} (type 'help' for commands)")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive BLINK inference demo")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Relative_Depth",
        choices=["Relative_Depth", "Spatial_Relation"],
        help="BLINK task to demo (default: Relative_Depth)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--num_world_steps",
        type=int,
        default=0,
        help="Number of world model steps (default: 0)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=10,
        help="Max tokens to generate (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token for private models",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Load model
    print(f"\n{'=' * 80}")
    print("Loading model...")
    print(f"{'=' * 80}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")

    import os

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    model = TheWorld.from_pretrained(
        args.model,
        device=args.device,
        hf_token=hf_token,
    )

    print("✓ Model loaded")

    # Create and run demo
    demo = BLINKDemo(
        model=model,
        task=args.task,
        split=args.split,
        num_world_steps=args.num_world_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    demo.run()


if __name__ == "__main__":
    main()
