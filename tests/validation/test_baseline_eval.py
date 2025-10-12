"""
Test script for baseline Spatial-RGPT evaluation.

This script tests the eval_spatial_rgpt_gemma.py script with a small test dataset
to verify the baseline evaluation pipeline works correctly.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = PROJECT_ROOT / "data/test/spatial_rgpt_test.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs/test"
OUTPUT_FILE = OUTPUT_DIR / "test_baseline_results.jsonl"


def main():
    print("=" * 80)
    print("Testing Baseline Spatial-RGPT Evaluation")
    print("=" * 80)

    # Check test data exists
    if not TEST_DATA.exists():
        print(f"❌ Test data not found at {TEST_DATA}")
        print("Please run this script from the project root.")
        sys.exit(1)

    print(f"✓ Test data found: {TEST_DATA}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")

    # Clean up previous test output
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        print(f"✓ Cleaned up previous test output")

    # Run evaluation script
    print("\n" + "=" * 80)
    print("Running evaluation (this will take a few minutes to load models)...")
    print("=" * 80 + "\n")

    cmd = [
        "uv",
        "run",
        "python",
        str(PROJECT_ROOT / "scripts/eval_spatial_rgpt_gemma.py"),
        "--data-path",
        str(TEST_DATA),
        "--output",
        str(OUTPUT_FILE),
        "--max-samples",
        "3",
        "--max-new-tokens",
        "50",
        "--temperature",
        "0.0",
    ]

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    print("\n" + "=" * 80)
    if result.returncode != 0:
        print("❌ Evaluation failed!")
        sys.exit(1)

    # Check output file was created
    if not OUTPUT_FILE.exists():
        print(f"❌ Output file not created: {OUTPUT_FILE}")
        sys.exit(1)

    print(f"✓ Output file created: {OUTPUT_FILE}")

    # Read and display results
    import json

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)

    with open(OUTPUT_FILE, "r") as f:
        for i, line in enumerate(f, 1):
            result = json.loads(line)
            print(f"\nExample {i}:")
            print(f"  ID: {result['id']}")
            print(f"  Prompt: {result['prompt'][:60]}...")
            print(f"  Response: {result['gemma_response'][:80]}...")
            print(f"  Parsed choice: {result['parsed_choice']}")
            print(f"  Ground truth: {result['ground_truth']}")
            print(f"  Has choices: {result['has_choices']}")

    print("\n" + "=" * 80)
    print("✓ Baseline evaluation test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
