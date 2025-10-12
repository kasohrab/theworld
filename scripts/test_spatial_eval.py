"""
Quick test of SpatialRGPT evaluation on a few samples.

This script tests the complete pipeline:
1. Load SpatialRGPT-Bench from HuggingFace
2. Draw bounding boxes on images
3. Generate answers with Gemma baseline
4. Evaluate with Gemma-as-judge
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from theworld import TheWorld
from theworld.datasets.spatial_rgpt import SpatialRGPTDataset
from theworld.evaluation import evaluate_with_gemma

# Load a few samples from HuggingFace
print("Loading dataset from HuggingFace...")
try:
    from datasets import load_dataset

    hf_dataset = load_dataset("a8cheng/SpatialRGPT-Bench", split="val")
    print(f"✓ Loaded {len(hf_dataset)} samples from HF")
except Exception as e:
    print(f"Error loading from HuggingFace: {e}")
    print("Note: SpatialRGPT-Bench may not be publicly available yet")
    sys.exit(1)

# Wrap in our dataset (with bbox drawing)
print("\nWrapping in SpatialRGPTDataset...")
ds = SpatialRGPTDataset(
    hf_dataset,
    num_samples=3,  # Just test 3 samples
    draw_bboxes=True,
)
print(f"✓ Created dataset with {len(ds)} samples")

# Load model (Gemma-only baseline)
print("\nLoading Gemma baseline model...")
model = TheWorld("google/gemma-3-4b-it", load_cosmos=False, device="cuda")
model.eval()
print("✓ Model loaded")

# Test on first sample
print("\n" + "=" * 60)
print("TESTING SAMPLE")
print("=" * 60)

ex = ds[0]
print(f"\nID: {ex['id']}")
print(f"Question: {ex['question']}")
print(f"Ground Truth: {ex['answer']}")
print(f"Question Type: {ex['qa_type']}")
print(f"Category: {ex['qa_category']}")
print(f"Image: {ex['image']}")

# Generate answer
print("\nGenerating answer...")
try:
    prediction = model.generate(
        image=ex["image"],
        prompt=ex["question"],
        max_new_tokens=128,
        temperature=0.0,
        skip_world_tokens=True,
    )
    print(f"Prediction: {prediction}")
except Exception as e:
    print(f"Error generating: {e}")
    prediction = f"<ERROR: {e}>"

# Evaluate with Gemma-as-judge
print("\nEvaluating with Gemma-as-judge...")
try:
    eval_result = evaluate_with_gemma(
        model,
        question=ex["question"],
        prediction=prediction,
        ground_truth=ex["answer"],
    )
    print(f"Score: {eval_result['score']}")
    print(f"Correct: {eval_result['correct']}")
    print(f"Judge Response: {eval_result['judge_response']}")
except Exception as e:
    print(f"Error evaluating: {e}")

print("\n" + "=" * 60)
print("✓ Test completed successfully!")
print("=" * 60)
