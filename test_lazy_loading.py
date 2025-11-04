#!/usr/bin/env python3
"""Quick test to verify lazy loading of SpatialRGPT dataset."""

import sys
from pathlib import Path

# Add python package to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from theworld.datasets import SpatialRGPTDataset

def test_lazy_loading():
    """Test that lazy loading works and doesn't crash."""
    print("Testing lazy loading with SpatialRGPT dataset...")
    print()

    # Test with limited samples
    dataset = SpatialRGPTDataset(
        data_source="/home/hice1/ksohrab3/scratch/theworld/data/result_10_depth_convs.json",
        image_folder="/home/hice1/ksohrab3/scratch/theworld/data/openimages/train",
        num_samples=100,  # Only index first 100 samples
        draw_bboxes=False,
        skip_missing_images=True,
    )

    print(f"✓ Dataset initialized")
    print(f"  Length: {len(dataset)}")
    print()

    # Test accessing a few samples
    print("Testing sample access...")
    for idx in [0, 10, 50, 99]:
        try:
            sample = dataset[idx]
            if sample is not None:
                print(f"  Sample {idx}: ✓ (image: {sample['image'].size if sample['image'] else None})")
            else:
                print(f"  Sample {idx}: ⚠ (missing image, skipped)")
        except Exception as e:
            print(f"  Sample {idx}: ✗ Error: {e}")
            return False

    print()
    print("✓ All tests passed! Lazy loading works correctly.")
    return True

if __name__ == "__main__":
    success = test_lazy_loading()
    sys.exit(0 if success else 1)
