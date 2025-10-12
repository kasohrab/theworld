"""
Visualize SpatialRGPT-Bench samples with bounding boxes.

Creates a folder for each sample containing:
  - image_with_bboxes.png: Image with drawn bounding boxes
  - prompt.txt: Question text
  - metadata.json: Full sample metadata (id, bbox coords, ground truth, etc.)

Usage:
    python scripts/visualize_spatial_bboxes.py --num-samples 5
    python scripts/visualize_spatial_bboxes.py --num-samples 10 --output-dir outputs/my_viz
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from theworld.datasets.spatial_rgpt import SpatialRGPTDataset


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Visualize SpatialRGPT bounding boxes")
    p.add_argument(
        "--data-path",
        type=str,
        default="a8cheng/SpatialRGPT-Bench",
        help="HuggingFace dataset ID or local JSONL path",
    )
    p.add_argument(
        "--image-folder",
        type=str,
        default="",
        help="Base folder for images (empty string for HF datasets)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize (default: 5)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/bbox_visualizations",
        help="Output directory for visualizations",
    )
    p.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in dataset (default: 0)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.data_path}")
    print(f"Visualizing {args.num_samples} samples starting from index {args.start_idx}")
    print(f"Output directory: {output_dir}")
    print()

    # Load dataset from HuggingFace or local path
    data_path_obj = Path(args.data_path)
    if data_path_obj.exists():
        # Local JSONL file
        dataset = SpatialRGPTDataset(
            data_path_obj,
            num_samples=args.num_samples,
            image_folder=args.image_folder if args.image_folder else None,
            draw_bboxes=True,
        )
    else:
        # HuggingFace dataset
        from datasets import load_dataset

        hf_dataset = load_dataset(args.data_path, split="val")
        dataset = SpatialRGPTDataset(
            hf_dataset,
            num_samples=args.num_samples,
            draw_bboxes=True,
        )

    print(f"✓ Loaded {len(dataset)} samples")
    print()

    # Process each sample
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_id = sample["id"]
        actual_idx = args.start_idx + i

        # Create folder for this sample
        sample_dir = output_dir / f"sample_{actual_idx:03d}_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save image with bboxes
        image = sample["image"]
        if image is not None:
            image_path = sample_dir / "image_with_bboxes.png"
            image.save(image_path)
            print(f"[{i+1}/{len(dataset)}] Saved: {image_path}")
        else:
            print(f"[{i+1}/{len(dataset)}] WARNING: No image for sample {sample_id}")
            continue

        # Save prompt
        prompt_path = sample_dir / "prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(sample["question"])

        # Save metadata (filter out non-serializable objects like PIL images)
        raw_metadata = sample.get("metadata", {})

        # Create clean metadata dict without PIL images
        clean_raw_metadata = {}
        for k, v in raw_metadata.items():
            # Skip PIL images and other non-serializable objects
            if k == "image" or hasattr(v, "save"):  # PIL images have a save method
                continue
            try:
                # Test if value is JSON serializable
                json.dumps(v)
                clean_raw_metadata[k] = v
            except (TypeError, ValueError):
                # Skip non-serializable values
                clean_raw_metadata[k] = str(v)

        metadata = {
            "id": sample_id,
            "question": sample["question"],
            "ground_truth": sample["answer"],
            "qa_type": sample.get("qa_type"),
            "qa_category": sample.get("qa_category"),
            "choices": sample.get("choices"),
            "raw_metadata": clean_raw_metadata,
        }

        # Extract bbox coordinates from raw metadata
        if "bbox" in raw_metadata:
            metadata["bounding_boxes"] = raw_metadata["bbox"]

        metadata_path = sample_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Total samples processed: {len(dataset)}")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Each folder contains:")
    print("  - image_with_bboxes.png  (image with drawn bounding boxes)")
    print("  - prompt.txt             (question text)")
    print("  - metadata.json          (full sample metadata)")
    print("=" * 60)


if __name__ == "__main__":
    main()
