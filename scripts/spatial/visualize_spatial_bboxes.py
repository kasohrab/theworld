"""
Visualize SpatialRGPT-Bench samples with bounding boxes.

Creates a folder for each sample containing:
  - image_with_bboxes.png: Image with drawn bounding boxes
  - prompt.txt: Question text
  - answer.txt: Ground truth answer
  - metadata.json: Full sample metadata (id, bbox coords, ground truth, etc.)

Usage:
    # Verify both eval and training datasets (10 samples each)
    python scripts/spatial/visualize_spatial_bboxes.py --dataset-type both --samples-per-dataset 10

    # Verify eval dataset only
    python scripts/spatial/visualize_spatial_bboxes.py --dataset-type eval --num-samples 10

    # Verify training dataset only
    python scripts/spatial/visualize_spatial_bboxes.py --dataset-type training --num-samples 10
"""

import argparse
import json
from pathlib import Path

from theworld.datasets.spatial_rgpt import SpatialRGPTDataset


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Visualize SpatialRGPT bounding boxes")
    p.add_argument(
        "--dataset-type",
        type=str,
        choices=["eval", "training", "both"],
        default="both",
        help="Which dataset(s) to verify: eval (SpatialRGPT-Bench), training (OpenSpatialDataset), or both",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default="a8cheng/SpatialRGPT-Bench",
        help="HuggingFace dataset ID or local JSONL path (for eval dataset)",
    )
    p.add_argument(
        "--training-data-path",
        type=str,
        default="/tmp/openspatial/result_10_depth_convs.json",
        help="HuggingFace dataset ID or local JSONL path (for training dataset)",
    )
    p.add_argument(
        "--image-folder",
        type=str,
        default="",
        help="Base folder for images (empty string for HF datasets, or path like /path/to/openimages)",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to visualize per dataset (overrides --samples-per-dataset)",
    )
    p.add_argument(
        "--samples-per-dataset",
        type=int,
        default=10,
        help="Number of samples to visualize per dataset when using --dataset-type both (default: 10)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/bbox_verification",
        help="Output directory for visualizations",
    )
    p.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in dataset (default: 0)",
    )
    return p.parse_args()


def process_dataset(dataset, output_dir, dataset_name, start_idx=0):
    """Process a single dataset and save visualizations.

    Args:
        dataset: SpatialRGPTDataset instance
        output_dir: Path to output directory
        dataset_name: Name of dataset (for logging)
        start_idx: Starting index for sample numbering
    """
    print(f"Processing {dataset_name} dataset...")
    print(f"✓ Loaded {len(dataset)} samples")
    print()

    # Process each sample
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_id = sample["id"]

        # Create folder for this sample (simplified naming: sample_001, sample_002, etc.)
        sample_dir = output_dir / f"sample_{i+1:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Get image (WITHOUT bboxes drawn yet)
        image = sample["image"]
        if image is None:
            print(f"[{i+1}/{len(dataset)}] WARNING: No image for sample {sample_id}")
            continue

        # Extract bbox array from raw metadata
        # IMPORTANT: The bbox array follows a specific structural invariant:
        #   - len(bbox_array) == total_<mask>_count in conversation
        #   - Nth <mask> token → bbox[N] in array (positional correspondence)
        #   - Same physical region may appear multiple times (one per <mask>)
        # See docs/spatial-bbox-correspondence.md for detailed proof
        raw_metadata = sample.get("metadata", {})
        raw_bboxes = raw_metadata.get("bbox", [])

        # Parse bbox if it's a string (HuggingFace serialization)
        if isinstance(raw_bboxes, str):
            import ast

            try:
                raw_bboxes = ast.literal_eval(raw_bboxes)
            except Exception:
                raw_bboxes = []

        # Convert Decimals to floats
        from decimal import Decimal

        def convert_decimals(obj):
            """Recursively convert Decimal objects to float."""
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(item) for item in obj]
            else:
                return obj

        raw_bboxes = convert_decimals(raw_bboxes)

        # Deduplicate bboxes to find unique physical regions
        # This is necessary because:
        #   1. TheWorld draws ALL unique regions on the image with global labels [0, 1, 2, ...]
        #   2. Text must reference these global labels, not turn-local numbering
        #   3. Same region referenced in multiple turns → multiple bbox entries → need deduplication
        # Example: Sample with 5 unique regions may have 18 bbox entries (one per <mask> appearance)
        unique_bboxes = []
        seen_bboxes = set()
        for bbox in raw_bboxes:
            # Ensure bbox is a list of 4 values
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            bbox_tuple = tuple(bbox)  # Convert to hashable type
            if bbox_tuple not in seen_bboxes:
                unique_bboxes.append(bbox)
                seen_bboxes.add(bbox_tuple)

        # Draw ONLY unique bboxes on image with global labels
        # The labels [0, 1, 2, ...] must match what spatial_rgpt.py generates in text
        # via the bbox_to_global_region mapping (see lines 368-394 in spatial_rgpt.py)
        if len(unique_bboxes) > 0:
            from theworld.datasets.bbox_utils import draw_bounding_boxes, clamp_bbox

            img_width, img_height = image.size
            clamped_bboxes = [clamp_bbox(bbox, img_width, img_height) for bbox in unique_bboxes]
            # Global labels: Region [0], Region [1], Region [2], ...
            labels = [f"Region [{i}]" for i in range(len(unique_bboxes))]
            image_with_bboxes = draw_bounding_boxes(image, clamped_bboxes, labels=labels)
        else:
            image_with_bboxes = image.copy()

        # Save image with unique bboxes
        image_path = sample_dir / "image_with_bboxes.png"
        image_with_bboxes.save(image_path)
        print(
            f"[{i+1}/{len(dataset)}] Saved: {image_path} ({len(unique_bboxes)} unique regions out of {len(raw_bboxes)} total bboxes)"
        )

        # Extract question and answer from messages
        messages = sample.get("messages", [])
        question = ""
        answer = ""
        if len(messages) > 0:
            question = messages[0].get("content", "")
        if len(messages) > 1:
            answer = messages[1].get("content", "")

        # Save prompt
        prompt_path = sample_dir / "prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(question)

        # Save answer
        answer_path = sample_dir / "answer.txt"
        with open(answer_path, "w", encoding="utf-8") as f:
            f.write(answer)

        # Save ORIGINAL raw JSON (before conversion)
        original_json_path = sample_dir / "original.json"
        # Filter out non-serializable objects
        original_data = {}
        for k, v in raw_metadata.items():
            if k == "image" or hasattr(v, "save"):
                continue
            original_data[k] = convert_decimals(v)

        with open(original_json_path, "w", encoding="utf-8") as f:
            json.dump(original_data, f, indent=2, ensure_ascii=False)

        # Save CONVERTED metadata (what training actually sees)
        # This shows Region [0], Region [1] instead of <mask> <depth>
        metadata = {
            "id": sample_id,
            "messages": messages,  # AFTER conversion (Region [0], Region [1], etc.)
            "qa_type": sample.get("qa_type"),
            "qa_category": sample.get("qa_category"),
            "choices": sample.get("choices"),
            "bbox_info": {
                "total_bboxes": len(raw_bboxes),
                "unique_bboxes": len(unique_bboxes),
                "note": f"Image shows {len(unique_bboxes)} unique regions. Bbox array has {len(raw_bboxes)} entries (one per QA turn mentioning regions).",
            },
        }

        # Add unique bbox coordinates
        if len(unique_bboxes) > 0:
            metadata["unique_bboxes"] = convert_decimals(unique_bboxes)

        metadata_path = sample_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()


def load_dataset(data_path, num_samples, image_folder=None, split="val", use_streaming=False):
    """Load dataset from HuggingFace or local path.

    Args:
        data_path: HuggingFace dataset ID or local JSONL path
        num_samples: Number of samples to load
        image_folder: Base folder for images (optional)
        split: Dataset split to load (default: "val")
        use_streaming: Use streaming mode for large datasets (default: False)

    Returns:
        SpatialRGPTDataset instance
    """
    import os

    # Check if it's a local file (same logic as train_hf.py)
    if os.path.exists(data_path):
        # Local JSON file - use SpatialRGPTDataset directly (same as training)
        # DON'T draw bboxes in dataset - we'll draw them manually after deduplication
        return SpatialRGPTDataset(
            data_source=data_path,
            image_folder=image_folder if image_folder else None,
            draw_bboxes=False,  # We'll draw manually
            num_samples=num_samples,
        )
    else:
        # HuggingFace dataset
        from datasets import load_dataset as hf_load_dataset

        hf_dataset = hf_load_dataset(data_path, split=split, streaming=use_streaming)
        return SpatialRGPTDataset(
            data_source=hf_dataset,
            image_folder=image_folder if image_folder else None,
            draw_bboxes=False,  # We'll draw manually
            num_samples=num_samples,
        )


def main():
    args = parse_args()

    # Determine number of samples per dataset
    num_samples = args.num_samples if args.num_samples is not None else args.samples_per_dataset

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BBOX VERIFICATION SCRIPT")
    print("=" * 80)
    print(f"Dataset type: {args.dataset_type}")
    print(f"Samples per dataset: {num_samples}")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 80)
    print()

    # Process evaluation dataset
    if args.dataset_type in ["eval", "both"]:
        print()
        print("=" * 80)
        print("EVALUATION DATASET (SpatialRGPT-Bench)")
        print("=" * 80)
        # Only add subdirectory when processing both datasets
        eval_output_dir = output_dir / "eval" if args.dataset_type == "both" else output_dir
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading dataset: {args.data_path}")
        eval_dataset = load_dataset(
            args.data_path,
            num_samples=num_samples,
            image_folder=args.image_folder if args.image_folder else None,
            split="val",
        )
        process_dataset(eval_dataset, eval_output_dir, "evaluation", start_idx=args.start_idx)

    # Process training dataset
    if args.dataset_type in ["training", "both"]:
        print()
        print("=" * 80)
        print("TRAINING DATASET (OpenSpatialDataset)")
        print("=" * 80)
        # Only add subdirectory when processing both datasets
        training_output_dir = output_dir / "training" if args.dataset_type == "both" else output_dir
        training_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading dataset: {args.training_data_path}")
        # Training dataset uses "train" split
        training_dataset = load_dataset(
            args.training_data_path,
            num_samples=num_samples,
            image_folder=args.image_folder if args.image_folder else None,
            split="train",
        )
        process_dataset(training_dataset, training_output_dir, "training", start_idx=args.start_idx)

    print()
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Each sample folder contains:")
    print("  - image_with_bboxes.png  (image with UNIQUE bboxes only)")
    print("  - prompt.txt             (question text with Region [N] tokens)")
    print("  - answer.txt             (ground truth answer)")
    print("  - metadata.json          (converted data - what training sees)")
    print("  - original.json          (raw data before conversion)")
    print()
    print("Note: Bbox arrays contain duplicates (one entry per QA turn).")
    print("      Images show only unique regions to avoid confusion.")
    print("=" * 80)


if __name__ == "__main__":
    main()
