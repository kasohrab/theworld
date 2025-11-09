#!/usr/bin/env python3
"""
Extract OpenImagesV7 image IDs for SpatialRGPT training.

This script extracts unique image IDs from result_10_depth_convs.json and writes
them to a text file compatible with the official OpenImages downloader script.

Usage:
    # Extract all image IDs (~909K images)
    python scripts/download_openimages.py

    # Extract subset (for testing)
    python scripts/download_openimages.py --num-images 1000

    # Custom paths
    python scripts/download_openimages.py \
        --json-path /path/to/result_10_depth_convs.json \
        --output /path/to/image_list.txt

After creating the image list, download using official OpenImages downloader:
    python downloader.py data/required_images.txt \
        --download_folder=/path/to/openimages \
        --num_processes=5
"""

import argparse
import sys
from pathlib import Path
from typing import List

try:
    import ijson
except ImportError:
    print("ERROR: ijson library not found. Install with: pip install ijson")
    sys.exit(1)


def extract_image_ids(json_path: Path, num_images: int = None) -> List[str]:
    """Extract unique image IDs from result_10_depth_convs.json using streaming.

    Args:
        json_path: Path to result_10_depth_convs.json
        num_images: Number of images to extract (None = all)

    Returns:
        List of image IDs (without .jpg extension)
    """
    print(f"Extracting image IDs from {json_path}...")
    print("Using incremental parsing (memory-efficient for 30GB file)")

    # Use ijson for incremental parsing
    image_ids = []
    seen = set()
    count = 0

    with open(json_path, "rb") as f:
        parser = ijson.items(f, "item")
        for entry in parser:
            filename = entry.get("filename")
            if filename and filename not in seen:
                seen.add(filename)
                # Remove .jpg extension if present
                image_id = filename.replace(".jpg", "")
                image_ids.append(image_id)
                if num_images and len(image_ids) >= num_images:
                    break

            count += 1
            if count % 100000 == 0:
                print(f"  Progress: {count:,} entries, {len(image_ids):,} unique images...")

    print(f"✓ Extracted {len(image_ids):,} unique image IDs from {count:,} entries")
    return image_ids


def write_image_list(image_ids: List[str], output_path: Path, split: str = "train") -> None:
    """Write image IDs to text file in format required by OpenImages downloader.

    The official downloader expects one line per image in format:
        train/IMAGE_ID

    Args:
        image_ids: List of image IDs (without extension)
        output_path: Output file path
        split: Dataset split (train/validation/test)
    """
    print(f"\nWriting {len(image_ids):,} image IDs to {output_path}")

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for image_id in image_ids:
            f.write(f"{split}/{image_id}\n")

    print(f"✓ Wrote {len(image_ids):,} lines to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract OpenImagesV7 image IDs for SpatialRGPT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example workflow:
  # Step 1: Extract image IDs
  python scripts/download_openimages.py --output data/required_images.txt

  # Step 2: Download using official OpenImages downloader
  python downloader.py data/required_images.txt \\
      --download_folder=/home/hice1/ksohrab3/scratch/theworld/data/openimages \\
      --num_processes=5

See docs/data/openimages-download.md for more information.
        """,
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("/home/hice1/ksohrab3/scratch/theworld/data/result_10_depth_convs.json"),
        help="Path to result_10_depth_convs.json (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/storage/ice1/7/7/ksohrab3/theworld/data/required_images.txt"),
        help="Output file for image list (default: %(default)s)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of images to extract (default: all ~909K images)",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "validation", "test"], help="Dataset split (default: %(default)s)"
    )

    args = parser.parse_args()

    # Check JSON file exists
    if not args.json_path.exists():
        print(f"ERROR: JSON file not found: {args.json_path}")
        sys.exit(1)

    # Extract image IDs
    image_ids = extract_image_ids(args.json_path, args.num_images)

    if not image_ids:
        print("ERROR: No image IDs extracted")
        sys.exit(1)

    # Write to file
    write_image_list(image_ids, args.output, args.split)

    # Print next steps
    print(f"\n{'='*70}")
    print("Next steps:")
    print(f"{'='*70}")
    print("\n1. Download the official OpenImages downloader script:")
    print("   wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py")
    print("\n2. Download images using the generated file:")
    print(f"   python downloader.py {args.output} \\")
    print("       --download_folder=/home/hice1/ksohrab3/scratch/theworld/data/openimages \\")
    print("       --num_processes=5")
    print("\n3. Update your training config:")
    print('   "image_folder": "/home/hice1/ksohrab3/scratch/theworld/data/openimages/train"')
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
