#!/usr/bin/env python3
"""
Find where to resume OpenImages download based on what's already downloaded.

This script:
1. Reads required_images.txt (all 909K image IDs)
2. Checks which images exist in download folder
3. Creates resume_images.txt with only undownloaded images

Usage:
    python scripts/find_download_resume_point.py
    python scripts/find_download_resume_point.py --image-folder /custom/path
"""

import argparse
from pathlib import Path


def find_resume_point(required_images_path: Path, image_folder: Path, output_path: Path):
    """Find which images still need to be downloaded.

    Args:
        required_images_path: Path to required_images.txt
        image_folder: Path to download folder
        output_path: Path to write resume_images.txt
    """
    print(f"Reading required images from: {required_images_path}")

    if not required_images_path.exists():
        print(f"ERROR: {required_images_path} not found!")
        print("Run: python scripts/download_openimages.py --output data/required_images.txt")
        return False

    # Read all required images
    with open(required_images_path, "r") as f:
        required_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    total_required = len(required_lines)
    print(f"Total images required: {total_required:,}")

    # Check which images exist
    print(f"Checking for existing images in: {image_folder}")

    if not image_folder.exists():
        print(f"WARNING: Image folder doesn't exist yet: {image_folder}")
        print(f"All {total_required:,} images need to be downloaded")
        downloaded_images = set()
    else:
        # Get all .jpg files
        downloaded_images = {f.stem for f in image_folder.glob("*.jpg")}

    num_downloaded = len(downloaded_images)
    print(f"Images already downloaded: {num_downloaded:,}")

    # Find images that still need downloading
    remaining = []
    for line in required_lines:
        # Extract image ID from format "train/IMAGE_ID"
        if "/" in line:
            image_id = line.split("/")[1]
        else:
            image_id = line

        if image_id not in downloaded_images:
            remaining.append(line)

    num_remaining = len(remaining)
    percent_complete = (num_downloaded / total_required * 100) if total_required > 0 else 0

    print(f"\n{'='*70}")
    print(f"Download Progress:")
    print(f"{'='*70}")
    print(f"  Downloaded: {num_downloaded:,} / {total_required:,} ({percent_complete:.1f}%)")
    print(f"  Remaining:  {num_remaining:,}")
    print(f"{'='*70}")

    if num_remaining == 0:
        print(f"\n✓ All images downloaded! No resume needed.")
        return True

    # Write remaining images to resume file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Add header comment
        f.write(f"# Resume file created: {Path(__file__).name}\n")
        f.write(f"# Downloaded: {num_downloaded:,} / {total_required:,}\n")
        f.write(f"# Remaining: {num_remaining:,} images\n")
        f.write(f"#\n")
        f.write(f"# Use with: python downloader.py {output_path.name} --download_folder=... --num_processes=24\n")
        f.write(f"#\n")

        # Write remaining images
        for line in remaining:
            f.write(f"{line}\n")

    print(f"\n✓ Created resume file: {output_path}")
    print(f"  Contains {num_remaining:,} images to download")
    print(f"\nTo continue downloading:")
    print(f"  python downloader.py {output_path} \\")
    print(f"      --download_folder={image_folder} \\")
    print(f"      --num_processes=24")
    print(f"\nOr submit SLURM job:")
    print(f"  sbatch scripts/openimages/download_openimages.sbatch")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Find resume point for OpenImages download",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example workflow:
  # 1. Find what needs downloading
  python scripts/find_download_resume_point.py

  # 2. Continue downloading
  python downloader.py data/resume_images.txt \\
      --download_folder=/home/hice1/ksohrab3/scratch/theworld/data/openimages \\
      --num_processes=24

Space-saving workflow (delete old images after training):
  # 1. Train on batch 1 (images 0-10K)
  sbatch scripts/train_slurm.sbatch configs/spatial_rgpt_training.json

  # 2. Before deleting, create resume file
  python scripts/find_download_resume_point.py
  # Creates: data/resume_images.txt (starting from image 10K+)

  # 3. Delete trained images to free space
  rm /path/to/openimages/batch1_*.jpg

  # 4. Continue downloading (uses resume file)
  python downloader.py data/resume_images.txt --download_folder=... --num_processes=24
        """,
    )

    parser.add_argument(
        "--required-images",
        type=Path,
        default=Path("data/required_images.txt"),
        help="Path to required_images.txt (default: %(default)s)",
    )

    parser.add_argument(
        "--image-folder",
        type=Path,
        default=Path("/home/hice1/ksohrab3/scratch/theworld/data/openimages"),
        help="Path to download folder (default: %(default)s)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/resume_images.txt"),
        help="Output file for resume list (default: %(default)s)",
    )

    args = parser.parse_args()

    success = find_resume_point(args.required_images, args.image_folder, args.output)

    if not success:
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
