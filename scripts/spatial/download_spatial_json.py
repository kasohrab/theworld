#!/usr/bin/env python3
"""
Download OpenSpatialDataset JSON for training.

Downloads result_10_depth_convs.json (~30GB) to shared location for multi-GPU training.
All ranks can access the same file, avoiding per-rank $TMPDIR issues.

Usage:
    # Download to /tmp (default, shared on node)
    python scripts/spatial/download_spatial_json.py

    # Download to custom location
    python scripts/spatial/download_spatial_json.py --output /path/to/scratch/openspatial

    # Skip download if exists
    python scripts/spatial/download_spatial_json.py --skip-if-exists

    # Check if file exists without downloading
    python scripts/spatial/download_spatial_json.py --check-only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def download_json(output_dir: Path, skip_if_exists: bool = False) -> Path:
    """Download result_10_depth_convs.json to output directory.

    Args:
        output_dir: Directory to download to (e.g., /tmp/openspatial)
        skip_if_exists: Skip download if file already exists

    Returns:
        Path to downloaded JSON file

    Raises:
        RuntimeError: If download fails
    """
    output_dir = Path(output_dir)
    json_path = output_dir / "result_10_depth_convs.json"

    # Check if already exists
    if json_path.exists():
        file_size_gb = json_path.stat().st_size / (1024**3)
        if skip_if_exists:
            print(f"✓ JSON already exists ({file_size_gb:.1f} GB): {json_path}")
            return json_path
        else:
            print(f"⚠ JSON already exists ({file_size_gb:.1f} GB): {json_path}")
            print("  Use --skip-if-exists to skip re-download")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading OpenSpatialDataset JSON to {output_dir}...")
    print("  Source: HuggingFace Hub (a8cheng/OpenSpatialDataset)")
    print("  File: result_10_depth_convs.json (~30GB)")
    print("  This may take 5-10 minutes...")

    try:
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                "a8cheng/OpenSpatialDataset",
                "result_10_depth_convs.json",
                "--repo-type",
                "dataset",
                "--local-dir",
                str(output_dir),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to download OpenSpatialDataset JSON: {e}\n"
            f"Check:\n"
            f"  1. HuggingFace token is set (HF_TOKEN or ~/.hf_token)\n"
            f"  2. Network connectivity\n"
            f"  3. Disk space available: {output_dir}"
        ) from e

    # Verify download
    if not json_path.exists():
        raise RuntimeError(f"Download completed but file not found: {json_path}\n" f"Check disk space and permissions")

    file_size_gb = json_path.stat().st_size / (1024**3)
    print(f"✓ Download complete ({file_size_gb:.1f} GB): {json_path}")

    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="Download OpenSpatialDataset JSON for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download to /tmp (default)
  python scripts/spatial/download_spatial_json.py

  # Skip re-download if exists
  python scripts/spatial/download_spatial_json.py --skip-if-exists

  # Download to scratch space
  python scripts/spatial/download_spatial_json.py --output data/openspatial

  # Check if file exists
  python scripts/spatial/download_spatial_json.py --check-only
        """,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/openspatial"),
        help="Output directory (default: /tmp/openspatial)",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip download if file already exists",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check if file exists without downloading",
    )

    args = parser.parse_args()

    # Check-only mode
    if args.check_only:
        json_path = args.output / "result_10_depth_convs.json"
        if json_path.exists():
            file_size_gb = json_path.stat().st_size / (1024**3)
            print(f"✓ JSON exists ({file_size_gb:.1f} GB): {json_path}")
            sys.exit(0)
        else:
            print(f"✗ JSON not found: {json_path}")
            sys.exit(1)

    # Download
    try:
        json_path = download_json(args.output, skip_if_exists=args.skip_if_exists)
        print(f"\n✓ JSON ready for training: {json_path}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Download failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
