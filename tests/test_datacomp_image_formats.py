"""Test DataComp image format handling.

This test verifies that the dataset loader and collator can handle
various PIL image formats (RGB, grayscale, palette, RGBA, etc.) without crashing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from theworld import TheWorld, TheWorldDataset, create_theworld_collator


def create_test_images():
    """Create test images in various PIL formats."""
    size = (224, 224)
    images = {}

    # RGB image (normal case)
    rgb_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    images["RGB"] = Image.fromarray(rgb_array, mode="RGB")

    # Grayscale (L mode) - this was causing crashes
    gray_array = np.random.randint(0, 255, size, dtype=np.uint8)
    images["L (grayscale)"] = Image.fromarray(gray_array, mode="L")

    # Grayscale with alpha (LA mode)
    la_array = np.random.randint(0, 255, (*size, 2), dtype=np.uint8)
    images["LA (gray+alpha)"] = Image.fromarray(la_array, mode="LA")

    # Palette mode (P) - common in web images
    p_img = Image.new("P", size)
    p_img.putpalette([i % 256 for i in range(768)])  # Simple palette
    for x in range(size[0]):
        for y in range(size[1]):
            p_img.putpixel((x, y), (x + y) % 256)
    images["P (palette)"] = p_img

    # RGBA (transparency)
    rgba_array = np.random.randint(0, 255, (*size, 4), dtype=np.uint8)
    images["RGBA (alpha)"] = Image.fromarray(rgba_array, mode="RGBA")

    # CMYK (printer color space)
    cmyk_array = np.random.randint(0, 255, (*size, 4), dtype=np.uint8)
    images["CMYK"] = Image.fromarray(cmyk_array, mode="CMYK")

    return images


def test_image_format_conversion():
    """Test that all image formats can be converted to RGB."""
    print("\n" + "=" * 60)
    print("Test 1: Image Format Conversion")
    print("=" * 60)

    images = create_test_images()

    for name, img in images.items():
        print(f"\nTesting {name} (mode={img.mode}, size={img.size}, bands={len(img.getbands())})")

        # Test direct conversion
        try:
            if img.mode != "RGB":
                if img.mode in ("P", "PA"):
                    converted = img.convert("RGBA").convert("RGB")
                else:
                    converted = img.convert("RGB")
            else:
                converted = img

            # Validate
            assert converted.mode == "RGB", f"Conversion failed: got mode {converted.mode}"
            assert len(converted.getbands()) == 3, f"Conversion failed: got {len(converted.getbands())} bands"
            print(f"  ✓ Converted successfully to RGB")

        except Exception as e:
            print(f"  ✗ Conversion failed: {e}")
            raise


def test_dataset_wrapper():
    """Test that TheWorldDataset handles various image formats."""
    print("\n" + "=" * 60)
    print("Test 2: Dataset Wrapper")
    print("=" * 60)

    images = create_test_images()

    # Create dataset with mixed image formats
    data = []
    for name, img in images.items():
        data.append({"image": img, "text": f"What is in this {name} image?", "label": "A test image"})

    dataset = TheWorldDataset(data)

    print(f"\nCreated dataset with {len(dataset)} samples")

    # Test __getitem__
    for i in range(len(dataset)):
        item = dataset[i]
        img = item["image"]
        print(f"  Sample {i}: mode={img.mode}, size={img.size}, bands={len(img.getbands())}")

        # Dataset should return PIL images (not yet converted)
        assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"

    print(f"\n✓ All {len(dataset)} samples loaded successfully")


def test_collator():
    """Test that the collator handles various image formats without crashing."""
    print("\n" + "=" * 60)
    print("Test 3: Collator (Critical Test)")
    print("=" * 60)

    # Initialize model (needed for collator)
    print("\nInitializing TheWorld model...")
    model = TheWorld(
        "google/gemma-3-4b-it",
        cosmos_model_name="nvidia/Cosmos-Predict2-2B-Video2World",
        device="cuda:1",
        load_full_cosmos_pipeline=True,
    )
    print(f"✓ Model loaded")

    # Create collator
    collate_fn = create_theworld_collator(model)
    print(f"✓ Collator created")

    # Create test batches with different image formats
    images = create_test_images()

    # Test each format individually first
    print("\n--- Testing individual formats ---")
    for name, img in images.items():
        print(f"\nTesting {name} (mode={img.mode})...")
        batch = [{"image": img, "text": "What is in this image?", "label": "A test image"} for _ in range(2)]

        try:
            result = collate_fn(batch)

            # Validate output
            assert "input_ids" in result
            assert "attention_mask" in result
            assert "pixel_values" in result
            assert "images" in result

            print(f"  ✓ Collator succeeded")
            print(f"    - input_ids: {result['input_ids'].shape}")
            print(f"    - pixel_values: {result['pixel_values'].shape}")
            print(f"    - images: {len(result['images'])} PIL images")

        except Exception as e:
            print(f"  ✗ Collator failed: {e}")
            raise

    # Test mixed batch (most realistic scenario)
    print("\n--- Testing mixed batch (all formats together) ---")
    mixed_batch = []
    for name, img in images.items():
        mixed_batch.append({"image": img, "text": f"Describe this {name} image.", "label": f"A {name} test"})

    print(f"\nTesting batch with {len(mixed_batch)} different image formats...")
    try:
        result = collate_fn(mixed_batch)

        print(f"✓ Mixed batch succeeded!")
        print(f"  - input_ids: {result['input_ids'].shape}")
        print(f"  - pixel_values: {result['pixel_values'].shape}")
        print(f"  - images: {len(result['images'])} PIL images")

        # All images should now be RGB after collator processing
        for i, img in enumerate(result["images"]):
            if isinstance(img, Image.Image):
                print(f"  - Image {i}: mode={img.mode}, channels={len(img.getbands())}")

    except Exception as e:
        print(f"✗ Mixed batch failed: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DataComp Image Format Handling Test")
    print("=" * 60)
    print("\nThis test verifies that the pipeline handles:")
    print("  - RGB images (normal)")
    print("  - Grayscale (L mode)")
    print("  - Grayscale with alpha (LA mode)")
    print("  - Palette images (P mode) - common in DataComp")
    print("  - RGBA images (transparency)")
    print("  - CMYK images (printer color space)")

    try:
        # Run tests
        test_image_format_conversion()
        test_dataset_wrapper()
        test_collator()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe fix is working correctly. Safe to run production training.")
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
