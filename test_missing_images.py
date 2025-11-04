"""Test that training handles missing images gracefully."""

import torch
from pathlib import Path
from theworld.datasets import SpatialRGPTDataset
from theworld.data import create_theworld_collator
from theworld import TheWorld

print("=" * 60)
print("Testing Missing Image Handling")
print("=" * 60)

# 1. Load dataset with skip_missing_images=True
print("\n1. Loading dataset with skip_missing_images=True...")
dataset = SpatialRGPTDataset(
    data_source="/home/hice1/ksohrab3/scratch/theworld/data/result_10_depth_convs.json",
    image_folder="/home/hice1/ksohrab3/scratch/theworld/data/openimages",
    num_samples=100,
    draw_bboxes=False,
    skip_missing_images=True,  # This makes dataset return None for missing images
)
print(f"✓ Dataset loaded: {len(dataset)} samples")

# 2. Check first few samples
print("\n2. Checking first 10 samples for missing images...")
none_count = 0
valid_count = 0
for i in range(min(10, len(dataset))):
    sample = dataset[i]
    if sample is None:
        none_count += 1
        print(f"  Sample {i}: MISSING (returns None)")
    else:
        valid_count += 1
        print(f"  Sample {i}: OK (has image)")

print(f"\nSummary: {valid_count} valid, {none_count} missing out of 10 samples")

if none_count == 10:
    print("\n⚠️  WARNING: All 10 samples have missing images!")
    print("   This means no images have been downloaded yet.")
    print("   Training will fail until images are downloaded.")
    exit(1)

# 3. Test collator with samples containing None
print("\n3. Testing collator with mixed batch (some None, some valid)...")

# Load model to get processor
print("  Loading model (for processor)...")
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    enable_world=True,
    dtype=torch.bfloat16,
    device_map="cpu",  # Use CPU for testing
)
print("  ✓ Model loaded")

# Create collator
collator = create_theworld_collator(model, max_length=2048)

# Create test batch with mix of valid and None samples
print("\n  Creating test batch...")
test_batch = []
for i in range(20):
    sample = dataset[i]
    test_batch.append(sample)

none_in_batch = sum(1 for s in test_batch if s is None)
valid_in_batch = sum(1 for s in test_batch if s is not None)
print(f"  Test batch: {valid_in_batch} valid, {none_in_batch} None samples")

# Try collating
print("\n  Collating batch...")
try:
    collated = collator(test_batch)
    print(f"  ✓ Collation succeeded!")
    print(f"    Batch size: {collated['input_ids'].shape[0]}")
    print(f"    Sequence length: {collated['input_ids'].shape[1]}")
    print(f"    Input IDs shape: {collated['input_ids'].shape}")
    print(f"    Attention mask shape: {collated['attention_mask'].shape}")
    print(f"    Labels shape: {collated['labels'].shape}")
    print(f"    Images: {len(collated['images'])}")

    # Check for empty batch
    if collated['input_ids'].shape[0] == 0:
        print("\n  ⚠️  WARNING: Collated batch is EMPTY!")
        print("     This happens when all samples in batch have missing images.")
        print("     Model forward pass needs to handle this.")
except Exception as e:
    print(f"  ✗ Collation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. Test model forward pass with the collated batch
print("\n4. Testing model forward pass with collated batch...")
try:
    # Move to device
    device = next(model.parameters()).device
    input_ids = collated['input_ids'].to(device)
    attention_mask = collated['attention_mask'].to(device)
    labels = collated['labels'].to(device)
    pixel_values = collated['pixel_values'].to(device)
    images = collated['images']

    if input_ids.shape[0] == 0:
        print("  Skipping forward pass - empty batch")
        print("  Model needs to handle empty batches in forward()")
    else:
        print(f"  Running forward pass with batch_size={input_ids.shape[0]}...")
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            images=images,
        )
        print(f"  ✓ Forward pass succeeded!")
        print(f"    Loss: {outputs.loss.item():.4f}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
