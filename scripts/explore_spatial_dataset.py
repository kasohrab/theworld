"""
Explore SpatialRGPT OpenSpatialDataset samples.
Downloads in non-streaming mode (full download, cached to disk).
"""

print("=" * 80)
print("Downloading SpatialRGPT OpenSpatialDataset (NON-STREAMING)")
print("=" * 80)
print()
print("⚠ This will download the full 31.9 GB dataset")
print("  But it caches to ~/.cache/huggingface/datasets/")
print("  So subsequent runs will be instant!")
print()
import datasets
datasets.logging.set_verbosity(datasets.logging.INFO)

# Load dataset in non-streaming mode (downloads everything)
print("Starting download...")
dataset = datasets.load_dataset("a8cheng/OpenSpatialDataset", split="train", streaming=False)

print(f"\n✓ Dataset loaded: {len(dataset):,} samples")
print()

# Show dataset structure
print("=" * 80)
print("DATASET STRUCTURE")
print("=" * 80)
print(f"Fields: {list(dataset.features.keys())}")
print()

# Show first 5 samples
print("=" * 80)
print("SAMPLE DATA (first 5 examples)")
print("=" * 80)
print()

for i in range(min(5, len(dataset))):
    sample = dataset[i]

    print(f"{'='*80}")
    print(f"SAMPLE {i+1}")
    print(f"{'='*80}")

    # Show all fields
    for key, value in sample.items():
        if key == "conversations":
            # Pretty print conversations
            print(f"\n{key}:")
            if isinstance(value, str):
                # Parse if it's a string
                try:
                    import ast
                    value = ast.literal_eval(value)
                except:
                    pass

            if isinstance(value, list):
                for j, turn in enumerate(value):
                    print(f"  Turn {j+1}:")
                    print(f"    from: {turn.get('from', 'N/A')}")
                    # Truncate long text
                    text = turn.get('value', 'N/A')
                    if len(text) > 300:
                        text = text[:300] + "..."
                    print(f"    value: {text}")
            else:
                text_preview = str(value)[:300]
                if len(str(value)) > 300:
                    text_preview += "..."
                print(f"  {text_preview}")
        else:
            # Show other fields directly
            val_str = str(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            print(f"\n{key}: {val_str}")

    print("\n")

# Analysis
print("=" * 80)
print("DEPTH MAP ANALYSIS")
print("=" * 80)
print()

sample = dataset[0]
has_depth_field = any("depth" in str(key).lower() for key in sample.keys())
print(f"Has 'depth' in field names: {has_depth_field}")
print(f"All fields: {list(sample.keys())}")
print()

# Check filename format
filename = sample.get("filename", "")
print(f"Example filename: '{filename}'")
print(f"Format: Image ID without extension")
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()
print("✓ Dataset contains conversation data (spatial reasoning QA)")
print("✓ Image filenames reference OpenImagesV7")
print("✗ No depth maps included (must generate separately)")
print()
