"""Quick diagnostic to check model structure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from theworld import TheWorld

print("Loading model...")
model = TheWorld("google/gemma-3-4b-it")

print("\n=== Checking for lm_head ===")
print(f"Has gemma.lm_head: {hasattr(model.gemma, 'lm_head')}")

print("\n=== All parameters with 'lm_head' in name ===")
lm_head_params = [(name, param.requires_grad) for name, param in model.named_parameters() if 'lm_head' in name.lower()]
if lm_head_params:
    for name, req_grad in lm_head_params[:10]:
        print(f"  {name}: requires_grad={req_grad}")
else:
    print("  No parameters found with 'lm_head' in name!")

print("\n=== Trainable parameters ===")
trainable = [(name, p.numel()) for name, p in model.named_parameters() if p.requires_grad]
print(f"Total trainable params: {len(trainable)}")
for name, size in trainable[:10]:
    print(f"  {name}: {size:,} params")

print("\n=== Checking gemma model structure ===")
print(f"Type: {type(model.gemma)}")
print(f"Attributes: {[attr for attr in dir(model.gemma) if not attr.startswith('_')][:20]}")
