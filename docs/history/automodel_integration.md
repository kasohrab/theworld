# Making TheWorld Compatible with HuggingFace AutoModel

This document outlines how to enable `AutoModel.from_pretrained("username/theworld-vsr")` support for TheWorld checkpoints.

## Current Status

**What doesn't work:**
```python
from transformers import AutoModel, AutoProcessor

# ❌ This fails
model = AutoModel.from_pretrained("kasohrab/theworld-vsr")
```

**Why it fails:**
1. **No `config.json`**: AutoModel doesn't know what class to instantiate
2. **No model code on Hub**: Transformers doesn't have a "TheWorld" class
3. **Partial checkpoints**: Only projection layers are saved, not full model
4. **Custom architecture**: TheWorld combines Gemma + Cosmos in a non-standard way

**Current workaround:**
```python
from theworld import TheWorld

# ✅ This works
model = TheWorld.from_checkpoint_hub("kasohrab/theworld-vsr")
```

## Solution Approaches

There are three main approaches to enable AutoModel support:

### Option 1: Register TheWorld with Transformers (Recommended)

This approach makes TheWorld a first-class citizen in the transformers library.

**Steps:**

#### 1.1. Create a TheWorld configuration class

```python
# python/theworld/modeling/configuration_theworld.py

from transformers import PretrainedConfig

class TheWorldConfig(PretrainedConfig):
    """Configuration class for TheWorld model.

    This class stores configuration for TheWorld, which fuses:
    - Gemma 3 vision-language model
    - Cosmos world model
    - Custom projection layers
    """

    model_type = "theworld"

    def __init__(
        self,
        gemma_model_name: str = "google/gemma-3-4b-it",
        cosmos_model_name: str = "nvidia/Cosmos-Predict2-2B-Video2World",
        enable_world: bool = True,
        load_full_cosmos_pipeline: bool = True,
        freeze_gemma_vision: bool = True,
        freeze_gemma_language: bool = True,
        freeze_cosmos_vae: bool = True,
        num_world_steps: int = 0,
        max_world_steps: int = 16,
        random_projection_init: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.gemma_model_name = gemma_model_name
        self.cosmos_model_name = cosmos_model_name
        self.enable_world = enable_world
        self.load_full_cosmos_pipeline = load_full_cosmos_pipeline
        self.freeze_gemma_vision = freeze_gemma_vision
        self.freeze_gemma_language = freeze_gemma_language
        self.freeze_cosmos_vae = freeze_cosmos_vae
        self.num_world_steps = num_world_steps
        self.max_world_steps = max_world_steps
        self.random_projection_init = random_projection_init
```

#### 1.2. Register the configuration

```python
# python/theworld/__init__.py

from transformers import AutoConfig

# Register TheWorld configuration
AutoConfig.register("theworld", TheWorldConfig)
```

#### 1.3. Update save_pretrained() to save config

```python
# python/theworld/modeling/theworld.py

def save_pretrained(self, save_directory: str, **kwargs):
    """Save model with proper HuggingFace config."""
    import os
    from safetensors.torch import save_file

    os.makedirs(save_directory, exist_ok=True)

    # Save config.json (standard HuggingFace format)
    config = TheWorldConfig(
        gemma_model_name=self.gemma_model_name,
        cosmos_model_name=self.cosmos_model_name,
        enable_world=self.enable_world,
        freeze_gemma_vision=self.freeze_gemma_vision,
        freeze_gemma_language=self.freeze_gemma_language,
        freeze_cosmos_vae=self.freeze_cosmos_vae,
    )
    config.save_pretrained(save_directory)

    # Save model weights (trainable parameters only)
    state = self.state_dict()
    save_path = os.path.join(save_directory, "model.safetensors")
    save_file(state, save_path)

    # Save custom metadata (optional, for backward compatibility)
    config_data = {
        "model_config": {...},
        "freeze_config": {...},
    }
    with open(os.path.join(save_directory, "checkpoint_config.json"), "w") as f:
        json.dump(config_data, f, indent=2)
```

#### 1.4. Register TheWorld model class

```python
# python/theworld/__init__.py

from transformers import AutoModel

# Register TheWorld model
AutoModel.register(TheWorldConfig, TheWorld)
```

#### 1.5. Upload model code to Hub (trust_remote_code)

When pushing to Hub, include the model code:

```python
model.push_to_hub(
    "username/theworld-vsr",
    commit_message="Upload TheWorld checkpoint",
    # This will automatically upload modeling_theworld.py and configuration_theworld.py
)
```

**Then users can load with:**

```python
from transformers import AutoModel

# ✅ This will now work!
model = AutoModel.from_pretrained(
    "username/theworld-vsr",
    trust_remote_code=True,  # Required for custom model code
)
```

**Pros:**
- ✅ Standard HuggingFace workflow
- ✅ Works with `AutoModel`, `pipeline()`, etc.
- ✅ Model code versioned with checkpoint

**Cons:**
- ❌ Requires `trust_remote_code=True` (security concern for users)
- ❌ Need to upload model code to every checkpoint repo
- ❌ Still need to reload base models (Gemma/Cosmos) on load

---

### Option 2: Save Full Model Checkpoints

Save the entire model (including frozen Gemma/Cosmos) instead of just projection layers.

**Implementation:**

```python
def save_pretrained(self, save_directory: str, save_full_model: bool = False, **kwargs):
    """Save model checkpoint.

    Args:
        save_full_model: If True, save entire model including frozen layers.
                        If False (default), save only trainable parameters.
    """
    os.makedirs(save_directory, exist_ok=True)

    if save_full_model:
        # Save ALL parameters (6B+ params)
        state = self.state_dict()  # Includes Gemma + Cosmos

        # Save config that describes the full architecture
        config = TheWorldConfig(...)
        config.save_pretrained(save_directory)

        # Save weights
        save_file(state, os.path.join(save_directory, "model.safetensors"))

        print(f"✓ Full model saved ({sum(p.numel() for p in self.parameters())} params)")
    else:
        # Current behavior: save only trainable params
        trainable_state = {k: v for k, v in self.state_dict().items()
                          if any(p.requires_grad for p in [self.get_parameter(k)])}
        save_file(trainable_state, os.path.join(save_directory, "model.safetensors"))
```

**Usage:**

```python
# Save full checkpoint (large, but works with AutoModel)
model.save_pretrained("./checkpoints/full", save_full_model=True)

# Push to Hub
model.push_to_hub("username/theworld-vsr-full", save_full_model=True)
```

**Pros:**
- ✅ Works with standard AutoModel (no custom code needed)
- ✅ Self-contained checkpoint
- ✅ No need to re-download base models

**Cons:**
- ❌ **Huge checkpoint size**: ~6GB instead of 146MB
- ❌ Wastes bandwidth/storage (frozen weights are unchanged)
- ❌ Still need AutoModel registration for model type

---

### Option 3: Create a Transformers-Native Integration (Long-term)

Submit TheWorld to the transformers library as an official model type.

**Steps:**

1. **Submit PR to transformers**:
   - Add `modeling_theworld.py` to `transformers/models/theworld/`
   - Add `configuration_theworld.py`
   - Add tests in `tests/models/theworld/`

2. **Follow transformers contribution guidelines**:
   - https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md
   - Add documentation to transformers docs

3. **Once merged**, users can use:
   ```python
   from transformers import TheWorldForConditionalGeneration, AutoModel

   # Native support, no trust_remote_code needed
   model = AutoModel.from_pretrained("username/theworld-vsr")
   ```

**Pros:**
- ✅ Official transformers support
- ✅ No `trust_remote_code` needed
- ✅ Better discoverability
- ✅ Maintained by transformers team

**Cons:**
- ❌ Long review process (weeks to months)
- ❌ High bar for acceptance (needs documentation, tests, etc.)
- ❌ Requires model to be stable/mature

---

## Recommended Approach

**For immediate use:**
- Use **Option 1** (register with trust_remote_code)
- Minimal code changes
- Works with current partial checkpoint design

**For production/long-term:**
- Pursue **Option 3** (transformers PR)
- Standard, no security warnings
- Better for community adoption

**Avoid Option 2** unless you have unlimited storage/bandwidth - 6GB checkpoints are impractical.

---

## Implementation Checklist

To implement Option 1 (recommended for now):

- [ ] Create `configuration_theworld.py` with `TheWorldConfig` class
- [ ] Update `save_pretrained()` to save `config.json`
- [ ] Register config with `AutoConfig.register()`
- [ ] Register model with `AutoModel.register()`
- [ ] Test loading with `AutoModel.from_pretrained(..., trust_remote_code=True)`
- [ ] Update documentation with AutoModel usage examples
- [ ] Add `modeling_theworld.py` to Hub repos (auto-uploaded by `push_to_hub`)

---

## Example: Before and After

**Before (current):**
```python
# Requires theworld package installed
from theworld import TheWorld

model = TheWorld.from_checkpoint_hub("kasohrab/theworld-vsr")
```

**After (with AutoModel):**
```python
# Works with just transformers
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "kasohrab/theworld-vsr",
    trust_remote_code=True,  # Needed because TheWorld isn't in transformers yet
)
```

---

## References

- [Transformers Custom Models Guide](https://huggingface.co/docs/transformers/custom_models)
- [AutoModel Registration](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModel)
- [trust_remote_code Security](https://huggingface.co/docs/hub/security-remote-code)
- [Contributing to Transformers](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md)
