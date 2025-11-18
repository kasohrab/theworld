# Guide: Adding New Configuration Parameters

This guide provides a complete checklist for adding new configuration parameters to TheWorld, ensuring they work correctly across initialization, training, checkpoint saving/loading, and Hub uploads.

## Overview

Adding a config parameter requires updating 6 categories of files across the codebase:

1. **Config Classes** (2 files) - Define the parameter
2. **Model Code** (2-3 files) - Use the parameter
3. **Training Configs** (all .json files) - Provide default values
4. **Documentation** (1-2 files) - Explain the parameter
5. **Tests** (1+ files) - Verify it works
6. **Validation** - Test backward compatibility

## Complete Checklist

### Phase 1: Config Definition

#### 1.1 Add to TrainingConfig (`python/theworld/config.py`)

```python
@dataclass
class TrainingConfig:
    """..."""

    # Add to docstring under appropriate section
    # my_parameter: Description of parameter and valid values

    # Add field with default value
    my_parameter: str = "default_value"  # Comment explaining choices
```

**Important:**
- Place in appropriate docstring section (Model Configuration, Training Hyperparameters, etc.)
- **Always provide a default value** for backward compatibility
- Use type hints (`str`, `bool`, `int`, `float`, `Optional[str]`, etc.)

#### 1.2 Add to TheWorldConfig (`python/theworld/modeling/config.py`)

```python
class TheWorldConfig(Gemma3Config):
    def __init__(
        self,
        gemma_model_name: Optional[str] = DEFAULT_GEMMA_MODEL,
        # ... existing parameters ...
        my_parameter: str = "default_value",  # Add here
        **kwargs,
    ):
        self.gemma_model_name = gemma_model_name
        # ... existing assignments ...
        self.my_parameter = my_parameter  # Store as instance variable
        super().__init__(**kwargs)
```

**Important:**
- Match the default value from TrainingConfig exactly
- Add to `__init__` parameter list
- Store as instance variable (`self.my_parameter = my_parameter`)
- This class is serialized to `config.json` in checkpoints

### Phase 2: Model Implementation

#### 2.1 Update `TheWorld.from_pretrained()` (`python/theworld/modeling/theworld.py`)

**Add parameter to method signature:**
```python
@classmethod
def from_pretrained(
    cls,
    pretrained_model_name_or_path: str,
    *model_args: Any,
    # ... existing parameters ...
    my_parameter: str = "default_value",  # Add here
    **kwargs: Any
) -> "TheWorld":
    """
    Args:
        # ... existing args ...
        my_parameter: Description of parameter
    """
```

**Add to docstring:**
```python
    Args:
        pretrained_model_name_or_path: HuggingFace model ID or path
        # ... existing args ...
        my_parameter: Description of what this parameter does
```

**Remove from gemma_config_dict (Case A - New Model):**
```python
# Create TheWorldConfig from Gemma config
gemma_config_dict = model.config.to_dict()
# Remove any conflicting keys that we're setting explicitly
gemma_config_dict.pop("gemma_model_name", None)
# ... existing pops ...
gemma_config_dict.pop("my_parameter", None)  # Add this line
```

**Pass to TheWorldConfig:**
```python
the_world_config = TheWorldConfig(
    gemma_model_name=pretrained_model_name_or_path,
    # ... existing parameters ...
    my_parameter=my_parameter,  # Add this line
    **gemma_config_dict
)
```

#### 2.2 Update `from_checkpoint()` (`python/theworld/modeling/theworld.py`)

**Pass parameter from loaded config:**
```python
# Stage 1: Initialize from base models
model = cls.from_pretrained(
    config.gemma_model_name,
    device=device,
    # ... existing parameters ...
    my_parameter=getattr(config, "my_parameter", "default_value"),  # Add this line
    **kwargs
)
```

**Important:** Use `getattr(config, "my_parameter", "default_value")` for backward compatibility with old checkpoints that don't have this field.

#### 2.3 Use Parameter in Implementation

Use the parameter wherever needed in the model code:

```python
# Access from config
if model.config.my_parameter == "some_value":
    # Do something
    pass

# Pass to submodules
my_module = MyModule(
    param=model.config.my_parameter,
    ...
)
```

### Phase 3: Training Configurations

#### 3.1 Update ALL JSON Config Files

Add the parameter to every config file in `configs/`:

```bash
# Quick method (automated)
for file in configs/*.json; do
  # Add your parameter after an existing field
  sed -i '/"existing_field":/a\  "my_parameter": "default_value",' "$file"
done

# Or manual method
# Edit each file individually to add:
{
  "model_name": "google/gemma-3-4b-it",
  "my_parameter": "default_value",  // Add this line
  ...
}
```

**Files to update:**
- `configs/default.json`
- `configs/smoke_test.json`
- `configs/datacomp_production.json`
- `configs/datacomp_test.json`
- All other config files in `configs/` directory

**Important:**
- Use the same default value everywhere for consistency
- Place logically near related parameters
- Add comment if value choices need explanation

### Phase 4: Documentation

#### 4.1 Update CLAUDE.md

Add documentation in the appropriate section:

```markdown
### My New Parameter

Description of what this parameter does and when to use it.

**Options:**
- `"option1"`: Description of option 1
- `"option2"`: Description of option 2

**Configuration:**
```json
{
  "my_parameter": "option1"
}
```

**Usage:**
```python
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    my_parameter="option2",
)
```

See `python/theworld/path/to/implementation.py` for details.
```

**Sections in CLAUDE.md:**
- Model Configuration (for model architecture choices)
- Training Hyperparameters (for training settings)
- Key Implementation Details (for technical explanations)

### Phase 5: Testing

#### 5.1 Create Unit Tests (`tests/test_*.py`)

Create tests for the new parameter:

```python
def test_my_parameter_default():
    """Test that default value works."""
    model = TheWorld.from_pretrained(
        "google/gemma-3-4b-it",
        enable_world=True,
    )
    assert model.config.my_parameter == "default_value"


def test_my_parameter_custom():
    """Test that custom value works."""
    model = TheWorld.from_pretrained(
        "google/gemma-3-4b-it",
        enable_world=True,
        my_parameter="custom_value",
    )
    assert model.config.my_parameter == "custom_value"


def test_my_parameter_checkpoint_compatibility():
    """Test loading checkpoint without parameter (backward compatibility)."""
    # Load checkpoint that doesn't have my_parameter field
    model = TheWorld.from_checkpoint(old_checkpoint_path)
    # Should default to "default_value"
    assert model.config.my_parameter == "default_value"
```

#### 5.2 Update Example Scripts

Add parameter to example scripts if relevant:

```python
# examples/simple_training.py

# Option 1: Using default
model = TheWorld.from_pretrained("google/gemma-3-4b-it")

# Option 2: Custom value
model = TheWorld.from_pretrained(
    "google/gemma-3-4b-it",
    my_parameter="custom_value",  # Use custom value
)
```

### Phase 6: Validation

#### 6.1 Test Backward Compatibility

```bash
# 1. Load existing checkpoint without the parameter
python -c "
from theworld import TheWorld
model = TheWorld.from_checkpoint('path/to/old/checkpoint')
print(f'Parameter value: {model.config.my_parameter}')
assert model.config.my_parameter == 'default_value'
print('✓ Backward compatibility verified')
"

# 2. Verify checkpoint saves new parameter
python scripts/train_hf.py --config configs/smoke_test.json

# 3. Load new checkpoint and verify parameter persists
python -c "
from theworld import TheWorld
model = TheWorld.from_checkpoint('./checkpoints/smoke_test/checkpoint-1')
print(f'Parameter value: {model.config.my_parameter}')
assert model.config.my_parameter == 'mlp'
print('✓ Parameter persistence verified')
"
```

#### 6.2 Test Different Values

```bash
# Test with each valid value
for value in option1 option2 option3; do
  echo "Testing with $value..."
  python -c "
from theworld import TheWorld
model = TheWorld.from_pretrained(
    'google/gemma-3-4b-it',
    enable_world=True,
    my_parameter='$value'
)
print(f'✓ {value} works')
"
done
```

#### 6.3 Run Full Test Suite

```bash
# Format code
make format

# Type checking
make typecheck

# Run tests
pytest tests/ -v

# Smoke test (full pipeline)
export HF_TOKEN=hf_your_token
make smoke-test
```

## Complete Example: `projection_architecture` Parameter

Here's a real example of adding the `projection_architecture` parameter:

### Files Modified (15 total):

**Config Classes:**
1. `python/theworld/config.py` - Added field to TrainingConfig
2. `python/theworld/modeling/config.py` - Added to TheWorldConfig.__init__()

**Model Code:**
3. `python/theworld/modeling/spatial_reducer.py` - Added to WorldProjectionConfig
4. `python/theworld/modeling/world_projector.py` - Implemented _build_projection()
5. `python/theworld/modeling/theworld.py` - Updated from_pretrained() and from_checkpoint()

**Training Configs:**
6-15. All JSON files in `configs/` directory

**Documentation:**
16. `CLAUDE.md` - Added "Projection Architectures" section
17. `docs/guides/adding-config-parameters.md` - This guide

**Tests:**
18. `tests/test_world_projector.py` - Architecture tests

## Common Pitfalls

1. **Forgetting to add default value** → Old checkpoints fail to load
2. **Mismatched defaults** → Inconsistent behavior between config classes
3. **Not using getattr() in from_checkpoint()** → Old checkpoints crash
4. **Forgetting to pop() in from_pretrained()** → Conflicts with Gemma config
5. **Not updating all JSON configs** → Inconsistent training behavior
6. **No backward compatibility tests** → Breaking changes for existing users

## Verification Checklist

- [ ] TrainingConfig has parameter with default value
- [ ] TheWorldConfig has parameter with same default
- [ ] from_pretrained() accepts parameter and passes to TheWorldConfig
- [ ] from_pretrained() pops parameter from gemma_config_dict
- [ ] from_checkpoint() uses getattr() with default for backward compatibility
- [ ] All JSON configs have the parameter
- [ ] CLAUDE.md documents the parameter
- [ ] Tests verify default value works
- [ ] Tests verify custom values work
- [ ] Tests verify backward compatibility
- [ ] make format passes
- [ ] make typecheck passes
- [ ] Smoke test passes

## References

- **Config Serialization**: TheWorldConfig extends Gemma3Config → inherits HuggingFace save/load
- **Backward Compatibility**: Use `getattr(config, "param", default)` pattern
- **Training Flow**: TrainingConfig → from_pretrained() → TheWorldConfig → checkpoint
- **Example PR**: See `projection_architecture` implementation (commit: adding projection architecture config)
