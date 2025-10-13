# Architecture Documentation

Documentation for TheWorld's architecture, design decisions, and implementation details.

## Quick Links

- **[Overview](overview.md)** - Core architecture concepts and design
- **[Token Flow](token-flow.md)** - How data flows through the model with shapes
- **[Cosmos Integration](cosmos-integration.md)** - World model details
- **[Tokenization](tokenization.md)** - Special tokens and chat templates
- **[Implementation Notes](implementation-notes.md)** - Technical details and lessons learned

## Overview

TheWorld fuses two pretrained models:
- **Gemma 3** (4B) - Vision-language understanding
- **Cosmos** (2B) - World model for temporal dynamics

The fusion is achieved through trainable projection layers that map Cosmos's 16-dim latent space to Gemma's 2304-dim embedding space.

## Key Concepts

### Inheritance Pattern

TheWorld **inherits from** `Gemma3ForConditionalGeneration`, not wraps it:

```python
class TheWorld(Gemma3ForConditionalGeneration):
    # Extends Gemma3, adds world model capability
```

**Benefits:**
- Perfect equivalence to Gemma3 when world tokens absent
- Automatic device management
- Standard HuggingFace patterns

### Token Sequence

```
[BOS] <world_start> [world_tokens×784] <world_end> [image_tokens×256] [text] [answer] [EOS]
```

- World tokens provide temporal/spatial context
- Image tokens provide visual features
- Text tokens are the only ones with loss computed

### Training Strategy

Default: Train only projection layers (1.72% of parameters)
- Gemma vision: Frozen
- Gemma language: Frozen
- Cosmos VAE: Frozen
- Projection: Trainable

## Document Guide

### Start Here

New to TheWorld? Read in this order:
1. [Overview](overview.md) - Understand the big picture
2. [Token Flow](token-flow.md) - See how data flows
3. [Implementation Notes](implementation-notes.md) - Technical details

### Deep Dives

For specific topics:
- **Cosmos details**: [Cosmos Integration](cosmos-integration.md)
- **Special tokens**: [Tokenization](tokenization.md)
- **Parameter counts**: [Implementation Notes - Parameter Breakdown](implementation-notes.md#parameter-breakdown)
- **Initialization**: [Implementation Notes - Model Initialization](implementation-notes.md#model-initialization-pattern)

## Related Documentation

- [Training Guide](../training/README.md) - How to train the model
- [Evaluation Guide](../evaluation/overview.md) - How to evaluate
- [Getting Started](../guides/getting-started.md) - Setup and quickstart
