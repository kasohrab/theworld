# Historical Documentation

This directory contains documentation of completed work, design decisions, and historical context for TheWorld development.

## Documents

### [Refactoring (January 2025)](refactoring-2025-01.md)

Complete summary of the major refactoring completed in January 2025.

**What changed:**
- Architecture: Composition → Inheritance pattern
- Interface: Standardized to match Gemma3 API
- Initialization: Fixed `from_pretrained()` pattern
- Compatibility: Made compatible with HuggingFace patterns

**Impact:**
- Code reduction: ~200 lines removed
- Correctness: Perfect equivalence to Gemma3 when world tokens absent
- Maintainability: Easier to understand and extend

### [AutoModel Integration](automodel-integration.md)

Design document for enabling HuggingFace AutoModel support.

**Goal:**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("username/theworld-vsr")
```

**Status:** Not yet implemented, workaround exists

### [World-Aware Generation](world_aware_generation_plan.md)

Plan for implementing world-aware generation using prefix-KV cache approach.

**Goal:** Enable world embeddings during autoregressive generation
**Status:** Design phase

## Why Keep Historical Docs?

Historical documentation helps:
1. **Understand context** - Why decisions were made
2. **Avoid rework** - What approaches were tried and abandoned
3. **Learn lessons** - Mistakes and insights from past work
4. **Onboard contributors** - Show evolution of project

## When to Archive

Move documentation to history/ when:
- Work is completed and merged
- Design is implemented or abandoned
- Document is superseded by current docs
- Information is historical context, not active guidance

## Archived Explorations

See `../archive/` for older design explorations and prototypes that didn't make it into the main codebase.

## Related Documentation

- [Main Documentation](../README.md) - Current documentation index
- [Architecture Overview](../architecture/overview.md) - Current architecture
- [Archive](../archive/) - Old design explorations
