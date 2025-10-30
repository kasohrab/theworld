# TheWorld Documentation

Welcome to the **TheWorld** documentation! TheWorld is a novel fused vision-language-world model that combines Google's Gemma 3 vision-language model with NVIDIA's Cosmos world model.

## Quick Navigation

### Getting Started
- **[Getting Started Guide](guides/getting-started.md)** - First-time setup and quickstart
- **[Inference Guide](guides/inference.md)** - How to run inference with TheWorld
- **[Troubleshooting](guides/troubleshooting.md)** - Common issues and solutions

### Architecture
- **[Architecture Overview](architecture/overview.md)** - Core concepts and design
- **[Token Flow](architecture/token-flow.md)** - How data flows through the model
- **[Cosmos Integration](architecture/cosmos-integration.md)** - World model details
- **[Tokenization](architecture/tokenization.md)** - Special tokens and chat templates
- **[Implementation Notes](architecture/implementation-notes.md)** - Technical details

### Training
- **[Training Guide](training/README.md)** - Overview and quickstart
- **[Infrastructure](training/infrastructure.md)** - Training design and setup
- **[Multi-Stage Training](training/multi-stage.md)** - Progressive unfreezing strategy
- **[Distributed Training](training/distributed.md)** - Accelerate and multi-GPU
- **[Hub Upload](training/hub-upload.md)** - Publishing to HuggingFace
- **[Datasets](training/datasets/)** - DataComp, SpatialRGPT, and more

### Evaluation
- **[Evaluation Overview](evaluation/overview.md)** - Metrics and baselines
- **[BLINK Benchmark](evaluation/benchmarks/blink.md)** - BLINK evaluation
- **[SpatialRGPT Benchmark](evaluation/benchmarks/spatial-rgpt.md)** - Spatial reasoning evaluation
- **[Baseline Comparisons](evaluation/baselines.md)** - Comparing against baselines

### Reference
- **[Project Instructions (CLAUDE.md)](../CLAUDE.md)** - Main project guide for development
- **[History](history/)** - Refactoring history and completed work
- **[Archive](archive/)** - Old explorations and design alternatives

## Documentation Organization

```
docs/
├── README.md (you are here)
├── guides/              # User-facing guides
├── architecture/        # Architecture and design docs
├── training/            # Training documentation
├── evaluation/          # Evaluation and benchmarks
├── history/             # Historical documentation (completed work)
└── archive/             # Archived explorations
```

## Quick Links

**For new users:**
1. Start with [Getting Started Guide](guides/getting-started.md)
2. Read [Architecture Overview](architecture/overview.md)
3. Try [Inference Guide](guides/inference.md)

**For training:**
1. Read [Training Guide](training/README.md)
2. Choose your [Dataset](training/datasets/)
3. Review [Multi-Stage Training](training/multi-stage.md) strategy

**For evaluation:**
1. Read [Evaluation Overview](evaluation/overview.md)
2. Pick a [Benchmark](evaluation/benchmarks/)
3. Compare against [Baselines](evaluation/baselines.md)

## Contributing

When adding new documentation:
- Place it in the appropriate category directory
- Update the relevant section README
- Keep this main README updated with links
- Follow existing documentation style

## Questions?

Check [Troubleshooting](guides/troubleshooting.md) for common issues or refer to [CLAUDE.md](../CLAUDE.md) for comprehensive project guidance.
