# TheWorld Unit Tests

This directory contains unit tests for TheWorld model components.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_fusion.py -v

# Run with coverage
pytest tests/ --cov=theworld --cov-report=html
```

## Test Structure

- `test_fusion.py` - Tests for EmbeddingFusion module (lightweight, no model loading)
- `test_cosmos_encoder.py` - Tests for CosmosEncoder (requires Cosmos model)
- `test_gemma_vision.py` - Tests for GemmaVisionEncoder (requires Gemma model)
- `test_theworld.py` - Integration tests for full TheWorld model

## Notes

- Fusion tests are lightweight and can run without GPU
- Encoder tests require models to be cached locally (use `local_files_only=True`)
- Integration tests require both Cosmos and Gemma models (~20GB disk space)
- Set `HF_TOKEN` environment variable for private model access
