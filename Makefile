.PHONY: format check install run train clean

# Format code with black
format:
	uv run black *.py

# Check formatting without modifying files
check:
	uv run black --check *.py

# Install dependencies including dev tools
install:
	uv sync --dev

# Run inference example
run:
	uv run main.py

# Run training example
train:
	uv run train.py

# Clean cache and temporary files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
