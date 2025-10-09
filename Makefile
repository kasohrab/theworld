.PHONY: setup pace-setup format check install run train train-simple train-hf eval-blink eval-gemma compare-results clean

# Complete setup (recommended for first-time setup)
setup:
	bash scripts/setup.sh

# PACE HPC-specific setup (for Georgia Tech PACE cluster)
pace-setup:
	bash scripts/pace_setup.sh

# Format code with black
format:
	uv run black python/theworld/*.py examples/*.py scripts/*.py

# Check formatting without modifying files
check:
	uv run black --check python/theworld/*.py examples/*.py scripts/*.py

# Install dependencies including dev tools (use 'make setup' instead for full setup)
install:
	uv sync --dev

# Run inference example
run:
	uv run python examples/inference.py

# Run simple training example (basic demo)
train-simple:
	uv run python examples/simple_training.py

# Run HuggingFace Trainer-based training (production)
train-hf:
	uv run python scripts/train_hf.py

# Run smoke test (2 samples, ~3 min, tests entire pipeline)
smoke-test:
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "Error: HF_TOKEN environment variable not set"; \
		echo "Please run: export HF_TOKEN=hf_your_token_here"; \
		exit 1; \
	fi
	uv run python scripts/train_hf.py --config configs/smoke_test.json

# Backward compatibility alias
train: train-simple

# Evaluate TheWorld on BLINK benchmark
eval-blink:
	@echo "Evaluating TheWorld on BLINK Relative_Depth..."
	uv run python scripts/evaluate_blink.py \
		--task Relative_Depth \
		--model ${MODEL} \
		--num_world_steps ${WORLD_STEPS:-0,4} \
		--output results/blink_theworld.json

# Evaluate Gemma baseline on BLINK
eval-gemma:
	@echo "Evaluating Gemma3 baseline on BLINK Relative_Depth..."
	uv run python scripts/evaluate_blink.py \
		--task Relative_Depth \
		--model gemma3-baseline \
		--output results/blink_gemma.json

# Compare evaluation results
compare-results:
	@echo "Comparing TheWorld vs Gemma baseline..."
	uv run python scripts/compare_results.py \
		--theworld results/blink_theworld.json \
		--baseline results/blink_gemma.json \
		--output results/comparison.md \
		--print

# Clean cache and temporary files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf checkpoints
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
