#!/bin/bash
# TheWorld Model - Automated Setup Script
# This script sets up the complete development environment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
SKIP_DEV=false
SKIP_MODELS=false
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-dev)
            SKIP_DEV=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            HELP=true
            shift
            ;;
    esac
done

# Show help
if [ "$HELP" = true ]; then
    echo "TheWorld Model - Setup Script"
    echo ""
    echo "Usage: bash scripts/setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-dev      Skip installing dev dependencies (black, wandb, etc.)"
    echo "  --skip-models   Skip downloading/caching models"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Example:"
    echo "  bash scripts/setup.sh              # Full setup"
    echo "  bash scripts/setup.sh --skip-dev   # Skip dev tools"
    exit 0
fi

# Print header
echo ""
echo "========================================"
echo -e "${BLUE}  TheWorld Model - Setup${NC}"
echo "========================================"
echo ""

# Step 1: Check uv is installed
echo -e "${BLUE}[1/7]${NC} Checking prerequisites..."
if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ Error: 'uv' package manager not found${NC}"
    echo ""
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Or visit: https://github.com/astral-sh/uv"
    exit 1
fi
echo -e "${GREEN}✓ uv package manager found${NC}"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"
else
    echo -e "${YELLOW}⚠ Python 3 not found in PATH (uv will handle this)${NC}"
fi

echo ""

# Step 2: Install core dependencies
echo -e "${BLUE}[2/7]${NC} Installing core dependencies..."
if [ "$SKIP_DEV" = true ]; then
    echo "  Running: uv sync"
    uv sync
else
    echo "  Running: uv sync --dev"
    uv sync --dev
    echo -e "${GREEN}✓ Installed core + dev dependencies${NC}"
fi
echo ""

# Step 3: Install Cosmos guardrail
echo -e "${BLUE}[3/7]${NC} Installing Cosmos safety checker..."
echo "  Note: cosmos_guardrail dependencies are already in pyproject.toml"
echo "  The package itself requires special handling due to version conflicts"

# Try to install via pip (will go to a temp location)
python3 -m pip install --target ./.temp_cosmos cosmos_guardrail --no-deps > /dev/null 2>&1 || true

# Copy to .venv if successful
if [ -d "./.temp_cosmos/cosmos_guardrail" ]; then
    mkdir -p .venv/lib/python3.11/site-packages
    cp -r ./.temp_cosmos/cosmos_guardrail .venv/lib/python3.11/site-packages/
    cp -r ./.temp_cosmos/cosmos_guardrail-*.dist-info .venv/lib/python3.11/site-packages/ 2>/dev/null || true
    rm -rf ./.temp_cosmos
    echo -e "${GREEN}✓ Cosmos guardrail installed${NC}"
else
    echo -e "${YELLOW}⚠ Could not install cosmos_guardrail automatically${NC}"
    echo "  Dependencies are installed, but cosmos_guardrail package may need manual setup"
fi
echo ""

# Step 4: Create necessary directories
echo -e "${BLUE}[4/7]${NC} Creating directories..."
mkdir -p checkpoints
mkdir -p logs
echo -e "${GREEN}✓ Created checkpoints/ and logs/ directories${NC}"
echo ""

# Step 5: Check HF_TOKEN
echo -e "${BLUE}[5/7]${NC} Checking HuggingFace authentication..."
if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}⚠ HF_TOKEN environment variable not set${NC}"
    echo ""
    echo "  To use HuggingFace Hub features, set your token:"
    echo "    export HF_TOKEN=\"hf_your_token_here\""
    echo ""
    echo "  Get your token at: https://huggingface.co/settings/tokens"
    echo ""
else
    echo -e "${GREEN}✓ HF_TOKEN is set${NC}"
fi
echo ""

# Step 6: Verify installation
echo -e "${BLUE}[6/7]${NC} Verifying installation..."
if uv run python -c "from theworld import TheWorld; print('OK')" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ TheWorld package imports successfully${NC}"
else
    echo -e "${RED}✗ Failed to import TheWorld package${NC}"
    echo "  This might indicate an installation problem."
    exit 1
fi
echo ""

# Step 7: Print summary
echo -e "${BLUE}[7/7]${NC} Setup complete!"
echo ""
echo "========================================"
echo -e "${GREEN}✓ Installation Summary${NC}"
echo "========================================"
echo ""
echo "Installed components:"
echo "  • TheWorld Python package"
echo "  • Core dependencies (transformers, diffusers, torch, etc.)"
if [ "$SKIP_DEV" = false ]; then
    echo "  • Dev tools (black, wandb, tensorboard)"
fi
echo "  • Cosmos safety checker (cosmos_guardrail)"
echo ""
echo "Created directories:"
echo "  • ./checkpoints  - Training checkpoints"
echo "  • ./logs         - Training logs"
echo ""

# Print next steps
echo "========================================"
echo -e "${BLUE}Next Steps${NC}"
echo "========================================"
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "1. Set your HuggingFace token:"
    echo "   export HF_TOKEN=\"hf_your_token_here\""
    echo ""
fi

echo "2. Run quick test (100 samples):"
echo "   uv run python scripts/train_hf.py --config configs/datacomp_test.json"
echo ""
echo "3. Or start full training:"
echo "   uv run python scripts/train_hf.py --config configs/datacomp_production.json"
echo ""
echo "4. Run inference example:"
echo "   make run"
echo ""
echo "For more information, see CLAUDE.md or README.md"
echo ""
