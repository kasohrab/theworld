#!/bin/bash
# TheWorld Model - PACE HPC Setup Script
# This script sets up the environment specifically for Georgia Tech PACE cluster

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_PATH="./env"
USE_ENV_FILE=false
HELP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-env-file)
            USE_ENV_FILE=true
            shift
            ;;
        --env-path)
            ENV_PATH="$2"
            shift 2
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
    echo "TheWorld Model - PACE Setup Script"
    echo ""
    echo "Usage: bash scripts/pace_setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --use-env-file  Use env-dev.yml for conda dependencies (includes nodejs)"
    echo "  --env-path PATH Custom path for conda environment (default: ./env)"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "Example:"
    echo "  bash scripts/pace_setup.sh                    # Basic setup"
    echo "  bash scripts/pace_setup.sh --use-env-file    # Use env-dev.yml"
    exit 0
fi

# Print header
echo ""
echo "========================================"
echo -e "${BLUE}  TheWorld - PACE HPC Setup${NC}"
echo "========================================"
echo ""

# Step 1: Load Anaconda module
echo -e "${BLUE}[1/7]${NC} Loading Anaconda3 module..."
if command -v module &> /dev/null; then
    module load anaconda3
    echo -e "${GREEN}✓ Anaconda3 module loaded${NC}"
else
    echo -e "${YELLOW}⚠ module command not found (not on PACE?)${NC}"
    echo "  Continuing anyway..."
fi
echo ""

# Step 2: Create conda environment
echo -e "${BLUE}[2/7]${NC} Creating conda environment..."
if [ "$USE_ENV_FILE" = true ] && [ -f "env-dev.yml" ]; then
    echo "  Using env-dev.yml to create environment at ${ENV_PATH}"
    conda env create -p "${ENV_PATH}" -f env-dev.yml
    echo -e "${GREEN}✓ Conda environment created from env-dev.yml${NC}"
elif [ -d "${ENV_PATH}" ]; then
    echo -e "${YELLOW}⚠ Environment already exists at ${ENV_PATH}${NC}"
    echo "  Skipping creation..."
else
    echo "  Creating Python 3.11 environment at ${ENV_PATH}"
    conda create -p "${ENV_PATH}" python=3.11 -y
    echo -e "${GREEN}✓ Conda environment created${NC}"
fi
echo ""

# Step 3: Activate environment
echo -e "${BLUE}[3/7]${NC} Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"
echo -e "${GREEN}✓ Environment activated: ${ENV_PATH}${NC}"
echo ""

# Step 4: Install uv package manager
echo -e "${BLUE}[4/7]${NC} Installing uv package manager..."
if command -v uv &> /dev/null; then
    echo -e "${YELLOW}⚠ uv already installed, skipping...${NC}"
else
    echo "  Running: pip install uv"
    pip install uv
    echo -e "${GREEN}✓ uv installed${NC}"
fi
echo ""

# Step 5: Install dependencies with uv
echo -e "${BLUE}[5/7]${NC} Installing Python dependencies..."
echo "  Running: uv sync --dev"
uv sync --dev
echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

# Step 6: Install Cosmos guardrail
echo -e "${BLUE}[6/7]${NC} Installing Cosmos safety checker..."
echo "  Note: cosmos_guardrail has dependency conflicts with newer transformers"
echo "  Installing without dependencies and copying from conda env..."

# Install cosmos_guardrail in conda env (where it can coexist)
pip install cosmos_guardrail > /dev/null 2>&1

# Copy the installed package to .venv (since uv can't install it due to conflicts)
if [ -d "${ENV_PATH}/lib/python3.11/site-packages/cosmos_guardrail" ]; then
    cp -r "${ENV_PATH}/lib/python3.11/site-packages/cosmos_guardrail" .venv/lib/python3.11/site-packages/
    cp -r "${ENV_PATH}/lib/python3.11/site-packages/cosmos_guardrail-"*.dist-info .venv/lib/python3.11/site-packages/ 2>/dev/null || true
    echo -e "${GREEN}✓ Cosmos guardrail installed${NC}"
else
    echo -e "${YELLOW}⚠ Could not find cosmos_guardrail package${NC}"
    echo "  You may need to install it manually"
fi
echo ""

# Step 7: Install nodejs (if not using env file)
echo -e "${BLUE}[7/7]${NC} Checking Node.js..."
if [ "$USE_ENV_FILE" = true ]; then
    echo -e "${GREEN}✓ Node.js should be installed from env-dev.yml${NC}"
elif command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓ Node.js already installed: ${NODE_VERSION}${NC}"
else
    echo "  Installing Node.js via conda..."
    conda install -c conda-forge nodejs -y
    echo -e "${GREEN}✓ Node.js installed${NC}"
fi
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p results
echo -e "${GREEN}✓ Created checkpoints/, logs/, results/ directories${NC}"
echo ""

# Verify installation
echo "Verifying installation..."
if uv run python -c "from theworld import TheWorld; print('OK')" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ TheWorld package imports successfully${NC}"
else
    echo -e "${RED}✗ Failed to import TheWorld package${NC}"
    echo "  This might indicate an installation problem."
    exit 1
fi
echo ""

# Print summary
echo "========================================"
echo -e "${GREEN}✓ PACE Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Environment details:"
echo "  • Conda environment: ${ENV_PATH}"
echo "  • Python version: $(python --version 2>&1)"
echo "  • uv version: $(uv --version 2>&1)"
if command -v node &> /dev/null; then
    echo "  • Node.js version: $(node --version)"
fi
echo ""

# Print activation instructions
echo "========================================"
echo -e "${BLUE}Important: Activate Your Environment${NC}"
echo "========================================"
echo ""
echo "Before running any commands, activate the environment:"
echo ""
echo "  conda activate ${ENV_PATH}"
echo ""
echo "Or add to your ~/.bashrc or PBS job scripts:"
echo "  module load anaconda3"
echo "  conda activate ${ENV_PATH}"
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

echo "2. Run smoke test to verify everything works:"
echo "   export HF_TOKEN=hf_your_token_here"
echo "   make smoke-test"
echo ""
echo "3. Or run inference example:"
echo "   make run"
echo ""
echo "For more information, see CLAUDE.md"
echo ""
