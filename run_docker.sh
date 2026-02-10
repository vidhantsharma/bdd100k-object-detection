#!/bin/bash
# BDD100K Object Detection - Docker Runner
# Usage: 
#   ./run_docker.sh                    # Start interactive bash
#   ./run_docker.sh --rebuild          # Rebuild image and start bash
#   ./run_docker.sh [command]          # Run a specific command
#   BDD100K_DATA_PATH=/path ./run_docker.sh  # Use custom data path

set -e

# Check for --rebuild flag
REBUILD=false
if [ "$1" == "--rebuild" ]; then
    REBUILD=true
    shift  # Remove --rebuild from arguments
fi

# ============================================================================
# CONFIGURATION: Edit this path to your BDD100K dataset parent folder
# ============================================================================
DEFAULT_BDD100K_PATH="/home/vidhant/bosch_applied_cv_assignment/data/assignment_data_bdd"
# 
# Your folder should contain a structure like:
#   bdd100k_images_100k/bdd100k/images/100k/{train,val,test}/
#   bdd100k_labels_release/bdd100k/labels/*.json
# OR simply:
#   images/{train,val,test}/
#   labels/*.json
# ============================================================================

echo "============================================================"
echo "BDD100K Object Detection - Docker Runner"
echo "============================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed!"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker permissions
if ! docker ps &> /dev/null; then
    echo ""
    echo "Docker permission denied. You have two options:"
    echo ""
    echo "1. Run with sudo:"
    echo "   sudo ./run_docker.sh"
    echo ""
    echo "2. Add your user to docker group (one-time setup):"
    echo "   sudo usermod -aG docker $USER"
    echo "   newgrp docker"
    echo ""
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Warning: docker-compose not found. Using 'docker compose' instead."
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Use environment variable if set, otherwise use default
if [ -z "$BDD100K_DATA_PATH" ]; then
    BDD100K_DATA_PATH="$DEFAULT_BDD100K_PATH"
    echo "Using default BDD100K path: $BDD100K_DATA_PATH"
else
    echo "Using BDD100K path from environment: $BDD100K_DATA_PATH"
fi

# Validate path exists
if [ ! -d "$BDD100K_DATA_PATH" ]; then
    echo "Error: BDD100K_DATA_PATH directory not found: $BDD100K_DATA_PATH"
    exit 1
fi

echo "✓ BDD100K data path: $BDD100K_DATA_PATH"

# Export for docker-compose
export BDD100K_DATA_PATH

echo ""
echo "Output directories (on host machine):"
echo "  Subset:     ./data/subset/"
echo "  Analysis:   ./data/analysis/"
echo "  Evaluation: ./data/evaluation/"
echo "  Training:   ./data/training/"

# Check for NVIDIA GPU support
echo ""
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "Warning: No NVIDIA GPU detected. Training will be slow on CPU."
fi

# Build image if not exists or if --rebuild is requested
echo ""
echo "Checking Docker image..."
if [ "$REBUILD" = true ]; then
    echo "Rebuilding Docker image (--rebuild flag detected)..."
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "$SCRIPT_DIR/docker" && $COMPOSE_CMD build && cd "$SCRIPT_DIR"
    echo "✓ Docker image rebuilt successfully!"
elif ! docker image inspect bdd100k-detection >/dev/null 2>&1; then
    echo "Building Docker image (this may take 5-10 minutes on first run)..."
    echo "Pulling base image and installing dependencies..."
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "$SCRIPT_DIR/docker" && $COMPOSE_CMD build && cd "$SCRIPT_DIR"
    echo "✓ Docker image built successfully!"
else
    echo "✓ Docker image exists (bdd100k-detection)"
    echo "  Tip: Use './run_docker.sh --rebuild' to rebuild after changing Docker files"
fi

echo ""
echo "============================================================"
echo "Starting Docker Container"
echo "============================================================"
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to docker directory for compose commands
cd "$SCRIPT_DIR/docker"

# If no arguments, run interactive bash
if [ $# -eq 0 ]; then
    echo "Starting interactive bash session..."
    echo ""
    echo "Inside the container, your data is available at /workspace/data/raw/"
    echo "You can run:"
    echo ""
    echo "  # Create subset (1000 train + 200 val)"
    echo "  python3 -m src.utils.create_subset --train-size 1000 --val-size 200"
    echo ""
    echo "  # Run analysis"
    echo "  python3 main.py --stage analysis"
    echo ""
    echo "  # Train model"
    echo "  python3 main.py --stage train"
    echo ""
    echo "  # Evaluate model"
    echo "  python3 main.py --stage evaluate"
    echo ""
    $COMPOSE_CMD run --rm --service-ports bdd100k
else
    # Run provided command
    echo "Running command: $@"
    $COMPOSE_CMD run --rm --service-ports bdd100k "$@"
fi
