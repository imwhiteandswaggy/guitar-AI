#!/bin/bash

# Guitar Teacher AI - Setup Script for Mac/Linux
# This script sets up the environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "Guitar Teacher AI - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Check if version is 3.8 or higher
if [ "$(printf '%s\n' "3.8" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.8" ]; then
    echo "⚠ Warning: Python 3.8+ recommended (found $PYTHON_VERSION)"
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes..."
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed"

# Download model if not present
echo ""
echo "Checking for model file..."
if [ ! -f "trained_models/real_guitar_test3/weights/best.pt" ]; then
    echo "Model file not found. Downloading..."
    python3 download_model.py
else
    echo "✓ Model file found"
fi

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To run the app:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run the app: python app.py"
echo "  3. Open browser: http://localhost:5000"
echo ""
echo "To deactivate virtual environment later:"
echo "  deactivate"
echo ""
