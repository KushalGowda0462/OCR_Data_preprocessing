#!/bin/bash
# scripts/setup_environment.sh

echo "Setting up Document Preprocessing Environment..."

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate document-preprocessing

# Install pip packages
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p data/processed

echo "Environment setup complete!"
echo "Run: conda activate document-preprocessing"
echo "Then: python main.py"