#!/bin/zsh

# Define the name of the virtual environment
VENV_NAME="chess_env"

# Create the virtual environment
echo "Creating virtual environment..."
python -m venv $VENV_NAME

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q kaggle chess numpy pandas torch

echo "Environment setup complete."
