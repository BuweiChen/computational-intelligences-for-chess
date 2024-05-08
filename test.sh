#!/bin/zsh

# Define the name of the virtual environment
VENV_NAME="chess_env"

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Run Python scripts
echo "Running Python scripts..."
python chess_net.py
python test_chess_agents.py --count=5 --time=5 --p1="alphabeta" --p2="mcts"
# python test_chess_agents.py --count=5 --time=5 --p1="alphabeta" --p2="ab_no_transposition"

echo "Tests completed."
