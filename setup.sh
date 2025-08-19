#!/bin/bash
echo "Setting up USD/JPY AI Trading System..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models logs notebooks

# Create __init__.py files for proper Python packages
touch __init__.py
touch config/__init__.py
touch src/__init__.py

# Create .env file for IBKR credentials
touch .env

echo "Setup complete! Please:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Add IBKR credentials to .env file"
echo "3. Run: python src/data_collector.py"
echo "4. Run: python src/train.py"