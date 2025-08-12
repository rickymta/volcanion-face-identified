#!/bin/bash

echo "Setting up Volcanion Face Identified project..."

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p temp
mkdir -p logs

# Set up MongoDB (if running locally)
echo "Note: Make sure MongoDB is running on localhost:27017"
echo "You can start MongoDB with: mongod"

# Run tests
echo "Running tests..."
python -m pytest tests/ -v

echo "Setup complete!"
echo "To start the application:"
echo "uvicorn main:app --reload --host 0.0.0.0 --port 8000"
