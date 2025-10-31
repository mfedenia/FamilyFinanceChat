#!/bin/bash

# Exit if any command fails
set -e

echo "=== Setting up environment and running backend + frontend ==="

# ---------- BACKEND SETUP ----------
cd backend
echo ">>> Setting up Python backend..."

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate venv
source venv/Scripts/activate

# Install Python requirements
if [ -f "requirements.txt" ]; then
    echo "Installing backend dependencies..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping pip install."
fi

# Run FastAPI backend
echo "Starting FastAPI backend on http://localhost:8001..."
# Run in background so frontend can start too
uvicorn backend:app --host 0.0.0.0 --port 8001 --reload &
BACKEND_PID=$!

# ---------- FRONTEND SETUP ----------
echo ">>> Setting up React frontend..."

# Go into frontend directory 
cd ..
pwd
cd frontend/upload/upload-app

# Install npm packages
if [ -f "package.json" ]; then
    echo "Installing frontend dependencies..."
    npm install
else
    echo "No package.json found in ./frontend. Skipping npm install."
fi

# Start React app
echo "Starting React frontend on http://localhost:3000..."
npm start & 
FRONTEND_PID=$!

# ---------- CLEANUP ----------
# Trap CTRL+C to kill both processes
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID" SIGINT

# Wait for processes
wait
