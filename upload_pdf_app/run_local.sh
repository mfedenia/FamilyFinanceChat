#!/bin/bash

# Exit if any command fails
set -e
# Get IP address for now
LOCAL_IP="34.85.238.142"
echo "=== Local IP Address: $LOCAL_IP ==="

BACKEND_PORT=8437  # i picked a random port but will change if its used
FRONTEND_PORT=3847  # i picked a random port but will change if its used

echo "=== Setting up environment and running backend + frontend ==="

# ---------- BACKEND SETUP ----------
cd backend
echo ">>> Setting up Python backend..."

# Create venv if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
# source venv/Scripts/activate # For Windows
source venv/bin/activate  # For Unix/Linux/Mac

# Install Python requirements
if [ -f "requirements.txt" ]; then
    echo "Installing backend dependencies..."
    pip install -r requirements.txt
    pip install playwright beautifulsoup4 pymupdf tqdm
else
    echo "No requirements.txt found. Skipping pip install."
fi

# Run FastAPI backend
echo "Starting FastAPI backend on http://$LOCAL_IP:$BACKEND_PORT..."
uvicorn backend:app --host 0.0.0.0 --port $BACKEND_PORT --reload &
BACKEND_PID=$!
sleep 2

# ---------- FRONTEND SETUP ----------
echo ">>> Setting up React frontend..."

# Go into frontend directory 
cd ../frontend/upload/upload-app

# Install npm packages
if [ -f "package.json" ]; then
    echo "Installing frontend dependencies..."
    npm install
else
    echo "No package.json found in ./frontend. Skipping npm install."
fi

# Start React app
echo "Starting React frontend on http://$LOCAL_IP:$FRONTEND_PORT..."
PORT=$FRONTEND_PORT REACT_APP_API_URL=http://$LOCAL_IP:$BACKEND_PORT HOST=0.0.0.0 npm start &
FRONTEND_PID=$!

# ---------- CLEANUP ----------
# Trap CTRL+C to kill both processes
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" SIGINT SIGTERM

# Wait for processes
wait
