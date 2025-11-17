#!/bin/bash
set -e

echo "Starting project..."

################################
# Backend setup and run
################################

echo "Setting up backend..."
cd backend

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install backend dependencies
echo "Installing backend requirements..."
pip install -r requirements.txt

# Start backend server
echo "Starting FastAPI backend..."
uvicorn main:app --host 0.0.0.0 --port 9500 --reload &
BACKEND_PID=$!

cd ..

################################
# Frontend setup and run
################################

echo "Setting up frontend..."
cd frontend

# Install node modules if missing
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo "Starting frontend..."
npm run dev &
FRONTEND_PID=$!

cd ..

################################
# Cleanup
################################

trap "echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID" EXIT

wait