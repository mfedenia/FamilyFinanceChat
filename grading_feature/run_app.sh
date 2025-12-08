#!/bin/bash
set -euo pipefail

echo "Starting project..."

################################
# Helper Functions
################################

check_version() {
    local name="$1"
    local current="$2"
    local required="$3"

    if [[ "$current" != "$required" ]]; then
        echo "$name version is $current but required is $required. Please update (refer to the docs)."
        exit 1
    else
        echo "$name version OK ($current)"
    fi
}

################################
# Check npm
################################

echo "Checking npm..."
if command -v npm &> /dev/null; then
    CURRENT_NPM=$(npm --version)
    REQUIRED_NPM="10.8.2"
    check_version "npm" "$CURRENT_NPM" "$REQUIRED_NPM"
else
    echo "npm is not installed. Refer to the docs."
    exit 1
fi

################################
# Check Node.js
################################

echo "Checking Node.js..."
if command -v node &> /dev/null; then
    CURRENT_NODE=$(node --version)
    REQUIRED_NODE="v20.19.5"
    check_version "Node.js" "$CURRENT_NODE" "$REQUIRED_NODE"
else
    echo "Node.js is not installed. Refer to the docs."
    exit 1
fi

################################
# Backend Setup
################################

echo "Setting up backend..."
cd backend

# Create venv if missing
if [[ ! -d "venv" ]]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python requirements..."
pip install -r requirements.txt --quiet

echo "Starting FastAPI backend on port 9500..."
uvicorn main:app --host 0.0.0.0 --port 9500 --reload &
BACKEND_PID=$!

cd ..

################################
# Frontend Setup
################################

echo "Setting up frontend..."
cd frontend

if [[ ! -d "node_modules" ]]; then
    echo "Installing frontend dependencies..."
    npm install --quiet
fi

echo "Starting frontend..."
npm run dev &
FRONTEND_PID=$!

cd ..

################################
# Cleanup and Exit Handling
################################

cleanup() {
    echo "Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo "Cleanup complete."
}

trap cleanup EXIT

echo "Project is running."
wait
