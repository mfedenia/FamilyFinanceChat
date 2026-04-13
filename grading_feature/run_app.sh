#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

echo "Starting project..."

if [[ -f "$ENV_FILE" ]]; then
    INVALID_ASSIGNMENTS="$(grep -nE '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]+=' "$ENV_FILE" || true)"
    if [[ -n "$INVALID_ASSIGNMENTS" ]]; then
        echo "ERROR: Invalid .env assignment syntax detected in $ENV_FILE"
        echo "Use VAR=value (no spaces around =)."
        echo "$INVALID_ASSIGNMENTS"
        exit 1
    fi

    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [[ -z "${OPENWEBUI_BASE_URL:-}" ]]; then
    echo "ERROR: OPENWEBUI_BASE_URL is not set. Put OPENWEBUI_BASE_URL in grading_feature/.env"
    exit 1
fi

if [[ -z "${OUTPUT_PATH:-}" ]]; then
    echo "ERROR: OUTPUT_PATH is not set. Put OUTPUT_PATH in grading_feature/.env"
    exit 1
fi

export DATA_PATH="${DATA_PATH:-$OUTPUT_PATH}"

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
cd "$SCRIPT_DIR/backend"

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

################################
# Frontend Setup
################################

echo "Setting up frontend..."
cd "$SCRIPT_DIR/frontend"

if [[ ! -d "node_modules" ]]; then
    echo "Installing frontend dependencies..."
    npm install --quiet
fi

echo "Starting frontend..."
npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

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
