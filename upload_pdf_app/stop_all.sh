#!/bin/bash

echo "Forcefully stopping all services..."

# Kill all uvicorn processes (except the main OpenWebUI one)
echo "Stopping uvicorn processes..."
pkill -f "uvicorn backend:app" 2>/dev/null || true

# Kill all npm/node processes related to our app
echo "Stopping npm/node processes..."
pkill -f "npm.*start" 2>/dev/null || true
pkill -f "react-scripts" 2>/dev/null || true

# Kill anything on our specific ports using fuser
echo "Clearing ports 8765 and 3456..."
fuser -k 8765/tcp 2>/dev/null || true
fuser -k 3456/tcp 2>/dev/null || true

# Check what's still running
echo ""
echo "Checking for remaining processes..."
ps aux | grep -E "(uvicorn backend|npm|react-scripts)" | grep -v grep || echo "No processes found."

echo "All processes stopped."