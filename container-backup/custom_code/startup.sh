#!/bin/bash
echo "[Startup] Applying customizations..."
python3 /app/custom_code/apply_customizations.py

if [ $? -eq 0 ]; then
    echo "[Startup] Customizations applied successfully"
else
    echo "[Startup] Warning: Customizations may have failed, continuing anyway..."
fi

# This is copied directly from /app/backend/start.sh
echo "[Startup] Starting OpenWebUI server on port 8080..."
cd /app/backend

# Use exec to replace the shell with uvicorn (important for Docker signal handling)
exec python3 -m uvicorn open_webui.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    --forwarded-allow-ips '*'