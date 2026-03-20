#!/usr/bin/env bash

# ------------------------------------------------------------
# Staging Deployment Script
#
# Deploys the staging environment using Docker Compose.
# - Validates required env files and variables
# - Ensures required files + directories exist
# - Builds and starts staging containers
# - Performs a basic health check
# ------------------------------------------------------------

# Fail fast on errors, undefined vars, or pipeline issues
set -euo pipefail

echo "==> Starting staging deploy"

# Move to repo root (script assumed to be in /scripts)
cd "$(dirname "$0")/.."

# Check required env files and docker-compose file exist
if [[ ! -f ./.env.staging.local ]]; then
  echo "ERROR: .env.staging.local not found"
  exit 1
fi

if [[ ! -f ./.env.staging ]]; then
  echo "ERROR: .env.staging not found"
  exit 1
fi

if [[ ! -f ./docker-compose.staging.yml ]]; then
  echo "ERROR: docker-compose.staging.yml not found"
  exit 1
fi


# Load and export local staging environment variables
set -a
source ./.env.staging.local
set +a

# Required environment variables for staging
required_vars=(
  STAGING_WEBUI_PORT
  STAGING_OPENWEBUI_DATA_DIR
  STAGING_UPLOADS_DIR
  STAGING_QDRANT_STORAGE_DIR
  STAGING_REDIS_DATA_DIR
  STAGING_REPO_DIR
)

echo "==> Validating required environment variables"

# Ensure all required env vars are set
for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "ERROR: Missing required env var: $var"
    exit 1
  fi
done

echo "==> Checking repo path"

# Validate repo directory exists
if [[ ! -d "$STAGING_REPO_DIR" ]]; then
  echo "ERROR: STAGING_REPO_DIR does not exist: $STAGING_REPO_DIR"
  exit 1
fi

# Required backend files for mounting into containers
required_files=(
  "$STAGING_REPO_DIR/custom-code/main.py"
  "$STAGING_REPO_DIR/custom-code/config.py"
  "$STAGING_REPO_DIR/custom-code/observability.py"
  "$STAGING_REPO_DIR/custom-code/integrated_backend/custom_pdf_router.py"
)

echo "==> Checking required mounted files"

# Ensure required files exist
for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "ERROR: Required file not found: $file"
    exit 1
  fi
done

echo "==> Ensuring staging data directories exist"

# Create persistent data directories if missing
mkdir -p \
  "$STAGING_OPENWEBUI_DATA_DIR" \
  "$STAGING_UPLOADS_DIR" \
  "$STAGING_QDRANT_STORAGE_DIR" \
  "$STAGING_REDIS_DATA_DIR"

echo "==> Running docker compose for staging"

# Start staging containers (build if needed)
sudo --preserve-env=STAGING_WEBUI_PORT,STAGING_OPENWEBUI_DATA_DIR,STAGING_UPLOADS_DIR,STAGING_QDRANT_STORAGE_DIR,STAGING_REDIS_DATA_DIR,STAGING_REPO_DIR \
docker compose \
  -f docker-compose.staging.yml \
  --env-file .env.staging \
  -p familyfinancechat-staging \
  up -d --build

echo "==> Waiting briefly for app startup"

# Allow containers time to initialize
sleep 5

echo "==> Checking staging HTTP response on port $STAGING_WEBUI_PORT"

# Basic health check via HTTP request
if curl -fsS "http://localhost:${STAGING_WEBUI_PORT}" >/dev/null; then
  echo "SUCCESS: Staging is responding on port $STAGING_WEBUI_PORT"
else
  echo "WARNING: Staging deployed, but HTTP check not ready yet"
fi

echo "==> Staging deploy finished"