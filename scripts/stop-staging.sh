#!/usr/bin/env bash

# ------------------------------------------------------------
# Staging Stop Script
#
# Stops and removes the staging environment using Docker Compose.
# - Validates required env files and variables
# - Ensures the staging compose file exists
# - Targets only the staging Docker Compose project
# - Does not affect production containers
# ------------------------------------------------------------

# Fail fast on errors, undefined vars, or pipeline issues
set -euo pipefail

echo "==> Starting staging shutdown"

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

echo "==> Stopping staging containers"

# Stop and remove only the staging project containers
sudo --preserve-env=STAGING_WEBUI_PORT,STAGING_OPENWEBUI_DATA_DIR,STAGING_UPLOADS_DIR,STAGING_QDRANT_STORAGE_DIR,STAGING_REDIS_DATA_DIR,STAGING_REPO_DIR \
docker compose \
  -f docker-compose.staging.yml \
  --env-file .env.staging \
  -p familyfinancechat-staging \
  down

echo "==> Staging shutdown finished"