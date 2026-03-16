#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

set -a
source ./.env.staging.local
set +a

sudo --preserve-env=STAGING_WEBUI_PORT,STAGING_OPENWEBUI_DATA_DIR,STAGING_UPLOADS_DIR,STAGING_QDRANT_STORAGE_DIR,STAGING_REDIS_DATA_DIR,STAGING_REPO_DIR \
docker compose \
  -f docker-compose.staging.yml \
  --env-file .env.staging \
  -p familyfinancechat-staging \
  down
