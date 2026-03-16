# Shared Staging Environment

## Purpose
This staging environment provides a safe place to test and demo changes without affecting the live customer-facing deployment.

## What it isolates
The staging stack runs separately from production and uses its own:

- Open WebUI app container
- SQLite app database
- uploads directory
- Redis instance
- Qdrant instance

Production remains on port `3000`.

Staging runs on port `3002`.

## Files added for staging
The staging environment depends on the following repo files:

- `docker-compose.staging.yml`
- `scripts/deploy-staging.sh`
- `scripts/stop-staging.sh`
- `.env.staging.example`
- `.env.staging.local.example`

## Local-only files
These should not be committed:

- `.env.staging`
- `.env.staging.local`
- `.env.runtime.dump`
- staging zip exports or other local data dumps

## Required local directories
Example paths used on the VM:

- `~/ffc-staging/openwebui-data`
- `~/ffc-staging/uploads`
- `~/ffc-staging/qdrant-storage`
- `~/ffc-staging/redis-data`

Create them with:

```bash
mkdir -p ~/ffc-staging/openwebui-data
mkdir -p ~/ffc-staging/uploads
mkdir -p ~/ffc-staging/qdrant-storage
mkdir -p ~/ffc-staging/redis-data
```

## Local env setup

Create the runtime env file:

```bash
cp .env.staging.example .env.staging
cp .env.staging.local.example .env.staging.local
```

Then edit `.env.staging.local` for your machine or VM:

```bash
nano .env.staging.local
```

Example configuration:

```
STAGING_WEBUI_PORT=3002
STAGING_OPENWEBUI_DATA_DIR=/home/USERNAME/ffc-staging/openwebui-data
STAGING_UPLOADS_DIR=/home/USERNAME/ffc-staging/uploads
STAGING_QDRANT_STORAGE_DIR=/home/USERNAME/ffc-staging/qdrant-storage
STAGING_REDIS_DATA_DIR=/home/USERNAME/ffc-staging/redis-data
STAGING_REPO_DIR=/home/USERNAME/projects/FamilyFinanceChat
```

Replace `USERNAME` with your own system username.

## Starting staging

From the repo root run:

```bash
bash scripts/deploy-staging.sh
```

This will start the staging stack using `docker-compose.staging.yml`, `.env.staging`, and `.env.staging.local`.

## Stopping staging

From the repo root run:

```bash
bash scripts/stop-staging.sh
```

This stops the staging containers without affecting production.

## Verifying staging

Check running containers:

```bash
sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | egrep "open-webui|staging|qdrant|redis"
```

You should see the staging containers running including:

- open-webui-staging

- redis-staging

- qdrant-staging

Then open the app in a browser:

```
http://localhost:3002
```

If using VS Code port forwarding, forward port `3002` and open the forwarded URL instead.

## Expected behavior

If the setup is correct:

- production remains on port `3000`
- staging runs separately on port `3002`
- staging uses isolated local data directories
- stopping staging does not stop production
- staging testing does not overwrite production data

## Current known limitation

The staging environment is isolated and safe for experimentation, but it does not automatically have full production retrieval parity.

Metadata and files can be present in staging while staged Qdrant collections are still empty. Full production-like RAG behavior requires re-ingestion or re-indexing into `qdrant-staging`.

## Troubleshooting

### Port 3002 already in use

If staging fails to start because port `3002` is already taken, change the port in `.env.staging.local`.

Example:

```
STAGING_WEBUI_PORT=3003
```

### Missing directory errors

If a container fails because a directory does not exist, create the required directories listed above and rerun:

```bash
bash scripts/deploy-staging.sh
```

### Changes are not appearing

Restart staging:

```bash
bash scripts/stop-staging.sh
bash scripts/deploy-staging.sh
```

### Production appears affected

Stop staging immediately and review `docker-compose.staging.yml`.

Staging should only use isolated directories and the staging port.

## Committing these changes

Run:

```bash
git add docs/staging-environment.md
git status
```

You should now see something like:

```
modified:   .gitignore
new file:   .env.staging.example
new file:   .env.staging.local.example
new file:   docker-compose.staging.yml
new file:   docs/staging-environment.md
new file:   scripts/deploy-staging.sh
new file:   scripts/stop-staging.sh
```

Then commit:

```bash
git commit -m "Add shared staging environment setup"
```