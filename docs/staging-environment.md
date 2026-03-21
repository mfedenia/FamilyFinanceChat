Shared Staging Environment
==========================

Purpose
-------

This staging environment provides a safe place to test and demo changes without affecting the live customer-facing deployment.

It is designed to be:

-   reproducible
-   isolated from production
-   safe for experimentation
-   easy for new developers to onboard into

What it isolates
----------------

The staging stack runs separately from production and uses its own:

-   Open WebUI app container
-   SQLite app database
-   uploads directory
-   Redis instance
-   Qdrant instance

Production remains on port `3000`.

Staging runs on port `3002`.

Files added for staging
-----------------------

The staging environment depends on the following repo files:

-   `docker-compose.staging.yml`
-   `scripts/deploy-staging.sh`
-   `scripts/stop-staging.sh`
-   `.env.staging.example`
-   `.env.staging.local.example`

Local-only files
----------------

These should not be committed:

-   `.env.staging`
-   `.env.staging.local`
-   `.env.runtime.dump`
-   staging zip exports or other local data dumps

Required local directories
--------------------------

Example paths used on the VM:

-   `~/ffc-staging/openwebui-data`
-   `~/ffc-staging/uploads`
-   `~/ffc-staging/qdrant-storage`
-   `~/ffc-staging/redis-data`

You do not strictly need to create these manually because the deploy script will create them if they are missing.

Optional manual creation:

mkdir -p ~/ffc-staging/openwebui-data\
mkdir -p ~/ffc-staging/uploads\
mkdir -p ~/ffc-staging/qdrant-storage\
mkdir -p ~/ffc-staging/redis-data

Local env setup
---------------

Create the runtime env file:

cp .env.staging.example .env.staging\
cp .env.staging.local.example .env.staging.local

Then edit `.env.staging.local` for your machine or VM:

nano .env.staging.local

Example configuration:

STAGING_WEBUI_PORT=3002\
STAGING_OPENWEBUI_DATA_DIR=/home/USERNAME/ffc-staging/openwebui-data\
STAGING_UPLOADS_DIR=/home/USERNAME/ffc-staging/uploads\
STAGING_QDRANT_STORAGE_DIR=/home/USERNAME/ffc-staging/qdrant-storage\
STAGING_REDIS_DATA_DIR=/home/USERNAME/ffc-staging/redis-data\
STAGING_REPO_DIR=/home/USERNAME/projects/FamilyFinanceChat

Replace `USERNAME` with your own system username.

Starting staging
----------------

From the repo root run:

bash scripts/deploy-staging.sh

What the deploy script does
---------------------------

The deploy script will:

-   verify required env files exist
-   validate required environment variables
-   check that the repo path exists
-   verify required backend override files exist
-   create missing staging data directories
-   build and start staging containers
-   perform a basic HTTP health check

This makes staging deployment reproducible and safe to run repeatedly.

Stopping staging
----------------

From the repo root run:

bash scripts/stop-staging.sh

This stops the staging containers without affecting production.

Verifying staging
-----------------

Check running containers:

sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | egrep "open-webui|staging|qdrant|redis"

You should see the staging containers running including:

-   open-webui-staging
-   redis-staging
-   qdrant-staging

Then open the app in a browser:

http://localhost:3002

Expected behavior
-----------------

If the setup is correct:

-   production remains on port `3000`
-   staging runs separately on port `3002`
-   staging uses isolated local data directories
-   stopping staging does not stop production
-   repeated deploys are safe and idempotent
-   invalid configs fail early with clear errors

Fresh-state behavior
--------------------

If staging data directories are deleted, a fresh deploy will:

-   successfully recreate all staging infrastructure
-   start containers normally

But:

-   Open WebUI state will reset to default
-   uploaded files will be lost
-   Qdrant collections will be empty
-   retrieval (RAG) will not reflect production

Current known limitation
------------------------

The staging environment is isolated and safe for experimentation, but it does not automatically have full production retrieval parity.

Specifically:

-   SQLite app data is not restored
-   uploads are not restored
-   Qdrant embeddings are not restored

As a result:

-   staging is infrastructure-complete
-   but not data-parity complete

To achieve production-like behavior, staging data must be:

-   re-ingested
-   manually restored
-   or seeded via future automation

Troubleshooting
---------------

### Port already in use

If staging fails to start because port `3002` is already taken, change the port in `.env.staging.local`.

Example:

STAGING_WEBUI_PORT=3003

### Missing directory errors

Run:

bash scripts/deploy-staging.sh

The script will create required directories automatically.

### Changes are not appearing

Restart staging:

bash scripts/stop-staging.sh\
bash scripts/deploy-staging.sh

### Staging looks empty or reset

This is expected after a fresh-state deploy.

You will need to:

-   re-upload data
-   re-run ingestion
-   or restore staging data manually

### Production appears affected

Stop staging immediately:

bash scripts/stop-staging.sh

Then review:

-   `docker-compose.staging.yml`
-   `.env.staging.local`

Staging should only use:

-   staging-specific directories
-   staging-specific port

Summary
-------

This staging system provides:

-   safe isolation from production
-   reproducible deployment via scripts
-   validated startup and failure handling
-   a foundation for CI/CD integration

It currently does not include automated data parity with production.