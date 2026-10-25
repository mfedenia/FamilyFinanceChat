# Setup Guide

How to get FamilyFinanceChat running from scratch.

---

## Prerequisites

| Requirement | Version / Notes |
|---|---|
| Docker | 24+ with Compose v2 (`docker compose`, not `docker-compose`) |
| GCP VM | e2-standard-4 or larger recommended (2 vCPU / 8 GB minimum; production uses 10 GB RAM limit for OpenWebUI alone) |
| Upload storage | `/mnt/gcs/fin602` is the Compose bind source; verify whether it is a GCS mount or a local directory |
| OpenAI API key | Required for chat completions and RAG embeddings |
| Python 3.10+ | Only needed if running the grading dashboard or RAG pipeline locally |
| Git | For cloning the repo |

---

## 1. Clone the Repository

```bash
git clone https://github.com/mfedenia/FamilyFinanceChat
cd FamilyFinanceChat
```

---

## 2. Environment Variables

Copy the example and fill in real values:

```bash
cp .env.staging.example .env   # use this as a starting template; rename to .env
```

**Every key you need to set:**

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** | OpenAI API key for chat completions |
| `RAG_OPENAI_API_KEY` | **Yes** | OpenAI API key for RAG embeddings (can be the same key) |
| `WEBUI_SECRET_KEY` | **Yes** | Random secret for JWT signing — generate with `openssl rand -hex 32` |
| `OPENAI_API_BASE_URL` | No | Defaults to `https://api.openai.com/v1` |
| `RAG_OPENAI_API_BASE_URL` | No | Defaults to `https://api.openai.com/v1` |
| `WEBUI_BUILD_VERSION` | No | Informational; set to the OW version string or leave blank |
| `DATA_DIR` | No | Defaults to `/app/backend/data` inside the container |
| `QDRANT_URI` | No | Defaults to `http://qdrant:6333` — only change if using an external Qdrant |
| `RAG_EMBEDDING_MODEL` | No | Defaults to `text-embedding-3-small` |
| `RAG_TOP_K` | No | Number of RAG chunks returned; defaults to `5` |
| `RAG_RELEVANCE_THRESHOLD` | No | Minimum similarity score (0–1); defaults to `0.30` |
| `REDIS_URL` | No | Defaults to `redis://redis:6379/0` |
| `WEBSOCKET_REDIS_URL` | No | Defaults to `redis://redis:6379/1` |

**Do not commit `.env` to git.** It is already in `.gitignore`.

---

## 3. Prepare Host Directories

These paths are bind-mounted from the host into containers. Create them before starting:

```bash
sudo mkdir -p /opt/openwebui/data
sudo mkdir -p /opt/qdrant/storage
sudo mkdir -p /opt/prometheus/data
sudo mkdir -p /opt/grafana/data
sudo chown -R 65534:65534 /opt/prometheus/data   # prometheus runs as nobody
sudo chown -R 472:0 /opt/grafana/data            # grafana user
```

Create `/mnt/gcs/fin602` and decide whether it is local storage or a GCS FUSE
mount before starting Docker. The 2026-10-17 VM inventory found it was not mounted;
do not assume that files in this path are protected by a bucket. For a GCS-backed
deployment, verify the bucket identity and mount before starting the stack.

---

## 4. Start the Stack

```bash
docker compose up -d
```

To check all containers came up healthy:

```bash
docker compose ps
docker compose logs open-webui --tail 50
```

OpenWebUI is available at `http://<VM-IP>:3000`. On first start it will prompt you to create an admin account.

---

## 5. Install the Chat Metrics Filter Function

This step is required after every fresh deployment. The Filter Function is not auto-installed.

1. Log into OpenWebUI as admin
2. Navigate to **Workspace > Functions**
3. Click **+** to create a new function
4. Name: `Chat Metrics`
5. Paste the contents of `monitoring/chat_metrics_filter.py`
6. Click **Save**
7. Toggle the **Global** switch to **ON**
8. Click the gear icon → set `pushgateway_url` to `http://pushgateway:9091`

Verify it is working:
```bash
curl -s http://localhost:9091/metrics | grep openwebui_chat
```
(metrics appear after the first chat message is sent)

---

## 6. Load Knowledge Base Documents

Use the KB Sync tool to load course documents into the OpenWebUI knowledge base:

```bash
cd tools/kb_sync
cp .env.example .env
# set OPENWEBUI_BASE_URL, OPENWEBUI_API_KEY, OPENWEBUI_KB_ID in .env
pip install -r requirements.txt
python sync_kb.py list                         # verify connection
python sync_kb.py sync ./path/to/documents/   # bulk upload
```

Alternatively, upload files directly via the OpenWebUI UI: **Workspace > Knowledge > drag-and-drop**.

---

## 7. Grafana Setup

Grafana runs at `http://<VM-IP>:3001`. Default login is `admin` / `admin` — change it on first login.

Add Prometheus as a data source:
1. Configuration > Data Sources > Add data source > Prometheus
2. URL: `http://prometheus:9090`
3. Save & Test

---

## 8. Running the Grading Dashboard (Local)

The grading dashboard runs on a developer workstation and connects to the production VM.

```bash
cd grading_feature
cp .env.example .env
# set OPENWEBUI_BASE_URL (the production VM URL), OPENWEBUI_API_KEY, DATA_PATH, OUTPUT_PATH
./run_app.sh
```

The dashboard will be available at `http://localhost:8000`. Click **Refresh** to pull the latest chat data from the VM.

---


---

## Common Errors

| Error | Likely Cause | Fix |
|---|---|---|
| `open-webui` exits immediately | Missing or invalid `OPENAI_API_KEY` or `WEBUI_SECRET_KEY` | Check `.env` values |
| Qdrant connection refused | `/opt/qdrant/storage` doesn't exist or wrong permissions | Create the directory; check Qdrant logs |
| Chat metrics missing in Grafana | Filter Function not installed or not enabled globally | Re-do step 5 |
| Grafana shows "No data" | Prometheus data source not configured, or no chat messages sent yet | Configure data source; send a test chat |
| Grading dashboard `/refresh` returns 401 | `OPENWEBUI_API_KEY` invalid or expired | Generate a new API key in OpenWebUI Settings > Account |
| `docker compose up` fails with port conflict | Port 3000, 8080, 9090, or 3001 in use | Stop conflicting services or change the host port mappings |
| Chat hangs after 7+ messages | Known OpenWebUI bug | Start a new chat session as a workaround |

---

## CI Smoke Test

`.github/workflows/smoke-test.yml` runs on every push. It starts the isolated
`docker-compose.test.yml` stack with the `docker-compose.ci.yml` override. The
override routes chat and embeddings to a deterministic OpenAI-compatible test
provider, so CI needs no external API key or paid model calls. This checks
OpenWebUI model routing and Qdrant retrieval, but does not check live OpenAI
availability or credentials. No production volume or port is used.

The test waits for `/health` and `/ready`, registers the first admin user, checks a
non-empty chat response, uploads a text document, adds it to a new knowledge base,
and queries that collection for a unique marker in the uploaded text. It fails at
the first broken step and prints the test stack logs on failure. The stack and
volumes are removed at the end of each run.

To run the CI stack locally, copy `.env.test.example` to `.env.test`. Then run:

```bash
docker compose -f docker-compose.test.yml -f docker-compose.ci.yml up -d --build open-webui-test qdrant-test redis-test mock-openai-test
python -m pip install requests==2.32.5
python scripts/smoke_test.py --model gpt-4o-mini
docker compose -f docker-compose.test.yml -f docker-compose.ci.yml down -v
```

To test a real provider locally, omit the CI override and set real
`OPENAI_API_KEY` and `RAG_OPENAI_API_KEY` values plus
`RAG_EMBEDDING_ENGINE=openai` in `.env.test`.

For the A3 red/green demonstration, push a commit on the feature branch with
`SMOKE_MODEL` set to a nonexistent model and capture the failed Actions run.
Restore the configured model in a follow-up commit, push, and capture the passing
run. Keep both commits on the PR so the failure and recovery are reviewable.

## A13 production upgrade runbook (preparation)

**Status: preparation for review; no production upgrade or restore has been executed
by this PR.** A13 closes only after the maintenance window, smoke checks, P2's
grading acceptance, and the execution record below. The target for this preparation
is the production Compose tag `v0.11.4`. A tag in Git is not evidence of the image
running on the server. The current isolated CI stack uses `v0.9.6`; its green run
does not validate a production-data migration to `v0.11.4`.

### 1. Review gates and deployment inventory

P1 and P2 must fill this table with evidence before scheduling execution. All entries
are **pending verification** in this preparation PR.

| Gate | Required evidence / decision | Record |
|---|---|---|
| A2 | Successful restore rehearsal; backup locations, retention and measured restore time | Pending |
| A9 | Restored-data v0.9.x smoke and concurrent metrics results; exact image tag/digest | Pending |
| A10 | Restored-data v0.10.x smoke; isolated downgrade failure and recovery record | Pending |
| A11 | Restored-data v0.11.4 smoke; UI changes handed to P4 | Pending |
| A12 | Team's SQLite/PostgreSQL decision and reasoning in HANDOFF.md | Pending |
| Window | Instructor-approved date/time/timezone, outside student deadline weeks; outage and rollback deadline | Pending |
| Operators | Named P1 executor and P2 grading reviewer, both available throughout | Pending |
| Upload protection | Verify whether the source is a bucket mount or local directory; matching snapshot or cold archive, restore procedure and retention | Pending |
| Deployment | Host/project, active Compose files/overrides, Git SHA, actual image tags/IDs/repository digests, DB backend, all mounts | Pending |
| Baseline | Existing-user smoke, known saved chats/files, P2 extraction and metrics results before change | Pending |

The commands below are for **Bash on the Linux deployment host**, with Docker
Compose v2, Python 3, GNU tar, sudo and enough space for backups, saved images and
a restore alongside the failed data. Rehearse this exact procedure on isolated
copies first. Do not paste the whole section as a script: there are human evidence
gates between blocks. Keep a durable, access-restricted backup outside the repository
and off the VM as required by A2; a VM-local archive alone is not disaster recovery.

Start with the actual deployed checkout, not a fresh clone of `main`. The VM inventory
from 2026-10-17 found an older, differently owned checkout, Open WebUI v0.8.12, no
hosted grading service, and an uploads directory that was not a mount. Recheck each
fact on the day of execution. Stop and revise/rehearse this procedure if the active
Compose files, services, mounts, or database differ from the inventory below.

```bash
set -euo pipefail
umask 077
read -rp 'Absolute path of the deployed checkout: ' REPO
REPO=$(realpath "$REPO")
test -f "$REPO/docker-compose.yml" && test -f "$REPO/.env"
# The deployed checkout has different ownership. Do not chown it or set a global exception.
deployed_git() { git -c "safe.directory=$REPO" -C "$REPO" "$@"; }
test "$(deployed_git rev-parse --show-toplevel)" = "$REPO"
test -z "$(deployed_git status --porcelain)"
read -rp 'Running Open WebUI container name: ' WEBUI_CONTAINER
test -n "$WEBUI_CONTAINER"
PROJECT=$(docker inspect "$WEBUI_CONTAINER" --format '{{index .Config.Labels "com.docker.compose.project"}}')
WEBUI_SERVICE=$(docker inspect "$WEBUI_CONTAINER" --format '{{index .Config.Labels "com.docker.compose.service"}}')
test -n "$PROJECT" && test "$PROJECT" != '<no value>'
test -n "$WEBUI_SERVICE" && test "$WEBUI_SERVICE" != '<no value>'
test "$WEBUI_SERVICE" = open-webui || {
  echo 'Open WebUI service name differs from this runbook; stop and adapt the overrides' >&2
  exit 1
}
COMPOSE_FILES=$(docker inspect "$WEBUI_CONTAINER" --format '{{index .Config.Labels "com.docker.compose.project.config_files"}}')
test "$COMPOSE_FILES" = "$REPO/docker-compose.yml" || {
  echo "Active Compose files differ from the deployed checkout; stop and adapt the runbook" >&2
  exit 1
}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BACKUP="/var/backups/familyfinancechat/$STAMP"
sudo install -d -m 700 -o "$(id -u)" -g "$(id -g)" "$BACKUP"
export BACKUP STAMP
dc() { docker compose -p "$PROJECT" --env-file "$REPO/.env" -f "$REPO/docker-compose.yml" "$@"; }
deployed_git rev-parse HEAD > "$BACKUP/git-sha.txt"
printf '%s\n' "$PROJECT" > "$BACKUP/project.txt"
dc config --quiet
dc ps -a > "$BACKUP/containers.txt"
dc config --services | sort > "$BACKUP/config-services.txt"
docker ps --filter "label=com.docker.compose.project=$PROJECT" \
  --format '{{.Label "com.docker.compose.service"}}' | sort -u > "$BACKUP/running-services.txt"
diff -u "$BACKUP/config-services.txt" "$BACKUP/running-services.txt" || {
  echo 'Running services differ from the deployed Compose file; stop and reconcile' >&2; exit 1;
}
for service in $(dc config --services); do
  cid=$(dc ps -a -q "$service")
  test -n "$cid" || { echo "Configured service $service is absent; stop and reconcile Compose" >&2; exit 1; }
  docker inspect "$cid" --format '{{.Name}} image={{.Config.Image}} id={{.Image}} mounts={{json .Mounts}}'
  docker image inspect "$(docker inspect "$cid" --format '{{.Image}}')" --format '{{json .RepoDigests}}'
done > "$BACKUP/images-and-mounts.txt"
dc exec -T "$WEBUI_SERVICE" python -c 'import os; u=os.getenv("DATABASE_URL", ""); print("database_backend=" + (u.split(":", 1)[0] if u else "default: verify SQLite path"))'
```

Inspect the recorded mounts and the private deployment configuration. Never print
the full container environment or resolved Compose configuration into PR logs.
Confirm `DATA_DIR`, any database URL, Qdrant and Redis endpoints actually match the
local services below. These backup/restore commands support **SQLite stored under
`/app/backend/data` only**; an external DB, custom SQLite location, external vector
store or A12 PostgreSQL migration requires a separate reviewed backup/restore plan.
Do not read or alter Open WebUI's database schema to perform these checks.

| Service | Expected persistent mount |
|---|---|
| `open-webui` | `/opt/openwebui/data` → `/app/backend/data` (SQLite and local app data) |
| `open-webui` uploads | Inspect the source of `/app/backend/data/uploads`; `/mnt/gcs/fin602` may be a plain VM directory |
| `qdrant` | `/opt/qdrant/storage` → `/qdrant/storage` |
| `redis` | Compose volume `redis-data` → `/data` |
| grading, if deployed | `grading-api` uses `grading-data` → `/app/data`; A15 `grading` uses `grading-dashboard-data` → `/data` |
| `prometheus`, `grafana` | `/opt/prometheus/data`, `/opt/grafana/data` |

Use the inspected volume names, which include the project prefix; do not guess them.
Confirm any named volumes are local Docker volumes without custom driver options and
no other process writes them. If uploads are a mount, identify the backing bucket and
test its point-in-time restore. If uploads are a plain directory, include them in the
cold archive. Every directory that rollback moves aside must be an ordinary directory,
not a symlink or mountpoint.

```bash
check_bind() {
  local service=$1 destination=$2 expected=$3 cid actual
  cid=$(dc ps -q "$service")
  test -n "$cid"
  actual=$(docker inspect "$cid" --format "{{range .Mounts}}{{if eq .Destination \"$destination\"}}{{.Source}}{{end}}{{end}}")
  test "$actual" = "$expected" || {
    echo "Unexpected mount for $service:$destination ($actual); stop and adapt" >&2; exit 1;
  }
}
check_bind open-webui /app/backend/data /opt/openwebui/data
check_bind qdrant /qdrant/storage /opt/qdrant/storage
check_bind prometheus /prometheus /opt/prometheus/data
check_bind grafana /var/lib/grafana /opt/grafana/data
REDIS_VOLUME=$(docker inspect "$(dc ps -q redis)" --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}')
test -n "$REDIS_VOLUME"
VOLUME_ITEMS=("redis:$REDIS_VOLUME")
GRADING_SERVICE=''
GRADING_VOLUME_KEY=''
if dc config --services | grep -qx 'grading' && dc config --services | grep -qx 'grading-api'; then
  echo 'Both grading services are configured; stop and choose the active data store' >&2
  exit 1
fi
if dc config --services | grep -qx 'grading'; then
  GRADING_SERVICE=grading; GRADING_VOLUME_KEY=grading-dashboard-data; GRADING_DATA_DEST=/data
elif dc config --services | grep -qx 'grading-api'; then
  GRADING_SERVICE=grading-api; GRADING_VOLUME_KEY=grading-data; GRADING_DATA_DEST=/app/data
fi
if [ -n "$GRADING_SERVICE" ]; then
  GRADING_CID=$(dc ps -q "$GRADING_SERVICE")
  test -n "$GRADING_CID" || { echo "Grading service is configured but not running" >&2; exit 1; }
  GRADING_VOLUME=$(docker inspect "$GRADING_CID" --format "{{range .Mounts}}{{if eq .Destination \"$GRADING_DATA_DEST\"}}{{.Name}}{{end}}{{end}}")
  test -n "$GRADING_VOLUME"
  VOLUME_ITEMS+=("grading:$GRADING_VOLUME")
fi
printf '%s\n' "${VOLUME_ITEMS[@]}" > "$BACKUP/volume-items.txt"
printf '%s\n' "$GRADING_VOLUME_KEY" > "$BACKUP/grading-volume-key.txt"
docker volume inspect "${VOLUME_ITEMS[@]#*:}" > "$BACKUP/volumes.json"
python3 - <<'PY'
import json, os, pathlib
volumes = json.loads((pathlib.Path(os.environ['BACKUP']) / 'volumes.json').read_text())
if not volumes or any(v.get('Driver') != 'local' or v.get('Options') for v in volumes):
    raise SystemExit('Non-local or custom-driver volume; stop and adapt the backup plan')
PY
UPLOADS_SOURCE=$(docker inspect "$WEBUI_CONTAINER" --format '{{range .Mounts}}{{if eq .Destination "/app/backend/data/uploads"}}{{.Source}}{{end}}{{end}}')
test -n "$UPLOADS_SOURCE" && test -d "$UPLOADS_SOURCE"
BIND_PATHS=(/opt/openwebui/data /opt/qdrant/storage /opt/prometheus/data /opt/grafana/data)
if mountpoint -q "$UPLOADS_SOURCE"; then
  UPLOADS_KIND=mount
else
  UPLOADS_KIND=local
  for path in "${BIND_PATHS[@]}"; do
    case "$UPLOADS_SOURCE/" in "$path/"* ) echo 'Uploads overlap another backup path; stop and adapt' >&2; exit 1;; esac
    case "$path/" in "$UPLOADS_SOURCE/"* ) echo 'Uploads contain another backup path; stop and adapt' >&2; exit 1;; esac
  done
  BIND_PATHS+=("$UPLOADS_SOURCE")
fi
printf '%s\n' "${BIND_PATHS[@]}" > "$BACKUP/bind-paths.txt"
printf 'source=%s\nkind=%s\n' "$UPLOADS_SOURCE" "$UPLOADS_KIND" > "$BACKUP/uploads.txt"
for path in "${BIND_PATHS[@]}"; do
  sudo test -d "$path"
  test ! -L "$path"
  if mountpoint -q "$path"; then echo "Unexpected data mountpoint: $path" >&2; exit 1; fi
done
```

### 2. Freeze the rollback release and approve the version sequence

Keep the deployed release, secrets (private backup only), and every running image,
including locally built grading and metrics images. Local rollback tags
below refer to the exact running image IDs; repository digests can be empty for local
builds. Do not prune them during the window. The helper image is only for archiving
volumes; record it and include it in the image archive.

```bash
mkdir "$BACKUP/release"
deployed_git archive HEAD | tar -xf - -C "$BACKUP/release"
install -m 600 "$REPO/.env" "$BACKUP/release/.env"
# Save only env files that the active Compose configuration actually uses.
for env_file in grading_feature/backend/grading.env grading_feature/.env; do
  if [ -f "$REPO/$env_file" ]; then install -m 600 "$REPO/$env_file" "$BACKUP/release/$env_file"; fi
done
: > "$BACKUP/frozen-images.tsv"
for service in $(dc config --services); do
  cid=$(dc ps -a -q "$service")
  image_id=$(docker inspect "$cid" --format '{{.Image}}')
  tag="localhost/ffc-a13/$service:$STAMP"
  docker tag "$image_id" "$tag"
  printf '%s\t%s\n' "$service" "$tag" >> "$BACKUP/frozen-images.tsv"
done
python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['BACKUP'])
services = {s: {'image': image} for s, image in
            (line.split() for line in (p / 'frozen-images.tsv').read_text().splitlines())}
(p / 'frozen-images.json').write_text(json.dumps({'services': services}, indent=2))
PY
docker pull alpine:3.21.3
docker image inspect alpine:3.21.3 --format '{{.Id}} {{json .RepoDigests}}' > "$BACKUP/archive-helper.txt"
# All local data and the archive must fit on the same filesystem for this procedure.
# Reserve room for the cold archive, restored copy, saved images and 5 GiB headroom.
DATA_DEVICE=$(stat -c %d "${BIND_PATHS[0]}")
test "$(stat -c %d "$BACKUP")" = "$DATA_DEVICE" || {
  echo 'Backup and data are on different disks; recalculate space and rehearse a revised plan' >&2; exit 1;
}
data_kb=0
for path in "${BIND_PATHS[@]}"; do
  test "$(stat -c %d "$path")" = "$DATA_DEVICE" || {
    echo "Data is on another disk: $path; revise the space check" >&2; exit 1;
  }
  data_kb=$((data_kb + $(sudo du -sk "$path" | cut -f1)))
done
volume_kb=0
for item in "${VOLUME_ITEMS[@]}"; do
  volume=${item#*:}
  mount=$(docker volume inspect "$volume" --format '{{.Mountpoint}}')
  test "$(stat -c %d "$mount")" = "$DATA_DEVICE" || {
    echo "Volume $volume is on another disk; revise the space check" >&2; exit 1;
  }
  volume_kb=$((volume_kb + $(sudo du -sk "$mount" | cut -f1)))
done
image_kb=0
for service in $(dc config --services); do
  cid=$(dc ps -a -q "$service")
  bytes=$(docker image inspect "$(docker inspect "$cid" --format '{{.Image}}')" --format '{{.Size}}')
  image_kb=$((image_kb + (bytes + 1023) / 1024))
done
reserve_kb=$((5 * 1024 * 1024))
need_kb=$((2 * (data_kb + volume_kb) + image_kb + reserve_kb))
free_kb=$(df -Pk "$BACKUP" | awk 'NR==2 {print $4}')
test "$free_kb" -gt "$need_kb" || {
  echo "Insufficient space: need ${need_kb} KiB, have ${free_kb} KiB; stop before saving images" >&2
  exit 1
}
printf 'data_kb=%s\nvolume_kb=%s\nimage_kb=%s\nreserve_kb=%s\nfree_kb=%s\n' \
  "$data_kb" "$volume_kb" "$image_kb" "$reserve_kb" "$free_kb" > "$BACKUP/space-check.txt"
docker image save -o "$BACKUP/images.tar" alpine:3.21.3 $(cut -f2 "$BACKUP/frozen-images.tsv")
```

P1/P2 now write the ordered **remaining** upgrade tags and digests in the execution
record. Use the fixed patch versions already rehearsed for A9–A11. The v0.10.x
patch is deliberately unspecified here because its rehearsal evidence is pending.
End at v0.11.4; include each intervening minor version from the real starting version.
Do not downgrade a server already at or above a planned step to recreate the sequence.
If already on v0.11.4, assess and record validation-only work; if newer, stop and
re-scope. A green blank-data v0.9.6 CI run does not fill A9–A11's restored-data gates.
Pull the approved images before the window and record their digests. Do not broadly
pull or rebuild the entire deployment; Qdrant/Valkey upgrades are separate changes.

### 3. Fresh, consistent backup inside the maintenance window

After the teacher-approved window starts, block public ingress and background jobs
using the deployment's reviewed maintenance procedure. Stop external grading refresh,
upload sync and all other writers. Run the baseline checks in section 4 and record
P2's baseline before stopping the stack. Record the outage start and rollback deadline.

The A2 `scripts/backup.sh` is a **live** application backup. Once A2 is complete,
run that reviewed script and verify its manifest and restore-drill record before the
window. This runbook retains the **cold** backup below because it also freezes the
other running services, local volumes and exact images at one rollback point. Do not
substitute one archive for the other. If A2's script or its restore drill is absent,
the A2 gate above remains pending and the upgrade cannot start.

```bash
read -rp 'Reviewed A2 backup.sh absolute path: ' A2_SCRIPT
test -f "$A2_SCRIPT"
sudo bash "$A2_SCRIPT"
read -rp 'Completed A2 backup directory (not .partial): ' A2_BACKUP
test -d "$A2_BACKUP" && test -f "$A2_BACKUP/SHA256SUMS"
case "$A2_BACKUP" in *.partial) echo 'Incomplete A2 backup' >&2; exit 1;; esac
(cd "$A2_BACKUP" && sudo sha256sum -c SHA256SUMS)
printf '%s\n' "$A2_BACKUP" > "$BACKUP/a2-backup-path.txt"
```

```bash
dc stop
test -z "$(dc ps --status running -q)"
sudo tar --numeric-owner --acls --xattrs -cpf "$BACKUP/bind-data.tar" -C / \
  "${BIND_PATHS[@]#/}"
sudo chown "$(id -u):$(id -g)" "$BACKUP/bind-data.tar"
for item in "${VOLUME_ITEMS[@]}"; do
  name=${item%%:*}; volume=${item#*:}
  docker run --rm --network none \
    --mount "type=volume,src=$volume,dst=/source,readonly" \
    --mount "type=bind,src=$BACKUP,dst=/backup" alpine:3.21.3 \
    tar -cpf "/backup/$name.tar" -C /source .
  sudo chown "$(id -u):$(id -g)" "$BACKUP/$name.tar"
done
tar -tf "$BACKUP/bind-data.tar" > "$BACKUP/bind-data.contents.txt"
checksum_files=(bind-data.tar images.tar)
for item in "${VOLUME_ITEMS[@]}"; do
  name=${item%%:*}
  tar -tf "$BACKUP/$name.tar" > "$BACKUP/$name.contents.txt"
  checksum_files+=("$name.tar")
done
(cd "$BACKUP" && sha256sum "${checksum_files[@]}" > SHA256SUMS)
(cd "$BACKUP" && sha256sum -c SHA256SUMS)
```

If `UPLOADS_KIND=local`, `bind-data.tar` includes the uploads directory. If it is
`mount`, the archive excludes it: while writers remain stopped, take the fresh
bucket snapshot/version manifest using the A2-tested procedure and record its
matching restore point. Confirm retention protects deleted/overwritten files and
restoration to that point. Copy the backup bundle to the A2 off-host destination
and verify its checksums.
Also take the fresh VM/disk snapshot required by A2 and record its ID. **Stop before
upgrade if these steps, archive inspection, or the rehearsed restore are unverified.**
Checksums confirm archive integrity, not application recoverability.

### 4. Baseline and per-version acceptance

Install the reviewed A13 `scripts/smoke_test.py` and `requests==2.32.5` in a dedicated
operator Python environment beforehand. Set `SMOKE_SCRIPT` to that reviewed absolute
path and `SMOKE_PYTHON` to its interpreter (the saved old release may lack this flag).
Use a dedicated existing test account with chat, upload, knowledge and retrieval
permissions; P2's extraction credential separately requires the users/chats API.
Enter only the token value, without a `Bearer ` prefix. Disable shell tracing.

```bash
set +x
read -rp 'Reviewed A13 smoke script absolute path: ' SMOKE_SCRIPT
read -rp 'Smoke Python interpreter absolute path: ' SMOKE_PYTHON
test -f "$SMOKE_SCRIPT" && test -x "$SMOKE_PYTHON"
read -rp 'Enabled smoke model ID: ' SMOKE_MODEL
read -rp 'Reviewed check_pushgateway_metric.py absolute path: ' METRICS_SCRIPT
read -rp 'Host-reachable Pushgateway URL: ' METRICS_URL
test -f "$METRICS_SCRIPT" && test -n "$METRICS_URL"
read -rsp 'Existing smoke account token: ' OPENWEBUI_SMOKE_TOKEN; printf '\n'
export OPENWEBUI_SMOKE_TOKEN
read -rp 'Evidence label (baseline or version, no slash): ' CHECK_LABEL
[[ "$CHECK_LABEL" =~ ^[a-zA-Z0-9._-]+$ ]]
SMOKE_STARTED=$(date +%s)
"$SMOKE_PYTHON" "$SMOKE_SCRIPT" --base-url http://127.0.0.1:3000 \
  --model "$SMOKE_MODEL" --existing-token-env OPENWEBUI_SMOKE_TOKEN \
  2>&1 | tee "$BACKUP/smoke-$CHECK_LABEL.log"
unset OPENWEBUI_SMOKE_TOKEN
python3 "$METRICS_SCRIPT" --url "$METRICS_URL" --since "$SMOKE_STARTED" \
  2>&1 | tee "$BACKUP/metrics-$CHECK_LABEL.log"
```

This waits for `/health` and `/ready`, requires a non-empty chat answer, processes an
uploaded document, attaches it to a knowledge base, and retrieves its unique marker.
It uses the deployment's real model/embedding provider (normal API usage applies).
A missing/blank environment variable fails before any HTTP request; HTTP 401/403
fails immediately without falling back to signup. Leave signup mode for the isolated
blank CI stack shown above. Never run the CI `down -v` command against production.

`Smoke run` identifies the run; `Smoke resource` lines record created user (fresh
mode only), knowledge and file IDs as soon as available, including on later failure.
The chat request uses `smoke-test-<run ID>` but does not create a saved chat fixture.
After retaining results, use `scripts/cleanup_smoke.py` with the printed run,
knowledge and file IDs. It first reads both resources and checks their exact names
and owner; it is a dry run unless `--execute` is supplied. Run the dry run, review
the IDs, then repeat with `--execute`. It never removes the smoke user. If a
timed-out create returned no ID, inspect the account's resources by the run marker
manually; the tool will not guess an ID. Tokens and raw HTTP error bodies are not
printed by the clients. Keep evidence private.

```bash
read -rp 'Reviewed cleanup_smoke.py absolute path: ' CLEANUP_SCRIPT
read -rp 'Smoke run ID: ' RUN_ID
read -rp 'Smoke knowledge ID: ' KNOWLEDGE_ID
read -rp 'Smoke file ID: ' FILE_ID
read -rsp 'Existing smoke account token: ' OPENWEBUI_SMOKE_TOKEN; printf '\n'
export OPENWEBUI_SMOKE_TOKEN
"$SMOKE_PYTHON" "$CLEANUP_SCRIPT" --base-url http://127.0.0.1:3000 \
  --run-id "$RUN_ID" --knowledge-id "$KNOWLEDGE_ID" --file-id "$FILE_ID" \
  --token-env OPENWEBUI_SMOKE_TOKEN
# Only after inspecting the dry-run result and retaining the evidence:
"$SMOKE_PYTHON" "$CLEANUP_SCRIPT" --base-url http://127.0.0.1:3000 \
  --run-id "$RUN_ID" --knowledge-id "$KNOWLEDGE_ID" --file-id "$FILE_ID" \
  --token-env OPENWEBUI_SMOKE_TOKEN --execute
unset OPENWEBUI_SMOKE_TOKEN
```

P2 must compare extraction against known **saved** baseline chats with user ownership,
message-pair counts and representative question/answer content. A successful exit or
an empty JSON alone is not acceptance. The smoke's synthetic completion alone is not
a saved conversation. Run the reviewed extractor from the operator's Python
environment into the restricted backup directory. This works when grading runs on
a laptop, as `grading-api`, or as A15's `grading` service; it does not overwrite any
dashboard data. Install its dependencies beforehand. The extraction token must have
access to the users/chats API.

```bash
read -rsp 'P2 extraction bearer token: ' OPENWEBUI_BEARER_TOKEN; printf '\n'
export OPENWEBUI_BEARER_TOKEN
read -rp 'Reviewed extract_chats.py absolute path: ' EXTRACT_SCRIPT
read -rp 'Extractor Python interpreter absolute path: ' EXTRACT_PYTHON
test -f "$EXTRACT_SCRIPT" && test -x "$EXTRACT_PYTHON"
OUTPUT_PATH="$BACKUP/extraction-$CHECK_LABEL.json" \
  OPENWEBUI_BASE_URL=http://127.0.0.1:3000 \
  "$EXTRACT_PYTHON" "$EXTRACT_SCRIPT" \
  > "$BACKUP/extraction-$CHECK_LABEL.log" 2>&1
unset OPENWEBUI_BEARER_TOKEN
test -s "$BACKUP/extraction-$CHECK_LABEL.json"
if [ "$CHECK_LABEL" != baseline ]; then
  read -rp 'Reviewed compare_extractions.py absolute path: ' COMPARE_SCRIPT
  test -f "$COMPARE_SCRIPT"
  python3 "$COMPARE_SCRIPT" \
    "$BACKUP/extraction-baseline.json" "$BACKUP/extraction-$CHECK_LABEL.json"
fi
```

P2 privately inspects `extraction-<label>.json`, confirms the comparison and the
agreed fixtures/counts, and signs each row below. Protect the JSON and logs: they
contain student data. Remove them after acceptance per the team's retention policy.
Verify the existing metrics filter still emits a **new** response-time sample after
the chat and check the grading UI's health; a previously pushed sample is insufficient.
The isolated CI stack has no installed metrics Filter, so this check runs only against
an environment where that Filter is installed and Pushgateway is reachable.

### 5. Execute one approved version at a time

Only proceed after sections 1–4's baseline and backup gates are signed. Keep ingress
and other writers blocked through validation. Each iteration requires a new evidence
label and P2 sign-off before the next version. Pin the selected rehearsed digest in
the step override; the frozen override keeps every other service on its original image.

```bash
read -rp 'Next approved full image reference (tag@sha256:digest): ' NEXT_IMAGE
[[ "$NEXT_IMAGE" =~ ^ghcr.io/open-webui/open-webui:v[0-9]+\.[0-9]+\.[0-9]+@sha256:[a-f0-9]{64}$ ]]
export NEXT_IMAGE
docker pull "$NEXT_IMAGE"
python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['BACKUP'])
(p / 'step.json').write_text(json.dumps({'services': {'open-webui': {'image': os.environ['NEXT_IMAGE']}}}))
PY
upgrade() { docker compose -p "$PROJECT" --env-file "$BACKUP/release/.env" \
  -f "$BACKUP/release/docker-compose.yml" -f "$BACKUP/frozen-images.json" \
  -f "$BACKUP/step.json" "$@"; }
upgrade config --quiet
test -d "$UPLOADS_SOURCE"
if [ "$UPLOADS_KIND" = mount ]; then mountpoint -q "$UPLOADS_SOURCE"; fi
upgrade up -d --no-build --pull never
upgrade ps
docker inspect "$WEBUI_CONTAINER" --format '{{.Config.Image}} {{.Image}}'
```

Re-run section 4 (health, chat, upload, retrieval, metrics and P2 extraction) after
**every** version, including v0.11.4. Record the selected reference, actual image ID,
timestamps and evidence before replacing `step.json` for the next step. Preserve each
approved step override with its evidence. Any migration error, unhealthy service,
failed assertion, extraction mismatch or rollback-deadline breach stops the sequence.
Keep maintenance in place, capture private logs, and use the rollback below if the
issue cannot be resolved within the agreed window. Never start an older image on the
migrated data as a production experiment; the A10 downgrade experiment belongs in test.

### 6. Roll back images AND the matching data

Restore the **pre-window** snapshot as one set: SQLite/local app data, Qdrant,
Redis, grading data when present, and matching uploads. Returning only the old
image can fail on the migrated schema. Keep ingress and all external writers blocked.
These commands retain failed bind data and use new Docker volumes instead of erasing
the originals. Verify space and the backup manifest before moving any directories.

Use the same shell variables from the window; after reconnecting, explicitly recover
`BACKUP`, `STAMP`, `PROJECT`, `UPLOADS_SOURCE`, `UPLOADS_KIND`, `BIND_PATHS` and
`VOLUME_ITEMS` from the recorded bundle and redefine the functions. Review every
value before running a mutating command.

```bash
upgrade stop
test -z "$(upgrade ps --status running -q)"
(cd "$BACKUP" && sha256sum -c SHA256SUMS)
docker image load -i "$BACKUP/images.tar"
mapfile -t BIND_PATHS < "$BACKUP/bind-paths.txt"
mapfile -t VOLUME_ITEMS < "$BACKUP/volume-items.txt"
GRADING_VOLUME_KEY=$(cat "$BACKUP/grading-volume-key.txt")
UPLOADS_SOURCE=$(sed -n 's/^source=//p' "$BACKUP/uploads.txt")
UPLOADS_KIND=$(sed -n 's/^kind=//p' "$BACKUP/uploads.txt")
test -n "$UPLOADS_SOURCE" && [[ "$UPLOADS_KIND" == local || "$UPLOADS_KIND" == mount ]]
export GRADING_VOLUME_KEY
for path in "${BIND_PATHS[@]}"; do
  sudo test -d "$path"
  test ! -L "$path"
  if mountpoint -q "$path"; then echo "Unexpected data mountpoint: $path" >&2; exit 1; fi
  sudo test ! -e "$path.failed-$STAMP"
done
for path in "${BIND_PATHS[@]}"; do
  sudo mv -- "$path" "$path.failed-$STAMP"
done
sudo tar --numeric-owner --acls --xattrs -xpf "$BACKUP/bind-data.tar" -C /
for item in "${VOLUME_ITEMS[@]}"; do
  name=${item%%:*}
  volume="ffc-a13-$name-$STAMP"
  if docker volume inspect "$volume" >/dev/null 2>&1; then
    echo "Restore volume already exists; stop and inspect before retrying" >&2
    exit 1
  fi
  docker volume create "$volume"
  docker run --rm --network none \
    --mount "type=volume,src=$volume,dst=/restore" \
    --mount "type=bind,src=$BACKUP,dst=/backup,readonly" alpine:3.21.3 \
    tar -xpf "/backup/$name.tar" -C /restore
done
python3 - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['BACKUP'])
keys = {'redis': 'redis-data', 'grading': os.environ.get('GRADING_VOLUME_KEY', '')}
names = [line.split(':', 1)[0] for line in (p / 'volume-items.txt').read_text().splitlines()]
volumes = {keys[name]: {'name': f"ffc-a13-{name}-{os.environ['STAMP']}", 'external': True}
           for name in names}
(p / 'restored-volumes.json').write_text(json.dumps({'volumes': volumes}))
PY
```

**Before starting any service**, if uploads were mounted, restore/verify the bucket
to the matching recorded point using the A2-approved procedure and verify the mount
and representative old uploads. If uploads were local, confirm they came back from
`bind-data.tar`. This is a hard gate even though the archive passed checksums.

```bash
rollback() { docker compose -p "$PROJECT" --env-file "$BACKUP/release/.env" \
  -f "$BACKUP/release/docker-compose.yml" -f "$BACKUP/frozen-images.json" \
  -f "$BACKUP/restored-volumes.json" "$@"; }
rollback config --quiet
test -d "$UPLOADS_SOURCE"
if [ "$UPLOADS_KIND" = mount ]; then mountpoint -q "$UPLOADS_SOURCE"; fi
rollback up -d --no-build --pull never
rollback ps
```

Run section 4 against the restored instance and obtain P2 sign-off before reopening
access. If it fails, keep maintenance active and escalate to P1/P2; do not continue
upgrading. Record the failure and retained `.failed-<timestamp>` directories/old volumes.
Rollback is not idempotent: inspect partial progress before retrying any restore step.
Record the **exact Compose invocation and bundle path** as the active deployment.
After rollback, bare `docker compose up` from the old checkout would select the
original volumes/images again; maintenance must use the recorded rollback overrides
until a reviewed follow-up reconciles the deployment configuration.

### 7. Execution record and closeout

Copy this table into the private window record. Put redacted outcomes and evidence
links in the PR/HANDOFF; never attach tokens, environment files, backups or student data.

| Stage / UTC start–end | Image tag, digest and Git SHA | Backup / snapshot / log location | Health / chat / upload / retrieval / metrics | P2 extraction result and signer | Exception / action / executor |
|---|---|---|---|---|---|
| Pre-window inventory and gates | Pending | Pending | Pending | Pending | Pending |
| Baseline and fresh backup | Pending | Pending | Pending | Pending | Pending |
| Each approved remaining minor step (one row each) | Pending | Pending | Pending | Pending | Pending |
| v0.11.4 final acceptance | Pending | Pending | Pending | Pending | Pending |
| Rollback, if needed | Pending | Pending | Pending | Pending | Pending |
| Reopen access and clean test resources | Pending | Pending | Pending | Pending | Pending |

Record the teacher's window confirmation, P1/P2 names, time budget, DB decision,
uploads restore point, archive checksums, off-host copy, actual migration observations
and any data loss/reconciliation needed after rollback. On success, retain the final
`step.json` and frozen images override as part of the recorded deployment command;
reconcile the checked-in deployment with that state in review before discarding the
overrides. Re-enable ingress/jobs only after both operators accept the result.

Local preparation checks (no deployment required):

```bash
python -m pip install requests==2.32.5
python -m unittest discover -s tests -p test_smoke_test.py -v
bash scripts/ci/compat_guards.sh
git diff --check
```

The mock tests validate both authentication paths and failure handling. CI additionally
runs the existing blank-data integration stack. Neither constitutes a restored-data
rehearsal or production acceptance. No Open WebUI core files are modified: the
extension surfaces are the public REST API and deployment configuration.

### What to add next

1. **Post-deploy Filter Function check** — after deploying, curl the Pushgateway and assert `openwebui_chat_completion_seconds` appears after a synthetic chat request. Automates the manual step 5 above.
2. **Grading dashboard smoke test** — `pip install -r requirements.txt && python -m pytest` for the backend once tests are written.
3. **Dependency pinning** — `qdrant:latest` and `cadvisor:latest` in `docker-compose.yml` are unpinned. Pin to specific versions so upgrades are explicit and tested.
