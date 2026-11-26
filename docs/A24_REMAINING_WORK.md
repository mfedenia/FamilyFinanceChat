<!--
A24 REMINDER: documentation work still open. Do this LAST, after the team's feature freeze.
  1. Professor-facing guides: rewrite for the v0.11 interface and re-shoot their screenshots.
  2. This file: add every screenshot marked "SCREENSHOT:" and confirm every click-path marked "TODO-UI".
  3. Resolve every "TODO-A19" and "TODO-A24" marker (pending decisions from P1/P2).
  4. Newcomer acceptance test on a fresh VM, then remove this comment and the status note below.
Full checklist: docs/A24_REMAINING_WORK.md
Find everything left:  grep -rn -E "TODO-UI|TODO-A19|TODO-A24|SCREENSHOT:" --include=*.md .
-->

# Setup Guide

How to get FamilyFinanceChat running from scratch on a fresh Linux VM. Follow the steps in
order. Each step tells you what you should see when it worked; if you see something else, check
[Common errors](#common-errors) before moving on.

> **Status (task A24, in progress):** every command in this guide matches the current `main`
> branch. Some menu click-paths are still being re-checked against the Open WebUI v0.11
> interface, and screenshots will be added at the end of the semester. If a menu doesn't match
> what you see, add a line to [`docs/A24_REMAINING_WORK.md`](docs/A24_REMAINING_WORK.md).

---

## What you will end up with

| Service | Address | What it is |
|---|---|---|
| `open-webui` | `http://<VM-IP>:3000` | The chat app students use (Open WebUI v0.11.4) |
| `grafana` | `http://<VM-IP>:3001` | Dashboards |
| `prometheus` | `http://localhost:9090` on the VM | Metrics database (30-day retention) |
| `pushgateway` | `http://localhost:9091` on the VM | Receives chat metrics from the Chat Metrics filter |
| `otel-collector` | `127.0.0.1:8889` on the VM | Receives Open WebUI's built-in OpenTelemetry metrics |
| `cadvisor` | `http://localhost:8080` on the VM | Container CPU/memory metrics |
| `metrics-exporter` | `http://localhost:8001` on the VM | Health-probe sidecar for Open WebUI |
| `grading-api` | `127.0.0.1:8000` on the VM | Grading dashboard backend (API only) |
| `qdrant` | internal only | Vector database for document search (RAG) |
| `redis` | internal only | Valkey, for WebSocket session state |

Only ports 3000 and 3001 should ever be reachable from outside the VM. The others have no login.

<!-- TODO-A19: pushgateway is commented out in docker-compose.yml on main and has no scrape job in
prometheus.yml. P1 must restore both (the team decided to keep the filter). Remove this
comment once merged. -->

---

## Prerequisites

| Requirement | Details |
|---|---|
| Linux VM | Ubuntu 22.04 or 24.04 LTS. GCP `e2-standard-4` (4 vCPU, 16 GB RAM) recommended; Open WebUI alone is allowed up to 10 GB RAM |
| Disk | Boot disk of at least 50 GB, with **at least 30 GB free** before you start. The Open WebUI image is several GB, and upgrades need room for the old image, the new image and a backup at the same time |
| Docker | Docker Engine 24+ with Compose v2.24.4+ (`docker compose`, with a space, not `docker-compose`) |
| Git | Any recent version |
| OpenAI API key | Used for chat, document embeddings and grading. Set a spending limit on it |
| Optional | Python 3.10+ (knowledge-base sync tool), Node.js **v20.19.5** with npm **10.8.2** (grading dashboard on your laptop) |

Check what you have:

```bash
docker --version          # 24.x or newer
docker compose version    # v2.24.4 or newer
git --version
df -h /                   # "Avail" should be 30G or more
```

If `docker` commands fail with `permission denied ... docker.sock`, add yourself to the
`docker` group, then **log out and back in**:

```bash
sudo usermod -aG docker "$USER"
```

### Make the web ports reachable (GCP)

GCP blocks incoming traffic by default. Choose one of these:

**Option A: SSH tunnel (simplest, nothing exposed).** From your laptop:

```bash
gcloud compute ssh <VM_NAME> --zone <ZONE> -- -L 3000:localhost:3000 -L 3001:localhost:3001
```

While that session is open, use `http://localhost:3000` and `http://localhost:3001` in your
browser wherever this guide says `http://<VM-IP>:...`.

**Option B: firewall rule limited to your IP.** Find your public IP (for example at
`https://ifconfig.me`), then:

```bash
gcloud compute firewall-rules create ffc-allow-web \
  --allow=tcp:3000,tcp:3001 --source-ranges=<YOUR_PUBLIC_IP>/32 --target-tags=ffc-web
gcloud compute instances add-tags <VM_NAME> --tags=ffc-web --zone <ZONE>
```

`<VM-IP>` is the VM's **external** IP, shown in the GCP console under
**Compute Engine → VM instances**. Never open ports 8000, 8001, 8080, 8889, 9090 or 9091 to the
internet.

<!-- TODO-A24: once A16 (reverse proxy + TLS + login) lands, replace this section with the real
hostname and remove the direct-port instructions. -->

---

## 1. Clone the repository

```bash
git clone https://github.com/mfedenia/FamilyFinanceChat
cd FamilyFinanceChat
```

All later commands are run from this `FamilyFinanceChat` directory unless a step says otherwise.

---

## 2. Create the configuration files

The stack needs two files that are **not** in Git, because they hold secrets. `docker compose`
refuses to start if either is missing.

### 2a. `.env` (Open WebUI and Grafana)

Create it with placeholders:

```bash
cat > .env <<'EOF'
# --- Chat model provider ---
OPENAI_API_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=REPLACE_ME
ENABLE_OLLAMA_API=false

# --- Document search (RAG) ---
RAG_EMBEDDING_ENGINE=openai
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_OPENAI_API_BASE_URL=https://api.openai.com/v1
RAG_OPENAI_API_KEY=REPLACE_ME
RAG_TOP_K=5
RAG_RELEVANCE_THRESHOLD=0.30
RAG_FILE_MAX_SIZE=50
RAG_ALLOWED_FILE_EXTENSIONS=pdf,md,txt,docx

# --- Security ---
WEBUI_SECRET_KEY=REPLACE_ME
ENABLE_PERSISTENT_CONFIG=true

# --- Grafana (read by docker compose) ---
GRAFANA_ADMIN_PASSWORD=REPLACE_ME
EOF
chmod 600 .env
```

Generate the secret key automatically:

```bash
sed -i "s/^WEBUI_SECRET_KEY=.*/WEBUI_SECRET_KEY=$(openssl rand -hex 32)/" .env
```

Open the file (`nano .env`) and replace the remaining `REPLACE_ME` values:

| Variable | What to put |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI key (`sk-...`) |
| `RAG_OPENAI_API_KEY` | The same key is fine |
| `GRAFANA_ADMIN_PASSWORD` | A password you choose for Grafana's `admin` user. Save it in a password manager |

Check that nothing is left:

```bash
grep -n REPLACE_ME .env    # should print nothing
```

Do **not** add `QDRANT_URI`, `REDIS_URL`, `WEBSOCKET_REDIS_URL`, `VECTOR_DB` or the `OTEL_*`
variables. `docker-compose.yml` already sets them.

> **Important:** with `ENABLE_PERSISTENT_CONFIG=true`, most Open WebUI settings (connections,
> document settings) are copied from `.env` into the database **on first start only**. After
> that, editing `.env` does nothing for those settings; change them in the Admin Panel instead.

`.env` is already in `.gitignore`. Never commit it.

<!-- TODO-A24 (P1): compare this template with the production .env (values redacted) and commit
it as .env.example, so this step becomes "cp .env.example .env". -->

### 2b. `grading_feature/backend/grading.env` (grading backend)

```bash
cat > grading_feature/backend/grading.env <<'EOF'
OPENAI_API_KEY=REPLACE_ME
OPENAI_MODEL=gpt-4o-mini
# 0 = real scoring through OpenAI (costs money). 1 = fake scores, for testing without API cost.
MOCK_SCORER=0
OPENWEBUI_API_KEY=FILL_IN_AT_STEP_7
EOF
chmod 600 grading_feature/backend/grading.env
nano grading_feature/backend/grading.env    # put your OpenAI key in OPENAI_API_KEY
```

Leave `OPENWEBUI_API_KEY` as it is for now. That key doesn't exist until you create an admin
account, and you will come back to it in [step 7](#7-create-an-api-key-for-the-grading-backend).
`docker-compose.yml` already sets `OPENWEBUI_BASE_URL`, `DATA_PATH` and `OUTPUT_PATH` for this
container.

Note: the grading backend quietly switches to fake scores if `OPENAI_API_KEY` in this file is
empty, even with `MOCK_SCORER=0`. Make sure the key is filled in.

### 2c. Check the configuration before downloading anything

```bash
docker compose config --quiet && echo "Compose config OK"
```

You should see `Compose config OK`. This catches a missing `.env` or `grading.env`, a missing
`GRAFANA_ADMIN_PASSWORD` and YAML mistakes in seconds, before step 4 downloads several GB. It does
**not** check whether your keys are valid; the `grep REPLACE_ME` check in 2a covers the
placeholders.

Always keep `--quiet`. Without it, the command prints the full configuration **including your
secrets**, so never paste its output into chat, issues or pull requests.

---

## 3. Prepare host directories

```bash
sudo mkdir -p /opt/openwebui/data /opt/qdrant/storage /mnt/gcs/fin602
```

- `/opt/openwebui/data` holds Open WebUI's database and app data.
- `/opt/qdrant/storage` holds the document vectors.
- `/mnt/gcs/fin602` holds uploaded files. For a new or test stack, the plain local directory
  created above is all you need.

Prometheus, Grafana, Redis and the grading backend use Docker named volumes, so there is nothing
to create for them.

<!-- TODO-A24 (P1): document the production uploads storage. The 2026-10-17 inventory found
/mnt/gcs/fin602 was a plain directory on the VM, not a GCS mount. State the decision here and,
if it is a bucket, the exact mount steps. -->

---

## 4. Start the stack

```bash
docker compose pull
docker compose up -d --build
```

The first run downloads several GB of images and builds two small images (`grading-api` and
`metrics-exporter`). Expect 5–15 minutes.

Check that everything is up:

```bash
docker compose ps
```

You should see these services with status `running` (or `Up`): `qdrant`, `redis`,
`otel-collector`, `open-webui`, `grading-api`, `cadvisor`, `prometheus`, `pushgateway`,
`metrics-exporter`, `grafana`. `open-webui` may show `health: starting` for the first minute or
two while it creates its database.

Then check Open WebUI directly:

```bash
curl -s http://localhost:3000/health
```

You should see `{"status":true}`. If you get nothing or an error, wait a minute and retry. If it
still fails, look at the logs:

```bash
docker compose logs open-webui --tail 100
```

---

## 5. Create the admin account

1. Open `http://<VM-IP>:3000` in your browser.
2. Click **Get started** and sign up. **The first account created becomes the administrator.** Use
   an email you control and save the password.

<!-- SCREENSHOT: setup/05-first-signup.png | first-run sign-up page -->
<!-- TODO-UI: confirm the first-run button and page wording on v0.11.4 -->

You should land on an empty chat page with a model selector at the top.

---

## 6. Send a test chat

1. Open the model selector and choose `gpt-4o-mini` (any OpenAI model in the list works).
2. Type `Hello` and send it.

You should get a reply within a few seconds.

<!-- SCREENSHOT: setup/06-model-picker.png | v0.11 model picker with OpenAI models listed -->
<!-- TODO-UI: confirm where the model picker is in v0.11.4 (A11 says it moved) -->

**If the model list is empty:** go to **Admin Panel → Settings → Connections** and check that
the OpenAI connection is enabled and shows your key. You reach the Admin Panel by clicking your
name or avatar in the bottom-left corner. <!-- TODO-UI -->

---

## 7. Create an API key for the grading backend

The grading backend and the knowledge-base sync tool talk to Open WebUI through its API.

1. Allow API keys: **Admin Panel → Settings → General → Enable API Keys** → on → **Save**.
   <!-- TODO-UI: confirm the toggle name and location on v0.11.4 -->
2. Create a key: click your name (bottom-left) → **Settings → Account → API keys → Create new
   key**. Copy the key; it starts with `sk-`. <!-- TODO-UI -->

<!-- SCREENSHOT: setup/07-api-key.png | Settings > Account > API keys, key blurred -->

3. Put it in the grading backend's config. Open `grading_feature/backend/grading.env`, replace
   `FILL_IN_AT_STEP_7` with the key, then recreate the container so it reads the new value:

```bash
docker compose up -d grading-api      # "restart" does NOT re-read env files; use "up -d"
curl -s http://127.0.0.1:8000/api/health
```

The health check should return a JSON success response.

This key has your admin account's permissions. Keep it private.

---

## 8. Install the Chat Metrics filter

**Why this is a manual step:** Open WebUI's built-in OpenTelemetry export (already running) does
not provide all the chat metrics we need, so the team kept our own filter. See
<!-- TODO-A19: link to the A19 justification paragraph --> the A19 decision for exactly which
metrics. Open WebUI has no documented way to install a filter from the command line, so it is
pasted in through the admin screens.

**When to do it:** once on every new deployment. The filter is stored in Open WebUI's
database, so it survives restarts and upgrades of the same data. Redo it whenever you start from
an empty database. <!-- TODO-A24: confirm on v0.11.4 that the filter survives an upgrade -->

1. Go to **Admin Panel → Functions**. <!-- TODO-UI: older guides say "Workspace > Functions";
   confirm the v0.11.4 location -->
2. Click **+** (new function).
3. Delete the example code, then paste the **entire** contents of
   `monitoring/chat_metrics_filter.py`. To print the file on the VM:
   `cat monitoring/chat_metrics_filter.py`.
4. Give it the name `Chat Metrics`, then click **Save**.
5. Back in the Functions list, switch **Chat Metrics** on, then open its **⋯** menu and turn
   **Global** on, so it runs for every model and every user. <!-- TODO-UI -->
6. Click the gear icon next to it (**Valves**) and check that `pushgateway_url` is
   `http://pushgateway:9091` (this is the default). Click **Save**. <!-- TODO-UI -->

<!-- SCREENSHOT: setup/08-functions-list.png | Chat Metrics enabled and Global on -->
<!-- SCREENSHOT: setup/08-valves.png | Valves with pushgateway_url -->

**Check that it works:** send one chat message (as in step 6), then on the VM run:

```bash
curl -s http://localhost:9091/metrics | grep openwebui_chat
```

You should see lines starting with `openwebui_chat`. Nothing appears until at least one chat
message has been sent after the filter was enabled.

---

## 9. Load the course documents (knowledge base)

### 9a. Create the knowledge base in the browser

1. Go to **Workspace → Knowledge → +** (new knowledge). Give it a name, for example
   `FIN602 Client Data`. <!-- TODO-UI -->
2. Upload files by dragging them onto the page. Wait until each file finishes processing.
3. Note the knowledge base's ID: it is the last part of the page address, after
   `/workspace/knowledge/`. You only need it for the sync tool in 9b. <!-- TODO-UI -->

<!-- SCREENSHOT: setup/09-knowledge-upload.png -->

### 9b. Bulk upload with the sync tool (optional)

For many files, use the command-line tool instead of dragging:

```bash
cd tools/kb_sync
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env
```

Set the three values in `tools/kb_sync/.env`:

| Variable | Value |
|---|---|
| `OPENWEBUI_BASE_URL` | `http://localhost:3000` if you run the tool on the VM. **The example file points at the production server; change it, or you will upload into production** |
| `OPENWEBUI_API_KEY` | The key from step 7 |
| `OPENWEBUI_KB_ID` | The ID from step 9a |

```bash
python sync_kb.py list                                  # should list the KB's files
python sync_kb.py sync ./path/to/documents/ --dry-run   # shows what would change
python sync_kb.py sync ./path/to/documents/             # does it
deactivate && cd ../..
```

### 9c. Check that document search works

Attach the knowledge base to a model: **Workspace → Models → (edit or create a model) →
Knowledge → select your knowledge base → Save**. <!-- TODO-UI --> Then start a new chat with that
model and ask a question that only the uploaded documents can answer. The answer should use the
document, and a source citation should appear under the reply.

The course's role-play "client" models (system prompts and attached documents) are set up by the
professor; see the professor guide. <!-- TODO-A24: link the rewritten professor guide -->

---

## 10. Grafana

1. Open `http://<VM-IP>:3001` and log in as `admin` with the `GRAFANA_ADMIN_PASSWORD` from your
   `.env`.
2. Go to **Dashboards → FamilyFinanceChat → Open WebUI OpenTelemetry**.

The Prometheus data source and this dashboard are created automatically from
`monitoring/grafana/`. There is nothing to configure. The panels (requests per second, 95th
percentile response time) fill in within about a minute of using the chat.

<!-- SCREENSHOT: setup/10-grafana-dashboard.png -->

To check that Prometheus is collecting from every source, run this on the VM:

```bash
curl -s http://localhost:9090/api/v1/targets | grep -o '"health":"[a-z]*"' | sort | uniq -c
```

Every target should be `"health":"up"`.

**Forgot the Grafana password?** Changing `GRAFANA_ADMIN_PASSWORD` later has no effect, because
it is only applied the first time Grafana starts. Reset it instead:

```bash
docker compose exec grafana grafana-cli admin reset-admin-password '<new-password>'
```

---

## 11. Grading dashboard

<!-- TODO-A24: once A15/A16 land, replace this whole step with "open https://<host>/grading and
log in", and move the laptop instructions to grading_feature/README.md. -->

The backend already runs in the stack as `grading-api` (step 7). It is an API only; the web
interface currently runs on a laptop.

**Running the full dashboard on your laptop:**

1. Install Python 3.10+, and Node.js **exactly v20.19.5** with npm **exactly 10.8.2**.
   `run_app.sh` refuses other versions. The easiest way is [nvm](https://github.com/nvm-sh/nvm):
   `nvm install 20.19.5 && nvm use 20.19.5`, then check with `node --version` and `npm --version`.
2. Make the VM's Open WebUI reachable from the laptop: the SSH tunnel from
   [Prerequisites](#make-the-web-ports-reachable-gcp) (then use `http://localhost:3000`), or the
   VM's IP if you used the firewall rule.
3. Configure and start it:

```bash
cd grading_feature
cp .env.example .env
nano .env
```

| Variable | Value |
|---|---|
| `OPENWEBUI_BASE_URL` | `http://localhost:3000` (tunnel) or `http://<VM-IP>:3000` |
| `OPENWEBUI_API_KEY` | Uncomment the line and paste the key from step 7 |
| `OUTPUT_PATH` and `DATA_PATH` | The same file path on your laptop, for example `/home/you/ffc-data/chats.json` (the folder must exist) |
| `OPENAI_API_KEY` | Add this line with your OpenAI key; it is used for scoring |

```bash
./run_app.sh
```

The backend starts on port 9500 and the frontend dev server starts after it. Open the address
the frontend prints (normally `http://localhost:5173`).

4. Click **Refresh** to pull the latest chats from Open WebUI.

**Refresh costs money:** it scores student questions through the OpenAI API and can take several
minutes on a full class. Click it once and wait.

<!-- SCREENSHOT: grading/home.png and grading/refresh.png (re-shoot; current ones are pre-v0.11) -->

---

## You're done when

- [ ] `docker compose ps` shows all ten services running
- [ ] A chat message gets a reply (step 6)
- [ ] A question about an uploaded document gets an answer with a citation (step 9c)
- [ ] `curl -s http://localhost:9091/metrics | grep openwebui_chat` prints lines (step 8)
- [ ] Grafana's **Open WebUI OpenTelemetry** dashboard shows data (step 10)
- [ ] `curl -s http://127.0.0.1:8000/api/health` succeeds (step 7)

---

## Day-to-day commands

```bash
docker compose ps                         # what is running
docker compose logs <service> --tail 100  # recent logs, e.g. open-webui
docker compose up -d <service>            # apply a changed config/env file to one service
docker compose down                       # stop everything (data is kept)
docker compose up -d                      # start everything again
```

**Never run `docker compose down -v` on a real deployment.** `-v` deletes the named volumes
(Redis, Prometheus, Grafana and grading data).

---

## Common errors

| What you see | Likely cause | Fix |
|---|---|---|
| `docker compose up` says `Set GRAFANA_ADMIN_PASSWORD` | Missing from `.env` | Add `GRAFANA_ADMIN_PASSWORD=...` to `.env` (step 2a) |
| `docker compose up` says `env file .../grading.env not found` | Step 2b skipped | Create `grading_feature/backend/grading.env` |
| `cp: cannot stat '.env.staging.example'` | Old instructions; that file doesn't exist | Use the template in step 2a |
| `permission denied ... docker.sock` | Your user isn't in the `docker` group | `sudo usermod -aG docker "$USER"`, then log out and in |
| `no space left on device` while pulling | Images need several GB | Free space (`docker image prune` removes unused images; check first with `docker images`), or enlarge the disk |
| Browser can't reach `http://<VM-IP>:3000` | GCP firewall | Use the SSH tunnel or firewall rule in [Prerequisites](#make-the-web-ports-reachable-gcp) |
| `open-webui` keeps restarting | Bad or missing `OPENAI_API_KEY` / `WEBUI_SECRET_KEY` | Check `.env`, then `docker compose logs open-webui --tail 100` |
| Model list is empty | OpenAI connection off or key wrong | **Admin Panel → Settings → Connections** (step 6) |
| Changed `.env` but Open WebUI behaves the same | Persistent config: `.env` only applies on first start | Change the setting in the Admin Panel instead |
| Changed `grading.env` but nothing changed | `docker compose restart` doesn't re-read env files | `docker compose up -d grading-api` |
| Uploading a document fails or search returns nothing | `RAG_EMBEDDING_ENGINE` not `openai`, or bad `RAG_OPENAI_API_KEY` | Fix in **Admin Panel → Settings → Documents** (persistent config) |
| No `openwebui_chat` lines from the Pushgateway | Filter not installed, not enabled, not Global, or no chat sent yet | Redo step 8 and send a message |
| `curl localhost:9091` refused | `pushgateway` service not running | `docker compose ps`; if it's not listed, your checkout predates the A19 change; pull the latest `main` |
| Grafana shows "No data" | No chat traffic yet, or a Prometheus target is down | Send a chat; run the targets check in step 10 |
| Can't log into Grafana | Password set on first start only | Reset with `grafana-cli` (step 10) |
| Grading scores look generic or identical for every question | Mock scoring: `MOCK_SCORER=1`, or `OPENAI_API_KEY` empty in the grading config | Set `MOCK_SCORER=0` and a real key, then `docker compose up -d grading-api` (or restart `run_app.sh`) |
| Grading `/refresh` returns 401 | API key wrong, expired, or API keys disabled | Redo step 7 |
| `run_app.sh` says the npm or Node version is wrong | Exact versions are required | `nvm install 20.19.5 && nvm use 20.19.5` |
| `docker compose up` fails with "port is already allocated" | Something else uses 3000, 3001, 8000, 8001, 8080, 8889, 9090 or 9091 | Stop the other program, or change the host port in `docker-compose.yml` |
| Chat hangs after 7+ messages | Known Open WebUI bug in earlier versions | Start a new chat. <!-- TODO-A24: re-test on v0.11.4; delete this row if fixed --> |

<!-- TODO-A24: add every stuck point from task 0.6 that isn't covered above. -->

---

## Upgrading Open WebUI

The short version of a safe upgrade:

1. **Back up** the database and data (`scripts/backup.sh`; see `docs/backup/backup-restore.md`).
2. **Free disk space** for the new image (`df -h /`). You need room for the old and the new image
   at the same time.
3. **Keep the old image** for rollback. Don't prune it until the new version is accepted.
4. Pull the new image, **stop** `open-webui`, change the `image:` tag in `docker-compose.yml`,
   and **start** it again (`docker compose up -d open-webui`).
5. **Test:** old chats and users are still there, a new chat works, document search works,
   metrics still arrive, and the grading backend can still extract chats.

Upgrade one minor version at a time. Some migrations are **one-way** (v0.10 changes the config
table), so going back means restoring the backup, not just switching the image. For production,
follow the full [A13 production upgrade runbook](#a13-production-upgrade-runbook-preparation)
below.

<!-- TODO-A24 (P1): this runbook is long for a setup guide. Consider moving it to
docs/runbooks/upgrade.md. If moved, update the anchor link in HANDOFF.md. -->

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
sudo test -d "$A2_BACKUP"
sudo test -f "$A2_BACKUP/SHA256SUMS"
case "$A2_BACKUP" in *.partial) echo 'Incomplete A2 backup' >&2; exit 1;; esac
sudo bash -euo pipefail -c 'cd -- "$1"; sha256sum -c SHA256SUMS' bash "$A2_BACKUP"
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
python3 "$METRICS_SCRIPT" --url "$METRICS_URL" --since "$SMOKE_STARTED" --model "$SMOKE_MODEL" \
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
The checker matches the model and the full Pushgateway grouping labels (including
job and instance) to a fresh push. It does not identify an individual request:
keep other chat traffic and producers for this metrics job stopped during the check.
For each supported version, rehearse the installed Filter's inlet/outlet lifecycle;
client and Filter unit tests alone do not prove the server invokes both hooks.

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
