# Setup Guide

How to get FamilyFinanceChat running from scratch.

---

## Prerequisites

| Requirement | Version / Notes |
|---|---|
| Docker | 24+ with Compose v2 (`docker compose`, not `docker-compose`) |
| GCP VM | e2-standard-4 or larger recommended (2 vCPU / 8 GB minimum; production uses 10 GB RAM limit for OpenWebUI alone) |
| GCS Bucket | Mounted on the VM at `/mnt/gcs/fin602` for file uploads |
| OpenAI API key | Required for chat completions and RAG embeddings |
| Python 3.10+ | Only needed if running the grading dashboard or RAG pipeline locally |
| Node.js 20+ | Only needed if running the scoring page locally |
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

The GCS bucket mount at `/mnt/gcs/fin602` must be set up separately via the GCP VM's FUSE mount config before starting Docker.

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

## 9. Running the Scoring Page (Local)

```bash
cd scoring_page
# set OPENAI_API_KEY and OPENAI_MODEL in scoring_page/backend/.env
./run.sh
```

Available at `http://localhost:8787`.

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

## CI/CD Recommendation

There is currently no CI/CD pipeline. The following is the recommended starting point for the next team.

### Minimum viable pipeline (GitHub Actions)

Create `.github/workflows/smoke-test.yml`:

```yaml
on: [push, pull_request]
jobs:
  smoke-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build OpenWebUI image
        run: docker build -t open-webui-test .
      - name: Start stack
        run: |
          cp .env.example .env
          echo "OPENAI_API_KEY=sk-test" >> .env
          echo "WEBUI_SECRET_KEY=$(openssl rand -hex 32)" >> .env
          docker compose up -d open-webui qdrant redis
      - name: Wait for health
        run: |
          for i in $(seq 1 30); do
            curl -sf http://localhost:3000/health && break
            sleep 5
          done
      - name: Smoke test
        run: curl -sf http://localhost:3000/health
```

### Why this matters

The v0.6.41 → v0.8.12 upgrade broke things that weren't caught until manual testing. A build + health-check gate would have caught the most common failure (image build error, OW failing to start) automatically on every PR.

### What to add next

1. **Post-deploy Filter Function check** — after deploying, curl the Pushgateway and assert `openwebui_chat_completion_seconds` appears after a synthetic chat request. Automates the manual step 5 above.
2. **Grading dashboard smoke test** — `pip install -r requirements.txt && python -m pytest` for the backend once tests are written.
3. **Dependency pinning** — `qdrant:latest` and `cadvisor:latest` in `docker-compose.yml` are unpinned. Pin to specific versions so upgrades are explicit and tested.
