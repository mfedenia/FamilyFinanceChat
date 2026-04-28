# Handoff Notes — Spring 2026

To the next team working on FamilyFinanceChat:

This document is what we wish we had when we started. Read it before touching any code.

---

## What Was Built This Semester

### OpenWebUI Upgrade and Decoupling (biggest accomplishment)

When we inherited the project it was locked to OpenWebUI v0.6.41 because our custom code replaced three of OpenWebUI's core internal files — ~6,500 lines of forked code that had to be manually re-merged with every upstream change. Upgrading was effectively impossible.

We decoupled the project completely. The Dockerfile is now a single line. All customization happens through OpenWebUI's official plugin system (Filter Functions) and public REST API. The project can now track upstream OpenWebUI releases without any forking.

See `legacy/openwebui_upgrade_decoupling.plan.md.done` for the full migration record.

### ABI Trust Scoring

We implemented an Ability / Benevolence / Integrity scoring framework on top of the question quality rubric. It maps 7 rubric dimensions (scored 0–2 each) through 12 sub-dimensions to produce three trust scores and a composite ABI total. The research pipeline for this is in `archive/abi_trust_pipeline/`; the production implementation is in `grading_feature/backend/scoring_service.py`.

### Grading Dashboard

A full FastAPI + React/Vite dashboard for professors to review student interactions. It pulls data from the OpenWebUI REST API (no direct database access), scores student questions via OpenAI, and presents per-student performance breakdowns with the ABI lens. Runs locally — see `SETUP.md` for how to run it.

### Monitoring Stack

Prometheus + Grafana + Pushgateway + cAdvisor + a custom sidecar exporter, all wired together in `docker-compose.yml`. Chat-level metrics (latency, token counts) flow through a Filter Function → Pushgateway → Prometheus → Grafana. Container-level metrics come from cAdvisor.

### Multi-Environment Infrastructure

We built staging and test Docker stacks. The staging environment was never fully validated after the OpenWebUI upgrade and has been removed. The test stack (`docker-compose.test.yml`) remains. A future team should build staging fresh once CI/CD is in place.

---

## Known Bugs and Limitations

### Filter Function must be manually re-installed after every deployment
The Chat Metrics filter (`monitoring/chat_metrics_filter.py`) is installed through the OpenWebUI admin UI and is not persisted in any config file. Every time you deploy a fresh instance you must re-install it. See `monitoring/README.md` for exact steps. Without it, Grafana shows no chat metrics.

### `rag_with_citations` prompt mode is broken
In `rag_bio_project/src/prompting.py`, the `_format_context_with_indices()` function has a Python f-string bug: it uses `{{i}}` and `{{it.get(...)}}` which produce literal `{i}` text instead of the actual index values. The citation numbers never appear correctly. This only affects the custom RAG pipeline, not the native OpenWebUI RAG.

### End-to-end upgrade validation was not completed
The decoupling migration plan has one item still marked pending: a full smoke test of KB upload + chat metrics via Filter Function + grading dashboard all working together post-upgrade. We tested components individually but never ran a formal end-to-end pass.

### Duplicate scoring logic
The 7-dimension rubric and ABI scoring formulas are implemented twice: once in `scoring_page/backend/server.js` (JavaScript) and once in `grading_feature/backend/scoring_service.py` (Python). These need to be kept in sync manually if the rubric changes. The `scoring_page` is a standalone prototype; `grading_feature` is the full platform. Consider consolidating.

### Grading dashboard is not hosted
Professors currently need to SSH into the VM and run `./run_app.sh` locally. This is a real barrier for non-technical instructors. Containerizing the grading app and putting it behind Nginx with basic auth should be a high-priority task next semester.

---

## Architectural Decisions and Why

### No Ollama
We evaluated running a local model via Ollama on the VM. The VM does not have a GPU; CPU inference was too slow for a live student experience. We kept the Ollama code paths in `rag_bio_project/src/llm.py` for future reference but the container is commented out.

### Pushgateway for chat metrics instead of direct Prometheus scraping
OpenWebUI does not expose a `/metrics` endpoint. We considered a scraping sidecar but settled on the Filter Function + Pushgateway approach because it uses OpenWebUI's official plugin mechanism and survives upgrades. The trade-off is that Pushgateway is a single-point accumulator — if it restarts, metrics between the last scrape and the restart are lost.

### Grading uses REST API, not SQLite
The previous implementation read OpenWebUI's SQLite database directly with raw SQL. This broke every time OpenWebUI changed its schema. We migrated to the public REST API (`/api/v1/users/all`, `/api/v1/chats/all/db`) which is versioned and stable. The `--legacy` flag in `extract_chats.py` is a stub placeholder in case the API ever becomes unavailable.

### GCS for file uploads
File uploads are routed to a GCS bucket mounted at `/mnt/gcs/fin602` on the VM. This means uploads persist independently of the container lifecycle. The downside is that the GCS mount must be configured on the host before `docker compose up` — it is not handled by Docker.

### ChromaDB for the custom RAG pipeline, Qdrant for native RAG
The `rag_bio_project/` pipeline uses ChromaDB because it was the first to be built and runs locally without a server. The native OpenWebUI RAG uses Qdrant because it is what OpenWebUI v0.8+ integrates with. They are completely separate systems.

---

## What We Would Do Next

**High priority:**

1. **Host the grading dashboard.** Add it to `docker-compose.yml`, put it behind Nginx with HTTP basic auth (or the university's SSO), and give professors a URL. One to two days of engineering.

2. **Wire `/ready` into GCP uptime monitoring.** OpenWebUI v0.8.9 added a `/ready` endpoint that only returns 200 when the DB and Redis are fully up. Set a GCP Uptime Check against it and alert the primary maintainer. The app went down once and we found out passively from students.

3. **Add CI/CD.** See the CI/CD section in `SETUP.md` for the recommended starting pipeline. The key insight from this semester: upgrades break things silently if there are no automated checks. A GitHub Actions workflow that builds the image and hits `/health` would catch 80% of issues with minimal effort.

**Medium priority:**

4. **Automate Filter Function installation.** The manual install-after-deploy step is fragile. Options: use the OpenWebUI REST API (`POST /api/v1/functions/`) to install it as part of a post-deploy script, or add it to a startup init container.

5. **Fix the `rag_with_citations` f-string bug** in `rag_bio_project/src/prompting.py` line 104.

6. **Pin `qdrant:latest` and `cadvisor:latest`** to specific versions in `docker-compose.yml`. Unpinned images will silently upgrade on the next `docker compose pull`.

7. **Consolidate scoring logic** — eliminate the duplicate JS/Python implementations or add a comment pointing to which is canonical.

**Longer term:**

8. **Multi-tenant packaging.** The long-term goal is expanding to other courses and universities. Everything is currently single-tenant. Start with a docker-compose profile per course and think about data isolation for vector stores.

9. **Skills for financial frameworks.** OpenWebUI v0.8.0 added Skills — reusable instruction sets attachable to models, manageable from the UI. The course frameworks (ABI, financial planning rubrics) should live as Skills so a professor can update them without touching config files or asking a developer.

---

## Gotchas That Took Us a Long Time to Figure Out

**The vendor fork was the root of everything.** Every weird environment issue traced back to the fact that we were running a fork of OpenWebUI internals. Once we removed it, almost every mysterious startup and upgrade issue disappeared.

**Filter Functions share in-memory state between requests.** The `_state` dict in `chat_metrics_filter.py` is keyed by `user_id` and held in the Filter object instance. If two users chat simultaneously or a request times out mid-flight, state can leak across requests. This is unlikely to cause visible problems at low traffic volumes but is worth knowing.

**GCS mount must exist before Docker starts.** If the GCS bucket isn't mounted at `/mnt/gcs/fin602` when you run `docker compose up`, OpenWebUI will start but file uploads silently fail. There is no error in the logs — the upload just disappears.

**Prometheus runs as UID 65534 (nobody).** The `/opt/prometheus/data` directory must be owned by that UID or Prometheus will fail to start with a cryptic permissions error.

**The scoring page `.env.example` has a Windows PowerShell line in it.** Line 6 of `scoring_page/backend/.env.example` reads `$env:OPENAI_API_KEY="your-api-key"` — that is a PowerShell syntax leftover, not a real env var format. Ignore it.

---

## Repository Map (quick reference)

```
FamilyFinanceChat/
├── Dockerfile                    # Single line: FROM ghcr.io/open-webui/open-webui:v0.8.12
├── docker-compose.yml            # Full production stack (8 services)
├── prometheus.yml                # Prometheus scrape config
├── tools/kb_sync/                # CLI to manage OpenWebUI knowledge bases via REST API
├── monitoring/
│   ├── chat_metrics_filter.py    # Filter Function — install manually in OW admin UI
│   ├── exporter/                 # Health-probe sidecar (FastAPI)
│   └── README.md                 # Filter install instructions
├── grading_feature/
│   ├── backend/                  # FastAPI: chat extraction, scoring, ABI
│   └── frontend/                 # React/Vite dashboard
├── scoring_page/
│   ├── backend/server.js         # Node.js scoring API (standalone prototype)
│   └── frontend/                 # Plain HTML scoring UI
├── rag_bio_project/
│   ├── src/                      # Custom RAG pipeline (standalone, not connected to OW)
│   ├── data_pdfs/                # FIN602 client biography PDFs
│   └── index/                   # Pre-built ChromaDB vector index
├── scripts/
│   ├── backfill_history.py       # One-time Prometheus history import utility
│   └── run_backfill.sh           # Wrapper for backfill (uses promtool)
├── docs/                         # PDF guides, setup notes, feature docs
├── legacy/                       # Old vendor fork code (reference only, not active)
├── archive/                      # Stale files moved here during Spring 2026 cleanup
├── README.md                     # Project overview
├── SETUP.md                      # This is how you run it
├── HANDOFF.md                    # This document
└── ARCHITECTURE.md               # Technical deep-dive
```
