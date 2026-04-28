# Architecture

**FamilyFinanceChat — Technical Deep-Dive**
Repository: https://github.com/mfedenia/FamilyFinanceChat

---

## Component Overview

```
                        ┌─────────────────────────────────────────────────────┐
                        │               GCP VM · Docker Compose                │
                        │                                                       │
  Student ──HTTPS──►   │  OpenWebUI :3000 ◄──── Qdrant (internal)            │
                        │       │    │    └──── Valkey/Redis (internal)        │
  Instructor            │       │    │                                          │
      │                 │       │    └──push metrics──► Pushgateway :9091      │
      ▼                 │       │                              │                │
  React Dashboard       │       └──chat completions──►  OpenAI API (external)  │
  (local workstation)   │       └──file uploads──────►  GCS Bucket (external)  │
      │                 │                                      │                │
      └──REST──►        │  Grading Backend (local workstation) │                │
                        │                                      ▼                │
                        │  Prometheus :9090 ◄── scrapes ── Pushgateway          │
                        │       │              ◄── scrapes ── cAdvisor :8080    │
                        │       │              ◄── scrapes ── metrics-exporter  │
                        │       ▼                                               │
                        │  Grafana :3001                                        │
                        └─────────────────────────────────────────────────────┘
```

---

## Services (docker-compose.yml)

All services run on a shared bridge network `ai-net`.

| Service | Image | Host Port | CPU / RAM Limit | Role |
|---|---|---|---|---|
| `open-webui` | `ghcr.io/open-webui/open-webui:v0.8.12` | 3000 | 2 CPU / 10 GB | Chat UI, LLM routing, user auth, chat history |
| `qdrant` | `qdrant/qdrant:latest` | none (internal) | — | Vector database for RAG |
| `redis` | `valkey/valkey:8.0.1-alpine` | none (internal) | 512 MB | WebSocket session state |
| `cadvisor` | `gcr.io/cadvisor/cadvisor:latest` | 8080 | 0.5 CPU / 256 MB | Container CPU/memory/network metrics |
| `prometheus` | `prom/prometheus:v2.51.2` | 9090 | — | Metrics collection, 30-day retention |
| `pushgateway` | `prom/pushgateway:v1.8.0` | 9091 | — | Receives chat metrics pushed by Filter Function |
| `metrics-exporter` | custom FastAPI build | 8001 | — | Probes OpenWebUI `/health` every 15s |
| `grafana` | `grafana/grafana:9.0.0` | 3001 | — | Dashboards (default login: admin/admin) |

**Key volume mounts:**
- `/mnt/gcs/fin602 → /app/backend/data/uploads` — file uploads route to a GCS bucket mounted on the VM
- `/opt/openwebui/data → /app/backend/data` — persistent SQLite DB and OpenWebUI config
- `/opt/qdrant/storage → /qdrant/storage` — vector index on disk
- `/opt/prometheus/data → /prometheus` — 30-day time-series store
- `/opt/grafana/data → /var/lib/grafana` — dashboard and datasource config

---

## The LLM

- **Provider:** OpenAI API (`https://api.openai.com/v1`)
- **Model:** `gpt-4o-mini` (default; configurable via `OPENAI_MODEL` env var)
- **Configured via:** `OPENAI_API_KEY`, `OPENAI_API_BASE_URL`, `OPENAI_MODEL` in `.env`
- Ollama support exists in the RAG pipeline code (`rag_bio_project/src/llm.py`) but is not running — it was commented out in `docker-compose.yml`

---

## RAG Pipeline

Two separate RAG systems exist in this project.

### 1. Native OpenWebUI RAG (active in production)

This is what students actually use when chatting.

```
PDF/TXT/DOCX files
  └─ Uploaded via OpenWebUI UI or tools/kb_sync/ CLI
       └─ OpenWebUI processes and embeds using OpenAI text-embedding-3-small
            └─ Vectors stored in Qdrant
                 └─ On each chat turn: MMR search (k=5, threshold=0.30)
                      └─ Retrieved chunks injected into LLM context
```

**Key config values** (from `.env`):
| Variable | Value |
|---|---|
| `VECTOR_DB` | `qdrant` |
| `QDRANT_URI` | `http://qdrant:6333` |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-small` |
| `RAG_TOP_K` | `5` |
| `RAG_RELEVANCE_THRESHOLD` | `0.30` |
| `RAG_TOP_K_RERANKER` | `3` (reranking model is not configured) |
| `RAG_FILE_MAX_SIZE` | `50` MB |
| `RAG_ALLOWED_FILE_EXTENSIONS` | `pdf,md,txt,docx` |

### 2. Custom Python RAG (`rag_bio_project/src/`)

A standalone research/utility pipeline for FIN602 client biography documents. Not connected to OpenWebUI — runs as Python scripts or notebooks.

```
PDF/TXT files (rag_bio_project/data_pdfs/)
  └─ loader.py — multi-source loader (PDF, TXT, URLs)
       └─ splitter.py — type-aware splitting
            PDF:  chunk_size=1000, overlap=150
            TXT:  chunk_size=800,  overlap=100
       └─ embeddings.py — builds embedding model
            Default: BAAI/bge-m3 (HuggingFace, runs locally)
            Optional: text-embedding-3-small (OpenAI), DashScope
       └─ vectorstore.py — persists to ChromaDB
            Collections named: {username}_{character}
       └─ retriever.py — MMR retrieval
            k=5, fetch_k=15, lambda=0.5
            Strictness: strict=0.70 / medium=0.60 / loose=0.50
       └─ prompting.py — 5 prompt templates, auto-selected by query keywords
            rag_concise          — default
            rag_with_citations   — "cite / citation / source"
            rag_compare          — "compare / vs / difference"
            rag_timeline         — "timeline / when / history"
            rag_extraction       — "extract / structured / json"
       └─ llm.py — LLM wrapper (OpenAI / DashScope / Ollama)
       └─ pipeline.py — run_pipeline(question, cfg) end-to-end entry point
```

A pre-built Chroma index is committed at `rag_bio_project/index/`.

---

## Chat Metrics Flow

```
Student sends a message
  └─ Filter Function inlet() — records start_time, message_count, estimated_tokens, model
  └─ OpenWebUI calls OpenAI API (LLM response)
  └─ Filter Function outlet() — computes elapsed time, extracts token usage
       └─ HTTP POST Prometheus text to http://pushgateway:9091/metrics/job/openwebui_chat_metrics

Prometheus (every 15s)
  └─ Scrapes Pushgateway → stores time-series
  └─ Scrapes cAdvisor → container CPU/mem/network
  └─ Scrapes metrics-exporter → OpenWebUI health probe latency + uptime

Grafana (port 3001)
  └─ Queries Prometheus via PromQL
```

**Metrics pushed per chat turn:**
| Metric | Description |
|---|---|
| `openwebui_chat_completion_seconds` | Total round-trip time (inlet → outlet), labeled by model |
| `openwebui_chat_context_length` | Number of messages in context |
| `openwebui_context_tokens_estimated` | Estimated tokens (total chars ÷ 4) |
| `openwebui_llm_prompt_tokens` | Actual prompt tokens from LLM response (when available) |
| `openwebui_llm_completion_tokens` | Actual completion tokens from LLM response (when available) |

**Important:** The Filter Function is not auto-deployed. After any new deployment it must be manually installed: Admin > Workspace > Functions > paste `monitoring/chat_metrics_filter.py` > enable globally. See `monitoring/README.md`.

**Prometheus scrape targets** (`prometheus.yml`):
| Job | Target | What it collects |
|---|---|---|
| `prometheus` | `prometheus:9090` | Self-metrics |
| `openwebui-exporter` | `metrics-exporter:8000` | Health probe latency and uptime |
| `cadvisor` | `cadvisor:8080` | Container resource usage |
| `pushgateway` | `pushgateway:9091` | Chat metrics from Filter Function |

---

## Grading & Scoring Pipeline

The professor grading tools run **locally on a developer workstation** — they are not in the Docker Compose stack.

```
OpenWebUI (production VM)
  └─ REST API: GET /api/v1/users/all + GET /api/v1/chats/all/db
       └─ extract_chats.py — normalizes into structured JSON
            {user_id, name, email, role, chats: [{title, message_pairs: [{timestamp, question, answer}]}]}
       └─ Stored at OUTPUT_PATH (local file)

Grading Backend (FastAPI, grading_feature/backend/)
  └─ GET /refresh — triggers extraction, overwrites JSON file atomically
  └─ GET /users — all students with chat counts
  └─ GET /user/{id} — full chat history
  └─ POST /api/score — score a list of questions via OpenAI
  └─ GET /api/student-feedback — per-student or aggregate feedback

Scoring (scoring_service.py):
  7 dimensions scored 0–2 each (total 0–14):
    relevance, politeness, on_topic, neutrality,
    non_imperative, clarity_optional, privacy_minimization_optional
  → Normalized to 0–100 overall score
  → Optional ABI (Ability, Benevolence, Integrity) trust framework
       12 sub-dimensions derived from rubric scores
       Weighted formulas → Ability / Benevolence / Integrity (each 0–1)
       ABI Total = average of the three

React Dashboard (grading_feature/frontend/):
  Home — metric cards, top-users chart, chats-per-day chart
  User Detail — paginated chat list with sliding transcript drawer
  Scoring — upload question list, score, paginate results
  Student Feedback — per-student performance band + synthesized habits
```

---

## Knowledge Base Sync Tool (`tools/kb_sync/`)

CLI tool for managing OpenWebUI knowledge bases programmatically via the public REST API. Useful for bulk updates to the course materials KB without using the UI.

```bash
python sync_kb.py list
python sync_kb.py add ./new_document.pdf
python sync_kb.py replace ./updated_document.pdf
python sync_kb.py sync ./context_files/ --dry-run
```

Requires `OPENWEBUI_BASE_URL`, `OPENWEBUI_API_KEY`, and `OPENWEBUI_KB_ID` (see `tools/kb_sync/.env.example`).

---

## External Services and Dependencies

| Service | Purpose | Required for |
|---|---|---|
| OpenAI API | LLM chat completions + text embeddings | Chat, RAG, scoring |
| GCS Bucket (`/mnt/gcs/fin602`) | Persistent file upload storage | File uploads via OpenWebUI |
| GCP VM | Hosts all Docker containers | Production deployment |

---

## What Prometheus Does NOT Scrape

OpenWebUI itself is not a Prometheus scrape target. Chat-level metrics only flow through the Pushgateway (pushed by the Filter Function). Infrastructure metrics come from cAdvisor. The `metrics-exporter` sidecar only measures whether OpenWebUI is reachable — it does not expose internal OpenWebUI metrics.
