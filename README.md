# FamilyFinanceChat

Repository: https://github.com/mfedenia/FamilyFinanceChat

An AI-powered financial advising practice platform for FIN 602 students. Students chat with an AI role-player pre-loaded with realistic family financial scenarios. Instructors get automated question quality scoring, ABI trust analysis, and a React dashboard for reviewing student interactions.

---

## What This System Does

FIN 602 students practice client-facing financial advising conversations. The platform gives them unlimited, on-demand AI sessions without requiring human role-players or scheduling. After sessions, professors use the grading dashboard to review chat transcripts, score student questions across seven quality dimensions, and get per-student performance summaries.

---

## Architecture Overview

Eight Docker containers on a GCP VM plus a grading tool that runs locally:

| Component | Role |
|---|---|
| **OpenWebUI** (port 3000) | Chat UI, LLM routing, user auth, chat history |
| **Qdrant** | Vector database for RAG document retrieval |
| **Valkey/Redis** | WebSocket session state |
| **Prometheus** (port 9090) | Metrics collection, 30-day retention |
| **Pushgateway** (port 9091) | Receives chat metrics from Filter Function |
| **cAdvisor** (port 8080) | Container CPU/memory metrics |
| **metrics-exporter** (port 8001) | Health probe sidecar for OpenWebUI |
| **Grafana** (port 3001) | Dashboards |
| **Grading Dashboard** (local) | FastAPI + React — pulls data from production VM |

Full details: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Setup

See [SETUP.md](SETUP.md) for step-by-step instructions including environment variables, host directory prep, Filter Function installation, and common errors.

**Quick start (assuming prerequisites are met):**
```bash
cp .env.staging.example .env   # fill in OPENAI_API_KEY and WEBUI_SECRET_KEY
docker compose up -d
```

---

## What Works

- Multi-user chat via OpenAI API with real-time WebSocket sessions
- RAG over course documents using Qdrant + `text-embedding-3-small`
- File upload to GCS bucket via native OpenWebUI UI or `tools/kb_sync/` CLI
- Chat metrics in Grafana (after manual Filter Function install — see SETUP.md)
- Grading dashboard: chat extraction, 7-dimension question scoring, ABI trust analysis
- Multi-environment support: production stack + test stack

---

## What Doesn't Work / Known Issues

- **Filter Function must be re-installed after every deployment** — not auto-deployed
- **Grading dashboard requires local run** — professors need to SSH to the VM or run it on their machine; not hosted yet
- **No CI/CD** — upgrades have broken things silently in the past; see SETUP.md for the recommended pipeline

---

## What the Next Team Should Work On

1. Host the grading dashboard on the VM behind Nginx with basic auth
2. Add CI/CD (GitHub Actions build + `/health` smoke test — see SETUP.md)
3. Wire the `/ready` endpoint into GCP uptime monitoring and alerting
4. Automate Filter Function installation as part of deployment

Full context and priority reasoning: [HANDOFF.md](HANDOFF.md)

---

## Project Structure

```
Dockerfile              # Single line — FROM ghcr.io/open-webui/open-webui:v0.8.12
docker-compose.yml      # Production stack
prometheus.yml          # Prometheus scrape config
tools/kb_sync/          # CLI for managing OpenWebUI knowledge bases
monitoring/             # Chat Metrics Filter Function + health exporter
grading_feature/        # Professor grading dashboard (FastAPI + React)
rag_bio_project/        # Custom Python RAG pipeline for biography documents
scripts/                # Prometheus backfill utility
docs/                   # PDF guides and setup notes
legacy/                 # Old vendor fork code (reference only)
archive/                # Stale files from Spring 2026 cleanup
```

---

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — component diagram, data flow, config values
- [SETUP.md](SETUP.md) — how to run this, environment variables, common errors, CI/CD recommendation
- [HANDOFF.md](HANDOFF.md) — what was built, known bugs, architectural decisions, what to do next
- [CLAUDE.md](CLAUDE.md) — working instructions: the one rule, where things are, conventions
- [docs/memory/](docs/memory/) — project memory: durable, non-obvious facts kept in the repo
- [capstone_fall2026/](capstone_fall2026/) — briefing package for the next team: project plan, the
  upstream-compatibility policy, the avatar research track, plus a slide deck and presentation
  script. Documentation only; not part of the running system.

---

## Maintainers

Primary maintainer: [@mfedenia](https://github.com/mfedenia)
