# Demo 8: OpenWebUI Upgrade Path Analysis

## Goal

Determine how to make FamilyFinanceChat upgradeable to newer versions of OpenWebUI without requiring extensive code changes each time.

## The Problem

The platform is currently locked to **OpenWebUI v0.6.41** (the current stable release is v0.8.12). Upgrading is effectively impossible because the project replaces three of OpenWebUI's core internal files at deploy time using Docker bind mounts:

| Replaced File | Size | Purpose |
|---|---|---|
| `main.py` | 2,518 lines | OpenWebUI's entire FastAPI application entrypoint |
| `config.py` | 3,876 lines | OpenWebUI's full configuration module |
| `observability.py` | 174 lines | Custom Prometheus metric definitions |

A fourth file, `custom_pdf_router.py`, is injected into OpenWebUI's routers directory and imports five of its internal Python modules directly.

In total, **~6,500 lines of forked OpenWebUI code** must be manually diffed and re-merged against upstream changes for every version upgrade. This is the sole reason the project cannot move past v0.6.41.

## What We Found

### Most of the forked code is no longer necessary

An audit of every modification in the forked files revealed that the custom code serves only three purposes:

1. **PDF Crawler** -- A custom router for uploading PDFs, crawling linked documents, and adding them to Knowledge Bases. This feature is no longer needed.

2. **Chat-stage Prometheus metrics** -- Per-stage timing of the RAG pipeline and LLM inference (how long retrieval takes, how long the model takes to respond, token counts). This is valuable and should be preserved.

3. **Compatibility shims** -- Workarounds for missing configuration flags specific to v0.6.41. These become unnecessary on newer versions.

### OpenWebUI has official extension mechanisms that replace our approach

OpenWebUI provides **Filter Functions** -- an official plugin system where Python code is installed through the admin UI and runs on every chat request/response. Filter Functions:

- Survive OpenWebUI upgrades (no source code modifications required)
- Have access to the full message payload, user context, and request metadata
- Can measure timing, count tokens, and push metrics to external systems

This is a direct replacement for the chat metrics currently embedded in the forked `main.py`.

### Knowledge Base management works natively

OpenWebUI's built-in UI already supports uploading files to Knowledge Bases via drag-and-drop. The project also already has a CLI tool (`tools/kb_sync/`) that manages KB files through OpenWebUI's public REST API. No custom code is needed for KB management.

### The grading dashboard has a separate fragility

The professor grading dashboard (`grading_feature/`) reads OpenWebUI's SQLite database directly with raw SQL queries. While this works today, the database schema can change across OpenWebUI versions. This should be migrated to use OpenWebUI's public REST API instead, which returns a stable JSON format regardless of internal schema changes.

## Proposed Plan

### Phase 1: Migrate Chat Metrics to a Filter Function

Build an OpenWebUI Filter Function (~50 lines of Python) that captures the same chat-stage timing data currently in the forked `main.py`. Metrics are pushed to a Prometheus Pushgateway container, which Prometheus already knows how to scrape. Grafana dashboards are updated to read from the new source.

### Phase 2: Migrate Grading Data Extraction to the Public API

Refactor `extract_chats.py` to pull user and chat data from OpenWebUI's REST API (`GET /api/v1/users/`, `GET /api/v1/chats/`) instead of querying the SQLite database directly. This eliminates the dependency on internal database schema.

### Phase 3: Remove the Vendor Fork and Upgrade

- Remove all five Docker bind mounts that replace OpenWebUI's internal files
- Update the Dockerfile to use a current OpenWebUI version (unpin from v0.6.41)
- Archive the `custom-code/` directory as reference (do not delete)
- Drop the PDF crawler, bookmarklet, and standalone upload app

### Phase 4: Validate

- Confirm Knowledge Base uploads work via native UI and `tools/kb_sync`
- Verify existing KB data (files and vector embeddings) survived the upgrade
- Confirm chat metrics appear in Grafana through the new pipeline
- Confirm the grading dashboard displays correct data via the API

## Impact Summary

| Area | Before | After |
|---|---|---|
| OpenWebUI version | Locked to v0.6.41 | Any version |
| Custom code replacing OW internals | ~6,500 lines | 0 lines |
| Code changes needed per OW upgrade | Manual diff/merge of 3 forked files | None (plugin API is stable) |
| Knowledge Base management | Custom router + bookmarklet | Native OW UI + existing CLI tool |
| Chat metrics | Embedded in forked main.py | OpenWebUI Filter Function (official plugin) |
| Grading data access | Direct SQLite queries (fragile) | Public REST API (stable) |

## Architecture: Before and After

**Current state** -- Custom code is tightly coupled to OpenWebUI internals:

```
OpenWebUI v0.6.41 (pinned)
  ├── main.py ← REPLACED by our fork (2,518 lines)
  ├── config.py ← REPLACED by our fork (3,876 lines)
  ├── observability.py ← REPLACED by our file (174 lines)
  └── routers/
      └── custom_pdf_router.py ← INJECTED (imports 5 internal modules)
```

**Target state** -- Zero modifications to OpenWebUI source:

```
OpenWebUI (any version, unmodified)
  ├── Filter Function installed via admin UI (chat metrics)
  ├── Native Knowledge Base UI (file uploads)
  └── Public REST API
      ├── → tools/kb_sync (programmatic KB management)
      └── → grading_feature (chat extraction for professor dashboard)

Prometheus Pushgateway (new container)
  └── receives metrics from Filter Function → Prometheus → Grafana
```
