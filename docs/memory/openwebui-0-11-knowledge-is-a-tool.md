---
name: openwebui-0-11-knowledge-is-a-tool
description: Open WebUI 0.11.x stops searching a model preset's attached knowledge on every turn and gives the model a search tool instead, unless the preset uses function_calling legacy — students' client chats change behaviour on the 0.8 → 0.11 upgrade.
metadata:
  type: project
---

Verified 2026-11-22 in Open WebUI source, `backend/open_webui/utils/middleware.py`:

- **v0.8.12 (production):** when a model preset has attached knowledge
  (`meta.knowledge`) and `function_calling` is not `native`, the middleware searches it
  automatically on **every** user message and injects the chunks (FFC: Qdrant,
  `text-embedding-3-small`, `RAG_TOP_K=5`).
- **v0.11.4 (pinned in `docker-compose.yml`):** the automatic search only runs when the preset's
  `function_calling` is `legacy`. Otherwise the model gets the knowledge as a tool
  (`query_knowledge_files`, advertised via `<knowledge …/>` tags) and decides itself whether to
  search.

**Why:** the course's client models answer from the Beckham data sheet. After the A13 upgrade,
a client may answer from memory without searching, so answers change silently: more invented
figures, fewer facts from the case. It also changes what B12 (per-turn retrieval vs. persona
preloading) is measuring. `liveavatar/` searches every turn on purpose, like production today,
so after the upgrade spoken and typed sessions would differ.

**How to apply:** on the upgrade, decide per client preset. Either set Advanced Params →
Function Calling to `legacy` to keep per-turn search, or accept model-decided search and
re-check answer quality. Include this in the A13 acceptance checks.
