---
name: 2027-knowledge-base-setup
description: The 2027 knowledge bases and models exist (private, empty) and copy the 2026 layout; the instance-wide RAG settings were changed on 2026-11-26 for them, and the DB config overrides env.
metadata:
  type: project
---

On 2026-11-26 the 2027 versions of the Beckham role-play were created on qasap-vm01 (Open WebUI
v0.13.4) through the public REST API, copying the 2026 layout:

| 2026 | 2027 (new) |
|---|---|
| KB "2026 personal data" `c60186c9-…` (Beckham bios PDF, FIN602 Client Data PDF) | KB "2027 personal data" `2d9cfa2a-739d-4b2e-a83c-d81aa852a1b5` — empty |
| KB "2026 Employee Benefits" `4e78577c-…` (WI ETF + Rockwell benefits, cleaned .md) | KB "2027 Employee Benefits" `cfaba129-260c-4653-9428-42e1299de115` — empty |
| model `2026` (gpt-5.5) | model `2027` — same system prompt and capabilities |
| model `2026-audio-21-mini` (gpt-realtime-2.1-mini) | model `2027-audio-21-mini` |

The 2027 items have **no access grants (admin-only)** until the files are uploaded; the 2026
ones are readable by all users. Same Qdrant, collection `open-webui_knowledge`; each KB is a
`tenant_id`, so 2027 is kept apart from 2026 automatically.

**System prompt (2026-11-26).** Both 2027 models carry the 2026 prompt plus a block between
`=== CLIENT FACTS (BEGIN) ===` and `=== CLIENT FACTS (END) ===`, inserted before `OVERALL GOAL`: the
2026 Client Data sheet, pages 1–14, re-extracted cleanly from the PDF (the job descriptions are
left to the KB). It is a **placeholder**. Once the 2027 knowledge base is built, Mark wants to
optimize what goes in it (swap in 2027 data, decide what belongs in the prompt and what in the KB). The
first data-use rule was changed to name the facts block as a source alongside the knowledge collection.

**Instance-wide RAG settings changed the same day** (they also apply to the 2026 models):
top_k 2→5, chunk_size 400→800, chunk_overlap 100→150, hybrid search off→on, top_k_reranker
3→5 (reranker `BAAI/bge-reranker-v2-m3`, relevance threshold 0.30 unchanged).

**Why:** Mark asked for "outstanding accuracy". The documents are mostly labelled numbers and
plan rules, which 400-character chunks split apart, and hybrid search catches exact terms
(WRS, HSA, names, amounts). The 2026 chunks were indexed in Feb 2026 at about 1,000 characters,
before the 2026-10-30 change to 400, so they were never built with the 400 setting; changing
chunk size only affects files indexed afterwards.

**How to apply:**
- Values saved in the `config` table override the env file (`RAG_TOP_K=5` in env was being
  ignored while the DB said 2). Change RAG settings in Admin → Settings → Documents or
  `POST /api/v1/retrieval/config/update`, then verify them in the DB.
- Next steps: upload the 2027 files; consider putting the small Client Data file's facts
  directly in the 2027 system prompt; build a 30–40 question answer key and measure before
  going live; time the audio model with the reranker on; then grant read access to all users.
