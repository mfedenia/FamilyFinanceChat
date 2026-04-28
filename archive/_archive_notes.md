# Archive Notes

Files moved here during the Spring 2026 repo cleanup and handoff. Nothing here is active code.

| File/Directory | Original Location | Why Archived |
|---|---|---|
| `ExtractChatByJson.py` | `grading_feature/backend/` | Interactive script using the old SQLite export JSON format (`chat.history.messages` dict structure). Superseded by `extract_chats.py` which uses the OpenWebUI REST API. |
| `analyze_openwebui_chats.py` | `grading_feature/backend/` | Multi-format interactive chat analyzer for old export format. Superseded by the grading dashboard. |
| `abi_trust_pipeline/` | `rag_bio_project/abi_trust_pipeline/` | Research prototype of the ABI trust scoring pipeline. The production implementation lives in `grading_feature/backend/scoring_service.py`. Kept here as lineage reference. |
| `demo8_upgrade_analysis.md` | `research_docs/demo8/README.md` | The R&D analysis document that drove the OpenWebUI decoupling plan (vendor fork → Filter Function + public API). Problem is solved; see `legacy/openwebui_upgrade_decoupling.plan.md.done` for the migration record. |
