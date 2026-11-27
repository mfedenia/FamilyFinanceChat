---
name: owui-013-knowledge-is-a-tool-and-reranker-is-slow
description: In Open WebUI 0.13 a model's attached knowledge is a tool the model may or may not call (not automatic RAG); the local CPU reranker made searches take ~30-160 s on qasap-vm01, so reranking now runs on hosted Cohere (~8 s answers).
metadata:
  type: project
---

Found by the 2027 answer-key test on 2026-11-26 (41 questions, `2027` model, results in
`artifacts/reports/kbtest-2027.md` on Mark's machine).

**Knowledge is a tool, not automatic retrieval.** In v0.13.4 the default `function_calling`
is `native`. Then a model's attached knowledge bases are *not* searched on every turn: the model
gets builtin tools (`list_attached_knowledge`, `query_knowledge_files`, `grep_knowledge_files`,
`view_knowledge_file`) and decides itself whether to use them. The builtin tools are only
injected for browser sessions (a socket `session_id`); a bare `POST /api/chat/completions` gets
the system prompt and no knowledge at all. Only `function_calling: legacy` (per model in
Advanced Params, or per request in `params`) gives the old inject-on-every-turn RAG.
In the test, the model skipped search on 15 of 41 questions, and that was fine: those answers came
from CLIENT FACTS in the prompt.

**The reranker is the latency.** Hybrid search was turned on 2026-11-26 with reranker
`BAAI/bge-reranker-v2-m3`, which runs on the VM's 2 vCPUs. Answers with no search took 2–5 s;
every answer that searched took 33–160 s (median ~85 s), and 4 of 41 exceeded 5 minutes. Legacy
mode, which searches both knowledge bases on every turn, was worse: 22 of 41 exceeded 5 minutes.
The log shows about 40 s of reranking per collection per query. The RAG settings are
instance-wide, so the `2026` models are affected too.

**Fix applied 2026-11-27: reranking moved to Cohere.** Admin Settings → Documents: Reranking Engine
`external`, model `rerank-v3.5`, URL `https://api.cohere.com/v2/rerank`, timeout 20 s; hybrid search
still on. The key is Mark's production Cohere key, stored only in Open WebUI's config (set by
`artifacts/scripts/rr1.sh`, which reads it from his clipboard). Open WebUI's external reranker speaks
the Cohere/Jina `/rerank` format (`results[].index`, `relevance_score`); Voyage's format does not fit.
Rerun of the test: 36 pass, 4 partial, 1 fail, 0 timeouts; median 8 s, searches 7–23 s, whole run
4 minutes (was over an hour). If Cohere is down, reranking returns nothing and search degrades
rather than hangs. Remaining misses are the model *skipping* search on back-story not in the prompt
(Darnell Whitaker, the $2,400 loss, the 529's history): a prompt nudge to search before answering
life-history questions would likely fix them.

**Accuracy is good when an answer arrives (run 2, local reranker).** In native mode 35 of 37 answered questions passed.
The misses were one back-story search that returned nothing (Tony's cashed-out 401(k)) and one
partial answer (the $2,400 gambling loss).

**How to test:** `artifacts/data/kbtest2027/run_kbtest.py MODEL native|legacy` runs inside the
container. It uses a fresh socket and a `temporary:<sid>:<uuid>` chat per question, so nothing is
saved to chat history, and grades with the model's base model. Long-lived sockets go stale after
a few minutes ("Temporary chat expired"). Call `GET /api/models` first or chat returns
"Model not found".

Related: [[2027-knowledge-base-setup]].
