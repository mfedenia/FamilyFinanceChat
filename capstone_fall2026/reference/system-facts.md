# System Facts and Q&A Prep

Verified facts behind the presentation, with where each one comes from. Use this to answer
questions without guessing, and to re-check the numbers before presenting again — several of
them will have moved.

**Checked:** August 2026. **Re-check before reuse:** the version numbers and all pricing.

---

## 1. What is actually deployed

| Fact | Value | Source in repo |
|---|---|---|
| Open WebUI version | `v0.8.12` | `Dockerfile`, `docker-compose.yml` |
| Dockerfile size | one `FROM` line | `Dockerfile` |
| Container count (production) | 8 | `docker-compose.yml` |
| LLM | `gpt-4o-mini` via OpenAI API | `.env`, `ARCHITECTURE.md` |
| Embeddings | `text-embedding-3-small` | `ARCHITECTURE.md` |
| Vector store (production) | Qdrant, `k=5`, relevance threshold `0.30` | `ARCHITECTURE.md` |
| Reranker | configured but **no model set** | `RAG_TOP_K_RERANKER=3`, model empty |
| Open WebUI resource limit | 2 CPU / 10 GB RAM | `docker-compose.yml` |
| Prometheus retention | 30 days | `docker-compose.yml` |
| Grafana version | `9.0.0`, default `admin/admin` | `docker-compose.yml` |
| Unpinned images | `qdrant:latest`, `cadvisor:latest` | `docker-compose.yml` |
| Uploads | GCS bucket mounted at `/mnt/gcs/fin602` | `docker-compose.yml` |
| Grading dashboard | FastAPI + React/Vite, **runs locally, not containerised** | `grading_feature/`, `SETUP.md` §8 |

**Compliance check performed for this deck:** no compose file in the repository mounts a host
path into `/app/backend/open_webui/**`. `docker-compose.test.yml` uses named volumes only. The
active stack is fork-free.

---

## 2. Scoring pipeline

- 7 dimensions, 0–2 each, total 0–14, normalised to 0–100: relevance, politeness, on-topic,
  neutrality, non-imperative, clarity (optional), privacy-minimisation (optional).
- ABI: 12 sub-dimensions derived from the rubric, weighted into Ability / Benevolence /
  Integrity (each 0–1); ABI Total is their average.
- Single implementation: `grading_feature/backend/scoring_service.py` (Python). The duplicate
  JavaScript implementation in `scoring_page/` was deleted on 2026-08-19.
- Question extraction is **heuristic** — punctuation and keyword based. Accuracy has never been
  measured.

---

## 3. The fork that was removed

| File | Lines |
|---|---:|
| `main.py` | 2,518 |
| `config.py` | 3,876 |
| `observability.py` | 174 |
| `custom_pdf_router.py` | injected into the vendor's `routers/` package |

Total ≈ 6,500 lines. Delivery mechanism for the custom UI was a browser bookmarklet:
`javascript:fetch('/api/v1/custom/inject-script?_='+Date.now()).then(r=>r.text()).then(eval)`.

Source: `legacy/openwebui_upgrade_decoupling.plan.md.done`,
`legacy/custom-code-vendor-fork/README.md`.

**The removal plan has one item still marked `pending`:** the end-to-end post-upgrade smoke
test (KB upload + chat metrics + grading, together). Components were tested individually. That
is the gap CI closes.

---

## 4. Upstream Open WebUI — what we are behind on

| Release | Change | Consequence |
|---|---|---|
| v0.8.0 | Skills, Analytics, message queuing | Skills are the professor-editable home for course frameworks |
| v0.8.9 | `/ready` endpoint; OpenTelemetry system metrics | enables real uptime checks and metrics without a plugin |
| **v0.9.0** | backend data layer fully **async**; SQLAlchemy async sessions | **our Filter Function must be migrated** |
| **v0.10.0** | `config` table → `config_old` + per-key table | **one-way migration**; an old instance fails against a migrated DB |
| **v0.11.0** | additive columns/indexes; `psycopg` v3; large UI reorganisation; sub-agents; shared folders; LDAP group sync | low migration risk, high documentation churn |

Upstream guidance: back up before releases with migrations; stop all replicas, let one instance
migrate, then bring the fleet back.

Sources: [plugin migration to 0.9.0](https://docs.openwebui.com/features/extensibility/plugin/migration/to-0.9.0/),
[updating](https://docs.openwebui.com/getting-started/updating/),
[v0.11.0 notes](https://openwebui.com/blog/v0-11-0-the-interface-reorganized).

---

## 5. Extension surfaces (why the avatar is a separate app)

- **Functions** (Filter / Pipe / Action) run *inside* Open WebUI's process; they cannot install
  new Python packages.
- **Pipelines** run on a *separate server* and are OpenAI-compatible, so they can carry any
  dependency, including GPU work.
- **Chat completions with RAG** from an external client:
  `POST /api/chat/completions` with `files: [{"type": "collection", "id": "<kb-id>"}]`,
  Bearer-token authenticated with an API key from Settings → Account.
- **There is no documented API for installing Functions** — which is exactly why the manual
  filter-install step cannot simply be scripted, and why moving to native OTel is the better fix.

Sources: [Tools & Functions](https://docs.openwebui.com/features/extensibility/plugin/),
[Pipelines](https://docs.openwebui.com/features/extensibility/pipelines/),
[API endpoints](https://docs.openwebui.com/reference/api-endpoints/).

---

## 6. Voice already in the box

Open WebUI ships hands-free **call mode**: the student speaks, the model answers, the reply is
read back sentence-by-sentence as it streams, and the microphone re-arms. Playback speed is
adjustable 0.5×–2×.

- **STT engines:** browser Web Speech API, Whisper, Mistral Voxtral.
- **TTS engines:** any OpenAI-compatible `/audio/speech` endpoint, plus ElevenLabs, Kokoro, and
  Chatterbox (voice cloning) — all documented integrations.
- Configured in Settings → Admin → Experience → Audio, or by environment variable.

Source: [Speech-to-Text & Text-to-Speech](https://docs.openwebui.com/features/chat-conversations/audio/).

**This is why B0 costs almost nothing to build.**

---

## 7. Avatar providers and orchestration

| Item | What we know (Aug 2026) |
|---|---|
| HeyGen **LiveAvatar** | real-time WebRTC avatar; credit-based; roughly \$0.10/min "lite" and \$0.20/min "full"; a separate Avatar Realtime tier is quoted per-second at 720p |
| Tavus **Phoenix-4** | claims sub-600 ms end-to-end and full-duplex (listens while speaking) |
| Simli, Anam, bitHuman, Beyond Presence, D-ID, others | LiveKit's avatar plugin ecosystem lists a dozen-plus providers |
| **LiveKit Agents** | orchestration framework; avatar renderer joins the room as its own participant and publishes synced A/V |
| **Pipecat** | orchestration framework; integrates Simli, HeyGen, Tavus, and fully local models |
| Open-source self-host | Whisper + open TTS (Kokoro / Chatterbox) + open lip-sync (MuseTalk / LivePortrait class) |

**Treat every latency and price figure above as a claim to be measured, not a fact.**

Sources: [HeyGen LiveAvatar](https://help.heygen.com/en/articles/12758516-introducing-liveavatar),
[HeyGen API pricing](https://www.heygen.com/api-pricing),
[Tavus + Pipecat](https://www.tavus.io/post/open-sourcing-ai-innovation-building-real-time-ai-interactions-with-pipecat-and-tavus),
[avatar/lip-sync in a video call](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/real-time-avatar-lipsync-in-call).

---

## 8. Cost arithmetic used in the deck

```
40 students x 4 sessions x 15 min = 2,400 min = 40 hours / semester
40 h x $0.10/min = $240      (lite tier)
40 h x $0.20/min = $480      (full tier)
+ STT $10-30, TTS $20-60, LLM $15-40
=> roughly $285-610 per semester
```

Self-hosting comparison: an L4-class GPU VM is on the order of \$0.70/hour on demand, i.e.
roughly \$500/month if left running. **At 40 hours a semester, metered streaming is cheaper than
a dedicated GPU.** Self-hosting wins on privacy and at much larger scale.

---

## 9. Likely questions, and honest answers

**"Why not just fork it again — it's faster in the short term?"**
Because we have the receipts. The last fork cost 6,500 lines of maintenance and froze the
platform across four minor releases. The plugin surfaces cover everything we actually need.

**"Can't we put the avatar inside the chat window?"**
Not without forking the Svelte frontend, which is the exact failure mode we removed. The
companion app gets the same result and can be deleted in one step if the track is killed.

**"Will the avatar be good enough to feel real?"**
Unknown — that is the research question. Our threshold is a median first-response under
1.2 seconds with working interruption. We will publish what we measure, including if it fails.

**"What if it costs too much?"**
Hard caps, quotas, and a spend alarm go in with the first prototype. And the fallback — spoken
practice with no avatar — costs essentially nothing and ships in week 5.

**"Is student voice data safe?"**
That is settled in Week 2 before any recording: written consent, an equivalent non-recorded
path, no real likenesses, documented retention, and a review of each vendor's data-processing
terms as part of the bake-off.

**"Is the upgrade risky?"**
Yes, specifically the v0.10.0 config migration, which is one-way. That is why we rehearse it on
a restored snapshot, upgrade one minor version at a time, and write the rollback before we start.

**"What if you don't finish the avatar?"**
Then B0 shipped, the platform is upgraded, hosted, monitored, and CI-guarded, and the next team
inherits a measured answer about avatars instead of an opinion. That is a good semester.
