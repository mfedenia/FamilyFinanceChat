# Upstream Compatibility Policy

**Project:** FamilyFinanceChat (FIN 602)
**Status:** Binding engineering constraint — not a guideline
**Owner:** Platform lead
**Applies to:** every line of code, every container, every configuration change

---

## 1. The rule

> **Open WebUI is a dependency, not a codebase we own.**
> We must be able to run `docker compose pull && docker compose up -d` against any
> published Open WebUI release and have the platform work — with no merge, no patch,
> and no re-fork.

Every feature must be delivered through a surface that Open WebUI publishes,
documents, and versions. If a feature cannot be built that way, the feature is
redesigned or dropped. The core is never modified.

---

## 2. Why this rule exists

This project already paid for the lesson once.

Through Spring 2026 the deployment worked by bind-mounting three replacement files
over Open WebUI's own source inside the container:

| Forked file | Lines | What it was really for |
|---|---:|---|
| `main.py` | 2,518 | mount one custom router + inject a metrics middleware |
| `config.py` | 3,876 | only existed to satisfy imports in the forked `main.py` |
| `observability.py` | 174 | define Prometheus metrics |
| `custom_pdf_router.py` | — | injected into Open WebUI's `routers/` package |

Roughly **6,500 lines of vendored fork to obtain two features.** The cost:

- The platform was frozen on **v0.6.41** while upstream shipped v0.7, v0.8, and beyond.
- Every upgrade required a manual three-way diff of someone else's application code.
- Unexplained startup and environment failures traced back to fork drift.
- A browser bookmarklet (`javascript:fetch('/api/v1/custom/inject-script')…eval`) was
  required for users to reach a custom feature — an unmaintainable, and frankly
  unsafe, delivery mechanism.

The Spring 2026 team removed all of it. The `Dockerfile` is now a single `FROM` line.
**This policy exists so that no future team quietly re-creates the fork** — which always
begins the same way: *"it's just one file, we'll mount it read-only."*

---

## 3. Allowed extension surfaces

These are supported, documented, versioned Open WebUI integration points. Build here.

| # | Surface | Use it for | Notes |
|---|---|---|---|
| A1 | **Environment variables / Admin settings** | model routing, RAG parameters, STT/TTS engines, auth, OTel export | first choice for anything configurable |
| A2 | **Filter Functions** (`inlet` / `outlet` / `stream`) | observing or lightly rewriting a request or response in flight | runs inside Open WebUI's process — keep it small and non-blocking |
| A3 | **Pipe Functions** | exposing a custom "model" that is really our own logic | appears in the model picker like any other model |
| A4 | **Action Functions** | per-message buttons that launch something (e.g. "Start voice session") | the sanctioned way to add a UI affordance |
| A5 | **Pipelines server** | anything needing its own dependencies, GPU, or heavy compute | separate container, OpenAI-compatible; Open WebUI just points at it |
| A6 | **Tools / OpenAPI servers / MCP servers** | letting the model call our services | tool schemas are ours; the host is untouched |
| A7 | **Public REST API** (`/api/chat/completions`, `/api/v1/*`) | grading extraction, KB sync, external clients, the avatar app | authenticate with an API key; treat as the only supported read path |
| A8 | **Sidecar containers on `ai-net`** | metrics exporters, our own web apps, avatar services | compose them alongside; never inside |
| A9 | **Reverse-proxy composition** (Nginx/Caddy in front) | TLS, auth, path routing, serving companion apps on one hostname | composition, not modification |
| A10 | **Knowledge Bases and Skills** | course documents and pedagogical frameworks | professor-editable from the UI, survives upgrades |

**Design heuristic:** if the feature needs to see or change something *inside* an Open
WebUI request, use A2–A4. If it needs its own runtime, use A5/A8 and talk to Open WebUI
over A7.

---

## 4. Forbidden — strip on sight, never reintroduce

| # | Forbidden | Why |
|---|---|---|
| F1 | Bind-mounting or `COPY`-ing any file into `/app/backend/open_webui/**` | this *is* the fork; it breaks silently on every release |
| F2 | Adding routers, models, or modules to the `open_webui` Python package | same as F1, plus import coupling to internals |
| F3 | Building a custom Open WebUI image from modified source (frontend or backend) | forces us to track upstream by hand forever |
| F4 | JavaScript injection into the Svelte frontend (bookmarklets, injected `<script>`) | no stable contract, no CSP story, breaks on any UI refactor |
| F5 | Direct SQL against Open WebUI's SQLite/Postgres schema | the schema is private and *did* change under us (see v0.10.0) |
| F6 | Monkeypatching Open WebUI internals from inside a Function | a fork wearing a plugin costume |
| F7 | Depending on undocumented internal endpoints or response fields | fine until it isn't, and the failure is silent |
| F8 | Pinning to an old release to avoid doing the migration work | this is exactly how v0.6.41 happened |

### The one-sentence test

> *If upstream cut a release tomorrow and we ran it unchanged, would this still work?*

If the answer requires "well, as long as they don't change…", it is forbidden.

---

## 5. Audit of the current repository (August 2026)

**Headline: the active production stack is compliant.** `docker-compose.yml` mounts only
data directories (`/opt/openwebui/data`, `/mnt/gcs/fin602`) — no source paths.
`docker-compose.test.yml` uses named volumes only. The `Dockerfile` is one `FROM` line.

The remaining work is peripheral coupling, plus loaded guns left in the tree.

| ID | Finding | Location | Risk | Required action |
|---|---|---|---|---|
| C-1 | Full vendor-fork source retained, with a README that instructs the reader to bind-mount `main.py` over Open WebUI | `legacy/custom-code-vendor-fork/` | **High** — it is a working recipe for re-forking | **Delete from the working tree.** It is preserved in git history; replace with a one-page tombstone (`legacy/README.md`) naming what it was, why it went, and the commit to recover it from |
| C-2 | `--legacy` flag advertised as a "direct SQLite extraction" escape hatch; the body is a `TODO` stub | `grading_feature/backend/extract_chats.py:512–531` | **Medium** — an advertised path back to F5 | Delete the flag and the stub. The REST API is the only supported extraction path |
| C-3 | Custom `build:` context for `open-webui` in compose | `docker-compose.yml` | Low — but it is the hook a fork hangs on | Replace `build: .` with `image: ghcr.io/open-webui/open-webui:<pinned tag>`; delete the `Dockerfile` |
| C-4 | Chat Metrics Filter uses **sync** `def inlet/outlet` with a blocking `urllib.request` call, plus a mutable `self._state` dict shared across concurrent users | `monitoring/chat_metrics_filter.py` | **High** — v0.9.0 made the data layer async; blocking I/O on the event loop degrades every user, and the shared dict leaks state between concurrent sessions | Rewrite per the upstream 0.9.0 plugin migration guide: `async def`, non-blocking HTTP, state keyed by `(chat_id, message_id)` — **or** retire it in favour of native OTel export (preferred; see C-5) |
| C-5 | Chat metrics depend on a Filter that a human must paste into the Admin UI after every deployment; there is no documented API to install Functions | `monitoring/`, `SETUP.md` §5 | **Medium** — an undeployable step, and silently absent metrics when skipped | Move measurement off the plugin surface entirely: enable Open WebUI's native OpenTelemetry export (v0.8.9+) into an OTel Collector → Prometheus. Removes both the manual step and the in-process filter |
| C-6 | `qdrant:latest` and `gcr.io/cadvisor/cadvisor:latest` unpinned | `docker-compose.yml` | Medium — a `pull` can change the vector store underneath us | Pin every image to an explicit tag, ideally a digest |
| C-7 | Grafana `9.0.0` (years old), Prometheus `2.51.2`, default `admin/admin` credentials in compose | `docker-compose.yml` | **High** (credentials), Medium (age) | Upgrade both; remove the default password; put Grafana behind the proxy |
| C-8 | A second, disconnected RAG stack (ChromaDB + BGE-M3) duplicating the native Qdrant RAG, sharing no code with it | `rag_bio_project/` | Medium — two retrieval systems, one of which nothing in production uses | Decide explicitly: retire it, or promote it behind a **Pipelines** server (A5) so it is reachable without touching the core. Do not leave it ambiguous |
| C-9 | Scoring rubric and ABI formulas implemented twice — JavaScript and Python — kept in sync by hand | `scoring_page/backend/server.js`, `grading_feature/backend/scoring_service.py` | Medium — the two will diverge and grades will disagree | Declare the Python implementation canonical; retire the Node prototype or make it call the Python service |
| C-10 | A provider API key was previously committed. It has been scrubbed from the tree, but **provider-side rotation is still outstanding** | history under `scoring_page/` | **High** | Rotate the key at the provider. Add `gitleaks` to CI so it cannot recur |
| C-11 | Stale analysis and one-off artifacts at the repo root | `archive/`, `metrics_snapshot.json` (1.6 MB) | Low | Move under `archive/` with a dated note, or delete. Keep the root legible |

---

## 6. How to add a feature under this policy

1. **Name the surface (A1–A10) the feature uses.** If you cannot name one, stop and redesign.
2. **Justify it in the pull request.** "Which surface, and what happens on the next upstream
   release?" is a required review question.
3. **Prove it survives an upgrade.** The feature must still work after the test stack is
   bumped to the newest Open WebUI tag. That is the acceptance test, not a nice-to-have.
4. **Design out manual steps.** If a feature needs a human to click something in the Admin
   UI after every deployment, that is a defect (see C-5), not a README entry.

### Enforced in CI

| Check | Fails the build when |
|---|---|
| Fork guard | any compose file mounts a host path into `/app/backend/open_webui/` or `/app/build/` |
| Dockerfile guard | the Open WebUI image is built from anything but an official base, with no `COPY` into the package |
| Schema guard | Python sources open SQLite or a SQLAlchemy session against the Open WebUI data directory |
| Secret scan | `gitleaks` finds a credential |
| Pin guard | any image reference ends in `:latest` |

That is roughly forty lines of grep plus one GitHub Action. Build it in Week 2 — it is what
makes this policy real instead of aspirational.

---

## 7. Applying the policy to the avatar track

The avatar work (see `AVATAR_TRACK.md`) is the first serious test of this policy, because
the obvious implementation is forbidden: **there is no supported way to embed a live video
avatar inside Open WebUI's chat page.** Doing it means editing the Svelte frontend, which
is F3 and F4 at once.

The policy therefore *dictates the architecture*. The avatar is a **companion web
application** in its own container (A8), served under the same hostname by the reverse
proxy (A9), which uses Open WebUI's REST API (A7) for the model, the knowledge base, and
the transcript of record. Open WebUI is never modified — it becomes the brain and the
system of record behind a second front end.

This is the better design regardless: it can be deployed, rolled back, and load-tested
independently, and it survives every upstream release by construction.
