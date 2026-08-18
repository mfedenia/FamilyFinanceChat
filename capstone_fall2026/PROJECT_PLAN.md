# FamilyFinanceChat — Fall 2026 Project Plan

**Audience:** the incoming development team, treated as professional engineers
**Course:** FIN 602 capstone / applied engineering team
**Term:** Fall 2026 (Week 1 begins 24 August 2026; final presentation week of 7 December 2026)
**Prepared:** August 2026
**Companion documents:** [`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) · [`AVATAR_TRACK.md`](AVATAR_TRACK.md) · [`slides/`](slides/) · [`script/`](script/)

---

## 1. Executive summary

FamilyFinanceChat is a working, in-production teaching platform. FIN 602 students practise
client-facing financial advising against an AI role-player loaded with realistic family
scenarios; instructors review the transcripts through a grading dashboard that scores
question quality across seven dimensions and an Ability–Benevolence–Integrity trust rubric.

The previous team's central achievement was **architectural, not featural**: they removed a
6,500-line fork of Open WebUI's internals and rebuilt every customisation on supported
plugin and API surfaces. That work is what makes this semester possible.

This semester has two workstreams:

- **Workstream A — Harden and modernise.** Close the version gap to current Open WebUI,
  make deployment repeatable, put the grading dashboard in front of professors without an
  SSH session, and permanently remove the remaining paths back to a fork.
- **Workstream B — Embodied advising (the new bet).** Investigate, prototype, and
  evaluate turning the text chatbot into a **spoken, face-to-face experience**: a visible
  avatar client that hears a student's question and answers out loud. This is a research
  and prototyping track with a defined kill switch, not a promised feature.

Both workstreams are governed by one non-negotiable constraint: **we never modify Open
WebUI's core.** Anything that cannot be built on a published extension surface gets
redesigned or dropped.

---

## 2. Where the system stands today

### 2.1 What runs

Eight containers on a single GCP VM, plus a grading tool that currently runs on a
developer's laptop.

| Component | Image / stack | Role |
|---|---|---|
| `open-webui` | `ghcr.io/open-webui/open-webui:v0.8.12` | chat UI, LLM routing, auth, chat history |
| `qdrant` | `qdrant/qdrant:latest` | vector store for course-document RAG |
| `redis` | `valkey/valkey:8.0.1-alpine` | WebSocket session state |
| `prometheus` | `prom/prometheus:v2.51.2` | metrics, 30-day retention |
| `pushgateway` | `prom/pushgateway:v1.8.0` | receives chat metrics from the Filter Function |
| `cadvisor` | `cadvisor:latest` | container CPU / memory |
| `metrics-exporter` | custom FastAPI | probes Open WebUI `/health` |
| `grafana` | `grafana/grafana:9.0.0` | dashboards |
| grading dashboard | FastAPI + React/Vite | **not containerised — runs locally** |

The LLM is `gpt-4o-mini` through the OpenAI API. Retrieval is native Open WebUI RAG over
Qdrant with `text-embedding-3-small`, `k=5`, relevance threshold `0.30`.

### 2.2 What is genuinely good

- **The decoupling holds.** No compose file mounts anything into Open WebUI's source tree.
  The `Dockerfile` is one line. This is the hard part, and it is already done.
- **Grading reads the public REST API**, not the database — it survives schema changes.
- **`tools/kb_sync/`** manages knowledge bases entirely through the public API.
- **The scoring pipeline is real work**: a defensible 7-dimension rubric mapped through 12
  sub-dimensions into an ABI trust model, with a React dashboard on top.

### 2.3 What hurts

| Problem | Consequence |
|---|---|
| Pinned to **v0.8.12** while upstream ships **v0.11.x** | three minor versions of security fixes, features, and — importantly — *migrations* deferred |
| Metrics Filter must be **hand-pasted into the Admin UI after every deployment** | metrics silently vanish; deployment is not reproducible |
| That Filter is **synchronous with blocking network I/O and shared mutable state** | on the v0.9.0+ async core this degrades every concurrent user |
| **Grading dashboard is not hosted** | a professor needs SSH and a terminal to grade — the single biggest adoption barrier |
| **No CI, no automated smoke test** | past upgrades broke things silently; nobody found out until a student did |
| No alerting on `/ready` | an outage was discovered passively, from students |
| Two RAG stacks, two scoring implementations | ambiguity about which is canonical; grades can disagree with themselves |
| Default `admin/admin` on Grafana; an unrotated leaked provider key | straightforward security debt |

None of this is exotic. It is the ordinary gap between "a system that works" and "a system
that can be handed to the next team."

---

## 3. Goals and non-goals

### 3.1 Goals

| # | Goal | Measured by |
|---|---|---|
| G1 | Run current Open WebUI and stay current | production on v0.11.x (or newer at the time); documented, rehearsed upgrade runbook |
| G2 | Deployment is reproducible from a clean VM with no manual UI steps | a scripted deploy passes a smoke suite end to end |
| G3 | Professors grade from a URL | hosted dashboard, authenticated, no SSH |
| G4 | Failures are noticed by us, not by students | `/ready` uptime check + alert; Grafana alert on memory and error rate |
| G5 | Every remaining path back to a fork is removed | items C-1…C-3 closed; CI fork guard blocks reintroduction |
| G6 | An evidence-based answer on embodied avatars | working prototype + evaluation report + go/no-go recommendation |

### 3.2 Non-goals for this semester

Stating these protects the team's time.

- **Not** replacing Open WebUI, and **not** writing our own chat frontend.
- **Not** full multi-tenant SaaS. Design decisions should not *preclude* it; building it is a
  later project.
- **Not** real-time adaptive difficulty — it needs a live scoring pipeline that does not exist.
- **Not** shipping an avatar to all students by December. The deliverable is a validated
  prototype and a recommendation, which may legitimately be "not yet, and here is why."

---

## 4. Team structure

Five roles. On a smaller team, one person holds two; the roles still exist, and so does
the accountability.

| Role | Owns | Primary workstream |
|---|---|---|
| **Platform / DevOps lead** | compose stack, upgrade path, CI/CD, proxy, secrets, backups | A |
| **Backend / integrations** | grading service, extraction, scoring consolidation, Pipelines | A |
| **Realtime / avatar engineer** | STT→LLM→TTS→avatar pipeline, latency budget, provider SDKs | B |
| **Frontend / UX** | hosted dashboard, avatar companion app UI, session flow | A + B |
| **Data / assessment** | evaluation design, metrics, transcript merge, learning-outcome study | A + B |

**Cross-cutting duty:** every member reviews pull requests against the compatibility
policy. The review question is fixed: *which extension surface, and what happens on the
next upstream release?*

---

## 5. Workstream A — Harden and modernise

Six epics. Each has a definition of done that someone else can verify without asking the
author.

---

### Epic A1 — Close the version gap (v0.8.12 → v0.11.x)

**Why this is first:** every other decision compounds on it, and the migrations are
one-way. The longer it waits, the more expensive it becomes — this is precisely the
dynamic that produced the v0.6.41 freeze.

**What upstream changed, and what it means for us:**

| Release | Change | Impact here |
|---|---|---|
| **v0.9.0** | the backend data layer became **async top to bottom**; model methods are `async def`, SQLAlchemy runs in async mode | **Our Filter Function must be migrated.** Sync plugin code with blocking I/O is now actively harmful. Follow the upstream 0.9.0 plugin migration guide |
| **v0.10.0** | the `config` table was renamed to `config_old` and replaced by a per-key table | **One-way migration.** An older instance pointed at a migrated database fails immediately. Back up before, and understand there is no rolling back to v0.9 without restoring the backup |
| **v0.11.0** | additive columns and indexes only; async driver moved to `psycopg` v3; substantial UI reorganisation (sidebar, settings, model picker); sub-agents; shared-folder collaboration; LDAP group sync | Low migration risk, **high documentation risk** — every screenshot and click-path in `SETUP.md` and the professor-facing guides is now wrong |

**Approach — staged, sequential, rehearsed:**

1. Snapshot production: `/opt/openwebui/data`, `/opt/qdrant/storage`, Grafana state. Verify
   the snapshot restores into the test stack. *A backup that has never been restored is not
   a backup.*
2. Restore the production snapshot into the **test stack** and upgrade there first.
3. Move **one minor version at a time** — 0.8.12 → 0.9.x → 0.10.x → 0.11.x. Migrations run
   sequentially; a single jump conflates failures and makes them unattributable.
4. Run the smoke suite (A2) at each step. Record what broke, in the runbook, as you go.
5. Evaluate **SQLite → PostgreSQL** during this epic. Postgres is the supported path for
   multi-replica operation and makes migrations and backups far less frightening. Decide
   explicitly and write down the decision either way.
6. Only then schedule the production window, with the rollback (restore snapshot, redeploy
   previous tag) written out *before* you start.
7. Re-shoot every screenshot and click-path in the docs for the v0.11 UI.

**Done when:** production runs the current release; the runbook is written and was actually
followed; the smoke suite passes; docs match what a user sees; the rollback path is proven,
not theorised.

---

### Epic A2 — CI/CD and the fork guard

**Pipeline, in order of value per hour spent:**

1. **Smoke test on every push** — `docker compose up` the test stack, wait for `/health`,
   then `/ready`, create a user via API, send a chat completion, assert a non-empty
   response, upload a document to a knowledge base and assert retrieval returns it.
2. **Fork guard** — the greps in `COMPATIBILITY_POLICY.md` §6. Fails the build on any mount
   into `/app/backend/open_webui/`, any `:latest` image, any direct-SQLite access to Open
   WebUI's data, any `COPY` into the package.
3. **Secret scanning** — `gitleaks` on every push and on history.
4. **Lint and unit tests** — `ruff` + `pytest` for Python, `eslint` for the frontend. Unit
   tests on `scoring_service.py` matter most: it produces numbers that become grades.
5. **Deploy** — on a tag, SSH to the VM, `docker compose pull && up -d`, re-run the smoke
   suite against production, and roll back automatically if it fails.

**Done when:** a red build blocks a merge; a tagged release deploys without anyone typing a
docker command; a deliberately broken commit is caught by CI and you can show the failure.

---

### Epic A3 — Host the grading dashboard

Today a professor must SSH into a GCP VM and run a shell script. That is the difference
between a tool that gets used and one that does not.

**Work:** containerise the FastAPI backend and build the Vite frontend to static assets;
add both to `docker-compose.yml`; put Nginx (or Caddy) in front of the whole stack with TLS
and a real hostname; authenticate — university SSO/OIDC if it can be arranged this term,
HTTP basic auth as the documented fallback; route `/` to Open WebUI, `/grading` to the
dashboard, `/grafana` to Grafana; keep the OpenAI key server-side only.

**Watch for:** the extraction job pulls every chat and scores questions through the OpenAI
API. Make `/refresh` asynchronous with visible progress, and cache scored results — a
professor clicking Refresh should not trigger an unbounded, uninformative wait, or an
unbounded bill.

**Done when:** a professor opens a URL, authenticates, and grades — with no terminal, no
VPN, and no help from a student.

---

### Epic A4 — Observability that survives deployment

The current chat-metrics path depends on a human pasting Python into an admin form after
every deployment. Fix the class of problem, not the instance.

**Work:** enable Open WebUI's native **OpenTelemetry** export (available since v0.8.9) into
an OTel Collector, and scrape the collector with Prometheus. Then either retire the Filter
Function entirely, or — if it still carries a metric OTel does not — rewrite it `async`
with non-blocking HTTP and per-message state keys, per C-4. Add a GCP uptime check against
`/ready` with alerting to a real person. Add Grafana alerts on container memory (a leading
indicator — the 10 GB limit has been approached) and on chat error rate. Upgrade Grafana and
Prometheus; delete the default `admin/admin`. Pin every image.

**Done when:** a fresh deployment produces metrics with zero manual steps, and an induced
outage pages someone within five minutes.

---

### Epic A5 — Consolidate the duplicated logic

Three ambiguities, each of which will eventually produce a wrong answer nobody can explain.

1. **Scoring exists twice** — `scoring_page/backend/server.js` (JavaScript) and
   `grading_feature/backend/scoring_service.py` (Python). Declare Python canonical. Retire
   the Node prototype, or reduce it to a thin client of the Python service. Add unit tests
   that pin the rubric arithmetic and the ABI weights, so a refactor cannot silently move a
   grade.
2. **RAG exists twice** — native Qdrant RAG (what students actually use) and the standalone
   ChromaDB pipeline in `rag_bio_project/`. Decide: retire it, or promote it behind a
   Pipelines server so it is reachable without touching the core. Note the already-fixed
   citation bug lived in the unused stack — dead code with live bugs is a tax.
3. **Question extraction is heuristic** — questions are identified by punctuation and
   keywords. Measure its accuracy against a hand-labelled sample before trusting scores
   derived from it. If accuracy is poor, that is a finding worth reporting, and a small
   model call fixes it.

**Done when:** for scoring and for retrieval there is exactly one implementation each, the
choice is documented, and tests cover the arithmetic that becomes a grade.

---

### Epic A6 — Strip the remaining fork surface

Execute C-1, C-2, C-3, C-10, C-11 from the compatibility audit:

- delete `legacy/custom-code-vendor-fork/`, leaving a tombstone README pointing at the commit;
- delete the `--legacy` SQLite flag and stub from `extract_chats.py`;
- replace `build: .` with a pinned `image:`, and delete the `Dockerfile`;
- rotate the previously-leaked provider key at the provider, and land `gitleaks` in CI;
- move stale artefacts (`archive/`, the 1.6 MB `metrics_snapshot.json`) out of the root.

**Done when:** the CI fork guard passes on a tree that contains no instructions for
re-forking, and the audit table in `COMPATIBILITY_POLICY.md` §5 is fully closed out.

---

## 6. Workstream B — Embodied advising

Full detail in [`AVATAR_TRACK.md`](AVATAR_TRACK.md). Summary for planning purposes:

| Phase | Weeks | Deliverable | Gate |
|---|---|---|---|
| **B0 — Voice, no face** | 3–5 | hands-free spoken practice using Open WebUI's built-in call mode with a persona-appropriate TTS voice | ships regardless; it is the fallback if avatars fail |
| **B1 — Provider bake-off** | 4–7 | two providers integrated behind one interface, measured on latency, cost, and interruption behaviour | pick one, or recommend stopping, with data |
| **B2 — Companion prototype** | 7–12 | avatar app in its own container, using Open WebUI for the model, the knowledge base, and the transcript | five students complete a full advising session |
| **B3 — Evaluate** | 11–15 | comparison of text vs voice vs avatar on question quality and student experience | evidence-backed go/no-go |

**Architectural consequence of the compatibility policy:** the avatar cannot live inside
Open WebUI's chat page — that would mean forking the Svelte frontend. It is therefore a
**companion application** in its own container, behind the same reverse proxy, using the
public REST API. This is the design the constraint produces, and it is the right one.

---

## 7. Semester timeline

Week 1 begins 24 August 2026. Thanksgiving falls in Week 14.

| Week | Dates | Workstream A | Workstream B |
|---|---|---|---|
| 1 | Aug 24 | environment set up; run the stack locally; read `HANDOFF.md` and this plan | read the compatibility policy; scope the avatar landscape |
| 2 | Aug 31 | **CI skeleton + fork guard + gitleaks** (A2) | requirements: what must a spoken session actually do? |
| 3 | Sep 7 | production snapshot; restore into test; A6 deletions | **B0**: enable STT/TTS, run a voice session end to end |
| 4 | Sep 14 | upgrade test stack to v0.9.x; migrate the Filter (A1, A4) | provider shortlist; accounts and cost model |
| 5 | Sep 21 | v0.10.x on test; smoke suite hardened | **B0 demo** to the instructor; latency baseline measured |
| 6 | Sep 28 | v0.11.x on test; docs and screenshots re-shot | provider #1 integrated in a bare harness |
| 7 | Oct 5 | **production upgrade window**; rollback rehearsed | provider #2 integrated; head-to-head measurement |
| 8 | Oct 12 | A3 hosted dashboard: container + proxy + TLS | **Gate 1** — bake-off report, choose a provider or stop |
| 9 | Oct 19 | A3 auth and professor walkthrough | **B2** companion app skeleton, wired to the chat API |
| 10 | Oct 26 | A4 OTel + alerting live | avatar joins the session; barge-in works |
| 11 | Nov 2 | A5 scoring consolidation + tests | transcript write-back into the grading pipeline |
| 12 | Nov 9 | A5 RAG decision executed | **Gate 2** — five students complete a full session |
| 13 | Nov 16 | performance and cost review; backlog burn-down | **B3** evaluation sessions run |
| 14 | Nov 23 | *(short week)* documentation and runbooks | evaluation data analysed |
| 15 | Nov 30 | freeze; handoff docs written | recommendation drafted with evidence |
| 16 | Dec 7 | **final presentation and handoff** | **final presentation and handoff** |

---

## 8. Engineering practices

These are the professional expectations, stated once.

- **Branch and PR for everything.** No direct commits to `main`. Every PR states the Open
  WebUI extension surface it uses.
- **The runbook is a deliverable.** If you did something to production, it is written down
  such that the next person can do it without you.
- **Snapshot before every migration, and restore the snapshot at least once.**
- **No secrets in the repository, ever.** `.env` files stay local; CI holds secrets in
  GitHub Actions secrets; `gitleaks` enforces it.
- **Measure before optimising.** "It feels slow" is a hypothesis; p50 and p95 are evidence.
- **Write down decisions you did not take.** The `HANDOFF.md` "Architectural Decisions and
  Why" section saved this team weeks. Extend it, don't restart it.
- **Demo every two weeks.** Working software over status updates.

---

## 9. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | The v0.10.0 config migration corrupts production data | Low | **Severe** | rehearse on a restored snapshot; verify the restore *before* touching production; schedule outside student deadlines |
| R2 | The upgrade breaks the Filter and metrics disappear unnoticed | Medium | Medium | migrate to OTel first (A4), so metrics no longer depend on the plugin |
| R3 | Avatar latency is bad enough to feel unnatural | **Medium-high** | Medium | measure early (Week 5); B0 voice-only is the fallback and it ships regardless |
| R4 | Avatar streaming cost surprises the department | Medium | Medium | model cost in Week 4 with real per-minute rates; enforce hard session caps and a spend alarm from the first prototype |
| R5 | Students spend the semester on the avatar and Workstream A slips | **High** | **High** | A1 and A2 must land before Week 8; the timeline front-loads them deliberately |
| R6 | Student voice and video recordings create privacy exposure | Medium | **Severe** | consent form; no student likeness used to build an avatar; documented retention and deletion; instructor and IRB sign-off before any recording |
| R7 | A provider changes pricing or deprecates an SDK mid-semester | Medium | Medium | build the provider-agnostic interface first (B1); keep a self-hosted option evaluated |
| R8 | Knowledge is lost at handoff again | Medium | High | `HANDOFF.md` updated continuously, not written in Week 15 |

---

## 10. Definition of done for the semester

The team is finished when all of the following are true:

1. Production runs a current Open WebUI release, upgraded via a written, rehearsed runbook.
2. A push to `main` runs CI; a tag deploys; a red build blocks the merge.
3. The CI fork guard exists and passes, and no re-forking instructions remain in the tree.
4. A professor grades from an authenticated URL with no terminal.
5. Metrics and alerting work from a clean deployment with zero manual steps.
6. Scoring and retrieval each have exactly one canonical implementation, with tests.
7. A working avatar prototype exists, has been used by real students, and is accompanied by
   a written evaluation and a defensible go/no-go recommendation.
8. `HANDOFF.md` is current enough that the Spring 2027 team can start in Week 1 instead of
   Week 4.

---

## 11. What the next team after this one should inherit

Say it out loud now, so it shapes the work: the goal is not to finish FamilyFinanceChat. It
is to leave a system where **the interesting work is the pedagogy, not the plumbing.** Every
hour spent this semester on upgrades, CI, and hosting is an hour a future team does not
spend re-learning why the fork was a mistake.
