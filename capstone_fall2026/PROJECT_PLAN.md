# FamilyFinanceChat — Fall 2026 Team Plan

**Audience:** the four-person development team, treated as professional engineers
**Term:** Fall 2026 · weekly check-in every **Tuesday, 15 September – 1 December 2026**
(no meeting 24 November, Thanksgiving break) · **11 meetings**
**Companion documents:** [`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) ·
[`AVATAR_TRACK.md`](AVATAR_TRACK.md) · [`reference/system-facts.md`](reference/system-facts.md)

---

## Contents

1. [How to use this document](#1-how-to-use-this-document)
2. [The assignment, in one page](#2-the-assignment-in-one-page)
3. [The team: four roles](#3-the-team-four-roles)
4. [Working agreement](#4-working-agreement)
5. [Before the first meeting](#5-before-the-first-meeting-15-september)
6. [Part A — Fix the existing system](#6-part-a--fix-the-existing-system)
7. [Part B — Build the interactive avatar](#7-part-b--build-the-interactive-avatar)
8. [The meeting calendar](#8-the-meeting-calendar)
9. [Deliverables checklist](#9-deliverables-checklist)
10. [Risks, and what to do when you fall behind](#10-risks-and-what-to-do-when-you-fall-behind)
11. [Appendix](#11-appendix)

---

## 1. How to use this document

This is the specification for the semester. It is written the way a work plan is written in
industry: every task has an **owner**, a **reason**, **steps**, and an **acceptance test** that
somebody other than the author can run.

- **Sections 6 and 7** are the task catalogue. Tasks are numbered `A1…A25` (fixing the existing
  system) and `B1…B18` (the avatar). Refer to them by number in commits, pull requests, and
  meetings — "A9 is blocked on A2" is a useful sentence; "the upgrade thing is stuck" is not.
- **Section 8** is the week-by-week calendar. Each meeting has an agenda, pre-work that is due
  *before* it starts, decisions that must be made *in* it, and the assignments each of the four
  of you leaves with.
- Read sections 1–5 in the first week. Read your own tasks in section 6/7 in full. Skim the
  others so you know who to ask.

**Capacity assumption:** this plan assumes **8–10 hours per person per week**, roughly 400
person-hours across the term. If your real capacity is closer to 6 hours, say so at meeting M1
— do not discover it at M6. The cut is specified in [section 10](#10-risks-and-what-to-do-when-you-fall-behind).

---

## 2. The assignment, in one page

FamilyFinanceChat is a live teaching platform. FIN 602 students practise client-facing
financial advising against an AI role-player loaded with realistic family scenarios; instructors
review the transcripts through a dashboard that scores question quality across seven dimensions
and an Ability–Benevolence–Integrity (ABI) trust rubric.

You have two jobs.

### Job one — fix and modernise what exists

The system works, but it is not yet something you can hand to somebody else:

- it runs **v0.8.12** of Open WebUI while upstream ships **v0.11.x** — three minor versions of
  fixes and, more importantly, three sets of one-way database migrations, deferred;
- the chat-metrics plugin must be **pasted into an admin form by hand after every deployment**,
  and it is written synchronously against a backend that went fully asynchronous in v0.9.0;
- the **grading dashboard runs on a laptop** — a professor needs SSH and a terminal to grade;
- there is **no CI and no automated smoke test**, so upgrades have broken things silently;
- there is **no alerting** — the last outage was reported by students;
- scoring and retrieval each exist **twice**, with no statement of which is canonical.

### Job two — investigate the interactive avatar

Turn the chatbot into a conversation: a visible AI client that hears a spoken question and
answers out loud, in character. Research it, cost it, prototype it, and **measure whether it
actually helps** using the scoring pipeline the platform already has. This is a research track
with real gates and permission to conclude "not yet." Full detail in
[`AVATAR_TRACK.md`](AVATAR_TRACK.md).

### The constraint that governs both

> **We never modify Open WebUI's core.** Every feature rides a published, documented, versioned
> extension surface. If a feature cannot be built that way, the feature changes — not the core.

This is not style guidance. The previous architecture carried ~6,500 lines of forked vendor code
to obtain two features, and it froze the platform on one release for a year. Read
[`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) in week one; it is binding, and it is
enforced in CI by task A4.

---

## 3. The team: four roles

Four people, four areas of ownership. **Owner** means: you do the work, you know its status
without being asked, and you are the one who says in the meeting whether it is on track.

| | Role | Owns | Primary tasks |
|---|---|---|---|
| **P1** | **Platform Lead** | Docker stack, the Open WebUI upgrade, CI/CD, reverse proxy, secrets, backups, monitoring and alerting | A2–A4, A6, A8–A13, A16, A18–A20 |
| **P2** | **Backend & Data Lead** | Grading service, chat extraction, scoring consolidation, transcript write-back, evaluation data and analysis | A7, A14, A17, A21–A23, B2, B12–B17 |
| **P3** | **Realtime & Avatar Lead** | The speech→model→speech→avatar pipeline, latency budget, provider evaluation | B1, B3–B10, B12, B18 |
| **P4** | **Frontend & Experience Lead** | Hosted dashboard UI, the avatar companion app UI, session flow, documentation and user testing | A15–A17, A24, B1, B2, B10, B11, B15, B16 |

**Everyone, every week:**

- review your teammates' pull requests, and ask the compatibility question every time:
  *which extension surface, and what happens on the next upstream release?*
- keep [`HANDOFF.md`](../HANDOFF.md) current as you learn things — do not write it in week eleven;
- bring a two-minute status to the Tuesday meeting: **done / blocked / next**.

**Pairing:** P3 and P4 will work closely from M5 onward (the avatar app is one product with two
halves). P1 and P2 pair on the upgrade window (A13) because the grading extractor is the thing
most likely to break. Nobody works alone on a production change.

---

## 4. Working agreement

### Meetings

**Every Tuesday, 15 September – 1 December. 60 minutes. No meeting on 24 November.**

The standing shape of every meeting (adjust the focus block, never the rest):

| Time | Block | Who |
|---|---|---|
| 0:00–0:10 | **Status round** — 2 minutes each: done / blocked / next | all four |
| 0:10–0:35 | **Focus topic** — the theme for that week (see section 8) | topic owner leads |
| 0:35–0:50 | **Decisions** — the named decisions for that meeting, resolved and written down | all |
| 0:50–1:00 | **Assignments and risks** — who does what by next Tuesday; anything newly at risk | all |

Rules that make this work:

- **Pre-work is due before the meeting, not during it.** If a demo is listed as pre-work, it is
  recorded or ready to run when the meeting starts. Debugging in the meeting wastes three
  people's time.
- **Decisions get written down the same day**, in the relevant document, with the reasoning.
  A decision nobody recorded will be re-litigated in three weeks.
- **"Blocked" is said out loud on Tuesday, not discovered on Sunday.** If you are blocked
  between meetings, message the team the day it happens.
- One person keeps the notes each week — rotate P1 → P2 → P3 → P4.

### Code

- **Branch and pull request for everything.** No direct commits to `main`.
- Every PR description states: what it does, which task number, and **which Open WebUI extension
  surface it uses** (A1–A10 in the compatibility policy).
- CI must be green to merge, from M2 onward.
- **No secrets in the repository, ever.** `.env` stays local; CI secrets live in GitHub Actions
  secrets; `gitleaks` enforces it (A4).
- Anything you did to production gets written into a runbook the same day, in enough detail that
  the next person can repeat it without you.

### Definition of done

A task is done when **all** of these are true — not when the code works on your machine:

1. it does what the acceptance test in this document says;
2. it is merged to `main` with CI green;
3. the documentation it invalidates has been updated;
4. if it touched production, the runbook exists and someone else has read it.

---

## 5. Before the first meeting (15 September)

*Collectively this is task **A1 · Environment and orientation**.*

Everything here is done by **all four people** before M1. It is roughly 4–6 hours.

| # | Task | Done when |
|---|---|---|
| 0.1 | Read this plan, `COMPATIBILITY_POLICY.md`, `AVATAR_TRACK.md`, and the existing `HANDOFF.md` and `ARCHITECTURE.md` | you can name the three things that hurt, and the one rule |
| 0.2 | Get repository access and clone it | `git log` works |
| 0.3 | Run the stack locally: copy the env template, `docker compose up -d`, create an admin account, send one chat message | you have chatted with the system on your own machine |
| 0.4 | Upload one PDF to a knowledge base and ask a question about it | the answer references your document |
| 0.5 | Run the grading dashboard locally against the test stack | you have seen a transcript scored |
| 0.6 | Write down every point where you got stuck | this list is pre-work for M1 — it is the honest state of our onboarding docs |

**Note on 0.6:** your confusion is data. The current setup docs were written by people who
already knew the system. Where you struggled is where the next team will struggle, and fixing
that is task A24.

---

## 6. Part A — Fix the existing system

Twenty-five tasks. Owner, why, steps, acceptance. Sized in hours for one person.

### Foundations — safety before speed

#### A2 · Verified backup and restore drill — **P1**, 4h
**Why.** Every upgrade in Part A depends on being able to go back. A backup that has never been
restored is not a backup; it is a hope.
**Steps.** Snapshot `/opt/openwebui/data`, `/opt/qdrant/storage`, and the Grafana data directory
from production. Restore all three into the **test stack**. Bring the test stack up on the
current version and confirm the restored data is really there — users, chats, and a working
knowledge-base query.
**Acceptance.** A teammate can follow your written restore procedure and end up with a working
test stack containing production data, without asking you anything.

#### A3 · CI smoke test — **P1**, 6h
**Why.** Past upgrades broke things silently. This is the single highest-value hour-for-hour
task in the plan.
**Steps.** A GitHub Actions workflow that, on every push: brings up the test stack; waits for
`/health` then `/ready`; creates a user via the API; sends a chat completion and asserts a
non-empty response; uploads a document to a knowledge base and asserts retrieval returns it.
**Acceptance.** Push a deliberately broken commit (e.g. an invalid model name) and show CI going
red. Then show it going green again.

#### A4 · CI compatibility guards — **P1**, 4h
**Why.** The compatibility policy is currently a document. A policy nobody enforces is a wish.
**Steps.** Add checks that fail the build when: any compose file mounts a host path into
`/app/backend/open_webui/` or `/app/build/`; the Open WebUI image is built from anything but an
official base; Python source opens SQLite or a SQLAlchemy session against Open WebUI's data
directory; `gitleaks` finds a credential; any image reference ends in `:latest`.
**Acceptance.** Add a line to a compose file that mounts into `open_webui/`, show the build
failing, then remove it. About forty lines of `grep` — do not over-engineer it.

#### A5 · Leaked key — closed, no action available — **nobody**, 0h
**Why this is here at all.** A provider API key was committed to this public repository and
stayed live on the default branch until 2026-08-19. You will find references to it in the git
history and may reasonably conclude someone should rotate it.
**Nobody on this project can.** The key belongs to a third party — a previous student's own
Qwen/DashScope account. It is not the department's and not Mark's. Nothing in this platform ever
authenticated with it, so there is no service to update and no data reachable through it.
**Steps.** None. Do not re-open this. The folder that held it (`scoring_page/`) was deleted on
2026-08-19; the value survives in git history and in a fork regardless of anything we do.
**What actually matters instead:** the prevention control, task **A4** — land `gitleaks` in CI so
a future key, one that *is* ours, cannot leak the same way. See
`docs/memory/leaked-key-not-ours-to-revoke.md`.

#### A6 · Delete the vendor fork from the tree — **P1**, 2h
**Why.** `legacy/custom-code-vendor-fork/` still contains the complete old fork *and a README
that walks the reader through bind-mounting `main.py` over Open WebUI*. It is a working recipe
for re-creating the exact mistake this project spent a semester undoing.
**Steps.** Delete the directory. Replace it with a one-page `legacy/README.md` tombstone: what it
was, why it was removed, what replaced it, and the commit hash to recover it from.
**Acceptance.** `grep -ri "bind-mount\|inject-script" legacy/` returns only the tombstone's
explanation of why those things are forbidden.

#### A7 · Delete the `--legacy` SQLite stub — **P2**, 1h
**Why.** `grading_feature/backend/extract_chats.py` advertises a `--legacy` flag as a
"direct SQLite extraction path". The body is an empty `TODO`. It is a signposted route back to
reading Open WebUI's private schema, which is forbidden (policy F5) and which broke this project
before.
**Steps.** Delete the flag, the branch, and the stub. Note the removal in `HANDOFF.md`.
**Acceptance.** `--legacy` is gone; the REST API is the only extraction path in the code.

#### A8 · Pin every image; drop the custom build — **P1**, 2h
**Why.** `qdrant:latest` and `cadvisor:latest` can change under you on any `pull`. And the
custom `build:` context for Open WebUI is the hook a future fork hangs on.
**Steps.** Replace `build: .` with `image: ghcr.io/open-webui/open-webui:<pinned tag>` and delete
the `Dockerfile`. Pin every other image to an explicit tag.
**Acceptance.** No `:latest` anywhere; the pin guard in A4 passes.

### The upgrade — v0.8.12 → v0.11.x

Read [`reference/system-facts.md`](reference/system-facts.md) §4 before starting. **Sequential,
rehearsed, one minor version at a time.** Never jump straight to the newest tag: the migrations
run in sequence anyway, and a single jump means that when something breaks you cannot attribute
it.

#### A9 · Test stack to v0.9.x, and migrate the metrics plugin — **P1**, 8h
**Why.** v0.9.0 moved Open WebUI's data layer from synchronous to asynchronous, top to bottom.
Our `monitoring/chat_metrics_filter.py` uses `def inlet/outlet` with a **blocking**
`urllib.request` call and a `self._state` dict shared across concurrent users. On an async core,
blocking I/O on the event loop degrades *every* user, and the shared dict leaks state between
sessions.
**Steps.** Restore the production snapshot into test (A2). Upgrade to v0.9.x. Follow the upstream
plugin migration guide: `async def`, non-blocking HTTP, and state keyed by
`(chat_id, message_id)` rather than by user. Run the A3 smoke suite.
**Acceptance.** Test stack runs v0.9.x with restored production data; metrics still arrive; two
simultaneous chats produce two correct, non-interleaved metric records.
**Note.** If A18 (native OpenTelemetry) lands first, this becomes *delete the filter* instead of
*port the filter*. That is the better outcome — check with P1 before doing the port.

#### A10 · Test stack to v0.10.x — **P1**, 6h
**Why.** v0.10.0 renames the `config` table to `config_old` and replaces it with a per-key table.
**This migration is one-way.** An older instance pointed at a migrated database fails
immediately. There is no rollback except restoring the snapshot.
**Steps.** Fresh restore. Upgrade. Run the smoke suite. **Then deliberately attempt to start the
previous version against the migrated database** and record exactly what the failure looks like,
so that if it happens in production nobody spends an hour diagnosing it.
**Acceptance.** Smoke suite passes on v0.10.x; the downgrade failure mode is documented in the
runbook.

#### A11 · Test stack to v0.11.x — **P1**, 5h
**Why.** v0.11.0 is additive (columns and indexes only) and moves the async driver to `psycopg`
v3, so migration risk is low. The risk is **documentation**: it substantially reorganised the
interface — sidebar, settings, model picker — so every screenshot and click-path we publish is
now wrong.
**Steps.** Upgrade. Run the smoke suite. File the UI changes that affect our documentation and
hand the list to P4 for A24.
**Acceptance.** Smoke suite passes; P4 has a written list of documentation to redo.

#### A12 · Decide SQLite vs PostgreSQL — **P1 leads, whole team decides**, 3h
**Why.** Postgres is the supported path for multi-replica operation and makes migrations and
backups far less frightening. It is also a migration in itself, and this semester is already
full. Either answer is defensible; leaving it undecided is not.
**Steps.** Write half a page: what we gain, what it costs this semester, what it costs the next
team if we defer. Bring it to M5 as a decision item.
**Acceptance.** The decision and its reasoning are recorded in `HANDOFF.md`, whichever way it goes.

#### A13 · Production upgrade window — **P1 + P2 pairing**, 6h
**Why.** This is the only irreversible thing you will do this semester.
**Steps.** Write the runbook **before** the window, including the rollback (restore snapshot,
redeploy previous tag) with exact commands. Schedule outside any week with a student deadline —
confirm the date with the instructor. Take a fresh snapshot immediately before. Upgrade through
each minor version in turn. Run the smoke suite after each. P2 verifies the grading extractor
against the upgraded instance before you call it done.
**Acceptance.** Production runs the current release; the smoke suite passes; grading still
extracts; the runbook records what actually happened, including anything that surprised you.

### Making it usable

#### A14 · Containerise the grading backend — **P2**, 5h
**Steps.** Dockerfile for the FastAPI service; add it to `docker-compose.yml` on `ai-net`;
configuration by environment variable; the OpenAI key stays server-side and never reaches the
browser.
**Acceptance.** `docker compose up -d` brings the grading API up alongside everything else.

#### A15 · Build the dashboard frontend for hosting — **P4**, 4h
**Steps.** Production Vite build to static assets; serve them from the backend container or the
proxy; make the API base URL configurable rather than hard-coded to localhost.
**Acceptance.** The dashboard loads from a server, not from a dev server on someone's laptop.

#### A16 · Reverse proxy, TLS, and authentication — **P1 + P4**, 8h
**Why.** Today a professor needs SSH and a shell script to grade. This is the single biggest
barrier to the platform actually being used.
**Steps.** Nginx or Caddy in front of the whole stack with a real hostname and TLS. Route `/` to
Open WebUI, `/grading` to the dashboard, `/grafana` to Grafana. Authenticate: university SSO/OIDC
if it can be arranged this term, HTTP basic auth as the documented fallback. Remove Grafana's
default `admin/admin`.
**Acceptance.** A professor opens a URL, authenticates, and grades — **no terminal, no VPN, and
no help from a student.** Test this with an actual non-developer.

#### A17 · Make `/refresh` asynchronous — **P2 + P4**, 5h
**Why.** Refresh pulls every chat and scores every question through a paid API. Today it is a
blocking request with no feedback. A professor clicking a button should not trigger an unbounded
wait — or an unbounded bill.
**Steps.** Background job with a progress endpoint; the UI shows progress and disables re-runs
while one is in flight; cache scored results so re-scoring is not repeated for unchanged chats.
**Acceptance.** Refresh on a realistic dataset shows visible progress and does not time out;
clicking twice does not double-charge.

### Observability that survives deployment

#### A18 · Native OpenTelemetry export — **P1**, 6h
**Why.** Chat metrics currently depend on a human pasting Python into an admin form after every
deployment, and there is **no documented API to install Functions** — so the manual step cannot
simply be scripted. Fix the class of problem: Open WebUI has exported OpenTelemetry metrics
natively since v0.8.9.
**Steps.** Enable OTel export; add an OpenTelemetry Collector to the stack; point Prometheus at
the collector; rebuild the Grafana panels on the new metric names.
**Acceptance.** A **fresh** deployment produces metrics with **zero** manual steps.

#### A19 · Retire or rewrite the metrics filter — **P1**, 2h
**Steps.** Once A18 is live, compare what the filter provides against what OTel provides. If OTel
covers it, delete the filter and the Pushgateway, and remove step 5 from `SETUP.md`. If a metric
is genuinely missing, keep the async rewrite from A9 and document exactly why.
**Acceptance.** Either the filter is gone, or there is a one-paragraph written justification for
keeping it.

#### A20 · Alerting — **P1**, 4h
**Why.** The last outage was reported by students. That is not acceptable for a course with
deadlines.
**Steps.** GCP uptime check against `/ready` (which returns 200 only when the database and Redis
are both up), alerting to a real person. Grafana alerts on container memory — a leading
indicator, since the Open WebUI container has approached its 10 GB limit — and on chat error
rate. Upgrade Grafana from 9.0.0 and Prometheus from 2.51.2.
**Acceptance.** Deliberately stop a container and show that someone is notified within five
minutes.

### Removing ambiguity

#### A21 · Pin the scoring logic with tests — **P2**, 3h
**Why.** The rubric and ABI formulas used to exist twice, in Python and JavaScript, kept in sync
by hand. **The duplicate was removed on 2026-08-19** — `scoring_page/` was deleted and
`grading_feature/backend/scoring_service.py` is canonical. What remains undone is the part that
stops the problem recurring: nothing currently fails if somebody changes a weight.
**Steps.** Add unit tests that pin the seven-dimension arithmetic and the ABI weights against
known inputs and expected outputs.
**Acceptance.** Change one weight in `scoring_service.py` and show a test failing.

#### A22 · Decide the fate of `rag_bio_project/` — **P2**, 3h
**Why.** There are two retrieval systems: the native Qdrant RAG that students actually use, and a
standalone ChromaDB pipeline that nothing in production touches. Dead code with live bugs is a
tax — the citation bug found last term lived in the stack nobody runs.
**Steps.** Choose: retire it (keeping any research notebooks), or promote it behind a **Pipelines**
server so it is reachable without touching the core. Write the reasoning down.
**Acceptance.** One canonical retrieval path; the decision is recorded.

#### A23 · Measure question-extraction accuracy — **P2**, 5h
**Why.** Student questions are currently identified by punctuation and keyword heuristics, and
**nobody has ever measured how accurate that is.** Every score downstream inherits that error
invisibly. If extraction is 70% accurate, so is every grade built on it.
**Steps.** Hand-label a sample of 200 messages from real transcripts. Run the extractor. Report
precision and recall. If accuracy is poor, propose a fix (a small model call is the obvious one)
and estimate its cost per run.
**Acceptance.** A short written report with the numbers. **A bad number is a finding, not a
failure** — report it either way.

#### A24 · Bring the documentation back into line — **P4**, 6h
**Steps.** Re-shoot every screenshot and rewrite every click-path in `SETUP.md` and the
professor-facing guides for the v0.11 interface. Fold in the stuck-points everyone recorded in
task 0.6. Delete the manual filter-install step if A19 removed it.
**Acceptance.** A person who has never seen this system follows `SETUP.md` and gets a running
stack without asking anyone a question.

#### A25 · Keep `HANDOFF.md` current — **everyone, continuously**, 1h/week
**Why.** The previous team's handoff document saved you weeks. The single highest-leverage thing
you can do for the next team is not to write it in the last week.
**Acceptance.** At M11, `HANDOFF.md` reflects reality without a scramble.

---

## 7. Part B — Build the interactive avatar

Full background, provider landscape, latency analysis, and evaluation design are in
[`AVATAR_TRACK.md`](AVATAR_TRACK.md). This section is the task breakdown.

**The architecture is decided by the compatibility rule and is not open for redesign:** the
avatar is a **companion application in its own container**, using Open WebUI's public REST API
for the model, the knowledge base, and the transcript. Embedding a video pane in Open WebUI's
frontend means forking it, which is forbidden (policy F3/F4). If someone proposes it in week
nine because it would be easier, the answer is no.

#### B1 · Requirements note — **P3 + P4**, 4h
**Steps.** Two pages: what a spoken advising session must do. Session start and persona
selection; who speaks first; what happens on interruption; what happens on silence; how a
session ends; what the student sees while waiting; what the instructor gets afterwards.
**Acceptance.** The instructor reads it and agrees that is the experience they want — **before**
anyone writes code.

#### B2 · Consent, privacy, and review package — **P2 + P4**, 6h · **due M2, non-negotiable**
**Why.** Student transcripts are education records under FERPA. Adding audio and video adds a
biometric-adjacent identifier on top of that. This gets settled **before the data exists**, not
in week eleven when it already does.
**Steps.** Draft a written consent form: what is captured, where stored, how long, who can see
it, how to opt out. Opting out must not affect a grade, so the text path stays a genuinely
equivalent alternative. Confirm with the instructor whether the learning-outcome study needs IRB
review, and start it if so — approval takes longer than students expect. Decide the retention
rule (default: keep the transcript, discard the audio).
**Acceptance.** Signed off by the instructor before any recording happens at M3.
**Hard rule.** No likeness of any real person is used to build an avatar. Use provider stock
avatars under licence. Do not clone a professor's face or voice "as a joke" — that is a deepfake
of a colleague and it will end this project.

#### B3 · B0: voice-only practice — **P3**, 6h
**Why.** Open WebUI **already ships** hands-free call mode: the student speaks, the model
answers, the reply is read back sentence-by-sentence as it streams, and the microphone re-arms.
STT and TTS are configuration, not engineering. This captures a large share of the pedagogical
benefit — speaking aloud, no backspace key, real-time pressure — at near-zero build cost and
zero compatibility risk.
**Steps.** Configure the STT engine and an OpenAI-compatible or ElevenLabs/Kokoro/Chatterbox TTS
voice. Pick a voice per family persona. Run a full advising session hands-free. Document it so a
professor can change voices without a developer.
**Acceptance.** Demo at M3. **This ships regardless of what happens to the avatar.**

#### B4 · Latency instrumentation — **P3**, 5h
**Why.** Conversation tolerates about a second. Past two seconds a person stops feeling *heard*
and starts feeling *processed*. One end-to-end number tells you it is slow; it does not tell you
what to fix.
**Steps.** Instrument each stage separately — endpointing, STT, retrieval, model
time-to-first-token, TTS first byte, avatar render — and log per-turn timings into the existing
Prometheus stack, on the same Grafana dashboard as the chat metrics.
**Acceptance.** A Grafana panel showing p50 and p95 **per stage**, populated from a real session,
by M4.
**Targets:** p50 under 1.2 s to first audible syllable; p95 under 2.0 s; interruption handled
within 300 ms.

#### B5 · Provider shortlist and cost model — **P3**, 5h
**Steps.** Shortlist two providers to actually test (HeyGen LiveAvatar, Tavus, Simli, Anam,
bitHuman, Beyond Presence and others all have agent-framework plugins). Create accounts. Build
the cost model from **live pricing pages**, recording the date checked. Include the self-hosted
comparison: an L4-class GPU VM is roughly $0.70/hour, about $500/month if left running — at ~40
hours of streaming a semester, **buying beats building**, and the report should say so plainly.
**Acceptance.** A cost table with sources and dates, presented at M4.
**Working figure:** 40 students × 4 sessions × 15 min = 40 hours ≈ $285–610 per semester all in.

#### B6 · Orchestration harness — **P3**, 8h
**Why.** Do not hand-roll WebRTC, voice activity detection, endpointing, or barge-in. That is an
entire semester of work and it is not the research question.
**Steps.** Stand up **LiveKit Agents** or **Pipecat**. Define a provider-agnostic interface so
providers can be swapped without touching the rest of the app. Get the fallback working first:
speech in, model, speech out — no avatar yet.
**Acceptance.** A scripted three-turn conversation runs end to end, with stage timings logged.

#### B7 · Integrate provider #1 — **P3**, 6h
#### B8 · Integrate provider #2 — **P3**, 5h
**Steps.** Same interface, same scripted conversation, same measurements for both. Measure:
stage-by-stage latency (p50/p95), barge-in responsiveness, lip-sync quality, **cost per minute
measured rather than quoted**, SDK maturity, behaviour when the connection drops, and whether
voice and appearance can be configured per family persona.
**Acceptance.** Identical measurements exist for both providers.

#### B9 · Bake-off report and recommendation — **P3**, 4h · **Gate 1 at M6**
**Steps.** Compare on measured data, not marketing claims — vendor latency figures were measured
under vendor conditions. Include each provider's **data-processing terms**: retention, training
use, sub-processors, region. For an education platform that can outweigh a latency win.
**Acceptance.** A recommendation the team can act on: pick one, **or recommend stopping**. Both
are acceptable outcomes. "We didn't measure" is not.

#### B10 · `avatar-app` container skeleton — **P4 + P3**, 8h
**Steps.** A new service in its own container on `ai-net`, behind the reverse proxy at
`/practice`, sharing the platform's authentication. Session creation, persona selection, hard
server-side session caps.
**Acceptance.** Deployed alongside the stack; the compatibility guards (A4) pass; a session can
be created and destroyed.

#### B11 · Session interface — **P4**, 8h
**Steps.** Video pane, a live captions track, a visible end-session control, connection status,
and a clear "this is an AI" disclosure. Enforce the 15-minute cap in the UI *and* on the server.
Sessions terminate on disconnect rather than idling and billing.
**Acceptance.** A student who has never seen it completes a session without instructions.

#### B12 · Persona loading strategy — **P3 + P2**, 4h
**Why.** The text platform runs a vector search on **every turn**. In a spoken loop that search
sits directly in the critical path, every turn, forever. The family scenarios are small.
**Steps.** Measure both: per-turn retrieval versus loading the whole persona into the system
prompt once at session start. Compare latency and answer quality.
**Acceptance.** A measured recommendation, in writing, with the numbers.

#### B13 · Transcript write-back — **P2**, 8h · **build this FIRST in the prototype phase**
**Why.** An avatar session that never reaches the grading dashboard is a demo, not a feature.
**Open question to answer, not assume:** does a chat created via `POST /api/chat/completions`
appear in Open WebUI's chat history — and therefore in `extract_chats.py` — or does it need to
be written back explicitly? **Test this in week one of the prototype phase.**
**Steps.** Preferred design: the avatar app creates a chat and appends each turn through the
public API, so there is one system of record and grading needs no changes. Fallback: the avatar
app keeps its own transcript store and `extract_chats.py` merges a second source by student and
timestamp.
**Acceptance.** A spoken session appears in the grading dashboard, attributed to the right
student, scored normally.

#### B14 · Spoken-session metrics — **P2**, 4h
**Why.** Speech exposes things typing hides, and this may be the most novel contribution the team
makes. Whether a student lets the client finish talking is arguably a more direct measure of
advising skill than anything the current rubric captures.
**Steps.** Capture per session: interruption count, talk-time ratio, filler/disfluency rate, and
the student's own hesitation before sensitive questions.
**Acceptance.** These land in the transcript record and are available to the evaluation.

#### B15 · Evaluation protocol — **P2 + P4**, 5h
**Steps.** Within-subjects, counterbalanced: each participating student practises in two or three
modalities (text / voice-only / avatar) with matched but different family scenarios, in varied
order to control for practice effects. Outcome measures: the existing 7-dimension score and ABI,
plus the B14 metrics, plus a short post-session survey on realism, social presence, and anxiety.
**Acceptance.** Protocol written and instructor-approved before M9, so recruiting can start.

#### B16 · Run the evaluation sessions — **all four**, 8h each
**When.** Between M10 (17 Nov) and M11 (1 Dec), including the Thanksgiving gap.
**Acceptance.** At least five students complete full sessions in at least two modalities, with
consent on file.

#### B17 · Analysis and evaluation report — **P2**, 8h
**Steps.** Analyse against the rubric. Report effect sizes and confidence intervals, and state
the limitations plainly.
**Acceptance.** *"Twelve students, counterbalanced, with these effect sizes and these confidence
intervals"* is a credible result. *"Avatars improve learning"* is not. **A well-designed study
with an honest negative result is the better deliverable, and it will be graded that way.**

#### B18 · Go/no-go recommendation — **P3 with all**, 4h
**Steps.** Combine cost, latency, student experience, and outcome data into a recommendation the
instructor can act on without redoing the analysis. Include a sizing estimate for what deploying
to a full cohort would require.
**Acceptance.** Presented at M11.

---

## 8. The meeting calendar

Eleven Tuesdays. Each block below lists **pre-work due before the meeting**, the **agenda**,
the **decisions** that must be resolved in the room, and the **assignments** each person leaves
with. The standing 60-minute shape is in [section 4](#4-working-agreement).

---

### M1 · Tuesday 15 September — Kickoff

**Pre-work (everyone):** section 5 complete — the stack running locally, a chat sent, a document
retrieved, the grading dashboard seen, and your list of stuck points written down.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Introductions and role assignment: who is P1, P2, P3, P4 |
| 0:10–0:25 | Walkthrough of the system as it exists — P1 drives, everyone else asks |
| 0:25–0:40 | **The one rule.** Read the compatibility policy §3–4 together. Everyone states the one-sentence test back in their own words |
| 0:40–0:50 | Stuck points from onboarding, collected into a list for A24 |
| 0:50–1:00 | Assignments; confirm real weekly capacity per person |

**Decisions:** role assignment; meeting time and place confirmed; note-taker rotation;
**honest capacity per person** (if it is 6 hours rather than 9, say so now).

**Assignments for 22 Sep**
- **P1** — Begin A2 backup/restore drill. *(Read A5 before anyone raises the leaked key as a task — it is closed.)*
- **P2** — A7 delete the `--legacy` stub; read `scoring_service.py` and `extract_chats.py` end to end; start the B2 consent draft.
- **P3** — Read `AVATAR_TRACK.md` in full; draft B1 requirements; begin the B5 provider shortlist.
- **P4** — Start B1 with P3; draft the B2 consent form with P2; collect the onboarding stuck points into a document.

---

### M2 · Tuesday 22 September — Safety net

**Focus:** we do not touch anything irreversible until we can undo it and until a broken change
gets caught automatically.

**Pre-work:** P1 — restore drill run at least once. P2/P4 — consent draft ready to read.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Status round |
| 0:10–0:25 | A2 restore drill: P1 demonstrates production data running in the test stack |
| 0:25–0:35 | A3/A4 CI plan — what the smoke test asserts, what the guards block |
| 0:35–0:50 | **B2 consent and privacy review** — read the draft aloud; agree the retention rule; confirm whether IRB review is needed |
| 0:50–1:00 | Assignments and risks |

**Decisions:** the retention rule (default: keep transcripts, discard audio); whether IRB review
is required, and who starts it this week; the two providers to shortlist for testing.

**Assignments for 29 Sep**
- **P1** — A3 CI smoke test; A4 guards; A8 pin images and drop the custom build.
- **P2** — Finalise B2 with the instructor; A21 scoring tests; start hand-labelling for A23.
- **P3** — B3 voice-only working end to end, **ready to demo next week**; open provider accounts.
- **P4** — A15 production frontend build; help P3 pick persona voices.

---

### M3 · Tuesday 29 September — Voice works

**Focus:** the fallback ships early, which is what makes the ambitious part safe to attempt.

**Pre-work:** P3 — B3 demo ready to run live. P1 — CI green on `main`.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Status round |
| 0:10–0:25 | **B3 live demo** — a full spoken advising session, hands-free |
| 0:25–0:35 | A3/A4 CI demonstration: push a broken commit, watch it fail |
| 0:35–0:50 | Upgrade plan review — read the A9/A10/A11 sequence together; agree the production window date with the instructor |
| 0:50–1:00 | Assignments |

**Decisions:** is B3 good enough to put in front of students now (yes/no, and if not, what is
missing); **the target date for the production upgrade window**; who is the second pair of eyes
on it.

**Assignments for 6 Oct**
- **P1** — A9: test stack to v0.9.x, plugin migrated to async.
- **P2** — A23 extraction accuracy measurement; A14 containerise the grading backend.
- **P3** — B4 latency instrumentation; B5 cost model with live pricing.
- **P4** — A16 proxy work with P1; B1 requirements finalised with the instructor.

---

### M4 · Tuesday 6 October — Numbers on the table

**Focus:** stop guessing about speed and money.

**Pre-work:** P3 — the latency dashboard exists and has real data in it; the cost model is written.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Status round |
| 0:10–0:25 | **B4 latency baseline** — walk through p50/p95 per stage. Where is the time actually going? |
| 0:25–0:40 | **B5 cost model** — per-minute rates, the semester total, and the self-hosting comparison |
| 0:40–0:50 | A9 report: what broke on v0.9.x, and what the plugin migration cost |
| 0:50–1:00 | Assignments |

**Decisions:** the spend cap and the alert threshold for avatar testing; whether per-turn
retrieval stays in the critical path or the persona moves into the system prompt (B12 — decide
by measurement, not preference).

**Assignments for 13 Oct**
- **P1** — A10 test stack to v0.10.x, including the documented downgrade failure; draft the A12 SQLite-vs-Postgres note.
- **P2** — A17 async refresh.
- **P3** — B6 orchestration harness; begin B7 provider #1.
- **P4** — A16 continues; begin B11 session interface sketches.

---

### M5 · Tuesday 13 October — Upgrade and integrate

**Pre-work:** P1 — v0.10.x passing the smoke suite in test; the A12 note circulated 24 hours in
advance so people can actually read it.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Status round |
| 0:10–0:25 | **A12 decision: SQLite or PostgreSQL** — P1 presents, team decides, decision recorded today |
| 0:25–0:40 | B7 provider #1 — first measurements against the harness |
| 0:40–0:50 | A16 dashboard hosting: what is left before a professor can log in |
| 0:50–1:00 | Assignments |

**Decisions:** **SQLite or Postgres, recorded in `HANDOFF.md` today**; the authentication method
for the hosted dashboard (SSO if achievable this term, basic auth as the documented fallback).

**Assignments for 20 Oct**
- **P1** — A11 test stack to v0.11.x; finalise the production runbook including rollback.
- **P2** — A22 decide the fate of `rag_bio_project/`.
- **P3** — B8 provider #2; assemble the B9 bake-off report.
- **P4** — A16 finish; A24 begin re-shooting documentation for the v0.11 interface.

---

### M6 · Tuesday 20 October — **GATE 1**

**Focus:** the first real decision point. Both outcomes are acceptable; not deciding is not.

**Pre-work:** P3 — the B9 bake-off report circulated 48 hours in advance. P1 — the production
runbook circulated, including the rollback procedure.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Status round |
| 0:10–0:30 | **B9 bake-off** — measured latency, measured cost, barge-in behaviour, data-processing terms |
| 0:30–0:40 | **GATE 1 decision: pick a provider, or stop the avatar track** |
| 0:40–0:50 | **Production upgrade go/no-go** — runbook reviewed, rollback confirmed, date locked |
| 0:50–1:00 | Assignments |

**Decisions:** the avatar provider — **or the decision to stop**, with reasons, and reallocation
of P3 to Part A if so; production upgrade go/no-go and the exact date.

**Assignments for 27 Oct**
- **P1** — A13 execute the production upgrade window, paired with P2.
- **P2** — Pair on A13; verify grading extraction post-upgrade; **B13 answer the write-back question** (does an API-created chat appear in chat history?).
- **P3** — B10 `avatar-app` skeleton with P4.
- **P4** — B10/B11; A24 continues.

---

### M7 · Tuesday 27 October — Production current

**Pre-work:** the upgrade has happened, or the reason it has not is written down.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Status round |
| 0:10–0:30 | **A13 debrief** — what happened, what surprised us, what the runbook now says. Anything that surprised you goes in `HANDOFF.md` today |
| 0:30–0:40 | A16 hosted dashboard: **walk a non-developer through logging in and grading**, live |
| 0:40–0:50 | B13 write-back: which design are we using, and why |
| 0:50–1:00 | Assignments |

**Decisions:** transcript write-back design (write into Open WebUI vs. merge a second source);
whether the hosted dashboard is ready to give to the instructor this week.

**Assignments for 3 Nov**
- **P1** — A18 native OTel export; A19 retire the filter; A20 alerting.
- **P2** — B13 implement write-back; B15 draft the evaluation protocol.
- **P3** — Avatar joins a live session; get barge-in working; B12 persona-loading measurement.
- **P4** — B11 session interface; hand the dashboard to the instructor and watch them use it.

---

### M8 · Tuesday 3 November — First conversation

**Pre-work:** P3 — an avatar session that runs, even if rough. Recorded is fine.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Status round |
| 0:10–0:30 | **First avatar session demo** — where does it break, and how does interruption feel? |
| 0:30–0:40 | A18/A20: metrics from a clean deployment with zero manual steps; induce an outage and watch the alert fire |
| 0:40–0:50 | B15 evaluation protocol review — recruiting starts next week |
| 0:50–1:00 | Assignments |

**Decisions:** evaluation protocol approved; how many students to recruit and how; the date of
the Gate 2 session block.

**Assignments for 10 Nov**
- **P1** — Finish A20; begin the deployment section of `HANDOFF.md`.
- **P2** — B13 end-to-end verified: a spoken session appears in the dashboard, scored; B14 spoken metrics.
- **P3** — Latency tuning against the p50/p95 targets; stability.
- **P4** — B11 polish; recruit evaluation participants; run one pilot session and fix what confused them.

---

### M9 · Tuesday 10 November — Wire it into grading

**Focus:** the avatar becomes a feature rather than a demo the moment its transcripts are graded.

**Pre-work:** P2 — a real spoken session visible in the grading dashboard, scored.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Status round |
| 0:10–0:25 | **Spoken session → dashboard → score**, demonstrated end to end |
| 0:25–0:35 | Latency check against targets: p50 < 1.2 s, p95 < 2.0 s, barge-in < 300 ms. Are we there? |
| 0:35–0:50 | Gate 2 readiness: what stands between us and five students completing full sessions |
| 0:50–1:00 | Assignments |

**Decisions:** are we ready to run students through it next week — yes, or what specifically
must be fixed first.

**Assignments for 17 Nov**
- **P1** — Finish outstanding Part A items; A24 review with P4.
- **P2** — Prepare the evaluation data pipeline; dry-run the analysis on pilot data.
- **P3** — Session reliability; make sure a dropped connection fails gracefully.
- **P4** — Schedule the five students; prepare consent forms and the session script.

---

### M10 · Tuesday 17 November — **GATE 2**

**Pre-work:** five students have completed, or are scheduled to complete, full spoken sessions.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Status round |
| 0:10–0:30 | **GATE 2: five students, full sessions, transcripts graded.** Show the transcripts and the scores |
| 0:30–0:45 | What the students said — first impressions from the survey and from watching them |
| 0:45–0:55 | Plan for the two-week Thanksgiving gap: who runs sessions, who analyses, who writes |
| 0:55–1:00 | Assignments |

**Decisions:** Gate 2 passed or not; the split of evaluation, analysis, and writing across the
gap; the outline of the final presentation.

**Assignments for 1 Dec** *(two weeks — no meeting 24 November)*
- **P1** — Deployment and operations sections of `HANDOFF.md`; confirm every Part A acceptance test passes.
- **P2** — B17 evaluation analysis and report.
- **P3** — B18 recommendation with production sizing; avatar sections of `HANDOFF.md`.
- **P4** — Finish A24 documentation; assemble the final presentation.

> **No meeting Tuesday 24 November — Thanksgiving break.** If something breaks in production
> during the break, P1 is on call and the rollback procedure is in the runbook. Do not attempt a
> production change during the break.

---

### M11 · Tuesday 1 December — Findings and handoff

**Pre-work:** the evaluation report and the recommendation are circulated 48 hours in advance.
This meeting is for discussion, not for first reading.

**Agenda**
| | |
|---|---|
| 0:00–0:10 | Status round: everything outstanding, honestly |
| 0:10–0:30 | **B17 evaluation findings** — what the data shows, including what it does not show |
| 0:30–0:40 | **B18 go/no-go recommendation** for the avatar |
| 0:40–0:50 | Handoff review: walk the [deliverables checklist](#9-deliverables-checklist) and mark each item done or not done |
| 0:50–1:00 | What the next team should do first, written into `HANDOFF.md` before you leave the room |

**Decisions:** the recommendation, in a form the instructor can act on; the top three items for
the next team.

---

## 9. Deliverables checklist

Walk this list at M11. Each item is verifiable by somebody who was not in the room.

**Part A — the platform**
- [ ] Production runs a current Open WebUI release, upgraded via a written, rehearsed runbook (A13)
- [ ] A push runs CI; a tag deploys; a red build blocks the merge (A3)
- [ ] The fork guard exists, passes, and no re-forking instructions remain in the tree (A4, A6)
- [ ] A professor grades from an authenticated URL with no terminal (A16) — *verified with an actual professor*
- [ ] Metrics and alerting work from a clean deployment with zero manual steps (A18, A20)
- [ ] An induced outage notifies a person within five minutes (A20)
- [ ] Scoring and retrieval each have one canonical implementation, with tests (A21, A22)
- [ ] Question-extraction accuracy has been measured and reported (A23)
- [ ] `SETUP.md` gets a newcomer to a running stack with no questions (A24)

**Part B — the avatar**
- [ ] Voice-only spoken practice is available to students (B3)
- [ ] Per-stage latency is measured and visible in Grafana (B4)
- [ ] Two providers were measured on identical conversations (B7, B8)
- [ ] A provider was chosen, or stopping was recommended, on evidence (B9)
- [ ] `avatar-app` runs in its own container behind the proxy (B10, B11)
- [ ] A spoken session appears in the grading dashboard, scored (B13)
- [ ] At least five students completed full sessions with consent on file (B16)
- [ ] A written evaluation reports findings **and** limitations (B17)
- [ ] A go/no-go recommendation with production sizing (B18)

**Both**
- [ ] `HANDOFF.md` lets the next team start in week one instead of week four (A25)

---

## 10. Risks, and what to do when you fall behind

| # | Risk | Severity | What we do about it |
|---|---|---|---|
| R1 | The v0.10.0 config migration corrupts production data | **Severe** | A2 restore drill first; rehearse on restored data; run outside deadline weeks; rollback written before the window |
| R2 | **The avatar absorbs the semester and Part A slips** | **Severe** | Expect this. A3/A4 land by M2 and the upgrade by M7 — deliberately front-loaded. Gate 1 at M6 is real |
| R3 | Avatar latency lands at 3+ seconds | Medium | Measure at M4, not M9. B3 voice-only ships regardless |
| R4 | Streaming costs run away during testing | Medium | Hard caps, quotas, and a billing alarm from the first prototype |
| R5 | Voice/video creates a privacy exposure | **Severe** | B2 settled at M2, before any recording. No real likenesses, ever |
| R6 | A provider reprices or deprecates mid-semester | Medium | Provider-agnostic interface (B6); keep the runner-up integrated |
| R7 | Avatar sessions never reach the grading dashboard | Medium | B13 is built **first** in the prototype phase, not last |
| R8 | Someone proposes embedding video in the chat page | Medium | Policy F3/F4 — the answer is no. That is what the companion app is for |
| R9 | Knowledge is lost at handoff again | High | A25 — `HANDOFF.md` updated weekly, not written in the last week |

### If you fall behind — cut in this order

Say it at the Tuesday meeting the week you realise it, not the week it becomes visible.

1. **B8 (second provider).** Evaluate one provider properly rather than two badly. Note the
   reduced confidence in the bake-off report.
2. **A22 (`rag_bio_project` decision).** Document the ambiguity for the next team instead of
   resolving it.
3. **A12 (Postgres migration), if the decision was to migrate.** Stay on SQLite, write down why,
   and hand the decision forward.
4. **B16 sample size.** Three students with a clean protocol beats eight with a sloppy one.

**Never cut:** A2 (backups), A3/A4 (CI and guards), B2 (consent), A13's
rollback plan, or A25 (`HANDOFF.md`). Those are the ones that hurt someone else if you skip them.

---

## 11. Appendix

### Meeting dates at a glance

| # | Date | Theme |
|---|---|---|
| M1 | Tue 15 Sep 2026 | Kickoff |
| M2 | Tue 22 Sep 2026 | Safety net — backups, CI, consent |
| M3 | Tue 29 Sep 2026 | Voice works (B3 demo) |
| M4 | Tue 6 Oct 2026 | Numbers — latency and cost |
| M5 | Tue 13 Oct 2026 | Upgrade and integrate |
| M6 | Tue 20 Oct 2026 | **Gate 1** — pick a provider or stop |
| M7 | Tue 27 Oct 2026 | Production current |
| M8 | Tue 3 Nov 2026 | First conversation |
| M9 | Tue 10 Nov 2026 | Wire it into grading |
| M10 | Tue 17 Nov 2026 | **Gate 2** — five students, graded |
| — | Tue 24 Nov 2026 | *No meeting — Thanksgiving break* |
| M11 | Tue 1 Dec 2026 | Findings and handoff |

### Where things live

| What | Where |
|---|---|
| Production stack | `docker-compose.yml` (8 services, one GCP VM) |
| Open WebUI version pin | `Dockerfile` — to become a pinned `image:` in compose (A8) |
| Chat metrics plugin | `monitoring/chat_metrics_filter.py` |
| Grading backend | `grading_feature/backend/` — `extract_chats.py`, `scoring_service.py`, `main.py` |
| Grading frontend | `grading_feature/frontend/` (React/Vite) |
| Scoring logic (canonical) | `grading_feature/backend/scoring_service.py` — the JS duplicate was deleted 2026-08-19 |
| Second retrieval stack | `rag_bio_project/` — fate to be decided (A22) |
| Knowledge-base CLI | `tools/kb_sync/` — already API-based and upgrade-safe |
| Old vendor fork | `legacy/custom-code-vendor-fork/` — **to be deleted** (A6) |

### Vocabulary

Use these precisely; they are not interchangeable.

- **Upstream** — the original open-source project we depend on (`open-webui/open-webui`).
- **Upgrade / version bump** — moving our pinned image tag to a newer published release. This is
  what we do.
- **Merging or syncing a fork** — reconciling upstream's changes with our own modified copy of
  their source. This is what we *used* to do, and it is forbidden.
- **Migration** — the schema change Open WebUI runs against its own database on first boot after
  an upgrade. Distinct from the upgrade, with distinct risk: you can roll back an image tag, but
  you cannot roll back a migration without restoring a backup.
- **Extension surface** — a published, documented, versioned integration point (Functions,
  Pipelines, Tools, the REST API, environment configuration). Everything we build rides one.

### Reference

- Facts, figures, sources, and Q&A prep: [`reference/system-facts.md`](reference/system-facts.md)
- The binding constraint and the repository audit: [`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md)
- Avatar background, providers, latency, evaluation design: [`AVATAR_TRACK.md`](AVATAR_TRACK.md)
- The five-minute overview deck: [`slides/`](slides/) · narration: [`script/`](script/)
