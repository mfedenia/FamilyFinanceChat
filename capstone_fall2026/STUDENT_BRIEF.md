# FamilyFinanceChat — Student Brief, Fall 2026

### A companion to the overview video

**Read this if you watched the video and are deciding whether to join.**

The video is a four-minute summary. This is the honest, longer version: what the system
actually is, what is actually broken, what you would actually do on a Tuesday afternoon in
October, how many hours it really takes, and what could go wrong. It is written to help you
say **no** as easily as yes — a capstone joined for the wrong reasons is worse for you than one
you skipped.

Everything here traces to a document in this repository, and links are given so you can check
any claim yourself before committing a semester to it.

---

## Contents

1. [The short version](#1-the-short-version)
2. [The teaching problem this exists to solve](#2-the-teaching-problem-this-exists-to-solve)
3. [What actually runs today](#3-what-actually-runs-today)
4. [The three things that hurt](#4-the-three-things-that-hurt)
5. [The one rule — and why it is the most valuable thing here](#5-the-one-rule--and-why-it-is-the-most-valuable-thing-here)
6. [Job one — make it solid](#6-job-one--make-it-solid)
7. [Job two — give the client a face](#7-job-two--give-the-client-a-face)
8. [The four roles: which one is you](#8-the-four-roles-which-one-is-you)
9. [The semester, week by week](#9-the-semester-week-by-week)
10. [What the workload really is](#10-what-the-workload-really-is)
11. [What you walk away with](#11-what-you-walk-away-with)
12. [Honest warnings](#12-honest-warnings)
13. [How to decide, and what to do first](#13-how-to-decide-and-what-to-do-first)
14. [Where to read more](#14-where-to-read-more)

---

## 1. The short version

FamilyFinanceChat is a **live teaching platform** — not a prototype, not a class exercise.
FIN 602 students use it to practise client-facing financial advising against an AI role-player
that plays a realistic family, grounded in the actual course documents so it does not invent
its own facts. Every question a student asks is scored automatically across seven quality
dimensions and an Ability–Benevolence–Integrity trust rubric, and the instructor reviews the
results in a dashboard.

It runs as eight Docker containers on a Google Cloud VM, built on top of an open-source project
called Open WebUI.

**You have two jobs this semester.**

| | Job | Character |
|---|---|---|
| **A** | Make the platform solid — upgrade it, host it, automate it, monitor it | Engineering. Unglamorous. The course depends on it. |
| **B** | Give the AI client a face and a voice, then measure whether it helps | Research. Genuinely uncertain. Allowed to conclude "no." |

Job A is what makes the semester safe. Job B is what makes it interesting. The plan is
deliberately built so that A ships even if B fails, and so that a failed B is still a
publishable result rather than a wasted semester.

**The one thing to understand before anything else:** this project is built *on top of* someone
else's open-source software, and we are forbidden from modifying it. That single constraint
shapes every technical decision you will make. Section 5 explains why, and why it is the most
transferable thing you will learn here.

---

## 2. The teaching problem this exists to solve

FIN 602 prepares students for a room with a real family in it.

What that room actually demands is: listening while someone is still talking, hearing
hesitation, holding eye contact through an uncomfortable question about money, and asking the
follow-up out loud without a backspace key.

Practising that traditionally requires human role-players — which means scheduling, cost, and a
hard ceiling on repetitions. Most students get two or three rehearsals. The ones who need ten
do not get ten.

FamilyFinanceChat removes the ceiling. A student can practise at 2 a.m., as many times as they
want, against a family with a consistent financial situation, and get scored feedback
afterwards. **The point is scale.**

The platform teaches the *content* of good advising well. What it cannot yet teach is the
*performance* of it — because typing removes every pressure that makes the real room hard. A
student can draft, delete and reword for ninety seconds before asking *"what happens to the
business if something happens to you?"* — a question they will have to ask once, in real time,
to a person whose face they can see.

That gap is Job B, and the plan states it as a hypothesis rather than a conclusion:

> A spoken, face-to-face AI client produces meaningfully better preparation than a text
> chatbot — **and we can measure whether that is true** using the scoring pipeline the platform
> already has.

That last clause is why this is a real research project and not a demo. The outcome measure
already exists and is already in production.

---

## 3. What actually runs today

This is a working system with real users, not a greenfield build. Joining means inheriting
something, which is a different skill from starting something.

### The production stack

Eight containers on one GCP VM, plus a grading tool that currently runs on a laptop:

| Component | Role |
|---|---|
| **Open WebUI** (`v0.8.12`) | Chat UI, model routing, user accounts, chat history |
| **Qdrant** | Vector database — retrieves relevant course documents per turn |
| **Valkey/Redis** | WebSocket session state |
| **Prometheus** | Metrics collection, 30-day retention |
| **Pushgateway** | Receives chat metrics from a plugin |
| **cAdvisor** | Container CPU/memory metrics |
| **metrics-exporter** | Health-probe sidecar |
| **Grafana** | Dashboards |
| **Grading dashboard** | FastAPI + React — **runs locally, not containerised** |

The language model is `gpt-4o-mini` via the OpenAI API; embeddings are
`text-embedding-3-small`. Retrieval uses MMR search with `k=5` and a relevance threshold of
`0.30`. Uploads land in a Google Cloud Storage bucket mounted into the VM.

Full detail: [`ARCHITECTURE.md`](../ARCHITECTURE.md). Every figure above is sourced in
[`reference/system-facts.md`](reference/system-facts.md).

### The scoring pipeline

This is the part that makes the platform more than a chatbot, and it is the measuring
instrument for Job B.

Each student question is scored **0–2 on seven dimensions** — relevance, politeness, on-topic,
neutrality, non-imperative, clarity, privacy-minimisation — for a raw total of 0–14, normalised
to 0–100.

Those scores then feed an **ABI trust rubric**: twelve sub-dimensions weighted into three
scores — **A**bility, **B**enevolence, **I**ntegrity — each 0–1, averaged into an ABI total. It
is a recognised framework from the trust literature, applied to advising transcripts.

One honest caveat, stated in the facts sheet: **question extraction is heuristic** —
punctuation and keyword based — and its accuracy has never been measured. Measuring it is task
A23. That is the kind of thing you find when you inherit a system.

### What already works well

- Multi-user chat with real-time WebSocket sessions
- RAG over course documents, so the AI client's facts come from the syllabus
- File upload through the UI or a command-line sync tool
- Chat metrics in Grafana
- The grading dashboard: extraction, seven-dimension scoring, ABI analysis, per-student feedback
- Separate production and test stacks

**Open WebUI already ships hands-free voice call mode** — speech in, speech out, with
sentence-by-sentence playback and automatic microphone re-arming, supporting Whisper, Voxtral,
ElevenLabs, Kokoro and others. This matters enormously: it means the voice-only version of Job
B costs almost nothing to build and can ship in week three as a safety net.

---

## 4. The three things that hurt

The video names three. Here they are with the detail behind them.

### 4.1 We are three versions behind, and the debt compounds

Production runs Open WebUI `v0.8.12`. Upstream ships `v0.11.x`. That is not just three sets of
bug fixes — it is **three sets of one-way database migrations that have been deferred**.

| Release | What changed | What it costs us |
|---|---|---|
| `v0.8.0` | Skills, Analytics, message queuing | Skills are where course frameworks should live, professor-editable |
| `v0.8.9` | `/ready` endpoint, OpenTelemetry system metrics | Real uptime checks without a custom plugin |
| **`v0.9.0`** | Backend data layer went fully **async** | **Our metrics plugin is written synchronously and must be migrated** |
| **`v0.10.0`** | `config` table split into per-key tables | **One-way migration.** An old instance cannot run against a migrated database |
| `v0.11.0` | `psycopg` v3, large UI reorganisation, sub-agents | Low migration risk, high documentation churn |

The `v0.10.0` migration is the sharp edge. You can roll back an image tag; you cannot roll back
a migration without restoring a backup. That is why the very first engineering task of the
semester is proving that the backup restores.

### 4.2 The grading tool runs on a laptop

To grade, a professor currently needs SSH access and a terminal. Realistically that means a
professor needs a student sitting next to them.

**This is the single biggest barrier to anyone else adopting this platform.** The fix is not
glamorous — containerise the backend, build the frontend for hosting, put it behind a reverse
proxy with TLS and authentication — but its acceptance test is unusually honest: *a professor
grades from a URL, verified with an actual professor.*

### 4.3 Every deployment needs manual steps people forget

The chat-metrics plugin is **pasted into an admin form by hand after every deployment**. There
is no documented API for installing Functions, so it cannot simply be scripted — which is why
the right fix is to move to Open WebUI's native OpenTelemetry export and retire the plugin
entirely.

Beyond that: no CI, no automated smoke test, no alerting. Upgrades have broken things silently.
**The last outage was reported by students.**

### 4.4 The quieter fourth problem

Some things exist twice with no statement of which is canonical. There is a second retrieval
stack (`rag_bio_project/`) that is not wired into production. A duplicate JavaScript scoring
implementation existed until it was deleted in August 2026. Ambiguity like this is how a
handoff goes wrong, and resolving it is explicit work in the plan (A21, A22).

---

## 5. The one rule — and why it is the most valuable thing here

> **We never modify Open WebUI's core.** Every customisation must ride a published, documented,
> versioned extension surface. If a feature cannot be built that way, **the feature changes —
> not the core.**

The test, applied in every code review:

> *If upstream cut a release tomorrow and we ran it unchanged, would this still work?*

If the answer requires "well, as long as they don't change…", it is forbidden.

### Why this rule exists

This project learned it the expensive way. A previous architecture carried roughly **6,500
lines of copied and modified vendor code** — `main.py` at 2,518 lines, `config.py` at 3,876,
plus a router injected into the vendor's own package — in order to obtain two features. The
custom UI was delivered by a **browser bookmarklet that fetched and `eval`'d remote code.**

The cost: every upgrade meant hand-merging someone else's software. So no upgrades happened.
The platform froze on one release for a year.

Removing all of it was the previous team's main achievement. **Today our Dockerfile is one
line.** We just run their release.

### What this means for you day to day

There are ten sanctioned places to build — environment variables and admin settings, Filter /
Pipe / Action Functions, a Pipelines server, Tools and MCP servers, the public REST API,
sidecar containers, reverse-proxy composition, and Knowledge Bases and Skills.

And eight forbidden patterns, stripped on sight: bind-mounting into the vendor package,
building a custom image from modified source, JavaScript injection into their frontend, direct
SQL against their schema, monkeypatching internals from inside a plugin, depending on
undocumented endpoints, and — notably — **pinning to an old release to avoid migration work.**

Every pull request you open states which surface it uses and what happens on the next upstream
release. It is enforced in CI.

### Why this is the thing worth learning

Most student projects never encounter this constraint, because most student projects are
abandoned before the dependency releases a breaking change. Working inside a hard architectural
boundary — and discovering that **the boundary chooses better designs for you** — is the single
most transferable skill in this capstone.

Section 7 is the proof. The rule forbids putting the avatar inside the chat window, which
forces it to be a separate application, which turns out to be independently deployable,
independently testable, upgrade-proof by construction, and deletable in one step if the
research says stop. The constraint did not get in the way of the design. It produced it.

Full policy: [`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md).

---

## 6. Job one — make it solid

Twenty-five tasks, each with an owner, a rationale, steps, an acceptance test, and an hour
estimate. Grouped by theme:

### Foundations — safety before speed *(weeks 1–2)*

- **A2 · Verified backup and restore drill.** Snapshot production, restore into the test stack,
  confirm the users, chats and knowledge-base queries are really there. *A backup that has never
  been restored is not a backup; it is a hope.* Acceptance: a teammate follows your written
  procedure and succeeds without asking you anything.
- **A3 · CI smoke test** and **A4 · CI compatibility guards** — including `gitleaks` for secrets
  and an automated check that nobody has re-forked the vendor.
- **A6, A7 · Delete the old vendor fork and a legacy stub** so nobody can reintroduce them by
  accident.
- **A8 · Pin every image.** Two images currently float on `:latest`.

### The upgrade *(weeks 2–7)*

`v0.8.12 → v0.9.x → v0.10.x → v0.11.x`, **one minor version at a time, on the test stack
first**, with the metrics plugin migrated to async along the way, a team decision on SQLite
versus PostgreSQL, and finally a production upgrade window run by two people pairing — because
the grading extractor is the thing most likely to break.

The rollback plan is written *before* the window opens. This is rehearsed on restored data, not
attempted live.

### Making it usable *(weeks 3–8)*

Containerise the grading backend, build the frontend for hosting, put Nginx or Caddy in front
with TLS and authentication, and make the slow `/refresh` operation asynchronous so the UI does
not hang.

### Observability that survives deployment *(weeks 5–9)*

Move to Open WebUI's **native OpenTelemetry export**, retire the hand-pasted metrics filter, and
add alerting. Acceptance test: *an induced outage notifies a person within five minutes.*

### Removing ambiguity *(throughout)*

Pin the scoring logic with tests, decide the fate of the second retrieval stack, **measure
question-extraction accuracy** for the first time, bring the documentation back into line, and
keep the handoff document current *weekly* — not written in the last week.

---

## 7. Job two — give the client a face

### The target experience

A student opens a link, sees a face, and says out loud:

> "Hi — I'm going to ask you a few questions about your family's finances. Is that alright?"

The client on screen looks at them, nods, and answers in their own voice, in character, with
their own financial situation drawn from the same course documents the text bot uses. **If the
student interrupts, the client stops talking.** If the student goes quiet, the client waits —
and may eventually prompt them, the way a real, slightly impatient client would.

At the end, the transcript lands in the same grading dashboard, scored on the same seven
dimensions and the same ABI rubric.

**What we are not building:** a photorealistic digital human, a likeness of any real person, or
a replacement for the text interface. Text stays.

### The architecture the rule chose

Not inside the chat window — that would mean forking a Svelte frontend we do not own, which is
exactly the failure mode this project spent a semester undoing.

Instead: **a companion web application in its own container**, served at `/practice` behind the
same reverse proxy. Open WebUI stays the brain (model routing, retrieval, knowledge base) and
the system of record (transcripts). The avatar app talks to it exclusively over the public REST
API.

The pattern that matters — and the one the industry converged on during 2026 — is that the
avatar renderer **joins the session as its own participant**, publishing synchronised audio and
video directly to the student, rather than being a post-processing step our server waits on.
The plan is explicit that you should *not* hand-roll the WebRTC and turn-taking layer:
**LiveKit Agents** or **Pipecat** give you voice activity detection, endpointing, barge-in and
provider swapping for free. Hand-rolling that is a semester by itself, and it is not the
semester's research question.

### The three numbers that decide it

**Speed.** Conversation tolerates about a second. Past roughly two seconds of silence a person
stops feeling heard and starts feeling processed. The budget from end of student speech to
first audible syllable:

| Stage | Realistic range |
|---|---:|
| Voice activity detection / endpointing | 150–300 ms |
| Speech to text | 150–300 ms |
| Retrieval (RAG) | 100–300 ms |
| LLM time-to-first-token | 300–600 ms |
| Text to speech, first byte | 150–300 ms |
| Avatar render + network | 200–500 ms |
| **Total** | **≈ 1.0–2.3 s** |

Targets: **p50 under 1.2 s, p95 under 2.0 s, barge-in interrupting within 300 ms.**

Note the trap hiding in that table: the text platform retrieves from the vector database on
every turn, which in a spoken loop sits directly in the critical path. The likely fix is loading
the family scenario into the system prompt once at session start.

**Cost.** Forty students × four sessions × fifteen minutes ≈ forty hours a semester. At roughly
$0.10–$0.20 per minute for avatar streaming, plus speech and model costs, that is **about
$285–$610 per semester** — with hard caps, quotas and a billing alarm from the first prototype,
not after the first bill. For comparison, a dedicated GPU VM would run around $500 a month. At
this scale, metered streaming wins.

**Permission.** Voice and video of students are education records under FERPA, and audio adds a
biometric-adjacent identifier. Written consent, an equivalent non-recorded path that does not
affect a grade, no likeness of any real person ever, documented retention, a review of each
vendor's data-processing terms, and a check on whether the study needs IRB review — **all
settled in week two, before anything is recorded.** This task is marked non-negotiable.

If any one of those three fails, the answer is no, and **no is an acceptable answer.**

### The gates

The track runs in four phases with two real gates:

| Phase | Window | Exit |
|---|---|---|
| **B0 · Voice, no face** | to 29 Sep | Spoken practice available to students. **Ships no matter what happens next.** |
| **B1 · Provider bake-off** | 29 Sep – 20 Oct | Two providers measured on identical conversations |
| — | **20 Oct** | **Gate 1: pick a provider, or stop** |
| **B2 · Companion prototype** | 20 Oct – 17 Nov | `avatar-app` running, transcripts reaching the dashboard |
| — | **17 Nov** | **Gate 2: five students complete graded spoken sessions** |
| **B3 · Evaluate** | 3 Nov – 1 Dec | Written findings and a go/no-go recommendation |

That B0 line is the whole reason the ambitious part is safe to attempt.

### The research design

Within-subjects and counterbalanced: each participating student does sessions in two or three
modalities (text / voice-only / avatar) with matched but different family scenarios, in varied
order to control for practice effects. Measures include the existing seven-dimension scores and
ABI trust scores, questions per session, interruptions and talk-time ratio, hesitation before
sensitive questions, and a short post-session survey on realism and anxiety.

The plan is blunt about how to report it:

> Sample size is small and the semester is short. Do not over-claim. *"Twelve students,
> counterbalanced, with these effect sizes and these confidence intervals"* is a credible
> result; *"avatars improve learning"* is not. A well-designed study with an honest negative
> result is a better deliverable than an over-claimed positive one, **and it will be graded that
> way.**

Full detail: [`AVATAR_TRACK.md`](AVATAR_TRACK.md).

---

## 8. The four roles: which one is you

Four people, four areas of ownership. **Owner** means: you do the work, you know its status
without being asked, and you are the one who says in the meeting whether it is on track.

### P1 · Platform Lead

**Owns:** Docker stack, the Open WebUI upgrade, CI/CD, reverse proxy, secrets, backups,
monitoring and alerting.

**You would spend the semester on:** the restore drill, GitHub Actions, four sequential version
upgrades with one-way database migrations, TLS and authentication, OpenTelemetry, and alerting.

**Fits you if** you like infrastructure, are calm about production, and get satisfaction from a
deployment that no longer needs a human to remember anything.

**You will learn:** staged upgrades of a dependency you do not control, migration risk and
rollback planning, CI as an enforcement mechanism rather than a formality.

### P2 · Backend & Data Lead

**Owns:** grading service, chat extraction, scoring consolidation, transcript write-back,
evaluation data and analysis.

**You would spend the semester on:** containerising the grading backend, making extraction
asynchronous, pinning the scoring logic with tests, **measuring question-extraction accuracy for
the first time**, building transcript write-back from spoken sessions, and then running the
statistical analysis that answers the semester's research question.

**Fits you if** you are the person who wants to know whether the number is actually right, and
you would enjoy writing the evaluation report.

**You will learn:** Python/FastAPI in production, working against someone else's REST API,
measurement design, and honest reporting of a small-N study.

### P3 · Realtime & Avatar Lead

**Owns:** the speech → model → speech → avatar pipeline, the latency budget, provider
evaluation.

**You would spend the semester on:** getting voice-only working in week three, instrumenting
per-stage latency, building a provider-agnostic orchestration harness on LiveKit Agents or
Pipecat, integrating two avatar providers, and writing the bake-off that decides Gate 1.

**Fits you if** real-time systems appeal to you and you are comfortable with a track where the
correct answer might be "stop."

**You will learn:** streaming media, WebRTC-adjacent orchestration, latency budgeting as an
engineering discipline, and evaluating vendors on measured data rather than marketing claims.

### P4 · Frontend & Experience Lead

**Owns:** the hosted dashboard UI, the avatar companion app UI, session flow, documentation and
user testing.

**You would spend the semester on:** building the grading dashboard for hosting, designing the
spoken-session interface (including what "the client is listening" looks like on screen), the
consent flow, running evaluation sessions with real students, and fixing the onboarding
documentation.

**Fits you if** you care about whether a professor can actually use the thing, and you want
design work with a measurable acceptance test attached.

**You will learn:** React in a real deployment, interface design for real-time conversation,
usability testing, and technical writing that gets verified against a newcomer.

### Everyone, every week

- Review your teammates' pull requests, asking the compatibility question every time: *which
  extension surface, and what happens on the next upstream release?*
- Keep [`HANDOFF.md`](../HANDOFF.md) current as you learn things.
- Bring a two-minute status: **done / blocked / next.**

**Nobody works alone on a production change.** P3 and P4 pair from mid-October onward — the
avatar app is one product with two halves. P1 and P2 pair on the upgrade window.

---

## 9. The semester, week by week

**Every Tuesday, 15 September – 1 December, 60 minutes.** No meeting 24 November.

| # | Date | Theme |
|---|---|---|
| M1 | Tue 15 Sep | Kickoff |
| M2 | Tue 22 Sep | Safety net — backups, CI, **consent** |
| M3 | Tue 29 Sep | **Voice works** — B0 demo |
| M4 | Tue 6 Oct | Numbers — latency and cost on the table |
| M5 | Tue 13 Oct | Upgrade and integrate |
| M6 | Tue 20 Oct | **Gate 1** — pick a provider or stop |
| M7 | Tue 27 Oct | Production current |
| M8 | Tue 3 Nov | First conversation |
| M9 | Tue 10 Nov | Wire it into grading |
| M10 | Tue 17 Nov | **Gate 2** — five students, graded |
| — | Tue 24 Nov | *No meeting — Thanksgiving* |
| M11 | Tue 1 Dec | Findings and handoff |

Every meeting has the same shape: ten minutes of status round, twenty-five minutes on that
week's focus topic, fifteen minutes resolving *named* decisions, ten minutes on assignments and
risks.

Three rules make it work:

- **Pre-work is due before the meeting, not during it.** If a demo is listed, it is ready to run
  when the meeting starts. Debugging live wastes three people's time.
- **Decisions get written down the same day**, with the reasoning. A decision nobody recorded
  gets re-litigated in three weeks.
- **"Blocked" is said out loud on Tuesday, not discovered on Sunday.**

Note the deliberate front-loading: CI and guards land by M2 and the production upgrade by M7.
That is not an accident — it is the defence against the most likely failure mode, which is the
avatar track eating the whole semester.

---

## 10. What the workload really is

Task estimates in the plan are given in hours per person. Summing the tasks assigned to each
role, every one of the four carries roughly **62–74 hours** of task work across eleven weeks —
the roles are deliberately balanced, and no role is a light one.

Budget realistically:

| | |
|---|---|
| Task work | ~6–7 hours/week |
| Weekly meeting | 1 hour |
| PR review + keeping the handoff current | ~1 hour/week |
| **Total** | **~8 hours/week** |

Plus **4–6 hours of onboarding before the first meeting** (section 13).

It is not evenly distributed. The upgrade window and the evaluation sessions in November are
heavier than the weeks around them, and the evaluation sessions require all four people.

### Definition of done

A task is done when **all** of these are true — not when the code works on your machine:

1. it does what the acceptance test says;
2. it is merged to `main` with CI green;
3. the documentation it invalidates has been updated;
4. if it touched production, the runbook exists **and someone else has read it.**

If you have not worked to that standard before, expect it to be the adjustment that costs you
the most in the first three weeks — and the habit you keep longest afterwards.

---

## 11. What you walk away with

**Concrete artefacts you can show someone:**

- A production system you upgraded across four releases, including a one-way database migration,
  with a written and rehearsed runbook
- A CI pipeline that blocks bad merges, including an architectural guard you wrote
- A hosted, authenticated web application replacing a laptop-only tool — verified with a real
  professor
- A real-time speech-to-speech application in its own container
- A measured, honestly-reported study with a go/no-go recommendation

**Skills that are hard to get anywhere else in a degree:**

- Working inside a hard architectural constraint and discovering it improves your design
- Inheriting an undocumented system and leaving it more legible than you found it
- Making an engineering recommendation on measured evidence, and being willing to recommend
  stopping
- Handing work to a successor team properly — the plan treats handoff as a deliverable, updated
  weekly

**And one thing worth stating plainly:** the users are real. If you break it, students in FIN
602 cannot practise. That pressure is the point — it is what makes this different from a project
graded on a demo.

---

## 12. Honest warnings

Do not join without reading these.

**This is maintenance work as much as invention.** A meaningful share of Job A is deleting
things, writing tests for logic that already works, and fixing documentation. If you only want
to build the shiny new thing, you will spend a lot of this semester unhappy.

**The avatar track may be killed in October.** Gate 1 is real. If latency lands at three
seconds, or the cost model does not close, or the consent question does not resolve cleanly, the
recommendation is to stop — and you will have to write that up as your deliverable. The plan
explicitly rates the risk of the avatar absorbing the semester as **severe**, and expected.

**The upgrade can genuinely damage production.** The `v0.10.0` config migration is one-way. This
is why the restore drill comes first, why it is rehearsed on restored data, why two people pair
on the window, and why the rollback is written before it opens. It is also why "pin to an old
release to avoid the work" is a forbidden pattern — that is exactly how the current situation
happened.

**You will be told "no" on the basis of policy.** If you propose embedding the avatar video in
the chat page, the answer is no, and it is written down in advance as a known risk. Working
inside a constraint you did not choose is part of the exercise.

**Some of this is unglamorous but non-negotiable.** Backups, CI, consent, the rollback plan and
the handoff document may never be cut, even when the semester slips — because those are the ones
that hurt someone else if you skip them. When there is not enough time, the plan says cut the
second avatar provider first, then the ambiguity-resolution work, then sample size.

**Documentation is a graded artefact here, not an afterthought.** There is a specific task for
bringing docs back into line, and one of the first things asked of you is to write down every
point where onboarding confused you — because your confusion is data.

---

## 13. How to decide, and what to do first

### Decide yes if

- You want production experience on a system with real users, not a demo
- You are interested in either careful infrastructure work **or** a genuinely uncertain research
  question — the team needs both kinds of person
- You can commit ~7–8 hours a week and be reliably present on Tuesdays
- You are comfortable saying "blocked" out loud on the day it happens

### Decide no if

- You want to build something from scratch with no inherited constraints
- A project that might conclude "this does not work" would feel like a wasted semester
- Tuesday afternoons are already contested
- You would find "which extension surface does this use?" on every pull request tedious rather
  than clarifying

### Before the first meeting — 15 September

This is task A1, done by **all four people**, roughly **4–6 hours**. Doing it early is also the
best way to test your own decision:

| # | Task | Done when |
|---|---|---|
| 0.1 | Read the project plan, the compatibility policy, the avatar track, and the existing handoff and architecture docs | You can name the three things that hurt, and the one rule |
| 0.2 | Get repository access and clone it | `git log` works |
| 0.3 | Run the stack locally: copy the env template, `docker compose up -d`, create an admin account, send one chat message | You have chatted with the system on your own machine |
| 0.4 | Upload a PDF to a knowledge base and ask a question about it | The answer references your document |
| 0.5 | Run the grading dashboard locally against the test stack | You have seen a transcript scored |
| 0.6 | **Write down every point where you got stuck** | This list is pre-work for the first meeting |

On 0.6 — the plan is explicit about why it matters:

> Your confusion is data. The current setup docs were written by people who already knew the
> system. Where you struggled is where the next team will struggle, and fixing that is a task.

If steps 0.3 through 0.5 feel interesting rather than tedious, that is your answer.

---

## 14. Where to read more

Everything in this brief traces to one of these. They are meant to be read together: the plan is
the plan, the policy is the constraint that shapes it, and the avatar track is the new work that
constraint turned out to improve.

| Document | What it gives you |
|---|---|
| [`PROJECT_PLAN.md`](PROJECT_PLAN.md) | The work plan: 43 task specifications, 4 roles, 11 dated meeting agendas |
| [`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) | The binding rule: 10 allowed surfaces, 8 forbidden patterns, repo audit |
| [`AVATAR_TRACK.md`](AVATAR_TRACK.md) | The research track: architecture, latency budget, providers, ethics, evaluation design |
| [`reference/system-facts.md`](reference/system-facts.md) | Every verified fact and figure, with its source — and honest answers to the hard questions |
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | Component diagram, data flow, configuration values |
| [`../SETUP.md`](../SETUP.md) | How to run it, environment variables, common errors |
| [`../HANDOFF.md`](../HANDOFF.md) | What was built, known bugs, decisions and why |
| [`slides/`](slides/) · [`script/`](script/) | The deck and its narration — the video this brief accompanies |

**A note on freshness:** this package makes claims about the repository and about third-party
products. Version numbers and audit findings should be re-checked against the tree; avatar
provider pricing and latency figures were checked in August 2026 and are moving quickly. Treat
every vendor latency and price claim as **something to measure, not a fact.**

---

*Prepared as a companion to the overview video. The video is the summary; this is the detail. If
a question is not answered here, it is answered in one of the four documents above — and if it
is answered in neither, that is worth raising at the first meeting.*
