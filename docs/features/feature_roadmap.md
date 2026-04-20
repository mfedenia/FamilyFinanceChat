# Future Feature Development
**FamilyFinanceChat — Post v0.6.41 → v0.8.12 Upgrade**

This document captures the product thinking and technical investigation that followed our OpenWebUI upgrade. It's meant to be a living reference — part roadmap, part rationale — for anyone on the team (dev or non-dev) who wants to understand where the platform is headed and why.

The upgrade from v0.6.41 to v0.8.12 was a significant jump. It unlocked a lot of new OpenWebUI capabilities that we weren't able to use before, and it also gave us a cleaner codebase to build on top of (see the decoupling work in `legacy/`). This document is the result of investigating those changes and asking: what should we actually build next?

---

## Context: What Changed in the Upgrade

A few headline capabilities that landed between our old and new versions that directly inform what's now possible:

- **Native function calling (v0.7.0)** — models can now autonomously invoke tools mid-conversation rather than needing manual file attachment or pre-loaded context
- **Skills (v0.8.0)** — reusable instruction sets that can be attached to specific models, updated from the UI without touching config files
- **Analytics dashboard (v0.8.0)** — admin-level visibility into model usage, token consumption, and user activity
- **Prompt version control (v0.8.0)** — full history, diffing, and rollback for prompts
- **Message queuing (v0.8.0)** — users can queue follow-up messages while a response is generating
- **Per-user resource sharing (v0.8.0)** — knowledge bases and prompts can now be shared to individual users, not just groups
- **`/ready` healthcheck endpoint (v0.8.9)** — returns 200 only after DB and Redis are fully up, useful for monitoring
- **OpenTelemetry system metrics (v0.8.9)** — CPU, memory, GC metrics now flow through the existing OTel pipeline

---

## The Roadmap

Features are ordered roughly chronologically — things we should do now first, bigger bets later. Each entry includes who it affects, how hard it is, and a concrete path to get it done.

---

### Phase 1 — Fix What's Broken and Add Visibility
*These are immediate. Low effort, high operational value. Do these before building anything new.*

---

#### 1.1 Healthcheck Alerting via `/ready`

**Who it helps:** Everyone. Students, professors, maintainers.

**Why it matters:** The app went down and we found out passively. For a course with assignment deadlines that's a real problem. The upgrade added a `/ready` endpoint that only returns 200 once the database and Redis are fully up — we should be using it.

**What to do:**
- Wire `/ready` into GCP uptime monitoring with an alert to the primary maintainer
- Set a simple status page or notification so students know when the system is down rather than assuming the chatbot is broken or they're doing something wrong
- Estimated time: under an hour once you're on the VM

---

#### 1.2 Verify the 7-Message Bug is Fixed

**Who it helps:** Students.

**Why it matters:** The known issue where sending 7+ messages caused the chat to get stuck was a real disruption mid-session. Message queuing in v0.8.0 likely resolved this upstream but we haven't confirmed it.

**What to do:**
- Run a manual test: open a fresh chat, send 10+ messages in sequence, verify no hanging
- If fixed, remove it from the known issues section of the main README
- If it's still happening it's now a custom issue specific to our deployment, worth isolating and filing properly

---

#### 1.3 Enable Streaming Output

**Who it helps:** Students (perceived performance).

**Why it matters:** Model latency is flagged as P0 in the existing roadmap. Streaming makes responses feel instant even if total generation time is the same — users see the first tokens immediately rather than waiting for the full response. This matters a lot during a timed advising simulation where dead air feels like something broke.

**What to do:**
- Verify streaming is enabled in the model configuration (it may already be on)
- If latency is still a problem after enabling streaming, evaluate whether a faster/smaller model could handle the scenario simulation without quality loss

---

### Phase 2 — Operational Foundations
*Slightly more work but these make the platform reliable enough to scale to more courses.*

---

#### 2.1 Host the Grading Dashboard

**Who it helps:** Professors and TAs significantly.

**Why it matters:** Right now accessing the grading dashboard requires SSH-ing into the GCP VM via VS Code Remote. That's a real barrier — if the one person who knows how to do this is unavailable, grading is blocked. This is the single highest-leverage improvement for making the platform usable by non-technical course staff.

**What to do:**
- Containerize the grading app and add it to the existing `docker-compose` stack
- Put it behind an Nginx reverse proxy with basic auth (username/password is fine for now)
- Give professors a URL they can just open in a browser
- Estimated effort: 1-2 days of engineering work

---

#### 2.2 Upgrade Monitoring with OTel System Metrics

**Who it helps:** Maintainers.

**Why it matters:** We already have a Prometheus + Grafana monitoring stack. The upgrade added CPU, memory, and garbage collection metrics through the existing OpenTelemetry pipeline — we just need to configure it. This gives us early warning on resource pressure before it causes an outage.

**What to do:**
- Enable system metrics in the OTel config
- Add a Grafana dashboard panel for CPU and memory alongside the existing chat metrics
- Set an alert threshold for memory usage as a leading indicator of container instability

---

#### 2.3 Prompt Version Control for Scenario Management

**Who it helps:** Whoever owns the family scenario content (likely professor or lead dev).

**Why it matters:** Family scenarios will evolve across semesters. Right now there's no history of what changed, why, or what an earlier version looked like. If a scenario update makes conversations worse there's no easy way to roll back.

**What to do:**
- Migrate all existing family scenario prompts into the OpenWebUI Prompts workspace if they aren't already there
- Start using commit messages when making updates ("updated Chen family income to reflect 2026 tax brackets")
- This is a process change, not an engineering one — just needs to become a habit

---

### Phase 3 — New Feature Development
*This is where the upgrade really opens up. Higher effort but directly improves the student experience.*

---

#### 3.1 Skills for Financial Frameworks

**Who it helps:** Professors primarily, students indirectly.

**Why it matters:** Course frameworks — how to assess a client's risk tolerance, how to structure a wealth management recommendation — are the pedagogical core of FIN 602. Right now injecting that context into the model likely requires touching config files or system prompts. Skills let a professor update the active framework week-to-week from the OpenWebUI UI with no technical help.

**What to do:**
- Create one Skill per major course framework used in FIN 602
- Attach relevant Skills to the FIN 602 model in the model editor
- Document the process for professors so they can manage it themselves
- This is essentially zero engineering work — it's a content and process design task

---

#### 3.2 Native Tool Calling for Scenario Context

**Who it helps:** Students. This one directly improves the advising simulation quality.

**Why it matters:** Right now students likely need files pre-loaded or manually attached for the model to know details about the family scenario. With native function calling the model can autonomously fetch family profile data, scenario context, or course reference material mid-conversation — much closer to how a real advising session would work where you pull up a client file during the meeting.

**What to do:**
- Define 2-3 tools: fetch family profile, retrieve scenario context, pull relevant course framework
- Register them as OpenWebUI tools and attach them to the FIN 602 model
- Enable native function calling mode in Chat Controls for the model
- Test thoroughly — native function calling can be unpredictable with models that aren't well-suited for it, so model selection matters here

---

#### 3.3 Per-User Scenario Assignment

**Who it helps:** Professors running differentiated assignments.

**Why it matters:** Right now all students probably get access to the same scenarios. v0.8.0 added per-user resource sharing, which means you can now assign specific knowledge bases or prompts to individual students without creating separate user groups for each assignment variant.

**What to do:**
- Decide on the assignment model: do all students get all scenarios, or are scenarios assigned per student or per cohort?
- Once decided, set up sharing accordingly in the admin panel
- No engineering required — this is an admin configuration task

---

### Phase 4 — Bigger Bets
*These require more design thinking and effort but represent the long-term direction of the platform.*

---

#### 4.1 Scenario Difficulty Progression

**Who it helps:** Students and professors — this is a pedagogical feature.

**Why it matters:** Students likely get the same scenario complexity regardless of where they are in the course. A tiered system — a straightforward young couple in week 3, a complex multi-generational household with business assets in week 10 — mirrors how real financial advising training programs are structured and gives students a sense of progression.

**What to do:**
- Work with the professor to design 3 tiers of scenario complexity mapped to course weeks
- Implement as separate Skills or prompt variants (no new infrastructure needed)
- Start with a semester-wide model where all students advance together before building adaptive per-student logic
- The harder version of this (adaptive based on grading performance) is interesting but needs the grading pipeline to be more real-time than it currently is

---

#### 4.2 Analytics-Informed Grading Signals

**Who it helps:** Professors and TAs.

**Why it matters:** The new analytics dashboard tracks token usage and conversation length per user. For FIN 602 this is a proxy signal for engagement — a student who has a 40-message high-token conversation is probably having a richer advising session than one who sent 3 messages. This doesn't replace qualitative grading but it adds a lightweight signal that's available without running the full extraction pipeline.

**What to do:**
- Enable the analytics dashboard for admin/professor accounts (costs nothing, it's already there)
- Use it as a supplemental signal — flag students with unusually low engagement for follow-up
- Longer term: explore whether the analytics API can feed directly into the grading dashboard so professors see engagement data alongside conversation quality scores in one view

---

#### 4.3 Multi-Tenant Packaging for Other Courses

**Who it helps:** The platform long-term. This is the expansion play.

**Why it matters:** The README already calls out expanding to other courses and eventually other universities as the long-term goal. Right now the deployment is fully single-tenant — one instance, one course, one set of scenarios. Making it easy to spin up an isolated instance per course is what enables that expansion without everything sharing the same database and config.

**What to do:**
- Design a containerized deployment recipe (Docker image + docker-compose profiles) for per-course instances
- Each instance gets its own scenarios, models, and grading pipeline
- This is a meaningful engineering project — probably 1-2 weeks — but it's the right architecture if the platform is going to grow beyond FIN 602

---

## What We're Not Doing (and Why)

A few things that came up during this investigation that aren't in the roadmap:

**Full CI/CD pipeline** — valuable eventually but not the bottleneck right now. Manual deploys are fine while the team is small.

**Adaptive per-student difficulty** — interesting but requires the grading pipeline to be real-time, which it isn't. Revisit after Phase 2 is done.

**Replacing the grading extraction pipeline with analytics** — the built-in analytics gives engagement signals but not conversation quality. The extraction pipeline is still needed for actual assessment.

---

## How to Contribute to This Document

If you're exploring a new feature or doing R&D on something not listed here, add it. The format is loose on purpose — what matters is capturing the reasoning (why does this add value, who does it help, what's the path forward) not just the idea. A bullet point with no rationale isn't useful to anyone picking this up later.

Last updated: April 2026 following v0.6.41 → v0.8.12 upgrade investigation.