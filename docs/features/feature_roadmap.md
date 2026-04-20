# Feature Roadmap
**FamilyFinanceChat — Post v0.6.41 → v0.8.12 Upgrade**

This document is the result of investigating what changed in our recent OpenWebUI upgrade and asking what we should actually build next. It's meant to be a living reference for anyone on the team, technical or not. The upgrade was a big jump and it unlocked a lot that we weren't able to do before. This is where we track what to do with that.

---

## What the Upgrade Unlocked

Before getting into the roadmap it's worth naming the capabilities that directly changed what's now possible for us.

**Skills (v0.8.0)** are reusable instruction sets attachable to specific models and manageable from the UI. No config file changes needed to update what frameworks the model knows about.

**Analytics (v0.8.0)** gives admins visibility into model usage, token consumption, and user activity per conversation.

**Message queuing (v0.8.0)** lets users send follow-up messages while a response is still generating. 

**The `/ready` endpoint (v0.8.9)** returns 200 only once the database and Redis are fully up. Useful for monitoring.

**OpenTelemetry system metrics (v0.8.9)** pipes CPU, memory, and garbage collection data through our existing monitoring stack.

---

## Phase 1 — Fix What's Broken

These are immediate. Mostly one-person tasks that make the platform more reliable before we build anything new.

**Healthcheck alerting.** Wire the `/ready` endpoint into GCP uptime monitoring and set an alert to the primary maintainer. The app went down recently and we found out passively. For a course with deadlines that's not acceptable. Add a basic status page so students know when the system is down rather than assuming they're doing something wrong. This should take under an hour on the VM.

---

## Phase 2 — Operational Foundations

Slightly more work but these make the platform reliable enough to hand off to course staff and eventually scale to other courses.

**Host the grading dashboard.** Right now professors have to SSH into the GCP VM to run the grading dashboard. That's a real barrier and a single point of failure. Containerize the grading app, add it to the docker-compose stack, put it behind Nginx with basic auth, and give professors a URL they can just open. Estimated effort is one to two days of engineering.

**OTel system metrics.** We already have Prometheus and Grafana running. The upgrade added CPU and memory metrics through the existing pipeline, we just need to turn them on. Add a Grafana panel for those metrics and set an alert on memory as a leading indicator before the container goes down.

---

## Phase 3 — New Features

This is where the upgrade directly improves the student experience.

**Skills for financial frameworks.** The pedagogical frameworks in FIN 602 should live as Skills attached to the model, not buried in system prompt config. A professor should be able to update what frameworks the model references week to week from the UI with no dev help. Create one Skill per major framework, attach them to the FIN 602 model, and document the process for the professor. Essentially zero engineering work.

---

## Phase 4 — Bigger Bets

These need more design thinking and represent where the platform goes longer term.

**Multi-tenant packaging.** The long-term goal is expanding to other courses and other universities. Right now everything is single-tenant. Building a containerized deployment recipe per course is the right architectural move if the platform grows beyond FIN 602. This is a real engineering project, probably one to two weeks, but it's the right time to start thinking about it now that the codebase is decoupled from OpenWebUI upstream.

---

## What We're Not Doing

**Full CI/CD** is valuable eventually but not the bottleneck right now.

**Adaptive per-student difficulty** requires the grading pipeline to be real-time. It isn't. Revisit after Phase 2.

---

If you're doing R&D on something not listed here, add it to this doc. What matters is capturing the reasoning, not just the idea. Last updated April 2026 following the v0.6.41 → v0.8.12 upgrade investigation.
