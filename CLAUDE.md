# CLAUDE.md — working instructions for this repository

Read this first. It is deliberately short; it points at the documents that carry the detail.

---

## What this is

FamilyFinanceChat is a live teaching platform for FIN 602. Students practise client-facing
financial advising against an AI role-player grounded in course documents; instructors review
transcripts through a grading dashboard that scores question quality on seven dimensions and an
Ability–Benevolence–Integrity (ABI) trust rubric.

It runs as ~8 Docker containers on a GCP VM, built **on top of** Open WebUI.

---

## The one rule

> **Open WebUI is a dependency, not a codebase we own.** Never modify its core. Every
> customisation must ride a published, documented, versioned extension surface — Functions,
> Pipelines, Tools, the public REST API, environment configuration, or a sidecar container.

The test, to apply in every code review:
*if upstream cut a release tomorrow and we ran it unchanged, would this still work?*

This is not style guidance. A previous architecture carried ~6,500 lines of forked vendor code
to obtain two features and froze the platform on one release for a year. Removing it was the
last team's main achievement.

**Full policy, including the ten allowed surfaces, the eight forbidden patterns, and the
outstanding audit items:** [`capstone_fall2026/COMPATIBILITY_POLICY.md`](capstone_fall2026/COMPATIBILITY_POLICY.md)

---

## Project memory

**Durable project knowledge lives in this repository, at
[`docs/memory/`](docs/memory/) — not in machine-local agent storage.**

- Read [`docs/memory/MEMORY.md`](docs/memory/MEMORY.md) at the start of a session; it is a
  one-line index of everything recorded.
- When you learn something durable and non-obvious, write it there as a new file and add a line
  to the index. Do **not** write it to home-directory auto-memory, where it is invisible to
  everyone else and lost with the machine.
- Conventions, format, and what does *not* belong: [`docs/memory/README.md`](docs/memory/README.md).

---

## Where things are

| What | Where |
|---|---|
| Architecture and data flow | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Running it, environment variables, common errors | [`SETUP.md`](SETUP.md) |
| What was built, known bugs, decisions and why | [`HANDOFF.md`](HANDOFF.md) |
| Current work plan, task specs, meeting calendar | [`capstone_fall2026/PROJECT_PLAN.md`](capstone_fall2026/PROJECT_PLAN.md) |
| Upstream compatibility policy and repo audit | [`capstone_fall2026/COMPATIBILITY_POLICY.md`](capstone_fall2026/COMPATIBILITY_POLICY.md) |
| Avatar research track | [`capstone_fall2026/AVATAR_TRACK.md`](capstone_fall2026/AVATAR_TRACK.md) |
| Verified facts, figures, and sources | [`capstone_fall2026/reference/system-facts.md`](capstone_fall2026/reference/system-facts.md) |
| Project memory | [`docs/memory/`](docs/memory/) |

---

## Conventions

- **Branch and pull request for everything.** No direct commits to `main`.
- Every PR states which Open WebUI extension surface it uses, and what happens on the next
  upstream release.
- **No secrets in the repository, ever.** `.env` stays local. A key was committed here once —
  see [`docs/memory/MEMORY.md`](docs/memory/MEMORY.md).
- Prefer editing an existing document over adding a new one. This repo has had documentation
  sprawl before.
- `.claude/` holds machine-local agent state and is gitignored, except `settings.json`. Nothing
  durable belongs there — put it in `docs/memory/`.
