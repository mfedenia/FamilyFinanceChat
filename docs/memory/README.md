# Project memory

Durable knowledge about this project that is **not derivable from the code**, kept in the
repository so it is versioned, reviewable, and shared by everyone — including AI assistants.

## Why it lives here and not in agent storage

Claude Code keeps per-project memory in a machine-local directory under the user's home folder
by default. That memory is invisible to teammates, absent from code review, and lost when the
machine changes or the project is cloned elsewhere.

On a project that hands over to a new team every semester, that is the wrong place for it.
Anything worth remembering is worth committing.

**So: `.claude/` stays sparse and machine-local; this directory is the real memory.**
[`CLAUDE.md`](../../CLAUDE.md) at the repository root points here.

## How it works

- **`MEMORY.md`** is the index — one line per fact, nothing more. Read it at the start of a
  session to see what is known.
- Each fact is **one file**, named in kebab-case after the thing it records.
- Link related facts with `[[file-name]]` (without the `.md`).

### File format

```markdown
---
name: <kebab-case-slug, matching the filename>
description: <one line — this is what someone reads to decide if the file is relevant>
metadata:
  type: project | reference | feedback
---

The fact itself, stated plainly. Convert relative dates to absolute ones.

**Why:** why this is worth recording — what goes wrong if nobody knows it.
**How to apply:** what a person or agent should actually do about it.
```

`type` values:

| Type | For |
|---|---|
| `project` | ongoing work, goals, constraints, and decisions not derivable from code or git history |
| `reference` | pointers to external resources — dashboards, tickets, provider consoles, URLs |
| `feedback` | how this team wants work done, including the reasoning behind a correction |

## What belongs here

Things that are true, non-obvious, and would cost someone real time to rediscover:

- decisions and the reasoning behind them, especially decisions *not* taken;
- obligations that are outstanding but invisible in the working tree (the leaked-key rotation is
  the canonical example — once scrubbed, nothing in the code reveals the job is unfinished);
- constraints that come from outside the codebase — the instructor's requirements, institutional
  policy, budget limits;
- traps that have already cost somebody a day.

## What does not belong here

- **Anything the repository already records.** Code structure, past fixes, git history, and
  anything in `ARCHITECTURE.md`, `SETUP.md`, or `HANDOFF.md`. If it belongs in one of those,
  put it there and do not duplicate it.
- **Secrets.** Record *that* a credential exists or was exposed, never its value.
- Session-specific chatter, status updates, or to-do lists. Those belong in the work plan or an
  issue tracker.

## Maintaining it

Before adding a file, check whether an existing one already covers it — update that instead of
creating a near-duplicate. **Delete facts that turn out to be wrong or that get resolved**; a
stale memory is worse than a missing one, because it will be trusted. If a memory names a file,
function, or flag, verify it still exists before acting on it.
