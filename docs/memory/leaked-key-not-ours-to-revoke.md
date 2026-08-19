---
name: leaked-key-not-ours-to-revoke
description: A Qwen/DashScope key was leaked in this public repo, but it belongs to a third party — Mark cannot revoke it, so treat the exposure as closed on our side and stop assigning it as a task.
metadata:
  type: project
---

`scoring_page/backend/README.md` published a live Qwen/DashScope API key in plain text from
2025-10-31 (commit `f86f8d3`) until 2026-08-19, in the public repository
`github.com/mfedenia/FamilyFinanceChat`, which has at least one fork.

**The key is not Mark's and not the project's.** It belongs to whoever created that DashScope
account — most likely a student on an earlier team. Confirmed with Mark on 2026-08-19:
**he has no ability to revoke or rotate it.**

That settles what happens next: **nothing, on our side.** Do not put "rotate the key" on a
sprint board, a handoff list, or a risk register again. It is not an action this project can
take, and carrying it as an open item makes the backlog dishonest.

## What was actually done

- The value was removed from the file on 2026-08-18, but that commit sat on the unmerged branch
  `cleanup/key-scrub-and-rag-citation-fix`, so it stayed live on the public default branch until
  the merge on 2026-08-19 (`911bcdc`).
- `scoring_page/` was **deleted entirely** on 2026-08-19 (`139a7a4`). Nothing consumed the key:
  `server.js` only ever read `OPENAI_API_KEY`, and its sole LLM dependency was the `openai`
  package. The `QWEN_*` variables were dead config and the "Qwen3-14B" branding was stale.
- The value remains in this repository's git history and in the fork's history. Deleting files
  does not remove it, and history rewriting would not reach the fork.

## What this means for risk

The residual exposure is to **the account owner's billing and quota**, not to this platform. No
FamilyFinanceChat service authenticates with it, and no student data is reachable through it. If
the owner is ever identified, telling them is a courtesy worth doing — but it is not a
prerequisite for anything here.

**Why:** without this note, every fresh reading of the repo re-derives "there is a leaked key,
someone must rotate it" and assigns it to Mark, who cannot act on it. That has already happened
more than once.

**How to apply:** treat the exposure as closed from this project's side. The transferable
control is prevention, not remediation — see [[secrets-never-in-repo]] and land `gitleaks` in
CI (task **A4**). If someone raises the key as an open risk, point them here.
