---
name: qwen-key-rotation-pending
description: A Qwen/DashScope API key was leaked in the public repo and still needs rotating at the provider — scrubbing the file did not close the exposure.
metadata:
  type: project
---

`scoring_page/backend/README.md` published a live Qwen/DashScope API key in plain text from
2025-10-31 (commit `f86f8d3`) until 2026-08-18, in the public repository
`github.com/mfedenia/FamilyFinanceChat`, which has at least one fork.

On 2026-08-18 the value was removed from the file on branch
`cleanup/key-scrub-and-rag-citation-fix`. **Rotation at the provider was still outstanding as of
2026-08-19.** The key remains in this repository's git history and in the fork's history, so
scrubbing the working tree does not close the exposure — only rotation does.

`scoring_page/` is an orphaned prototype: nothing else in the repo references it, and it is
superseded by `grading_feature/`. No running service depends on the key, so **retiring the
component outright is a reasonable alternative to rotating** — and retiring it is already
planned as task A21 in the work plan.

**Why:** once the key was scrubbed, nothing in the working tree reveals that a rotation is still
owed. This is exactly the kind of obligation that disappears silently at handover.

**How to apply:** confirm with Mark whether the key was rotated at the provider, or the
component retired, before treating this as closed. Tracked as **A5** in
[`capstone_fall2026/PROJECT_PLAN.md`](../../capstone_fall2026/PROJECT_PLAN.md) and **C-10** in
[`capstone_fall2026/COMPATIBILITY_POLICY.md`](../../capstone_fall2026/COMPATIBILITY_POLICY.md).
Delete this file once it is genuinely resolved. See also [[secrets-never-in-repo]].
