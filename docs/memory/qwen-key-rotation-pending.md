---
name: qwen-key-rotation-pending
description: A Qwen/DashScope API key was leaked in the public repo and still needs rotating at the provider — scrubbing the file did not close the exposure.
metadata:
  type: project
---

`scoring_page/backend/README.md` published a live Qwen/DashScope API key in plain text from
2025-10-31 (commit `f86f8d3`) until 2026-08-18, in the public repository
`github.com/mfedenia/FamilyFinanceChat`, which has at least one fork.

The removal was committed on 2026-08-18 to branch `cleanup/key-scrub-and-rag-citation-fix`,
**but that branch was never merged** — so the key stayed live in plaintext on `main` and on the
public `origin/main` until 2026-08-19, when the branch was finally merged (commit `911bcdc`).
Add roughly a further day to the exposure window on that account.

**Rotation at the provider was still outstanding as of 2026-08-19.** The key remains in this
repository's git history and in the fork's history, so removing it from the working tree does
not close the exposure — only rotation does.

**A scrub sitting on an unmerged branch protects nothing.** Verify the fix is on the default
branch, not merely committed somewhere.

**Nothing ever consumed the key.** `scoring_page/backend/server.js` reads only
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `MOCK_SCORER` and `PORT`; its sole LLM
dependency is the `openai` package. The `QWEN_*` variables were dead config and the "Qwen3-14B"
branding was stale — somebody prototyped against Qwen, moved to OpenAI, and never updated the
docs. All of it was removed on 2026-08-19, along with a hardcoded `OPENAI_API_KEY` export in
`scoring_page/run.sh` that both invited committed secrets and silently overrode `.env`.

**`scoring_page/` was deleted entirely on 2026-08-19**, so nothing in the working tree refers
to the key any more. Since nothing ever consumed it, **revoking it outright is simpler than
rotating it** — but the value is still in this repository's git history and in the fork's, so
the exposure stays open until it is revoked at the provider.

The DashScope provider path in `rag_bio_project/src/llm.py` is unrelated: it reads a different
variable (`DASHSCOPE_API_KEY`), defaults to `openai`, and was never wired to this key.

**Why:** once the key was scrubbed, nothing in the working tree reveals that a rotation is still
owed. This is exactly the kind of obligation that disappears silently at handover.

**How to apply:** confirm with Mark whether the key was rotated at the provider, or the
component retired, before treating this as closed. Tracked as **A5** in
[`capstone_fall2026/PROJECT_PLAN.md`](../../capstone_fall2026/PROJECT_PLAN.md) and **C-10** in
[`capstone_fall2026/COMPATIBILITY_POLICY.md`](../../capstone_fall2026/COMPATIBILITY_POLICY.md).
Delete this file once it is genuinely resolved. See also [[secrets-never-in-repo]].
