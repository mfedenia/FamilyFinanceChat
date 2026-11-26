---
name: agents-do-not-merge-prs
description: Claude Code's auto mode refuses to merge a PR that has no review; a person merges, and the agent syncs local main afterwards.
metadata:
  type: feedback
---

On 2026-10-13 Claude Code was asked to merge PR #8 with `gh pr merge`. The auto-mode safety
classifier refused it as "Merge Without Review", because the PR had no approving review.

**Why:** with no branch protection on `main` (see [[repo-write-access]]), this classifier is
the only automated thing between an agent and an unreviewed merge. Working around it, for
example through the GitHub MCP merge tool, defeats the point of it.

**How to apply:** an agent prepares the PR, resolves its conflicts, and confirms it is
mergeable. A person then clicks **Merge pull request** on GitHub, or adds an explicit
permission rule in Claude Code settings. After the merge, the agent pulls `main`.
