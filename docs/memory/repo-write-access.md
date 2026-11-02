---
name: repo-write-access
description: Who can push and merge on GitHub as of 2026-10-25 (Mark trimmed the list on 2026-10-13), and why main is deliberately left without branch protection.
metadata:
  type: project
---

On 2026-10-13 Mark cut write access to `mfedenia/FamilyFinanceChat` down to the Fall 2026
team. As of 2026-10-25 GitHub lists:

| Name | GitHub username | Role |
|---|---|---|
| Mark Fedenia | `mfedenia` | admin |
| Allan Biju Gregory | `AllanBijuGregory` | write |
| Jason Jisen Li | `jason-jisen-li` | write |
| Lily Wang | `lnwang2-cmd` | write (invited 2026-10-13, accepted) |
| Mohammed Al Hinai | `m8oh` | write |
| Taehyun Park | `iamtaehyunpark` | write |
| Tongzhou Li | `tongzhou1215` | write |

Eleven earlier collaborators, from previous cohorts and people who did not join, were removed
that day. Their old commits and PRs stay in the history. Two profiles show no real name, so
match people by username, not by display name: `lnwang2-cmd` has none, and `tongzhou1215`
shows only "tong".

**`main` has no branch protection, and Mark decided on 2026-10-13 that it should stay that
way.** The branch-and-PR rule in `CLAUDE.md` is a team convention, not something GitHub
enforces. Any account in the table can merge, or push straight to `main`.

**Why:** a merge or a push to `main` that bypasses review is not blocked by GitHub, so it is
easy to assume the platform would have caught it. Before Lily was invited, her merge rights
could not be confirmed from the code, only from GitHub's collaborator API.

**How to apply:** check access with `gh api repos/mfedenia/FamilyFinanceChat/collaborators`
rather than trusting this table, and update the table when the roster changes. Don't propose
branch protection again unless the team's situation changes.
