---
name: secrets-never-in-repo
description: Scrubbing a committed credential is not remediation — rotation is; and CI secret scanning is what stops it recurring.
metadata:
  type: feedback
---

This repository has published a live API key once already (see
[[leaked-key-not-ours-to-revoke]]). The lesson is narrower and more useful than "don't commit
secrets":

1. **Deleting a secret from a file does not close the exposure.** It stays in git history, in
   every clone, and in every fork. The only remediation is rotating the credential at the
   provider. Treat the scrub as a cleanup step, never as the fix.
2. **You may not be able to remediate at all.** The key leaked here belonged to a third party —
   a student's own provider account — so nobody on this project could revoke it. When the
   credential is not yours, there is no fix available after the fact, only the consequence.
   That makes prevention the only control this project actually owns.
3. **The exposure window is from the commit, not from the discovery.** Assume the credential
   was harvested; automated scanners find public keys within minutes.
4. **A fix on an unmerged branch is not a fix.** The scrub here was committed on 2026-08-18 but
   sat on a side branch while the key stayed live on the public default branch for another day.
   For a security fix, "committed" is not the bar — "on `main`, and pushed" is.
5. **A secret nobody is looking for comes back.** The only durable defence is an automated
   check on every push.

**Why:** the first exposure here went unnoticed for roughly nine months, and the remaining
obligation became invisible the moment the file was cleaned up.

**How to apply:**

- `.env` files stay local and gitignored; CI credentials live in GitHub Actions secrets.
- Record *that* a credential exists or was exposed — never its value, not even a prefix.
- Land `gitleaks` in CI (task **A4** in
  [`capstone_fall2026/PROJECT_PLAN.md`](../../capstone_fall2026/PROJECT_PLAN.md)) so this cannot
  recur silently, and run it over history, not just the current tree.
- If you find a committed credential: establish **whose it is** first. If it is ours, rotate,
  then scrub, then confirm the scrub landed on the default branch. If it is not ours, record
  that fact and stop — see [[leaked-key-not-ours-to-revoke]].
