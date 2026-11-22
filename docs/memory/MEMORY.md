# Memory index

One line per recorded fact. Read this at the start of a session; open a file when its
description is relevant. Format and conventions: [`README.md`](README.md).

- [Leaked key is not ours to revoke](leaked-key-not-ours-to-revoke.md) — a third party's API key was exposed in this public repo; Mark cannot revoke it, so stop assigning it as a task and treat it as closed on our side.
- [Secrets never in the repo](secrets-never-in-repo.md) — why scrubbing is not remediation, why a fix on an unmerged branch is not a fix, and why prevention is the only control we actually own.
- [VM SSH needs project-level read](vm-ssh-access-needs-project-level-read.md) — instance-level osAdminLogin alone fails gcloud compute ssh on compute.projects.get; who has access and how it expires.
