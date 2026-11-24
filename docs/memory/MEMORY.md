# Memory index

One line per recorded fact. Read this at the start of a session; open a file when its
description is relevant. Format and conventions: [`README.md`](README.md).

- [Leaked key is not ours to revoke](leaked-key-not-ours-to-revoke.md) — a third party's API key was exposed in this public repo; Mark cannot revoke it, so stop assigning it as a task and treat it as closed on our side.
- [VM state differs from the repo](vm-state-differs-from-repo.md) — production runs OW 0.8.12 / Qdrant 1.17.1 while `main` pins 0.11.4 / 1.19.1; staging still runs the forbidden fork mounts; student-data copies are world-readable. Inspect the VM, not the compose file.
- [Secrets never in the repo](secrets-never-in-repo.md) — why scrubbing is not remediation, why a fix on an unmerged branch is not a fix, and why prevention is the only control we actually own.
- [Chat import owner is the caller](openwebui-chat-import-owner-is-caller.md) — Open WebUI files API-created chats under whoever calls; saving avatar meetings for grading needs the student's own token (handed over by the "Talk to the avatar" button).
- [auths/ rejects API keys on 0.13](openwebui-auths-endpoint-rejects-api-keys.md) — from v0.13 `GET /api/v1/auths/` is browser-session only (403 for API keys); check admin rights on an admin-only route instead.
