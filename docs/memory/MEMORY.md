# Memory index

One line per recorded fact. Read this at the start of a session; open a file when its
description is relevant. Format and conventions: [`README.md`](README.md).

- [Leaked key is not ours to revoke](leaked-key-not-ours-to-revoke.md) — a third party's API key was exposed in this public repo; Mark cannot revoke it, so stop assigning it as a task and treat it as closed on our side.
- [VM state differs from the repo](vm-state-differs-from-repo.md) — production runs OW 0.8.12 / Qdrant 1.17.1 while `main` pins 0.11.4 / 1.19.1; staging still runs the forbidden fork mounts; student-data copies are world-readable. Inspect the VM, not the compose file.
- [Secrets never in the repo](secrets-never-in-repo.md) — why scrubbing is not remediation, why a fix on an unmerged branch is not a fix, and why prevention is the only control we actually own.
- [Open WebUI chat import: owner is the caller](openwebui-chat-import-owner-is-caller.md) — `/api/v1/chats/import` puts the chat in the requesting account's history, so spoken-session write-back needs the student's own token; grading drops a second assistant message in a row, so merge multi-speaker runs.
- [Open WebUI 0.11 turns attached knowledge into a tool](openwebui-0-11-knowledge-is-a-tool.md) — 0.8.12 searches a client preset's knowledge every turn; 0.11.x lets the model decide unless `function_calling: legacy`. Client answers change silently on the A13 upgrade.
- [The main .env is Open WebUI's env_file](ffc-main-env-is-openwebui-env-file.md) — every variable there reaches the open-webui container; it reads `GEMINI_API_KEY` and `PORT`, so other services must use prefixed names.
