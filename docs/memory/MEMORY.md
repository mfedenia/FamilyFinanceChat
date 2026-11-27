# Memory index

One line per recorded fact. Read this at the start of a session; open a file when its
description is relevant. Format and conventions: [`README.md`](README.md).

- [Leaked key is not ours to revoke](leaked-key-not-ours-to-revoke.md) — a third party's API key was exposed in this public repo; Mark cannot revoke it, so stop assigning it as a task and treat it as closed on our side.
- [VM state differs from the repo](vm-state-differs-from-repo.md) — production runs OW 0.8.12 / Qdrant 1.17.1 while `main` pins 0.11.4 / 1.19.1; staging still runs the forbidden fork mounts; student-data copies are world-readable. Inspect the VM, not the compose file.
- [Secrets never in the repo](secrets-never-in-repo.md) — why scrubbing is not remediation, why a fix on an unmerged branch is not a fix, and why prevention is the only control we actually own.
- [Repo write access](repo-write-access.md) — who can push and merge since Mark trimmed the list on 2026-10-13 (six collaborators plus Mark), and why `main` deliberately has no branch protection.
- [Agents do not merge PRs](agents-do-not-merge-prs.md) — Claude Code's auto mode refuses to merge an unreviewed PR; a person clicks Merge, and the agent pulls `main` afterwards.
- [VM SSH needs project-level read](vm-ssh-access-needs-project-level-read.md) — instance-level osAdminLogin alone fails gcloud compute ssh on compute.projects.get; who has access and how it expires.
- [2027 knowledge base setup](2027-knowledge-base-setup.md) — IDs of the private 2027 KBs and models, the instance-wide RAG settings changed for them on 2026-11-26, and why the DB config overrides env.
- [2027 Beckham case is current](beckham-case-2027-is-current.md) — ignore 2026 material; where the 2027 sources (Drive Knowledgebase) and the master system prompt (Drive Prompts) live.
- [OWUI 0.13: knowledge is a tool; reranker is slow](owui-013-knowledge-is-a-tool-and-reranker-is-slow.md) — the model decides whether to search attached knowledge (native function calling); the CPU reranker made each search 30–160 s, so reranking moved to Cohere (~8 s); how to run the answer-key test.
