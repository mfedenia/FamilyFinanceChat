---
name: beckham-case-2027-is-current
description: The 2027 Beckham case is the only current one; where its source files and its system prompt live (Drive), and which 2026 material to ignore.
metadata:
  type: project
---

From 2026-11-26 the **2027 Beckham case is the current case** for this repo and the role-play.
Treat everything 2026 (the "2026" models, the 2026 knowledge bases, `beckham-bios.pdf`,
`FIN602_ Client Data (2026).pdf`, the 2026 benefit extracts) as last year's archive. Don't
use it as a source, and don't "fix" 2027 to match it.

**Sources (Mark's Drive, UW shared drive, folder `Knowledgebase`):**
https://drive.google.com/drive/folders/1DUdVdCNsidS6z-hrb7ZVgKd6Q-FNV_25
- `FIN602: Client Data (2027)` (Google Doc, v2026.11.26.01) is the authority for every number.
- `beckham-family-bios-2027.llm.md` holds the biographies, written as retrieval records. Its
  "HOW TO USE" records set the rules: the client data wins on numbers, ages don't advance, and
  sensitive subjects are disclosed only when asked.
- Rockwell 2027 benefits guide and three WI ETF 2027 extracts (`.clean.md`).
- `client-data-2026-vs-2027.pdf` lists what changed, and `Sources/README` says how each file is
  refreshed. The scripts are in the `investment-module` repo.

**System prompt:** Drive folder `Prompts`
https://drive.google.com/drive/folders/1oz5F13ia954BUJTOPJtOaEj2fLQ2UyNL, Doc "FIN602 Beckham
role-play system prompt (2027)". It holds the rules plus a CLIENT FACTS block built from the 2027
sheet. The Doc is the master copy: edit it there, then push it to the `2027` and
`2027-audio-21-mini` models. v3 (2026-11-27) added the search-first rule. The Docs API is off in the
service account's project, but the SA can replace the Doc's text with a Drive `files.update` media
upload (`text/plain`, `supportsAllDrives`).

**What's loaded in Open WebUI** (2026-11-26): KB "2027 personal data" holds the bios and the
Client Data (exported to markdown with the images stripped). KB "2027 Employee Benefits" holds
the Rockwell guide and the three ETF extracts. All of them are still private.

**Open decisions for Mark:** see [[2027-knowledge-base-setup]] and the PR that added this note.
