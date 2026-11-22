---
name: ffc-main-env-is-openwebui-env-file
description: The repo-root .env is the open-webui container's env_file, so every variable added there reaches Open WebUI; Open WebUI reads GEMINI_API_KEY and PORT, so other services must not use those names in it.
metadata:
  type: project
---

`docker-compose.yml` gives the `open-webui` service `env_file: .env` (the repo-root file).
Everything in that file is passed into the Open WebUI container, not just the variables
SETUP.md §2a lists. Verified 2026-11-22 in Open WebUI source (v0.8.12 / v0.11.4):

- `GEMINI_API_KEY` is read by Open WebUI's `config.py`, so adding it quietly gives Open WebUI a
  Gemini connection;
- `PORT` is read by Open WebUI's `backend/start.sh` (`PORT="${PORT:-8080}"`), so adding it moves
  Open WebUI off 8080 and the proxy loses it.

**Why:** `liveavatar/` shares the main `.env` on purpose (it reuses `OPENAI_API_KEY`), and the
next tool that does the same will hit this. The failure is silent: nothing errors, Open WebUI
just behaves differently.

**How to apply:** anything that is not Open WebUI configuration goes into the main `.env` under
a prefixed name (`LIVEAVATAR_GEMINI_API_KEY`, `LIVEAVATAR_PORT`, `OPENWEBUI_*` for clients of
Open WebUI). Remember that secrets added there also sit in the Open WebUI container's
environment. Before adding a plain, generic name, grep Open WebUI's `config.py`, `env.py` and
`start.sh` for it.
