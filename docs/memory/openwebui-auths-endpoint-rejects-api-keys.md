---
name: openwebui-auths-endpoint-rejects-api-keys
description: From Open WebUI v0.13, GET /api/v1/auths/ answers only browser-session tokens (403 for API keys), so "who am I / am I admin" checks with an API key break.
metadata:
  type: reference
---

In Open WebUI v0.13.4, `GET /api/v1/auths/` depends on `get_human_user`, which rejects anything
but a browser session JWT with `403 A human session is required.` On v0.11.x the same call with
an API key returned the user. Found 2026-11-24 while wiring the avatar's chat write-back.

**Why:** scripts that check "is this key an admin?" through `/api/v1/auths/` stop working after
an upgrade to 0.13, with an error that looks like a permissions problem.

**How to apply:** with an API key, test an admin-only route instead (e.g.
`GET /api/v1/functions/list`), as `liveavatar-gpt-live-demos/scripts/openwebui-setup.mjs` now
does. The avatar app treats a 403 there as "valid key, identity unknown". Related:
[[openwebui-chat-import-owner-is-caller]].
