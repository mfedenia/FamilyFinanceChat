---
name: openwebui-chat-import-owner-is-caller
description: Open WebUI files a chat created through the API (chats/new, chats/import) under whoever calls it, so saving an avatar meeting for grading needs the student's own token, not an admin key.
metadata:
  type: project
---

Checked in Open WebUI v0.13.4 source (`routers/chats.py`), 2026-11-24, and the same in v0.11.4
(AVATAR_TRACK AV4): `POST /api/v1/chats/new` and `POST /api/v1/chats/import` store the chat
under the **calling user** and overwrite every message's `user_id` with the caller. There is no
admin route that creates a chat for another user.

The grading extraction (`grading_feature/backend/extract_chats.py`) groups chats by owner, so a
meeting saved with an admin key would be graded as the admin's.

**Why:** the obvious fix — "save it with the admin key we already have" — silently files every
student's meeting under the admin.

**How to apply:** the couple avatar app (`liveavatar-gpt-live-demos`) gets the student's token
from Open WebUI's "Talk to the avatar" button: the button's JavaScript runs in the Open WebUI
page, reads `localStorage.token`, and passes it to the avatar page in the URL fragment
(`#owui_token=…`). The server keeps it in memory for that meeting only. Any other write-back
path needs the same: the student's credential. Related: [[openwebui-auths-endpoint-rejects-api-keys]],
[[multi-persona-switching-requirement]].
