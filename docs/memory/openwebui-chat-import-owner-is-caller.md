---
name: openwebui-chat-import-owner-is-caller
description: Open WebUI's chat import assigns the chat to whoever makes the request, so writing a spoken session into a student's history needs the student's own token; and grading drops a second assistant message in a row.
metadata:
  type: project
---

Verified 2026-11-22 by reading Open WebUI source at v0.8.12 (production) and v0.11.4 (pinned in
`docker-compose.yml`), and by running a sample through `grading_feature/backend/extract_chats.py`
with its API calls stubbed. Not yet tested against a running instance.

- `POST /api/v1/chats/import` with `{"chats": [{chat, meta, created_at, updated_at, …}]}` exists
  in both versions and stores the `chat` object verbatim, keeping the given timestamps. The same
  format is what Settings → Data Controls → Import Chats reads. In v0.11.4 it is gated by the
  `chat.import` user permission (`USER_PERMISSIONS_CHAT_IMPORT`, default on); v0.8.12 has no gate.
- **The imported chat is always owned by the authenticated caller.** There is no way for an admin
  or service key to import into another user's history through this endpoint.
- `extract_chats.py` reads only the flat `chat.messages` list and pairs each `user` message with
  the **next** `assistant` message; a second assistant message in a row is skipped with a warning.
  The web UI renders `chat.history.messages` instead, so an importer must write both.

**Why:** task B13 (transcript write-back) and any avatar or multi-persona session depend on this.
The obvious design — a backend service key that writes every session — silently puts every chat in
the service account's history, where grading attributes it to the wrong person. And in a
multi-speaker session, any persona that speaks right after another vanishes from grading.

**How to apply:** for write-back, obtain the *student's* credential (their API key, or launch
the avatar page from inside an Open WebUI session) and import as them. Merge consecutive messages
from the same side before import, labelling each speaker inside the content (`Tony: …`).
`liveavatar/shared/openwebui.ts` implements this format; verify a real import against the version
production actually runs before relying on it.
