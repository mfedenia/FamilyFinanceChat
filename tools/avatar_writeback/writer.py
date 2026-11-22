"""Avatar-to-Open WebUI transcript persistence for FamilyFinanceChat (B13).

Works with the Open WebUI v0.11.4 public chat APIs. No database access,
no Open WebUI core modifications, and no separate transcript database.

Important: each writer uses the *student's own* authenticated API credential.
An admin/service account's key creates a chat owned by the admin instead of
that student and will break grading attribution.
"""
from __future__ import annotations

import asyncio
import copy
import json
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from typing import Any


class WritebackError(RuntimeError):
    """Open WebUI chat persistence or attribution verification failed."""


class JsonAPI:
    def __init__(self, base_url: str, bearer_token: str, timeout: float = 20):
        if not (base_url.startswith("http://") or base_url.startswith("https://")):
            raise ValueError("base_url must be http(s)")
        if not bearer_token or bearer_token.strip() in {"REPLACE_ME", "YOUR_API_KEY"}:
            raise ValueError("A real, student-scoped Open WebUI bearer token is required")
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Accept": "application/json",
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(self.base_url + path, method=method, data=data, headers=headers)
        try:
            with urlopen(req, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            # Do not print response body: it may contain sensitive chat content.
            raise WritebackError(f"Open WebUI {method} {path}: HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise WritebackError(f"Open WebUI {method} {path}: connection failure") from exc
        except (ValueError, TypeError) as exc:
            raise WritebackError(f"Open WebUI {method} {path}: invalid JSON response") from exc
        if not isinstance(payload, dict):
            raise WritebackError(f"Open WebUI {method} {path}: expected JSON object")
        return payload


class AvatarWriteback:
    """Saves complete STT/AI turns to a student-owned Open WebUI chat.

    Create one instance per student/session. Retain ``chat_id`` in the avatar
    session state, and use a unique stable ``turn_id`` per spoken exchange.
    Resubmitting the same turn_id does not append it again.

    ``api`` may be a JsonAPI or a fake implementation for isolated tests.
    """

    def __init__(self, api: JsonAPI, *, model: str, expected_user_id: str):
        if not model.strip():
            raise ValueError("model must be the model used by the avatar")
        if not expected_user_id.strip():
            raise ValueError("expected_user_id is required for attribution verification")
        self.api = api
        self.model = model
        self.expected_user_id = expected_user_id

    def _check_owner(self, record: dict[str, Any]) -> None:
        owner = record.get("user_id")
        if owner is None:
            # A chat GET response should include the owner ID. Fail closed.
            raise WritebackError("Open WebUI response omitted user_id; cannot verify student attribution")
        if str(owner) != self.expected_user_id:
            raise WritebackError("Chat owner does not match expected student; refusing write-back")

    def start_session(self, *, title: str = "Avatar advising practice") -> str:
        """Creates an empty chat under the student's authenticated identity."""
        record = self.api.request("POST", "/api/v1/chats/new", {
            "chat": {
                "title": title,
                "models": [self.model],
                "messages": [],
                "history": {"currentId": None, "messages": {}},
            }
        })
        self._check_owner(record)
        chat_id = record.get("id")
        if not isinstance(chat_id, str) or not chat_id:
            raise WritebackError("Chat creation did not return an ID")
        return chat_id

    def get_chat(self, chat_id: str) -> dict[str, Any]:
        record = self.api.request("GET", f"/api/v1/chats/{chat_id}")
        self._check_owner(record)
        chat = record.get("chat")
        if not isinstance(chat, dict):
            raise WritebackError("Chat record has no JSON chat payload")
        return record

    def append_turn(
        self,
        chat_id: str,
        *,
        turn_id: str,
        student_text: str,
        avatar_text: str,
        timestamp: int | None = None,
    ) -> bool:
        """Append one finalized turn. Return False when already persisted.

        Intended to be called after STT and avatar replies are finalized;
        never write partial transcripts. Never accept a user_id from the browser
        in place of the authenticated account bound to the API token.
        """
        if not turn_id or not turn_id.strip():
            raise ValueError("turn_id must be stable and nonempty")
        if not student_text or not student_text.strip():
            raise ValueError("student_text must not be blank")
        if not avatar_text or not avatar_text.strip():
            raise ValueError("avatar_text must not be blank")
        if not chat_id.strip():
            raise ValueError("chat_id must not be blank")

        # Deterministic IDs allow safe retries of POST update when the result
        # of a previous request is unknown. Different chats cannot collide.
        student_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ffc-avatar:{chat_id}:{turn_id}:student"))
        reply_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ffc-avatar:{chat_id}:{turn_id}:avatar"))
        record = self.get_chat(chat_id)
        chat = copy.deepcopy(record["chat"])
        messages = chat.get("messages", [])
        history = chat.get("history", {"currentId": None, "messages": {}})
        if not isinstance(messages, list) or not isinstance(history, dict):
            raise WritebackError("Unexpected chat message/history shape")
        by_id = history.get("messages", {})
        if not isinstance(by_id, dict):
            raise WritebackError("Unexpected indexed chat history")
        if student_id in by_id and reply_id in by_id:
            return False
        if student_id in by_id or reply_id in by_id:
            raise WritebackError("Incomplete prior turn; refusing to duplicate it")

        now = int(time.time()) if timestamp is None else int(timestamp)
        previous_id = history.get("currentId")
        if not previous_id and messages:
            previous_id = messages[-1].get("id")
        student_msg = {
            "id": student_id,
            "role": "user",
            "content": student_text.strip(),
            "timestamp": now,
            "models": [self.model],
            "childrenIds": [reply_id],
        }
        if previous_id:
            student_msg["parentId"] = previous_id
        assistant_msg = {
            "id": reply_id,
            "role": "assistant",
            "content": avatar_text.strip(),
            "timestamp": now,
            "parentId": student_id,
            "childrenIds": [],
            "model": self.model,
            "modelName": self.model,
            "modelIdx": 0,
            "done": True,
        }

        # Keep both the legacy flat list used by extract_chats.py AND the
        # indexed history tree used by Open WebUI's UI.
        if previous_id and previous_id in by_id:
            old = dict(by_id[previous_id])
            old["childrenIds"] = list(dict.fromkeys([*old.get("childrenIds", []), student_id]))
            by_id[previous_id] = old
        elif previous_id and messages:
            # Avoid silent history corruption on a malformed existing chat.
            raise WritebackError("Previous message is not in indexed history")

        by_id[student_id] = student_msg
        by_id[reply_id] = assistant_msg
        history["messages"] = by_id
        history["currentId"] = reply_id
        messages.extend([student_msg, assistant_msg])
        chat["messages"] = messages
        chat["history"] = history
        chat.setdefault("models", [self.model])

        self.api.request("POST", f"/api/v1/chats/{chat_id}", {"chat": chat})
        # Read-after-write verification, not just HTTP 200.
        stored = self.get_chat(chat_id)["chat"]
        saved = stored.get("messages", [])
        expected = [("user", student_id, student_text.strip()), ("assistant", reply_id, avatar_text.strip())]
        for role, message_id, content in expected:
            if not any(isinstance(m, dict) and m.get("role") == role and m.get("id") == message_id and m.get("content") == content for m in saved):
                raise WritebackError("Write-back response succeeded but transcript did not persist")
        return True

    async def append_turn_async(self, chat_id: str, **kwargs: Any) -> bool:
        """Use from an async avatar server without blocking its event loop."""
        return await asyncio.to_thread(self.append_turn, chat_id, **kwargs)
