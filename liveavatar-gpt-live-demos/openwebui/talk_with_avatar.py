"""
title: Talk with avatar
author: FamilyFinanceChat
version: 0.2.0
required_open_webui_version: 0.6.0
description: Opens the spoken avatar meeting for the family you are chatting with.
icon_url: data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cmVjdCB4PSI5IiB5PSIyIiB3aWR0aD0iNiIgaGVpZ2h0PSIxMiIgcng9IjMiLz48cGF0aCBkPSJNNSAxMWE3IDcgMCAwIDAgMTQgMCIvPjxwYXRoIGQ9Ik0xMiAxOHY0Ii8+PC9zdmc+
"""

# Open WebUI Action Function (extension surface: Functions). Attached to the
# family model presets via meta.actionIds by scripts/openwebui-setup.mjs, so
# the button shows under replies from those models only.
#
# It opens the avatar app (a separate web app, not part of Open WebUI) at
# ?family=<id>, where <id> is the model id minus the "family-" prefix. The
# tab is opened by the chat page through the documented "execute" event; if
# the browser blocks the pop-up, a small link appears instead. Like the
# banner, it hands the signed-in student's token to the avatar page in the
# URL fragment, so the meeting is saved as the student's chat when it ends.

import json
from urllib.parse import quote

from pydantic import BaseModel, Field


class Action:
    class Valves(BaseModel):
        AVATAR_APP_URL: str = Field(
            default="http://localhost:5173",
            description="Where the avatar app is served, as the student's browser sees it.",
        )
        MODEL_PREFIX: str = Field(
            default="family-",
            description="Family model ids are this prefix plus the avatar app's family id.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def action(
        self,
        body: dict,
        __user__=None,
        __model__=None,
        __event_emitter__=None,
        __event_call__=None,
    ):
        model_id = (body or {}).get("model") or (__model__ or {}).get("id") or ""
        prefix = self.valves.MODEL_PREFIX
        family = model_id[len(prefix):] if prefix and model_id.startswith(prefix) else model_id
        name = (__model__ or {}).get("name") or family or "the family"
        url = f"{self.valves.AVATAR_APP_URL.rstrip('/')}/?family={quote(family)}"

        if __event_call__ is None:
            return

        # Runs in the chat page. window.open stays within the click's user
        # activation in practice; the fallback link covers browsers that block it.
        code = f"""
const t = localStorage.getItem("token");
const url = {json.dumps(url)} + (t ? "#owui_token=" + encodeURIComponent(t) : "");
const label = {json.dumps(f"Open the avatar meeting with {name}")};
const w = window.open(url, "_blank");
if (w) {{ try {{ w.opener = null; }} catch (e) {{}} return "opened"; }}
document.getElementById("ffc-avatar-link")?.remove();
const box = document.createElement("div");
box.id = "ffc-avatar-link";
box.style.cssText = "position:fixed;right:20px;bottom:90px;z-index:9999;background:#111827;color:#fff;padding:12px 16px;border-radius:12px;box-shadow:0 6px 24px #0006;font:14px system-ui;display:flex;gap:12px;align-items:center";
const a = document.createElement("a");
a.href = url; a.target = "_blank"; a.rel = "noopener";
a.textContent = label;
a.style.cssText = "color:#93c5fd;font-weight:600;text-decoration:underline";
const x = document.createElement("button");
x.textContent = "\\u00d7"; x.title = "Close";
x.style.cssText = "background:none;border:0;color:#9ca3af;font-size:18px;cursor:pointer";
x.onclick = () => box.remove();
a.onclick = () => setTimeout(() => box.remove(), 100);
box.append(a, x);
document.body.appendChild(box);
return "blocked";
"""
        result = await __event_call__({"type": "execute", "data": {"code": code}})

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": (
                            f"Opened the avatar meeting with {name} in a new tab"
                            if result == "opened"
                            else f"Pop-up blocked: use the link at the bottom right to meet {name}"
                        ),
                        "done": True,
                    },
                }
            )
