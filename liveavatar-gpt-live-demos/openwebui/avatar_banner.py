"""
title: Avatar banner
author: FamilyFinanceChat
version: 0.2.0
required_open_webui_version: 0.6.0
description: Shows a large "Talk to them face to face" banner under each reply from a family model.
"""

# Open WebUI Filter Function (extension surface: Functions). Attached to the
# family model presets via meta.filterIds by scripts/openwebui-setup.mjs.
#
# After each reply (outlet), it attaches an embed to that message: a banner
# with one big button. Embeds are stored beside the message, not in its text,
# so the transcript (and the grading that reads it) is unchanged.
#
# Why the button does not just link to the avatar app: Open WebUI renders
# embeds in a sandboxed iframe, and a tab opened from inside it inherits that
# sandbox (opaque origin) — the avatar app then cannot load its scripts, call
# its API or use the microphone. So the click is relayed instead: the button
# posts a message to the chat page, where a small listener (installed through
# the documented "execute" event, on each send and each reply) opens the tab
# from the unsandboxed page. The listener only opens URLs on the avatar app's
# own origin.

import asyncio
import html
import json
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, Field

OPEN_MSG = "ffc:open-avatar"
ACK_MSG = "ffc:open-avatar:ack"


class Filter:
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

    def _family(self, body: dict, model: dict):
        model_id = (body or {}).get("model") or (model or {}).get("id") or ""
        prefix = self.valves.MODEL_PREFIX
        if prefix and not model_id.startswith(prefix):
            return None
        return model_id[len(prefix):]

    async def _install_listener(self, event_call) -> None:
        """Let the chat page open the avatar app when a banner asks it to."""
        if event_call is None:
            return
        parts = urlsplit(self.valves.AVATAR_APP_URL)
        origin = f"{parts.scheme}://{parts.netloc}"
        code = f"""
if (window.__ffcAvatarRelay) return "already";
window.__ffcAvatarRelay = true;
const allowed = {json.dumps(origin)};
window.addEventListener("message", (e) => {{
  const d = e.data || {{}};
  if (d.type !== {json.dumps(OPEN_MSG)} || typeof d.url !== "string") return;
  let u;
  try {{ u = new URL(d.url); }} catch (err) {{ return; }}
  if (u.origin !== allowed) return;
  const w = window.open(u.href, "_blank");
  if (w) {{ try {{ w.opener = null; }} catch (err) {{}} }}
  if (e.source) e.source.postMessage({{ type: {json.dumps(ACK_MSG)}, opened: !!w }}, "*");
}});
return "installed";
"""
        try:
            # Short timeout: with no browser attached (API use) nobody answers.
            await asyncio.wait_for(event_call({"type": "execute", "data": {"code": code}}), timeout=3)
        except Exception:
            pass

    async def inlet(self, body: dict, __event_call__=None, __model__=None) -> dict:
        if self._family(body, __model__) is not None:
            await self._install_listener(__event_call__)
        return body

    async def outlet(self, body: dict, __event_emitter__=None, __event_call__=None, __model__=None) -> dict:
        family = self._family(body, __model__)
        if family is None or __event_emitter__ is None:
            return body
        meta = ((__model__ or {}).get("info") or {}).get("meta") or {}
        people = meta.get("avatar_members") or (__model__ or {}).get("name") or "the family"
        url = f"{self.valves.AVATAR_APP_URL.rstrip('/')}/?family={quote(family)}"

        await self._install_listener(__event_call__)
        await __event_emitter__(
            {"type": "embeds", "data": {"embeds": [banner_html(url, people)], "replace": True}}
        )
        return body


def banner_html(url: str, people: str) -> str:
    p = html.escape(people)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  html,body{{margin:0;padding:0;background:transparent;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
  .b{{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:4px 2px 8px;padding:16px 18px;border-radius:16px;
     background:linear-gradient(135deg,#1d4ed8,#7c3aed);color:#fff;box-shadow:0 6px 20px #1d4ed844}}
  .i{{font-size:34px;line-height:1}}
  .t{{flex:1;min-width:200px}}
  .t b{{display:block;font-size:18px}}
  .t span{{font-size:14px;opacity:.9}}
  button.go{{border:0;cursor:pointer;background:#fff;color:#1e1b4b;font:700 16px system-ui,sans-serif;padding:12px 22px;
        border-radius:12px;white-space:nowrap;box-shadow:0 2px 8px #0003}}
  button.go:hover{{background:#e0e7ff}}
  .note{{flex-basis:100%;font-size:13px;background:#0003;border-radius:8px;padding:6px 10px;display:none}}
</style></head><body>
<div class="b">
  <div class="i">🎥</div>
  <div class="t"><b>Talk to {p} face to face</b><span>Open the video meeting, press Start, and speak to them out loud.</span></div>
  <button class="go" id="go" type="button">Talk to the avatar →</button>
  <div class="note" id="note"></div>
</div>
<script>
  const URL_ = {json.dumps(url)};
  const note = document.getElementById("note");
  const h = () => parent.postMessage({{ type: "iframe:height", height: document.documentElement.scrollHeight }}, "*");
  const say = (t) => {{ note.textContent = t; note.style.display = "block"; h(); }};
  document.getElementById("go").addEventListener("click", () => {{
    let acked = false;
    const onAck = (e) => {{
      const d = e.data || {{}};
      if (d.type !== {json.dumps(ACK_MSG)}) return;
      acked = true;
      removeEventListener("message", onAck);
      if (!d.opened) say("Your browser blocked the new tab. Allow pop-ups for this site, then press the button again.");
    }};
    addEventListener("message", onAck);
    parent.postMessage({{ type: {json.dumps(OPEN_MSG)}, url: URL_ }}, "*");
    setTimeout(() => {{
      if (acked) return;
      removeEventListener("message", onAck);
      say("Not connected yet (this happens after reloading the page). Send any message, then press the button again — or use the microphone button under this reply.");
    }}, 800);
  }});
  addEventListener("load", h); addEventListener("resize", h); h();
</script>
</body></html>"""
