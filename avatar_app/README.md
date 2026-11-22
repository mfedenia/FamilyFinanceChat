# avatar_app — the avatar bridge

The server that lets a student **talk out loud to a role-played client with a face**.
It sits between three things:

```
Student browser ──mic audio──▶ BRIDGE ──▶ GPT-Live (OpenAI: listens, thinks, talks back)
       ▲                          │
       │                          └──voice audio──▶ LiveAvatar (HeyGen: moves the face)
       └────────── video of the face (LiveKit) ◀──────────────┘

BRIDGE ◀──▶ Open WebUI REST API   (from AV3/AV4: who is this student, save the transcript)
```

It is a **sidecar container** (compatibility surface **A8**), served behind Caddy at
`/practice/` (**A9**), and talks to Open WebUI only over its public REST API (**A7**).
Open WebUI is never modified. See
[`../capstone_fall2026/COMPATIBILITY_POLICY.md`](../capstone_fall2026/COMPATIBILITY_POLICY.md) §7.

**Task list, status and acceptance criteria:**
[`../capstone_fall2026/AVATAR_TRACK.md` §16](../capstone_fall2026/AVATAR_TRACK.md#16-implementation-the-gpt-live-avatar-bridge-av0av7).

---

## Where this code came from

A fork of [heygen-com/liveavatar-gpt-live-demos](https://github.com/heygen-com/liveavatar-gpt-live-demos)
(MIT) at commit **`daa8cc92ab34c75c0bbed201c94fb98360da2179`**.

- **AV0** imported it verbatim (commit `4ad6924` on `feat/avatar-bridge`), so
  `git diff 4ad6924 -- avatar_app/` shows every change we have made to upstream.
- **AV1** removed the demo's Japanese-tutor logic: the lesson script, term-card and
  recap tools, the review-break machinery, the on-screen overlay player and its
  vendored GSAP file, and the upstream docs that only described those features.
  The domain-independent parts — audio path, turn projection, barge-in, greeting gate,
  silence check-in, teardown — are unchanged.

Do not "fix" the invariants listed in [`AGENTS.md`](AGENTS.md). They are measured
behaviour, not style.

## Layout

```
server/src/
  index.ts        HTTP + WebSocket routes (/api/personas, /api/session/start|stop, /ws/:id, /healthz)
  session.ts      one live conversation: wiring, barge-in, silence check-in, time limit
  gptlive.ts      the GPT-Live websocket (session.start, transcripts, delegation)
  liveavatar.ts   starts/stops LiveAvatar LITE sessions
  mediaServer.ts  sends GPT-Live's audio into the avatar
  turns.ts        stitches ~200 ms transcript fragments into turns
  prompts.ts      mechanics prompts (speak-first, role-play rules, silence, wrap-up)
  personas.ts     AV2 persona loader: file or Open WebUI source, token-budget check
  openwebui.ts    EVERY call to Open WebUI lives here (public REST API only)
  metrics.ts      AV5a Prometheus metrics, served on their own internal port
  registry.ts     live sessions + the 60 s idle reaper
personas/         file-source personas (*.md); placeholder-dana.md is NOT a course persona
shared/messages.ts  the browser ↔ bridge wire protocol
web/              developer test harness — NOT the student-facing page
docs/ARCHITECTURE.md  upstream design rationale (read the fork note at the top)
```

## Running it locally

Needs Node ≥ 20.12 and corepack (ships with Node).

```bash
cd avatar_app
corepack pnpm install
cp .env.example .env          # then fill in the two keys — .env is gitignored
corepack pnpm -r typecheck
corepack pnpm --parallel -r dev   # server :8787 + test page :5173
```

Open <http://localhost:5173>, pick a client, press **Start talking**, and allow the
microphone. You play the advisor; the avatar plays the client.

Keys:

| Variable | Where from | Notes |
|---|---|---|
| `LIVEAVATAR_API_KEY` | app.liveavatar.com → Developers | Free plan: 2-minute sessions with a watermark — fine for development |
| `OPENAI_API_KEY` | platform.openai.com | Needs GPT-Live access |

`corepack pnpm run setup` will prompt for and verify both keys instead of
editing `.env` by hand. Every other setting is optional and documented in
[`.env.example`](.env.example).

> ⚠ **No authentication yet.** Until AV3 (login check) lands, anyone who can reach
> the server can start a billed session. Run it on `localhost` only.

## Personas (AV2)

Which client the avatar plays is chosen per session. The whole persona is sent to
GPT-Live once, at the start: GPT-Live cannot change its instructions mid-session,
so there is no per-turn document search in the speech loop.

- **File source** (default, `AVATAR_PERSONA_SOURCE=file`): one Markdown file per
  persona in [`personas/`](personas/). See [`personas/_TEMPLATE.md`](personas/_TEMPLATE.md).
- **Open WebUI source** (`AVATAR_PERSONA_SOURCE=openwebui`): the professor's own
  role-play models. Tag a model **`avatar`** in Workspace → Models and it appears
  in the picker. Its system prompt becomes the persona. An optional tag such as
  `voice:cedar` picks the voice. This needs `OPENWEBUI_BASE_URL` and an **admin**
  `OPENWEBUI_API_KEY`, because Open WebUI hides system prompts from everyone else.
  - Only the system prompt is loaded. **Documents attached to the model are not**,
    and the log warns when a persona model has any.
  - `{{CURRENT_DATE}}`-style variables are filled in. `{{USER_NAME}}` becomes
    "the advisor" until AV3 knows the student.

Personas over GPT-Live's 16,384-token limit are **refused, not cut off**. The
estimate is deliberately pessimistic, about 40,000 characters of persona.

## Guardrails (AV5a)

| Setting | Default | What it does |
|---|---|---|
| `AVATAR_ENABLED` | `true` | Kill switch: `false` refuses every new session with a plain message |
| `AVATAR_MAX_SESSION_MIN` | `15` | Hard cap. About a minute before it, the client says in character that they need to leave and asks for a last question |
| `AVATAR_METRICS_PORT` | `9464` | Prometheus `/metrics` on its own port, so the public `/practice/` route can never expose it |

Students never see raw upstream errors: they get a plain message, and the detail
goes to the server log.

Metrics: `avatar_sessions_active`, `avatar_sessions_started_total`,
`avatar_session_start_failures_total{reason}`, `avatar_sessions_ended_total{reason}`,
`avatar_session_seconds_total` (multiply by the per-second price to get cost), and
`avatar_session_ready_seconds` (from start until the voice model is ready). The
Prometheus scrape job is added with the container in AV6.

## Every session costs money

A session bills from `POST /api/session/start` on both services: roughly
**$0.09/min LiveAvatar LITE + $0.05/min GPT-Live ≈ $0.14/min**. Always end calls
with **End call**. The server also reaps a session after 60 s without mic audio,
and stops all sessions on shutdown.
