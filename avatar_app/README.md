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
  index.ts        HTTP + WebSocket routes (/api/session/start|stop, /ws/:id, /healthz)
  session.ts      one live conversation: wiring, barge-in, silence check-in
  gptlive.ts      the GPT-Live websocket (session.start, transcripts, delegation)
  liveavatar.ts   starts/stops LiveAvatar LITE sessions
  mediaServer.ts  sends GPT-Live's audio into the avatar
  turns.ts        stitches ~200 ms transcript fragments into turns
  prompts.ts      mechanics prompts (speak-first, role-play rules, silence)
  registry.ts     live sessions + the 60 s idle reaper
server/prompts/   PLACEHOLDER client persona — replaced by the AV2 persona loader
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

Open <http://localhost:5173>, press **Start talking**, and allow the microphone.
You play the advisor; the avatar plays the placeholder client.

Keys:

| Variable | Where from | Notes |
|---|---|---|
| `LIVEAVATAR_API_KEY` | app.liveavatar.com → Developers | Free plan: 2-minute sessions with a watermark — fine for development |
| `OPENAI_API_KEY` | platform.openai.com | Needs GPT-Live access |

`corepack pnpm run setup` will prompt for and verify both keys instead of
editing `.env` by hand.

> ⚠ **No authentication yet.** Until AV3 (login check) and AV5 (caps, kill switch)
> land, anyone who can reach the server can start a billed session. Run it on
> `localhost` only.

## Every session costs money

A session bills from `POST /api/session/start` on both services: roughly
**$0.09/min LiveAvatar LITE + $0.05/min GPT-Live ≈ $0.14/min**. Always end calls
with **End call**. The server also reaps a session after 60 s without mic audio,
and stops all sessions on shutdown.
