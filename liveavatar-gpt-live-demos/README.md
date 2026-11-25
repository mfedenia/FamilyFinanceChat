# LiveAvatar × GPT-Live — language demo

A minimal, readable reference integration: **OpenAI GPT-Live** (a full-duplex
speech-to-speech model) driving a **HeyGen LiveAvatar** — a realtime voice
agent with a face, whose tool calls land on screen as animated overlays.

Out of the box it is a **Japanese tutor**: it teaches you words out loud,
puts a term card (term · pronunciation · meaning) on screen as it says them,
and every few words shrinks itself to the corner and reviews everything taught
so far — a recap panel rendered from the server's own record of the session,
not the model's memory. The persona is two markdown files (`server/prompts/`)
and each visual is one tool plus one composition — swap those and it is any
other demo.

Barebones on purpose: the wiring is the thing you read, not a framework around
it. Fork it, swap the persona, keep the face.

```
                        ┌────────────────────────┐
                 ws     │      orchestrator      │   ws    ┌──────────────┐
   mic audio ──────────►│       (server/)        │◄───────►│   GPT-Live   │
   transcripts ◄────────│                        │         │  + Responses │
   tool calls  ◄────────│  audio ──► media server│         │  (tools)     │
                        └───────────────┬────────┘         └──────────────┘
  ┌─────────┐                           │ ws (LITE session)
  │ browser │      LiveKit      ┌───────▼────────┐
  │ (web/)  │◄─────────────────►│   LiveAvatar   │
  └─────────┘  avatar A/V       └────────────────┘
```

The browser never holds an API key. It gets a LiveKit token to watch the
avatar, and a websocket for mic audio (up) and transcripts + visuals (down).
The avatar's voice takes the short path: GPT-Live → orchestrator → media
server, with the browser out of the loop.

## Quickstart

Requirements: Node ≥ 20.12, pnpm, a [LiveAvatar API key](https://app.liveavatar.com),
and an OpenAI API key with GPT-Live access (generally available; this
integration speaks the v3 contract, `gpt-live-1.1`).

```bash
pnpm install
pnpm run setup   # prompts for the two API keys, verifies each, writes .env
pnpm dev         # server on :8787, web on :5173
```

`pnpm run setup` walks you through it: it asks for each key (with the URL to
create one), verifies it against the live API before accepting it, and writes
`.env` at the repo root. Safe to re-run — existing values are kept and
re-verified, not re-asked. (The `run` matters: bare `pnpm setup` is pnpm's own
built-in command.)

Prefer doing it by hand? `cp .env.example .env` and fill it in. Either way,
`pnpm dev` and `pnpm start` check the required variables before starting and
name exactly what's missing.

No avatar to pick, no prompt to write: a default avatar id ships in
`.env.example` (swap `LIVEAVATAR_AVATAR_ID` for one of your own, or unset it
and the server uses the first public avatar and logs which), and the persona
ships in `server/prompts/`.

Open http://localhost:5173 → **Start** → allow the mic → talk. The tutor
greets you, teaches こんにちは, and the term card lands on screen while it
keeps talking. Say the word back; ask it for another one.

To change what the demo is: edit `server/prompts/instructions.md` (who the
avatar is) and `server/prompts/greeting.md` (how it opens), restart the
server. An empty `greeting.md` means the user speaks first.

## FIN 602 couple meeting (Tony and Antonella)

Open http://localhost:5173 after `pnpm dev`: two avatars, Tony and Antonella
Beckham, each a GPT-Live session driving its own LiveAvatar LITE session
(`server/src/couple.ts`). Both hear the student's mic; the server decides who
answers (a name said wins at once, otherwise OpenAI's
[Decisions API](https://developers.openai.com/api/docs/guides/decisions),
`POST /v1/decisions` with `gpt-6-luna`: a `choice` question for who answers,
with a probability per spouse, and a `predicate` for whether the other should
chime in) and only lets that spouse's voice through. The OpenAI key needs the
`api.decisions.write` permission. The other spouse hears what was said as silent context. Wear
headphones. A meeting runs two LiveAvatar sessions, so it uses 2 credits per
minute.

Added on top of that (branch `feat/couple-gptlive-combined`), the useful parts
of Taehyun Park's duo (`feat/liveavatar-duo`) plus transcript saving:

| Feature | What it does |
| --- | --- |
| **Transcripts saved automatically** | When a meeting ends (End, closing the tab, 60 s without mic audio, Ctrl+C), `server/src/transcript.ts` writes one JSON file to `transcripts/` (gitignored): every heard turn with speaker, text, times, cut-off flag, which spouse answered each adviser turn, every floor decision with its reason, and per-person metrics. Text only, never audio. |
| **Handoff** (Park) | A spouse who ends by asking the other ("Antonella, what do you think?") hands over the floor; at most two in a row, then the floor returns to the adviser. |
| **Backchannel** (Park) | "Mm-hmm" or "okay" (under three words, no question mark) while a spouse is still talking keeps that spouse speaking instead of opening a new decision. |
| **Cut-off notice** (Park) | A spouse the adviser talks over is told, silently, that the end of its reply was not heard. |
| **Decision shown on the page** | Under the controls: who has the floor and why (a name was said; the decision model, its time and each spouse's probability; chime-in, handoff, backchannel). The same reasons are in the transcript's `floor` list. |
| **Personas from Open WebUI** (Park, optional) | With the four variables below set, each spouse's persona is the system prompt of an Open WebUI client model preset, read through the public REST API before any avatar starts (a wrong id or key costs nothing). Otherwise `server/prompts/couple/*.md` is used. |

Optional settings, in this folder's `.env` or the shell:

```
# Transcripts (defaults shown)
TRANSCRIPTS_DIR=./transcripts
STUDENT_ID=local-test-student
STUDENT_NAME=Local Test Student
STUDENT_EMAIL=local-test@example.invalid

# Personas from Open WebUI (all four, or none)
OPENWEBUI_BASE_URL=https://your-open-webui
OPENWEBUI_API_KEY=            # the presets' owner or an admin (Open WebUI hides system prompts from read-only keys)
COUPLE_OPENWEBUI_MODEL_TONY=
COUPLE_OPENWEBUI_MODEL_ANTONELLA=
```

### Knowledge and saving to Open WebUI

With the presets configured, the avatar uses the same Open WebUI knowledge a
text chat with that preset uses (endpoints checked against Open WebUI v0.13.4):

1. **Knowledge in the instructions.** At meeting start, the files attached to
   each spouse's preset (`meta.knowledge`: knowledge bases and single files)
   are read and placed in that spouse's GPT-Live instructions, up to
   `OPENWEBUI_KNOWLEDGE_MAX_CHARS`. Most questions are answered with no lookup.
2. **Retrieval on a delegated turn.** A spouse whose preset has knowledge runs
   GPT-Live with `delegation: "client"`. When the live model needs a fact it
   does not have, it says something like "hmm, let me think" and hands the
   turn over. The server sends the adviser's words to Open WebUI's
   `POST /api/v1/retrieval/query/collection` over the preset's collections, so
   Open WebUI's own embedding model, vector DB, top-k, threshold and hybrid
   settings apply. `GPT_LIVE_RESPONSES_MODEL` writes the spouse's reply from
   the hits, and GPT-Live says it (`session.commentary.append`). A spouse with
   no knowledge keeps OpenAI's Responses delegation, as before.
3. **Saved as the student's chat.** When the meeting ends, it is saved with
   `POST /api/v1/chats/new` as an ordinary Open WebUI chat, titled
   `[Avatar] The Beckhams — <time>`, tagged `avatar`, and filed under the
   family preset (`family-beckham`). It shows in the student's sidebar and in
   the grading extraction. Open WebUI files a new chat under whoever creates
   it, so this uses the **student's** token: the "Talk to the avatar" button
   passes the signed-in student's token to this page in the URL fragment. A page
   not opened from the button saves locally only, unless
   `OPENWEBUI_STUDENT_TOKEN` is set for testing. Back-to-back spouse turns are
   merged into one reply, because the grader pairs strictly question → answer.
   The local transcript file is still written, with the chat id and a log of
   every delegated turn.

```
OPENWEBUI_KNOWLEDGE_MAX_CHARS=32000   # knowledge text per spouse in the instructions
OPENWEBUI_RETRIEVAL_TIMEOUT_MS=4000   # wait for Open WebUI's search on a delegated turn
COUPLE_OPENWEBUI_FAMILY_MODEL=family-beckham  # the preset the saved chat is filed under
OPENWEBUI_STUDENT_TOKEN=              # testing only: save meetings under this student
GPT_LIVE_ACK_DELAY_MS=                # 0–800: wait before the spoken "hmm" (API default 0)
```

Open WebUI's inlet/outlet Filter functions do not run on spoken turns: no chat
completion passes through Open WebUI.

Tests for these additions: `cd server && npx tsx --test test/*.test.ts`.

### "Talk to the avatar" button in Open WebUI

Students start from Open WebUI: they chat with **The Beckhams** model, and
every reply carries a large **Talk to the avatar** banner that opens this app
in a new tab (press *Start meeting*, then speak). A small microphone button in
the reply's toolbar does the same. The empty new-chat screen shows a
**▶ Start the meeting** card: one click sends a greeting, and the video button
arrives on the reply. (It cannot appear before any reply: Open WebUI runs no
function code until a message is sent.)

```bash
pnpm openwebui:setup
```

installs it over Open WebUI's public REST API (re-runnable):

- `openwebui/avatar_banner.py`, a **Filter** function: after each reply it
  attaches the banner as an embed, stored beside the message and not in its
  text, so transcripts and grading are unchanged. Open WebUI sandboxes embeds,
  and a tab opened from inside one inherits the sandbox (this app then cannot
  load), so the click is relayed to the chat page, which opens the tab. The
  relay only opens `AVATAR_APP_URL`'s origin.
- `openwebui/talk_with_avatar.py`, an **Action** function: the toolbar button.
- the model preset `family-beckham` ("The Beckhams"): a text-chat role-play of
  both spouses built from `server/prompts/couple/*.md`, readable by all users,
  with both functions attached.

Open WebUI extension surface: Functions (Filter + Action) and the public REST
API; nothing in Open WebUI is changed. Checked against Open WebUI v0.11.4.
After reloading the chat page, banners on earlier replies need one new message
in that chat before they open (the banner says so).

Settings, in this folder's `.env`:

```
OPENWEBUI_BASE_URL=http://localhost:3000
OPENWEBUI_API_KEY=            # an admin's key (API keys must be enabled), or instead:
OPENWEBUI_EMAIL=              # an admin login; on a brand-new Open WebUI
OPENWEBUI_PASSWORD=           #   this account is created as the admin
OPENWEBUI_BASE_MODEL=gpt-5.4-mini   # what the text chat runs on
AVATAR_APP_URL=http://localhost:5173 # this app, as students' browsers reach it
```

`OPENWEBUI_BASE_URL` and `OPENWEBUI_API_KEY` alone do not switch the avatar's
personas to Open WebUI; that also needs the two `COUPLE_OPENWEBUI_MODEL_*`.

## Troubleshooting a fresh clone

**"Missing required env: …" when you run `pnpm dev`.** The preflight check
(`scripts/check-env.mjs`) refuses to start until the required variables are
set, and names them. Run `pnpm run setup` — it prompts for each key and
verifies it against the live API — or fill in `.env` by hand. The server also
re-checks per session (`/api/session/start` 500s naming what's absent), so a
deployment with broken ambient env explains itself too. The server reads
`.env` only at startup.

**A key that doesn't work.** `pnpm run setup` verifies both keys before saving
them, so a typo or revoked key fails there with a pointer to the right
dashboard — re-run it any time keys change.

**The avatar appears and blinks but never speaks.** LiveAvatar is up; GPT-Live
isn't. Almost always a bad `OPENAI_API_KEY` or no GPT-Live access on the
account. Set `GPT_LIVE_DEBUG=1` and restart to watch every upstream event
(audio elided) — a session that authenticates but errors will say why.

**Any other error on Start.** Upstream error bodies are passed through
verbatim on purpose — a 401 from LiveAvatar or OpenAI means that key; read the
message, it is the real one.

**It speaks but never hears you.** The status line under the stage says
`microphone unavailable` if permission was denied — re-allow it in the
browser's site settings and Start again. There is no push-to-talk and no VAD:
the mic streams continuously (meter next to the status line) and the model
decides when you're done talking.

**No greeting when the session opens.** An empty `server/prompts/greeting.md`
is a feature, not a bug: it means "say nothing, let the user speak first".

**Port 8787 is taken.** Set `PORT` in `.env` — and mirror it in
`web/vite.config.ts`, whose dev proxy points at `:8787`. (Vite itself moving
off 5173 is fine; the proxy target is the only coupling.)

## What's in the box

| Path | What it is |
| --- | --- |
| `shared/messages.ts` | The wire protocol — every message both sides speak, typed once |
| `shared/tools.ts` | The tool registry — schemas the model sees, co-located with their arg types |
| `server/prompts/*.md` | The persona — edit these to change what the demo is |
| `server/src/prompts.ts` | Loads the persona; holds the delegation/tool mechanics prompts |
| `server/src/gptlive.ts` | The GPT-Live v3 bridge: session start, audio, transcripts, tool calls |
| `server/src/turns.ts` | Turn projection over v3 transcript deltas — the API has no turn events |
| `server/src/mediaServer.ts` | The avatar's ear: LITE media-server websocket |
| `server/src/session.ts` | Wires the legs together; owns barge-in |
| `web/src/livekitRoom.ts` | Joins the room, attaches the avatar's video + audio |
| `web/src/micCapture.ts` | AudioWorklet → 24kHz PCM16 base64 (no VAD — the model owns turn-taking) |
| `web/src/overlays/` | The single switch over `widget`; one renderer per widget |
| `web/public/overlays/term-card.html` | The overlay composition (a self-contained animated page) |

Deeper docs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (how and why),
[`AGENTS.md`](AGENTS.md) (a map for coding agents, including the add-a-tool
recipe).

## How tool calls become visuals

1. The live model holds **no tools**. When a visual is wanted it delegates the
   turn to its backend Responses model — which does hold them
   (`shared/tools.ts`).
2. The Responses model answers in words **and** calls e.g. `show_term_card` in
   the same reply. The words are injected back into the live session and
   spoken; the tool call surfaces on the orchestrator's socket.
3. The orchestrator validates the call (`server/src/tools.ts`) and forwards
   one `{ type: "ui", widget, props }` message to the browser. Term cards are
   also recorded per-session — `show_learned_words` renders the recap from
   that server-side store, so the list is never the model's to get wrong.
4. The browser's widget switch (`web/src/overlays/`) plays the matching
   composition — a transparent animated page layered over the avatar's video.
   Staging is per-widget: the term card overlays the full-frame avatar, the
   recap panel shrinks it to the corner. Nothing is composited into the
   stream itself.

Adding a tool is three small edits — see [`AGENTS.md`](AGENTS.md).

## Iterating on overlays

Render a widget without burning session minutes — from the browser console:

```js
window.__ui({ widget: "term_card", props: { term: "こんにちは", reading: "kon-ni-chi-wa", meaning: "hello" } })
window.__ui({ widget: "learned_words", props: { title: "Words so far", words: [{ term: "こんにちは", reading: "kon-ni-chi-wa", meaning: "hello" }, { term: "お茶", reading: "o-cha", meaning: "tea" }] } })
```

## Production notes

This is a starter, not a deployment. Before exposing it publicly: add auth on
`/api/session/start` and the websocket upgrade, hide upstream error bodies,
and read the hardening list in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#production-hardening).

## License

MIT. One vendored exception: the bundled GSAP
(`web/public/overlays/vendor/gsap.min.js`) stays under its own
[GSAP Standard License](https://gsap.com/standard-license) — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
