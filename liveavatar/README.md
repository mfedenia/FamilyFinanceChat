# LiveAvatar Duo

**Two talking avatars in one live, spoken conversation.** You speak; a decision
model works out which of the two characters should answer; that character's
avatar answers out loud, in its own face and voice. Both characters hear the
whole conversation, so they can pick up each other's threads, disagree, or
hand the floor to each other.

It lives inside FamilyFinanceChat as its own app (outside Open WebUI, per the
[compatibility rule](../capstone_fall2026/COMPATIBILITY_POLICY.md)) and ships with the
**FIN 602 Beckham case** (see [Connecting to FamilyFinanceChat](#connecting-to-familyfinancechat)): a student practises
client advising with **Tony and Antonella Beckham**, a married couple meeting a
financial adviser for the first time. Ask about Tony's pension and Tony
answers; ask about Antonella's 401(k) and Antonella answers; ask "do you two
have a will?" and either may. When the call ends, **Export** saves the
dialogue as an Open WebUI chat, so it can be graded like a typed session.

> This repo also contains the original **single-avatar** reference app (one
> avatar driven by OpenAI GPT-Live or Gemini Live) at `/`. It is documented in
> [docs/SINGLE_AVATAR.md](docs/SINGLE_AVATAR.md) and kept for possible OpenAI
> use later. Everything below is about the duo at **`/duo.html`**.

---

## Contents

- [Quickstart](#quickstart)
- [Using it](#using-it)
- [How it works](#how-it-works)
- [Scenarios](#scenarios)
- [Exporting to Open WebUI](#exporting-to-open-webui)
- [Connecting to FamilyFinanceChat](#connecting-to-familyfinancechat)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Repository layout](#repository-layout)

---

## Quickstart

You need Node ≥ 20.12, pnpm, a **LiveAvatar API key**
([app.liveavatar.com/developers](https://app.liveavatar.com/developers)) and a
**Gemini API key** ([aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

```bash
pnpm install
cp .env.example .env     # set LIVEAVATAR_API_KEY and GEMINI_API_KEY
pnpm dev                 # server on :8787, web on :5173
```

Open **http://localhost:5173/duo.html** in Chrome or Safari, click **Start**,
allow the microphone, and talk.

> **Cost.** Every call runs **two** LiveAvatar sessions (one per avatar), and
> both bill for as long as the call is open. **End**, closing the tab, or 60
> seconds without microphone audio stops both. If start fails with
> `No credits available for start session`, the LiveAvatar account is out of
> credit.

`pnpm dev` refuses to start without `LIVEAVATAR_API_KEY`, and without the key
for the single-avatar page's provider (`OPENAI_API_KEY`, or `GEMINI_API_KEY`
with `LIVE_PROVIDER=gemini`). The duo itself needs only the LiveAvatar and
Gemini keys.

---

## Using it

```
┌───────────────────────────┬───────────────────────────┐  ┌────────────────────┐
│                           │                           │  │ CONVERSATION       │
│       Tony (video)        │    Antonella (video)      │  │ Tony: Hi, I'm Tony │
│                           │                           │  │ You (Adviser): …   │
├───────────────────────────┴───────────────────────────┤  │ Antonella: …       │
│ DECISION ████░░░░  → Antonella · Tony 6% / Antonella 94% · 280ms │                    │
├────────────────────────────────────────────────────────┤  │                    │
│ ● live — talk to Tony or Antonella  Mute  End  Export  Start │                    │
└────────────────────────────────────────────────────────┘  └────────────────────┘
```

- **Start** brings up both avatars (about 10 seconds). In the Beckham scenario
  Tony opens with a short, guarded hello and lets you lead.
- **Just talk.** There is no push-to-talk. When you pause, the system decides
  who answers. To steer, use a name ("Antonella, how do you feel about…") or
  ask about something that clearly belongs to one of them.
- **The decision bar** shows each answer's routing: who was chosen, the
  decision model's probability for each persona, and how long the decision
  took. A handoff ("handed the floor") is shown too.
- **Interrupting works.** Talk over an avatar with a real sentence and it
  stops; the transcript marks the cut line with "—". Short sounds like
  "mm-hmm" or "yeah" do not interrupt; they are noted, not answered.
- **Mute** silences the microphone.
- **End** stops both avatars, and the billing.
- **Export** (enabled after End) downloads the call as an Open WebUI chat file —
  see [Exporting to Open WebUI](#exporting-to-open-webui).

**Use headphones.** The microphone can pick up an avatar's voice from your
speakers, and anything the microphone hears is treated as you. Browser echo
cancellation helps but is not perfect.

---

## How it works

### The big picture

```
                 ┌──────────────────────── server: the "director" ────────────────────────┐
 browser mic ─ws─► Listener — Gemini Live, transcription only                              │
                 │   (the ONLY thing that hears the mic)    barge-in ─► cut audible avatar │
                 │      │ utterance                                                       │
                 │      ▼                                                                 │
                 │ Router — gemini-decision-1: "which persona should answer?"             │
                 │      ├─ say("Adviser: …") ───────► Persona A — Gemini Live, own prompt │
                 │      └─ addContext("Adviser: …") ► Persona B — same conversation, quiet│
                 │                                      │ audio                           │
                 │                           media leg A ▼                 media leg B     │
                 └────────────────────────── LiveAvatar A ──────────── LiveAvatar B ──────┘
                                                  │ LiveKit               │ LiveKit
 browser ◄─ws─ transcript, floor decisions        └─► video tile A        └─► video tile B
```

Four kinds of model are involved:

| Role | Model | What it does |
|---|---|---|
| Listener (the ears) | `gemini-3.5-transcribe-live` | hears the mic, produces text |
| Router (who answers) | `gemini-decision-1` | reads recent lines, returns a choice + probabilities |
| Persona × 2 (the voices) | `gemini-3.8-live` | text turns in, speech out |
| Avatar × 2 (the faces) | LiveAvatar LITE sessions | speech in, lip-synced video out |

The key design choice: **the personas never hear audio.** If both did, both
would start answering before anyone could decide who should. Instead, one
listener turns speech into text, the router picks a speaker, and only then does
a persona get the words, as text.

### 1. Starting a call

`POST /api/duo/start` (`server/src/index.ts`):

1. Loads the scenario (`server/scenarios/<name>/`) and its two avatar ids.
2. Starts **two LiveAvatar LITE sessions** in parallel. If either fails, both
   are stopped before the error is returned.
3. Creates a `DuoStage` (`server/src/duo/stage.ts`), the director, which opens
   one **media leg** per avatar: a websocket into that avatar's media server,
   where audio becomes lip-synced video.
4. Waits until **both media legs report `connected`** (up to 15s). LiveAvatar
   drops audio sent before that, with no replay, so the opening line would be
   lost.
5. Opens both persona sessions and the listener. When all are ready, the
   scenario's opening persona (Tony) gets its greeting instruction, and the
   browser is told `ready`, which starts the microphone.

The browser receives one LiveKit token per avatar and joins both rooms. It
never holds an API key.

### 2. Listening

`server/src/duo/listener.ts`. The browser streams microphone audio
continuously (PCM16, 24 kHz, no voice detection in the browser). The listener
forwards it to a transcription-only Gemini Live session, with end-of-speech
silence set by `DUO_SILENCE_MS` (600 ms).

What the transcription model actually does, as observed:

- `ACTIVITY_START` about 0.3s after you start speaking.
- Interim transcripts about twice a second.
- **With a real microphone it almost never sends a final transcript**, and its
  transcript is **cumulative for the whole connection**: each update repeats
  everything said so far, glued together without spaces, with earlier
  punctuation revised ("…why I guess.enzo en zoHow's your…").

So the listener does its own bookkeeping:

- **Utterance end:** if the interim text has not changed for **1.2s**, the
  utterance is over. A real final, when one comes, also ends it.
- **Only new words:** it remembers how much of the running transcript it has
  already delivered, counted in **letters and digits only**, so re-spacing and
  re-punctuation cannot shift it. Each utterance is just the part after that
  point.
- **Fresh buffer:** once the running transcript passes 1,500 characters, the
  listener reconnects at the next pause (about a 0.2s gap) to reset it.

### 3. Deciding who answers

`server/src/duo/decision.ts` calls `POST /v1beta/decisions` with:

- **input:** who is on the call, who spoke last, and the **last 8 lines** of
  the conversation, ending with the new utterance;
- **one `choice` question:** the scenario's `routing` instructions, with each
  persona's `role` text as that option's criterion.

It returns a choice and per-persona probabilities, typically in
**0.22–0.47s**. If you keep talking while a decision is in flight, the new
words are merged and the decision is redone on all of it. If the decision call
fails, whoever spoke last keeps the floor.

Routing tested on the Beckham scenario with 10 typical adviser questions (10
right): "are you using your 403(b) or 457?" → Tony, 1.00; "catch-up
contributions to your 401(k)?" → Antonella, 0.95; "your father's plans for his
money?" → Antonella, 0.97; "do the two of you have a will?" → split
0.54 / 0.46, as a joint question should be.

### 4. Speaking

Each persona is one Gemini Live session (`server/src/duo/personaBridge.ts`).
Its system prompt is the persona's own prompt, plus a shared "how this call
works" block, plus the scenario's knowledge files in full. The chosen persona
receives `Adviser: <your words>` as a turn and answers in audio (24 kHz PCM),
forwarded chunk by chunk into its avatar's media leg (`agent.speak`). First
audio typically arrives **0.9–1.5s** after the turn is sent.

From you stopping to the avatar speaking takes about **2.5s**: ≈1.0–1.2s to
detect the end of your turn, ≈0.3s to decide, ≈1s to first audio, plus the
avatar's own rendering.

### 5. One shared conversation

Both personas must know everything that was said, by anyone, so every line
goes to **both** of them:

| Line | Persona with the floor | The other persona |
|---|---|---|
| your utterance | as a **turn** (it answers) | as **silent context** |
| a persona's reply | (it said it) | as silent context: `Tony: …` |

Silent context is a Gemini Live client message with `turnComplete: false`. It
was tested to never trigger a reply, and to be remembered. One subtlety:
sending *anything* to a Gemini Live session while it is replying cuts that
reply off. So context for a persona that is mid-reply is held, and delivered
when its reply ends.

### 6. Interruptions (barge-in)

- The director estimates when each avatar's queued audio will finish playing:
  from the bytes sent, plus a 1s margin for the avatar's rendering delay.
- If you say **3 or more words** while an avatar is audible, it is cut. Its
  avatar buffer is cleared (`agent.interrupt`), and the rest of that reply,
  which Gemini keeps generating, is discarded until the reply ends. Each reply
  is cut at most once.
- The cut persona is told the user cut it off. The other persona sees its line
  marked "(cut off by the adviser)".
- **Backchannels** (fewer than 3 words, no question mark) over a speaking
  avatar are recorded in both personas' context, but not answered.

### 7. Handoffs

When a persona finishes a reply that mentions the other persona by name, the
decision model is asked one yes/no question: did it hand the floor over
("Antonella, you read that letter, right?")? Above **0.6**, the other persona
answers at once. At most **2** persona-to-persona exchanges happen in a row,
and anything you say cancels a pending handoff.

### 8. Ending and billing

**End**, closing the tab (via `sendBeacon`), the browser websocket closing, any
leg failing, or **60 seconds with no microphone audio** all go through one
`stopSession`. It closes the models, closes the media legs, and stops **both**
LiveAvatar sessions. Stopping twice is harmless.

---

## Scenarios

A scenario is a folder in `server/scenarios/`, chosen with `DUO_SCENARIO`
(default `beckham`). Adding a family or a different cast is a new folder — no
code.

```
server/scenarios/beckham/
  scenario.json    who is on the call, routing, avatars, voices, opening line
  tony.md          Tony's own system prompt
  antonella.md     Antonella's own system prompt
  case.md          knowledge: the full client data sheet (both personas get it)
```

### `scenario.json`

```jsonc
{
  "title": "The Beckham Case",
  "subtitle": "FIN 602 · first meeting with Tony and Antonella Beckham",
  "userLabel": "Adviser",            // what the human is called in prompts and transcripts
  "knowledge": ["case.md"],          // files given IN FULL to both personas
  "routing": "…which of them should answer the adviser's latest message? …",
  "greeting": {                      // or null: nobody speaks until the user does
    "persona": "tony",
    "directive": "[director] The meeting is starting… say a brief hello…"
  },
  "personas": [
    {
      "id": "tony",
      "name": "Tony",                // omit to use the avatar's first name
      "tagline": "54 · UW-Milwaukee, freelance web developer · risk-tolerant",
      "role": "Tony Beckham, the husband (54). Answers about his … pension, the 403(b) and 457 …",
      "prompt": "tony.md",
      "voice": "Charon",             // a Gemini prebuilt voice
      "avatarId": "dff45d08-…"       // omit for a random public avatar
    },
    { "id": "antonella", "…": "…" }
  ]
}
```

- **`role`** is what the router reads to choose between the personas. Write it
  as "send this kind of question here": the topics this persona owns, and its
  temperament.
- **Persona prompts** can use `{{name}}`, `{{other}}` and `{{user}}`. The
  director appends a shared block explaining how the call works: lines from
  the other persona, `[director]` messages, speak only as yourself.
- **Knowledge** goes into the system prompt rather than being retrieved per
  turn. The Beckham case is about 6k tokens; putting it in costs nothing per
  turn, while retrieval would add latency to every reply. FamilyFinanceChat's
  avatar plan ([`AVATAR_TRACK.md`](../capstone_fall2026/AVATAR_TRACK.md) §5) recommends the same.
- **Avatars and voices.** The public avatar list has no gender or age field,
  so pick avatars by their preview images
  (`GET https://api.liveavatar.com/v1/avatars/public` → `preview_url`) and give
  each a fitting voice. Avatars shot in the same style look like a pair.

### Included scenarios

| Scenario | Personas | Avatar / voice |
|---|---|---|
| `beckham` (default) | Tony (husband, risk-tolerant) and Antonella (wife, risk-averse). Each answers about their own job, benefits, accounts and side of the family, and defers to the other on theirs. | David (Friendly) / Charon; Helen (Friendly) / Kore |
| `cohosts` | an optimist and a skeptic co-host | two random public avatars, named after them |

The Beckham prompts make both spouses behave as **clients**, never advisers:
short spoken answers, nothing volunteered, unsure about jargon, skeptical of
advisers, and "I don't know" rather than invented figures. Tony's prompt is the
course's Tony brief, plus a section on Antonella being in the meeting.
Antonella's prompt is written in the same style from the client data sheet.

`case.md` is FIN 602 course material (© Mark Fedenia), transcribed from
[`FIN602 Client Data 2025.pdf`](../rag_bio_project/data_pdfs/FIN602%20Client%20Data%202025.pdf). **Do not publish it outside the course without
permission.**

---

## Exporting to Open WebUI

After **End**, **Export** downloads `<scenario>-session-<time>.openwebui.json`.
The converter is `shared/openwebui.ts`, a pure function.

- **The file** is a JSON array of chat rows: the format Open WebUI's own
  "Export chats" writes and **Settings → Data Controls → Import Chats** reads.
  Each row's `chat` object is stored verbatim in Open WebUI's `chat` table.
- **Both message forms** are included: `history.messages` (the id-linked tree
  the web UI renders) and a flat `messages` list (what grading reads).
- **Speakers:** persona lines are labelled `Tony: …`, each reply's model name
  is its speaker, and cut-off lines end with "— (cut off)".
- **Merging:** consecutive lines from the same side become one message.
  FamilyFinanceChat's grading pairs each student message with the *next*
  assistant message and skips a second one in a row; without merging, the
  second speaker in a handoff would be lost.

What has been verified, and how:

| Check | Result |
|---|---|
| Open WebUI's chat format and import endpoint, read from source at v0.8.12 (production) and v0.11.4 (pinned) | matches |
| A sample export run through FamilyFinanceChat's real [`extract_chats.py`](../grading_feature/backend/extract_chats.py) (with its API calls stubbed) | 3 of 3 student questions paired with their full answers; nothing skipped |
| A real import into a running Open WebUI | **not yet done** |

---

## Connecting to FamilyFinanceChat

This repo is a working prototype of two items in FamilyFinanceChat's avatar
plan ([`capstone_fall2026/`](../capstone_fall2026/)): the mentor's **multi-persona switching**
requirement ([`docs/memory/multi-persona-switching-requirement.md`](../docs/memory/multi-persona-switching-requirement.md); Team 3 in
[`AVATAR_SPRINT.md`](../capstone_fall2026/AVATAR_SPRINT.md)) and task **B13, transcript write-back**
([`PROJECT_PLAN.md`](../capstone_fall2026/PROJECT_PLAN.md)). The import facts below are also recorded in
[`docs/memory/openwebui-chat-import-owner-is-caller.md`](../docs/memory/openwebui-chat-import-owner-is-caller.md).

### What Open WebUI provides (verified in its source)

- `POST /api/v1/chats/import` with `{"chats": [rows]}` exists in both v0.8.12
  and v0.11.4. It stores the `chat` object as-is and keeps the given
  `created_at` / `updated_at`. In v0.11.4 it requires the `chat.import` user
  permission, which is on by default.
- **The chat always belongs to the account making the call.** An admin key
  cannot import into a student's history; the request must carry the
  **student's** credentials.
- It is a public REST endpoint, so it fits FamilyFinanceChat's rule: build on
  Open WebUI's extension surfaces, never modify its core.

### Suggested path

1. **Now — manual.** The student clicks Export, then imports the file in Open
   WebUI (Settings → Data Controls → Import Chats). The chat appears in their
   history and in the grading dashboard. Confirm this once against a real
   instance first; [`docker-compose.test.yml`](../docker-compose.test.yml) runs one locally.
2. **Automatic import.** On End, the server posts the same rows to
   `/api/v1/chats/import` with the student's token. The only new piece is
   getting that token: the student pastes an Open WebUI API key once (an admin
   must enable API keys), or the page receives it from step 3.
3. **Launch from inside Open WebUI.** Open the avatar page from Open WebUI (a
   link, a tool or a function) with the signed-in student's session. There are
   then no keys to paste, and the export goes straight into that student's
   history.
4. **Scenarios from the course.** Load scenario prompts and knowledge from
   wherever FamilyFinanceChat keeps its personas (task B12), instead of the
   `server/scenarios/` folder.
5. **Spoken-session metrics** (task B14). The director already sees what typing
   hides: who interrupted whom, how long the student waited, and the talk-time
   split. These can be added to the export as extra fields.

Open questions for the integration:

- **Which Open WebUI version.** Production runs 0.8.12 while the repo pins
  0.11.4. Import works on both, but test against the one that will actually
  run.
- **Consent.** Students' voices go to Google (Gemini) and to LiveAvatar. Get
  consent first, as the avatar track's ethics section requires.
- **Auth.** This server has none. Before students use it, put `/api/duo/start`
  and the websocket behind a login.

---

## Configuration

All in `.env` at the repo root (see `.env.example`). The browser never sees any
of it.

| Variable | Default | Purpose |
|---|---|---|
| `LIVEAVATAR_API_KEY` | — (required) | starts the two avatar sessions |
| `GEMINI_API_KEY` | — (required) | listener, router and both personas |
| `DUO_SCENARIO` | `beckham` | folder in `server/scenarios/` |
| `DUO_AVATAR_IDS` | — | `id1,id2`: overrides the scenario's avatars |
| `DUO_VOICE_<PERSONA_ID>` | from the scenario | e.g. `DUO_VOICE_TONY=Fenrir` |
| `DUO_SILENCE_MS` | `600` | listener end-of-speech silence (below 500, turns never ended) |
| `DUO_LISTENER_MODEL` | `gemini-3.5-transcribe-live` | the ears |
| `DUO_DECISION_MODEL` | `gemini-decision-1` | the router |
| `GEMINI_LIVE_MODEL` | `gemini-3.8-live` | the personas' voices |
| `GEMINI_LIVE_DEBUG` | off | log every line sent to the personas, and every listener stall |
| `PORT` | `8787` | server port |

Tuning constants live at the top of `server/src/duo/stage.ts` (barge-in word
count, audible margin, handoff threshold, context lines) and
`server/src/duo/listener.ts` (stall time, buffer size).

---

## Troubleshooting

The server log (the `pnpm dev` terminal) narrates every call: each decision
with its scores, each barge-in, each handoff. Start there.

| Symptom | Cause / fix |
|---|---|
| `No credits available for start session` | The LiveAvatar account is out of credit. |
| `missing env vars: …` | Fill in `.env` and restart `pnpm dev`; `.env` is only read at startup. |
| No Start button visible | The tiles are sized to fit the window; make the window taller. |
| An avatar answers its own words, or odd lines like "enzo en zo" appear | The mic is hearing the speakers. Use headphones. |
| An avatar does not stop when you talk over it | Say at least 3 words; "mm-hmm" is deliberately ignored. |
| Every answer comes from the same persona | Read the `floor →` log lines: the scores show how sure the router was. Make the personas' `role` text more distinct, or address them by name. |
| Long pause before answers | About 2.5s is expected. Lowering `DUO_SILENCE_MS` below 500 breaks end-of-turn detection. |
| A persona's voice does not suit its face | Set `voice` in `scenario.json`, or `DUO_VOICE_<ID>`. |

---

## Known limitations

- **Latency** is about 2.5s per answer: the price of deciding who speaks before
  anyone does.
- **The transcription model's end-of-turn is unreliable** with a real
  microphone. The 1.2s fallback makes it work, at the cost of that delay.
- **Speaker echo** is heard as the user unless you use headphones.
- **No real Open WebUI import has been tested yet** (see above).
- **No authentication**: this is a local prototype.
- **Two billable avatar sessions per call.**

---

## Repository layout

```
server/src/index.ts              HTTP + websocket entrypoint
server/src/duo/stage.ts          the director: floor, shared context, barge-in, handoffs
server/src/duo/listener.ts       transcription listener (the only thing that hears the mic)
server/src/duo/decision.ts       gemini-decision-1: who answers; did they hand off
server/src/duo/personaBridge.ts  one persona: text turns in, audio out
server/src/duo/scenario.ts       loads server/scenarios/<name>/
server/scenarios/                beckham/, cohosts/
server/src/mediaServer.ts        one LiveAvatar media leg (audio into an avatar)
server/src/liveavatar.ts         LiveAvatar session API
server/src/registry.ts           live sessions + idle watchdog
shared/messages.ts               browser ⇄ server protocol
shared/openwebui.ts              call transcript → Open WebUI chat
web/duo.html, web/src/duo.ts     the duo page

Single-avatar app (see docs/SINGLE_AVATAR.md):
server/src/{session,gptlive,geminiLive,prompts,tools,turns}.ts, server/prompts/,
shared/tools.ts, web/index.html, web/src/{main,socket}.ts, web/src/overlays/, web/public/
```

Notes for coding agents (invariants, measured numbers, gotchas) are in
[AGENTS.md](AGENTS.md).

## License

MIT — see [LICENSE](LICENSE). The vendored GSAP used by the single-avatar
overlays has its own terms ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
`server/scenarios/beckham/case.md` is FIN 602 course material and not covered
by the MIT license.
