# AGENTS.md

Operational map for coding agents (and fast-moving humans) working in
`avatar_app/`. What it is and how to run it: [README.md](README.md). Design
rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Task list:
`../capstone_fall2026/AVATAR_TRACK.md` §16. Repository-wide rules — above all,
**never modify Open WebUI** — are in `../CLAUDE.md` and
`../capstone_fall2026/COMPATIBILITY_POLICY.md`, and they apply here too.

## Shape

pnpm workspace, TypeScript everywhere, deliberately few dependencies. Forked
from heygen-com/liveavatar-gpt-live-demos @ `daa8cc9`; AV1 removed the
upstream tutor domain.

```
shared/     The contract. messages.ts (wire protocol).
            NOT a package — both sides import it by relative path.
server/     The bridge. Node + ws only. `pnpm dev` = tsx watch.
            prompts/*.md is a PLACEHOLDER client persona until AV2.
            src/prompts.ts holds the non-negotiable mechanics prompts.
web/        Developer test harness. Vite + livekit-client. Not the student UI.
```

## Commands

Run from `avatar_app/`. pnpm comes from corepack: prefix with `corepack` if
`pnpm` is not on PATH (and then call workspace scripts directly — the root
scripts shell out to a bare `pnpm`).

```bash
corepack pnpm install
corepack pnpm run setup          # interactive: collects + verifies API keys, writes .env
corepack pnpm --parallel -r dev  # server :8787 + harness :5173
corepack pnpm -r typecheck       # tsc --noEmit in server and web
corepack pnpm -r build           # web/dist — the server serves it in production
```

`scripts/check-env.mjs` mirrors `missingConfig()` in `server/src/config.ts` —
change one, change both.

## Invariants — do not "fix" these

- **One continuous utterance.** The avatar never gets a per-turn `speak_end`
  or interrupt. GPT-Live owns turn-taking; the bridge only moves audio, in
  arrival order.
- **Turns are projected, not received.** v3 has no turn events, only timed
  transcript deltas. `server/src/turns.ts` groups them per speaker with a gap
  timeout (`TURN_GAP_MS`). A user turn *opening* there is the barge-in signal.
  Tune the gap, don't bypass the projector. AV4's saved transcript is built
  from these same turns.
- **No VAD in the browser.** `micCapture.ts` streams every frame, silence
  included. Turn detection belongs to the model.
- **Barge-in is two-step.** A user turn opening only *starts a watch*
  (`session.ts`); the buffer is cleared only if the model actually stops
  producing audio. Acting on the turn alone cuts the avatar off on every
  "mm-hmm".
- **Every tool call gets a result AND a continue** (`gptlive.ts`). No tools
  are registered today, but the loop stays: an unanswered call blocks every
  later delegation. If tools return, restore `parallel_tool_calls: false`.
- **The greeting is gated on the media socket** (`session.ts`). Audio sent
  before the media leg is up is dropped with no replay. The greeting is a
  `session.instructions.append`, not commentary.
- **Append kinds mean what v3 says they mean** (`bridge.append`).
  `instructions` = directive to follow; `commentary` = information to SAY;
  `thinking` = silent context.
- **Startup instructions are immutable.** Everything the persona needs must be
  in `session.start` (≤ 16k tokens). Mid-session appends are ≤ 500 tokens and
  the model acts on them immediately — they are for nudges, not content.
- **Model output is untrusted.** Render transcript text with `textContent`
  only; never interpolate it into HTML.

## FamilyFinanceChat rules that bite here

- **No secrets in the repo.** `.env` is gitignored at both levels; keep it so.
- **Open WebUI only over its public REST API**, and all of those calls in one
  module (`openwebui.ts`, from AV3) so an upstream change breaks in one place.
- **Every billed session must reach `stopSession`.** The 60 s idle reaper in
  `registry.ts` is the backstop, not the plan. The hard session cap arrives in
  AV5.

## Gotchas

- The session is billable from `/api/session/start`.
- `stopSession` claims via Map delete, so racing stops are idempotent.
- One browser per session: the second websocket for the same id is rejected.
- Media-server readiness means the server said `session.state_updated:
  "connected"` — NOT that the socket opened.
- GPT-Live v3 shapes only: don't reintroduce v2 events (`turn.*`,
  `output_audio.delta.audio`, `delegation.function_call_output.create`,
  `channel:`).
- LITE protocol reference: https://docs.liveavatar.com/docs/lite-mode/events.md
  — `agent.speak` / `agent.interrupt` / `session.keep_alive` are the public
  contract.
