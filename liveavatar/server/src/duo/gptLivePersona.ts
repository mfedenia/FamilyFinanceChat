/**
 * One persona's voice on OpenAI GPT-Live (DUO_PERSONA_ENGINE=gptlive), behind
 * the same PersonaVoice interface as the Gemini bridge. Protocol: the GPT-Live
 * v3 contract already used by the single-avatar app (../gptlive.ts) — the event
 * names and session.start shape come from there.
 *
 * ⚠ UNTESTED. Written without an OpenAI key; verify before relying on it:
 *  - GPT-Live is built to HEAR a user. Here it never gets input audio, only
 *    text appends. Whether it answers an `instructions` append with no audio
 *    stream at all is the first thing to check (the single-avatar greeting
 *    proves it speaks on an instructions append, but with a mic attached).
 *  - Appends are "guidance, not commands" and capped at 500 tokens each, so
 *    retrieved documents are split into `thinking` appends and trimmed.
 *  - v3 has no reply-boundary event and no cancel. A reply is considered over
 *    after REPLY_GAP_MS without output (or NO_REPLY_MS with none at all);
 *    `silence()` can only drop audio, not stop generation.
 *  - `delegation` is omitted from session.start (no tools here). The
 *    single-avatar app always sends one; if the API requires it, add the same
 *    block without tools.
 *  - The default model id `gpt-live-1.1` (DUO_GPT_LIVE_MODEL) is as requested,
 *    not verified against the API.
 */

import WebSocket from "ws";
import { config } from "../config";
import type { PersonaEvents, PersonaVoice } from "./personaBridge";
import type { Persona } from "./scenario";

const BASE_URL = "wss://api.openai.com/v1/live/sessions";
const CLOSE_DRAIN_TIMEOUT_MS = 3_000;
/** Same gap the single-avatar app's turn projector uses (../turns.ts TURN_GAP_MS). */
const REPLY_GAP_MS = 1_500;
/** A turn that produced nothing at all for this long is treated as answered. */
const NO_REPLY_MS = 8_000;
/** ≈450 tokens — appends are capped at 500 tokens server-side. */
const APPEND_MAX_CHARS = 1_800;
/** How many document appends go with one turn. */
const MAX_DOCUMENT_APPENDS = 2;

type Json = Record<string, any>;

// Custom voices are sent as {id}; named voices ("marin", "cedar", …) as strings.
const voiceField = (voice: string): string | { id: string } =>
  voice.startsWith("voice_") ? { id: voice } : voice;

export class GptLivePersona implements PersonaVoice {
  private ws: WebSocket | null = null;
  private started = false;
  private closedByServer: (() => void) | null = null;
  private seq = 0;
  /** Appends queued until session.started. */
  private readonly outbox: { type: string; content: string }[] = [];
  /** Context held while a reply is in flight (an append could disturb it). */
  private readonly heldContext: string[] = [];
  private replying = false;
  private muted = false;
  private gotOutput = false;
  private replyTimer: NodeJS.Timeout | null = null;

  constructor(
    readonly persona: Persona,
    private readonly instructions: string,
    private readonly events: PersonaEvents,
    private readonly log: (msg: string) => void,
    /** OpenAI's endpoint; a parameter only so tests can point it at a fake server. */
    private readonly url: string = BASE_URL,
  ) {}

  get busy(): boolean {
    return this.replying;
  }

  run(): Promise<void> {
    return new Promise((resolve) => {
      const ws = new WebSocket(this.url, {
        headers: { Authorization: `Bearer ${config.gptlive.apiKey}` },
        maxPayload: 8 * 1024 * 1024,
      });
      this.ws = ws;
      ws.on("open", () =>
        this.send("session.start", {
          event_id: "start",
          session: {
            model: config.duo.gptLiveModel,
            instructions: this.instructions,
            audio: {
              format: { type: "audio/pcm", rate: 24000 },
              output: { voice: voiceField(this.persona.voice) },
            },
          },
        }),
      );
      ws.on("message", (raw) => {
        try {
          this.handle(JSON.parse(raw.toString()) as Json);
        } catch {
          this.log("gptlive: malformed event");
        }
      });
      ws.on("error", (err) => this.log(`gptlive socket error: ${err.message}`));
      ws.on("close", (code, reason) => {
        this.log(`gptlive socket closed (code=${code}${reason.length ? `, ${reason}` : ""})`);
        this.ws = null;
        if (this.replying) this.endReply(true);
        if (!this.started) this.events.onError(`${this.persona.name}: GPT-Live session did not start`);
        resolve();
      });
    });
  }

  say(text: string, documents?: string): void {
    for (const line of this.heldContext.splice(0)) this.append("session.thinking.append", line);
    if (documents) {
      for (const chunk of chunks(documents, APPEND_MAX_CHARS).slice(0, MAX_DOCUMENT_APPENDS)) {
        this.append("session.thinking.append", chunk);
      }
    }
    // A new turn replaces whatever was playing; its audio is wanted from byte one.
    this.muted = false;
    this.replying = true;
    this.gotOutput = false;
    this.armReplyTimer(NO_REPLY_MS);
    this.append(
      "session.instructions.append",
      `${text.slice(0, APPEND_MAX_CHARS - 200)}\n\n[director] It is your turn: answer that now, as ${this.persona.name}, briefly.`,
    );
  }

  addContext(text: string): void {
    if (this.replying) this.heldContext.push(text);
    else this.append("session.thinking.append", text);
  }

  silence(): void {
    if (this.replying) this.muted = true;
  }

  async close(): Promise<void> {
    const ws = this.ws;
    if (!ws) return;
    const drained = new Promise<boolean>((resolve) => {
      const timer = setTimeout(() => resolve(false), CLOSE_DRAIN_TIMEOUT_MS);
      this.closedByServer = () => {
        clearTimeout(timer);
        resolve(true);
      };
      ws.once("close", () => {
        clearTimeout(timer);
        resolve(true);
      });
    });
    this.send("session.close", { event_id: "close" });
    if (!(await drained)) ws.close();
  }

  // ── internals ──────────────────────────────────────────────────────────────

  private append(type: string, content: string): void {
    const trimmed = content.trim().slice(0, APPEND_MAX_CHARS);
    if (!trimmed) return;
    if (!this.started) {
      this.outbox.push({ type, content: trimmed });
      return;
    }
    this.seq += 1;
    this.send(type, { event_id: `duo_${this.seq}`, delegation_id: null, content: trimmed });
    if (config.gptlive.debug) this.log(`${type} → ${trimmed.slice(0, 140).replace(/\n/g, " ⏎ ")}`);
  }

  private send(type: string, payload: Json): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ ...payload, type }));
  }

  private handle(event: Json): void {
    switch (event.type) {
      case "session.started":
        this.started = true;
        this.log(`gptlive ready (model=${config.duo.gptLiveModel}, voice=${this.persona.voice})`);
        for (const a of this.outbox.splice(0)) this.append(a.type, a.content);
        this.events.onReady();
        break;
      case "session.output_audio.delta":
        if (typeof event.delta === "string" && event.delta) {
          this.noteOutput();
          if (!this.muted) this.events.onAudio(event.delta);
        }
        break;
      case "session.output_transcript.delta":
        if (typeof event.delta === "string" && event.delta) {
          this.noteOutput();
          if (!this.muted) this.events.onText(event.delta);
        }
        break;
      case "session.closed":
        this.closedByServer?.();
        break;
      case "error":
        // Errors on an append are not fatal to the call; a session that never
        // starts is reported when the socket closes.
        this.log(`gptlive error: ${JSON.stringify(event).slice(0, 400)}`);
        break;
      default:
        break;
    }
  }

  /** Output keeps the reply open; quiet for REPLY_GAP_MS ends it. */
  private noteOutput(): void {
    if (!this.replying) return; // output outside a turn (e.g. a late tail)
    this.gotOutput = true;
    this.armReplyTimer(REPLY_GAP_MS);
  }

  private armReplyTimer(ms: number): void {
    if (this.replyTimer) clearTimeout(this.replyTimer);
    this.replyTimer = setTimeout(() => {
      this.replyTimer = null;
      if (!this.gotOutput) this.log("gptlive: no reply to the turn");
      this.endReply(false);
    }, ms);
  }

  private endReply(dropped: boolean): void {
    if (this.replyTimer) clearTimeout(this.replyTimer);
    this.replyTimer = null;
    if (!this.replying) return;
    const cut = dropped || this.muted;
    this.replying = false;
    this.muted = false;
    this.events.onTurnEnd(cut);
    for (const line of this.heldContext.splice(0)) this.append("session.thinking.append", line);
  }
}

function chunks(text: string, size: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out;
}
