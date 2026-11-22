/**
 * The duo's single ear: a transcription-only Gemini Live session
 * (gemini-3.5-transcribe-live) that hears every mic frame and reports
 * utterances. Neither persona hears audio — they get text from the director —
 * which is what lets the decision model choose a speaker BEFORE anyone talks.
 *
 * What the model sends (observed):
 *   voiceActivity ACTIVITY_START          ~0.3s after speech begins
 *   serverContent.interimInputTranscription  the whole utterance so far, ~2/s
 *   serverContent.inputTranscription      the final text  ┐ together, ~1.0s
 *   voiceActivity ACTIVITY_END                            ┘ after speech ends
 * Interim texts are cumulative (each repeats the full utterance), not deltas.
 *
 * Its end-of-turn is not reliable. With a real mic in a live session it never
 * sent a final at all, and without one its transcript is cumulative for the
 * whole connection: every update repeats everything said so far, glued
 * together without spaces and with punctuation revised along the way
 * ("…that's why I guess.enzo en zoHow's your…"). So:
 *  - a stalled interim is promoted to final after STALL_MS;
 *  - delivery is tracked as an offset into that running transcript, counted
 *    in letters and digits only — spacing/punctuation/case revisions cannot
 *    shift it — and each utterance is only the part past the offset;
 *  - once the running transcript passes MAX_BUFFER_CHARS, the connection is
 *    recycled at the next pause, which resets the model's buffer.
 *
 * There is no conversation state worth keeping, so a dropped or goAway'd
 * connection just reconnects fresh.
 */

import WebSocket from "ws";
import { config } from "../config";

const BASE_URL =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
const INPUT_MIME = "audio/pcm;rate=24000";
const MAX_RECONNECTS = 3;
/** No new interim text for this long → the utterance is over (fallback). */
const STALL_MS = 1_200;
/** Recycle the connection (at a pause) once its running transcript is this long. */
const MAX_BUFFER_CHARS = 1_500;

type Json = Record<string, any>;

export interface ListenerEvents {
  onReady(): void;
  /** The user started talking (voice activity) — the barge-in signal. */
  onSpeechStart(): void;
  /** The utterance so far (cumulative). */
  onInterim(text: string): void;
  /** The utterance is over; this is its final text. */
  onFinal(text: string): void;
  onError(message: string): void;
}

export class Listener {
  private ws: WebSocket | null = null;
  private closing = false;
  private setupDone = false;
  private ready = false;
  /** Final text already delivered for the current utterance. */
  private finalSent = false;
  private lastInterim = "";
  private stallTimer: NodeJS.Timeout | null = null;
  /** The model's latest full transcript (cumulative), as sent. */
  private lastRaw = "";
  /** Letters+digits of the running transcript already delivered as finals. */
  private consumed = 0;
  /** Recycle requested: close at the next pause, reconnect fresh. */
  private recycling = false;

  constructor(
    private readonly events: ListenerEvents,
    private readonly log: (msg: string) => void,
  ) {}

  async run(): Promise<void> {
    let failures = 0;
    while (!this.closing) {
      const healthy = await this.connectOnce();
      if (this.closing) break;
      this.recycling = false;
      failures = healthy ? 0 : failures + 1;
      if (failures >= MAX_RECONNECTS) {
        this.events.onError("the listener could not stay connected");
        return;
      }
      this.log("listener reconnecting");
    }
  }

  sendMicAudio(audioB64: string): void {
    if (!this.setupDone || this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ realtimeInput: { audio: { data: audioB64, mimeType: INPUT_MIME } } }));
  }

  close(): void {
    this.closing = true;
    this.clearStall();
    this.ws?.close(1000, "session ended");
  }

  private connectOnce(): Promise<boolean> {
    this.setupDone = false;
    return new Promise((resolve) => {
      const ws = new WebSocket(`${BASE_URL}?key=${encodeURIComponent(config.gemini.apiKey)}`);
      this.ws = ws;
      ws.on("open", () => {
        ws.send(
          JSON.stringify({
            setup: {
              model: `models/${config.duo.listenerModel.replace(/^models\//, "")}`,
              inputAudioTranscription: {},
              realtimeInputConfig: {
                automaticActivityDetection: { silenceDurationMs: config.duo.silenceMs },
              },
            },
          }),
        );
      });
      ws.on("message", (raw) => {
        try {
          this.handle(JSON.parse(raw.toString()) as Json);
        } catch (err) {
          this.log(`listener: bad message ${err instanceof Error ? err.message : err}`);
        }
      });
      ws.on("error", (err) => this.log(`listener socket error: ${err.message}`));
      ws.on("close", (code, reason) => {
        this.log(`listener socket closed (code=${code}${reason.length ? `, ${reason}` : ""})`);
        const healthy = this.setupDone;
        this.setupDone = false;
        this.ws = null;
        // An utterance cut by the reconnect still counts — deliver what we heard.
        this.flushInterim();
        resolve(healthy);
      });
    });
  }

  private handle(msg: Json): void {
    if (msg.setupComplete) {
      this.setupDone = true;
      this.resetBuffer();
      this.log(`listener ready (model=${config.duo.listenerModel}, silence=${config.duo.silenceMs}ms)`);
      if (!this.ready) {
        this.ready = true;
        this.events.onReady();
      }
      return;
    }
    const activity = msg.voiceActivity?.type;
    if (activity === "ACTIVITY_START") this.events.onSpeechStart();
    const sc = msg.serverContent ?? {};

    const raw = sc.interimInputTranscription?.text;
    if (typeof raw === "string" && raw.trim()) {
      const text = this.unconsumed(raw);
      if (text && text !== this.lastInterim) {
        // New words past the offset are a new utterance, ACTIVITY_START or not.
        this.finalSent = false;
        this.lastInterim = text;
        this.events.onInterim(text);
        this.armStall();
      }
    }

    const finalRaw = sc.inputTranscription?.text;
    if (typeof finalRaw === "string" && finalRaw.trim()) {
      // No finalSent guard: the offset already excludes anything delivered,
      // and a final can carry words past a fallback-closed utterance.
      const text = this.unconsumed(finalRaw);
      if (text) {
        this.finalSent = true;
        this.lastInterim = "";
        this.clearStall();
        this.events.onFinal(text);
      }
      // A real final closes the model's turn; its next transcript starts over.
      this.resetBuffer();
    }
    // Activity ended without a final transcript: promote the last interim.
    if (activity === "ACTIVITY_END") this.flushInterim();
    if (msg.goAway) this.log(`listener goAway (timeLeft=${msg.goAway.timeLeft ?? "?"})`);
  }

  /**
   * The part of the model's cumulative transcript `raw` past what was already
   * delivered. Counting only letters and digits makes the offset survive the
   * model's re-spacing and re-punctuating of earlier text.
   */
  private unconsumed(raw: string): string {
    this.lastRaw = raw;
    const total = alnumLength(raw);
    // Shorter than what we delivered: the model started a fresh transcript.
    if (total < this.consumed) this.consumed = 0;
    // The model glues utterances on without a space; drop the seam's punctuation.
    return sliceAfterAlnum(raw, this.consumed).replace(/^[\s\p{P}]+/u, "").trim();
  }

  private resetBuffer(): void {
    this.consumed = 0;
    this.lastRaw = "";
  }

  private armStall(): void {
    this.clearStall();
    this.stallTimer = setTimeout(() => {
      this.stallTimer = null;
      if (this.finalSent || !this.lastInterim) return;
      if (config.gemini.debug) this.log(`listener: no final after ${STALL_MS}ms — closing the utterance`);
      this.flushInterim();
    }, STALL_MS);
  }

  private clearStall(): void {
    if (this.stallTimer) clearTimeout(this.stallTimer);
    this.stallTimer = null;
  }

  private flushInterim(): void {
    this.clearStall();
    if (this.finalSent || !this.lastInterim.trim()) return;
    this.finalSent = true;
    const text = this.lastInterim.trim();
    this.lastInterim = "";
    // Everything the model has transcribed so far is now delivered.
    this.consumed = alnumLength(this.lastRaw);
    this.events.onFinal(text);
    // A pause is the one safe moment to recycle an overgrown buffer.
    if (this.lastRaw.length > MAX_BUFFER_CHARS && !this.recycling && this.ws) {
      this.recycling = true;
      this.log(`listener: transcript buffer at ${this.lastRaw.length} chars — recycling the connection`);
      this.ws.close(1000, "recycle");
    }
  }
}

/** Letters and digits in `s` — the transcript offset's unit. */
function alnumLength(s: string): number {
  let n = 0;
  for (const ch of s) if (/[\p{L}\p{N}]/u.test(ch)) n++;
  return n;
}

/** `s` after its first `n` letters/digits (whatever spacing surrounds them). */
function sliceAfterAlnum(s: string, n: number): string {
  if (n <= 0) return s;
  let seen = 0;
  let i = 0;
  for (const ch of s) {
    i += ch.length;
    if (/[\p{L}\p{N}]/u.test(ch) && ++seen === n) return s.slice(i);
  }
  return "";
}
