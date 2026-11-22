/**
 * One persona's voice: a Gemini Live session that takes TEXT turns from the
 * director and answers in audio. It never hears the mic — the listener does —
 * so it only ever speaks when it has been given the floor.
 *
 * Shared context: every line of the conversation reaches BOTH personas. The
 * one with the floor gets it as a turn (`say`, turnComplete=true); the other
 * gets it as silent context (`addContext`, turnComplete=false — verified to
 * never trigger a reply, and remembered). Context for a persona that is
 * mid-reply waits for that reply to end: any client turn interrupts Gemini.
 *
 * Generation boundaries (verified): a reply ends with `turnComplete`, always —
 * a reply replaced by a newer turn sends `interrupted` THEN `turnComplete`,
 * and only then does the new reply start. So `turnComplete` is the one
 * boundary; `interrupted` only colors it.
 *
 * Cutting it off: the director clears the avatar's buffer, then `silence()`
 * drops whatever still streams from that reply until its boundary.
 *
 * Connections last ~10 minutes; sessionResumption handles + goAway reconnect
 * transparently (same scheme as geminiLive.ts).
 */

import WebSocket from "ws";
import { config } from "../config";
import type { Persona } from "./scenario";

const BASE_URL =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
const OUTPUT_RATE = 24000;
const MAX_RECONNECTS = 3;
const CLOSE_TIMEOUT_MS = 1_000;

type Json = Record<string, any>;

export interface PersonaEvents {
  onReady(): void;
  onAudio(audioB64: string): void;
  /** Transcript delta of what the persona is saying. */
  onText(delta: string): void;
  /** The reply ended — finished, or abandoned because a new turn arrived. */
  onTurnEnd(interrupted: boolean): void;
  onError(message: string): void;
}

export class PersonaBridge {
  private ws: WebSocket | null = null;
  private closing = false;
  private setupDone = false;
  private ready = false;
  private resumeHandle: string | null = null;
  private goAway = false;
  private warnedRate = false;
  /** Client turns held while (re)connecting; flushed on setupComplete. */
  private readonly outbox: { text: string; complete: boolean }[] = [];
  /** Silent context held while a reply is in flight. */
  private readonly heldContext: string[] = [];
  /** Turns sent whose reply has not reached its turnComplete yet. */
  private pending = 0;
  /** The current reply was replaced (`interrupted` seen before its boundary). */
  private replaced = false;
  /** Drop audio + text until the current reply's boundary. */
  private muted = false;

  constructor(
    readonly persona: Persona,
    private readonly instructions: string,
    private readonly events: PersonaEvents,
    private readonly log: (msg: string) => void,
  ) {}

  async run(): Promise<void> {
    let failures = 0;
    while (!this.closing) {
      const out = await this.connectOnce();
      if (this.closing) break;
      failures = out.healthy ? 0 : failures + 1;
      const resumable =
        this.resumeHandle !== null && (out.goAway || out.healthy) && failures < MAX_RECONNECTS;
      if (!resumable) {
        this.events.onError(`${this.persona.name}: ${out.reason || `socket closed (code ${out.code})`}`);
        return;
      }
      this.log(`reconnecting with resumption handle (${out.goAway ? "goAway" : `code ${out.code}`})`);
    }
  }

  /** Give this persona the floor with `text` as the user turn. */
  say(text: string): void {
    // Anything held back goes first, so the turn lands on a complete context.
    // (Sending while a reply is in flight replaces that reply — which is what
    // a new turn does anyway.)
    for (const line of this.heldContext.splice(0)) this.queue(line, false);
    this.pending += 1;
    this.queue(text, true);
  }

  /** Something the persona should know but not answer (the other side's line). */
  addContext(text: string): void {
    if (this.pending > 0) this.heldContext.push(text);
    else this.queue(text, false);
  }

  /** Stop hearing the current reply (the avatar's buffer was just cleared). */
  silence(): void {
    if (this.pending > 0) this.muted = true;
  }

  get busy(): boolean {
    return this.pending > 0;
  }

  async close(): Promise<void> {
    this.closing = true;
    const ws = this.ws;
    if (!ws) return;
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        ws.terminate();
        resolve();
      }, CLOSE_TIMEOUT_MS);
      ws.once("close", () => {
        clearTimeout(timer);
        resolve();
      });
      ws.close(1000, "session ended");
    });
  }

  private queue(text: string, complete: boolean): void {
    if (this.setupDone) this.sendTurn(text, complete);
    else this.outbox.push({ text, complete });
  }

  private sendTurn(text: string, complete: boolean): void {
    this.send({
      clientContent: { turns: [{ role: "user", parts: [{ text }] }], turnComplete: complete },
    });
    if (config.gemini.debug) {
      this.log(`${complete ? "turn" : "context"} → ${text.slice(0, 160).replace(/\n/g, " ⏎ ")}`);
    }
  }

  private connectOnce(): Promise<{ healthy: boolean; goAway: boolean; code: number; reason: string }> {
    this.setupDone = false;
    this.goAway = false;
    return new Promise((resolve) => {
      const ws = new WebSocket(`${BASE_URL}?key=${encodeURIComponent(config.gemini.apiKey)}`, {
        maxPayload: 16 * 1024 * 1024,
      });
      this.ws = ws;
      let errored = "";
      ws.on("open", () => this.sendSetup());
      ws.on("message", (raw) => {
        try {
          this.handle(JSON.parse(raw.toString()) as Json);
        } catch (err) {
          this.log(`bad message: ${err instanceof Error ? err.message : err}`);
        }
      });
      ws.on("error", (err) => {
        errored = err.message;
        this.log(`socket error: ${err.message}`);
      });
      ws.on("close", (code, reasonBuf) => {
        const reason = reasonBuf.toString() || errored;
        this.log(`socket closed (code=${code}${reason ? `, ${reason}` : ""})`);
        this.ws = null;
        const healthy = this.setupDone;
        this.setupDone = false;
        // Replies cut by the drop are over as far as the director is concerned.
        if (this.pending > 0) {
          this.pending = 0;
          this.muted = false;
          this.replaced = false;
          this.events.onTurnEnd(true);
          for (const line of this.heldContext.splice(0)) this.queue(line, false);
        }
        resolve({ healthy, goAway: this.goAway, code, reason });
      });
    });
  }

  private sendSetup(): void {
    const model = config.gemini.model.replace(/^models\//, "");
    this.send({
      setup: {
        model: `models/${model}`,
        generationConfig: {
          responseModalities: ["AUDIO"],
          speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: this.persona.voice } } },
        },
        systemInstruction: { parts: [{ text: this.instructions }] },
        outputAudioTranscription: {},
        contextWindowCompression: { slidingWindow: {} },
        sessionResumption: this.resumeHandle ? { handle: this.resumeHandle } : {},
      },
    });
  }

  private send(msg: Json): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  private handle(msg: Json): void {
    if (msg.setupComplete) {
      this.setupDone = true;
      this.log(`ready (model=${config.gemini.model}, voice=${this.persona.voice})`);
      for (const t of this.outbox.splice(0)) this.sendTurn(t.text, t.complete);
      if (!this.ready) {
        this.ready = true;
        this.events.onReady();
      }
    }

    const sc = msg.serverContent;
    if (sc) {
      for (const part of sc.modelTurn?.parts ?? []) {
        const data = part?.inlineData;
        if (!data || typeof data.data !== "string" || !data.data) continue;
        const mime = String(data.mimeType ?? "");
        if (!mime.startsWith("audio/")) continue;
        const rate = /rate=(\d+)/.exec(mime)?.[1];
        if (rate && Number(rate) !== OUTPUT_RATE && !this.warnedRate) {
          this.warnedRate = true;
          this.log(`WARNING: output audio is ${mime}, media leg expects ${OUTPUT_RATE}Hz`);
        }
        if (!this.muted) this.events.onAudio(data.data);
      }
      const said = sc.outputTranscription?.text;
      if (typeof said === "string" && said && !this.muted) this.events.onText(said);

      if (sc.interrupted) this.replaced = true;
      if (sc.turnComplete && this.pending > 0) {
        const cut = this.replaced || this.muted;
        this.pending -= 1;
        this.replaced = false;
        this.muted = false;
        this.events.onTurnEnd(cut);
        if (this.pending === 0) {
          for (const line of this.heldContext.splice(0)) this.queue(line, false);
        }
      }
    }

    if (msg.sessionResumptionUpdate) {
      const u = msg.sessionResumptionUpdate;
      if (u.resumable && typeof u.newHandle === "string" && u.newHandle) this.resumeHandle = u.newHandle;
    }
    if (msg.goAway) {
      this.goAway = true;
      this.log(`goAway (timeLeft=${msg.goAway.timeLeft ?? "?"})`);
    }
  }
}
