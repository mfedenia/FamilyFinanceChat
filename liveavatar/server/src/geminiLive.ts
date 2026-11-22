/**
 * Per-session Gemini Live upstream websocket (BidiGenerateContent), selected
 * with LIVE_PROVIDER=gemini. A drop-in for GptLiveBridge: same LiveBridge
 * surface, same GptLiveEvents sink, so session.ts and everything downstream
 * (media leg, turns, cards, recap) are unchanged.
 *
 * Where Gemini differs from GPT-Live v3, and what this file does about it:
 *  - Tools live on the live model itself — no delegation, no backend model.
 *    `toolCall.functionCalls[]` is answered with ONE `toolResponse` carrying
 *    every result. Declared NON_BLOCKING, so the model keeps speaking while a
 *    card renders; `scheduling` in each response says whether the result
 *    should be talked about (WHEN_IDLE) or just absorbed (SILENT).
 *  - The model reports barge-in itself (`serverContent.interrupted`): it has
 *    already abandoned that output, which is exactly the state session.ts's
 *    two-step watch infers for GPT-Live — so it goes out as onModelYielded.
 *  - Turns: the model sends `turnComplete`, which closes the assistant turn;
 *    transcripts still flow through TurnProjector (no timeline offsets used).
 *  - One text channel. All three append kinds become `clientContent` user
 *    turns tagged `[system]` (the prompt tells the model what that means);
 *    `thinking` is queued with turnComplete=false so it does not trigger speech.
 *  - A connection lives ~10 minutes. `sessionResumption` handles are tracked
 *    and the bridge reconnects transparently on goAway / an unexpected close;
 *    `contextWindowCompression` lifts the 15-minute audio-session cap.
 */

import WebSocket from "ws";
import { geminiFunctionDeclarations } from "../../shared/tools";
import { config } from "./config";
import type { AppendKind, GptLiveEvents, LiveBridge } from "./gptlive";
import {
  DEFAULT_GREETING,
  DEFAULT_INSTRUCTIONS,
  GEMINI_TOOL_DIRECTIVE,
  GREETING_PREAMBLE,
  LESSON_DIRECTIVE,
} from "./prompts";
import { TurnProjector } from "./turns";

const BASE_URL =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

// The mic worklet and the media-server leg are both fixed at 24kHz PCM16.
// Gemini accepts any input rate named in the mime type and outputs 24kHz.
const INPUT_MIME = "audio/pcm;rate=24000";
const OUTPUT_RATE = 24000;

// Back-to-back reconnects without a healthy connection in between before the
// leg gives up (and session.ts ends the session).
const MAX_RECONNECTS = 3;
const CLOSE_TIMEOUT_MS = 1_000;
// Gemini's transcript arrives in bursts with multi-second gaps mid-sentence,
// and `turnComplete` closes assistant turns explicitly — so the projector's
// gap only has to catch a turn that never completes.
const TRANSCRIPT_GAP_MS = 5_000;

type Json = Record<string, any>;

interface ConnectionOutcome {
  /** setupComplete arrived — the connection worked before it ended. */
  healthy: boolean;
  goAway: boolean;
  code: number;
  reason: string;
}

export class GeminiLiveBridge implements LiveBridge {
  readonly reportsYield = true;
  private ws: WebSocket | null = null;
  private closing = false;
  private setupDone = false;
  private ready = false;
  private resumeHandle: string | null = null;
  private goAway = false;
  private warnedRate = false;
  private appendSeq = 0;
  private readonly handledCalls = new Set<string>();
  private readonly seenTypes = new Set<string>();
  private readonly turns: TurnProjector;

  constructor(
    private readonly events: GptLiveEvents,
    private readonly log: (msg: string) => void = () => {},
  ) {
    this.turns = new TurnProjector({
      onTurn: (turn) => this.events.onTurn(turn),
      onUserTurnStarted: () => this.events.onUserTurnStarted(),
    }, TRANSCRIPT_GAP_MS);
  }

  async run(): Promise<void> {
    let failures = 0;
    try {
      while (!this.closing) {
        const out = await this.connectOnce();
        if (this.closing) break;
        failures = out.healthy ? 0 : failures + 1;
        const resumable =
          this.resumeHandle !== null &&
          (out.goAway || (out.healthy && out.code !== 1000)) &&
          failures < MAX_RECONNECTS;
        if (!resumable) {
          if (out.code !== 1000 || !out.healthy) {
            const why = out.reason || `socket closed (code ${out.code})`;
            this.events.onError(`Gemini Live: ${why}`);
          }
          break;
        }
        this.log(
          `gemini reconnecting with resumption handle (${out.goAway ? "goAway" : `code ${out.code}`})`,
        );
      }
    } finally {
      this.turns.dispose();
    }
  }

  sendMicAudio(audioB64: string): void {
    // Anything before setupComplete is a protocol error; during a reconnect
    // there is nowhere for it to go.
    if (!this.setupDone) return;
    this.send({ realtimeInput: { audio: { data: audioB64, mimeType: INPUT_MIME } } });
  }

  append(kind: AppendKind, text: string): void {
    const trimmed = text.trim();
    if (!trimmed || !this.setupDone) return;
    this.appendSeq += 1;
    const body =
      kind === "commentary"
        ? `[system] Tell the learner this, in your own words: ${trimmed}`
        : kind === "thinking"
          ? `[system] Context for later — do not respond to this now: ${trimmed}`
          : `[system] ${trimmed}`;
    this.send({
      clientContent: {
        turns: [{ role: "user", parts: [{ text: body }] }],
        // true starts a response; `thinking` is silent context only.
        turnComplete: kind !== "thinking",
      },
    });
    if (config.gemini.debug) this.log(`gemini append #${this.appendSeq} (${kind})`);
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

  // ── connection ──────────────────────────────────────────────────────────────

  private connectOnce(): Promise<ConnectionOutcome> {
    this.setupDone = false;
    this.goAway = false;
    return new Promise((resolve) => {
      const url = `${BASE_URL}?key=${encodeURIComponent(config.gemini.apiKey)}`;
      const ws = new WebSocket(url, { maxPayload: 16 * 1024 * 1024 });
      this.ws = ws;
      let errored = "";

      ws.on("open", () => this.sendSetup());
      ws.on("message", (raw) => {
        let msg: Json;
        try {
          msg = JSON.parse(raw.toString()) as Json;
        } catch {
          this.log("gemini: malformed message");
          return;
        }
        try {
          this.handleMessage(msg);
        } catch (err) {
          this.log(`gemini: handler error ${err instanceof Error ? err.message : err}`);
        }
      });
      ws.on("error", (err) => {
        errored = err.message;
        this.log(`gemini socket error: ${err.message}`);
      });
      ws.on("close", (code, reasonBuf) => {
        const reason = reasonBuf.toString() || errored;
        this.log(`gemini socket closed (code=${code}${reason ? `, reason=${reason}` : ""})`);
        this.ws = null;
        const healthy = this.setupDone;
        this.setupDone = false;
        resolve({ healthy, goAway: this.goAway, code, reason });
      });
    });
  }

  private sendSetup(): void {
    const model = config.gemini.model.startsWith("models/")
      ? config.gemini.model
      : `models/${config.gemini.model}`;
    this.send({
      setup: {
        model,
        generationConfig: {
          responseModalities: ["AUDIO"],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: config.gemini.voice } },
          },
        },
        systemInstruction: {
          parts: [{ text: DEFAULT_INSTRUCTIONS + GEMINI_TOOL_DIRECTIVE + LESSON_DIRECTIVE }],
        },
        tools: [{ functionDeclarations: geminiFunctionDeclarations() }],
        inputAudioTranscription: {},
        outputAudioTranscription: {},
        contextWindowCompression: { slidingWindow: {} },
        sessionResumption: this.resumeHandle ? { handle: this.resumeHandle } : {},
      },
    });
  }

  private send(msg: Json): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  // ── server messages ─────────────────────────────────────────────────────────

  private handleMessage(msg: Json): void {
    this.debugShape(msg);

    if (msg.setupComplete) {
      this.setupDone = true;
      this.log(`gemini setup complete (model=${config.gemini.model}, voice=${config.gemini.voice})`);
      if (!this.ready) {
        // First connection only — a resumed session already greeted.
        this.ready = true;
        const greeting = DEFAULT_GREETING;
        if (greeting) this.append("instructions", GREETING_PREAMBLE + greeting);
        this.events.onReady();
      }
    }

    if (msg.serverContent) this.handleServerContent(msg.serverContent);
    if (msg.toolCall) this.handleToolCall(msg.toolCall);

    if (msg.toolCallCancellation) {
      this.log(`gemini tool calls cancelled: ${JSON.stringify(msg.toolCallCancellation.ids ?? [])}`);
    }

    if (msg.sessionResumptionUpdate) {
      const u = msg.sessionResumptionUpdate;
      if (u.resumable && typeof u.newHandle === "string" && u.newHandle) {
        this.resumeHandle = u.newHandle;
      }
    }

    if (msg.goAway) {
      this.goAway = true;
      this.log(`gemini goAway (timeLeft=${msg.goAway.timeLeft ?? "?"})`);
    }
  }

  private handleServerContent(sc: Json): void {
    for (const part of sc.modelTurn?.parts ?? []) {
      const data = part?.inlineData;
      if (!data || typeof data.data !== "string" || !data.data) continue;
      const mime = String(data.mimeType ?? "");
      if (!mime.startsWith("audio/")) continue;
      const rate = /rate=(\d+)/.exec(mime)?.[1];
      if (rate && Number(rate) !== OUTPUT_RATE && !this.warnedRate) {
        this.warnedRate = true;
        this.log(`gemini WARNING: output audio is ${mime}, media leg expects ${OUTPUT_RATE}Hz`);
      }
      this.events.onAudio(data.data);
    }

    const heard = sc.inputTranscription?.text;
    if (typeof heard === "string" && heard) {
      this.turns.fragment("user", heard, null, null);
    }

    const said = sc.outputTranscription?.text;
    if (typeof said === "string" && said) {
      // The model is answering: whatever the learner said is over.
      this.turns.close("user");
      this.turns.fragment("assistant", said, null, null);
    }

    if (sc.interrupted) {
      this.log("gemini: model interrupted by user");
      this.turns.close("assistant");
      this.events.onModelYielded?.();
    }
    if (sc.turnComplete) this.turns.close("assistant");
  }

  /**
   * Every call gets exactly one response, batched into one toolResponse —
   * rendered, rejected, or malformed alike, as with GPT-Live.
   */
  private handleToolCall(toolCall: Json): void {
    const responses: Json[] = [];
    for (const call of toolCall.functionCalls ?? []) {
      const id = typeof call?.id === "string" ? call.id : "";
      if (id && this.handledCalls.has(id)) continue;
      if (id) this.handledCalls.add(id);
      const name = typeof call?.name === "string" && call.name ? call.name : null;
      const args =
        call?.args && typeof call.args === "object" ? (call.args as Record<string, unknown>) : {};
      const result = this.events.onToolCall(name, args);
      // The recap result carries the words to walk through, and a failure
      // should be acknowledged — both worth speaking about. A shown card is
      // not: the model is already talking about the word.
      const speak = name === "show_learned_words" || result.shown !== true;
      responses.push({
        id,
        name: name ?? "",
        response: { ...result, scheduling: speak ? "WHEN_IDLE" : "SILENT" },
      });
      this.log(`gemini tool result (${name ?? "unnamed"} → shown=${result.shown})`);
    }
    if (responses.length) this.send({ toolResponse: { functionResponses: responses } });
  }

  private debugShape(msg: Json): void {
    if (!config.gemini.debug) return;
    const key = Object.keys(msg).sort().join(",");
    const sub = msg.serverContent ? `/${Object.keys(msg.serverContent).sort().join(",")}` : "";
    if (!this.seenTypes.has(key + sub)) {
      this.seenTypes.add(key + sub);
      this.log(`gemini message shape: [${key}]${sub}`);
    }
    if (!msg.serverContent?.modelTurn) this.log(`gemini raw: ${JSON.stringify(msg).slice(0, 600)}`);
  }
}
