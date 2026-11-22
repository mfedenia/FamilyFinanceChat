/**
 * Wiring between a session's three websockets.
 *
 * A `Session` owns the two upstream legs (GPT-Live, media server) and the
 * single browser socket, and implements the sink the bridge emits into: audio
 * goes to the media server; transcripts go to the browser.
 *
 * Note what does NOT go to the browser: avatar audio. It reaches the browser
 * through LiveKit, because this server threads it into the media server
 * directly.
 *
 * FamilyFinanceChat fork (AV1): the upstream tutor machinery — lesson-card
 * push, learned-words store, auto-recap and review-break fallback — is
 * removed. Barge-in, the greeting gate, the silence watchdog and teardown are
 * domain-independent and kept unchanged.
 *
 * AV5a adds the hard time limit (with a spoken wrap-up first) and keeps raw
 * upstream errors out of the browser. AV2 makes the persona per-session.
 */

import type WebSocket from "ws";
import type { ServerMessage, Turn } from "../../shared/messages";
import { GptLiveBridge, type GptLiveEvents, type GptLivePersona } from "./gptlive";
import { MediaServerLeg } from "./mediaServer";
import { metrics } from "./metrics";
import { SILENCE_CHECKIN, WRAP_UP_PROMPT } from "./prompts";

// How long GPT-Live waits for the media socket before the session is written
// off. The gate is load-bearing: the bridge triggers the opening greeting as
// soon as GPT-Live reports session.started, and audio that arrives before the
// media socket exists is dropped outright — the greeting would be lost.
const MEDIA_READY_TIMEOUT_MS = 15_000;

// ── Barge-in constants ───────────────────────────────────────────────────────
// Not every user turn is an interruption: a "mm-hmm" over the avatar is a
// backchannel, and the model talks through it. Acting on the turn alone cut
// the avatar off mid-sentence — and because GPT-Live believes it already said
// the rest, it never came back. What separates the two is what the model does
// next, so a user turn only starts a watch; the timings below decide.

// How long to watch before deciding. GPT-Live yields on its own when genuinely
// interrupted (measured around 420ms), so this is long enough to see it.
const YIELD_GRACE_MS = 600;
// No audio for this long means the model stopped producing — the only state
// where clearing the buffer is right: what is queued downstream is then speech
// the model has already abandoned.
const YIELD_QUIET_MS = 350;
// Past this there is nothing left playing; clearing would be a wasted trip.
const YIELD_STALE_MS = 6_000;
// Floor between interrupts, so a re-projected turn cannot chop the avatar twice.
const INTERRUPT_COOLDOWN_MS = 1_000;

// ── Silence watchdog ─────────────────────────────────────────────────────────
// When NOBODY has produced anything — no avatar audio, no student turn — for
// this long, the avatar is prodded (an appended instruction) to check in on the
// student. Note `lastAudioAt` is generation time and generation runs seconds
// ahead of the avatar's voice, so the threshold has to absorb that lead: by
// the time this fires, the played-back silence is shorter than the number says.
const SILENCE_CHECKIN_MS = 12_000;
const SILENCE_POLL_MS = 3_000;
// Floor between check-ins, for a model that ignores the first prod — without
// it every poll tick past the threshold re-prods, which is nagging, not care.
const CHECKIN_COOLDOWN_MS = 20_000;

// ── Time limit (AV5a) ────────────────────────────────────────────────────────
// The wrap-up nudge goes out this long before the hard stop, so the client
// can say goodbye in character. Capped at half the limit for very short
// limits (tests).
const WRAP_UP_LEAD_MS = 60_000;

/** What the browser is told, in plain words, when a session fails. */
const STUDENT_ERROR = "Something went wrong with the practice client. Please start a new session.";
const STUDENT_DROPPED = "The connection to the practice client dropped. Please start a new session.";

export interface SessionOptions {
  persona: GptLivePersona & { id: string; name: string };
  /** Hard cap on session length (AV5a). */
  maxSessionMs: number;
  /** When the student pressed start — the zero point for the ready metric. */
  requestStartedAt: number;
}

export class Session implements GptLiveEvents {
  readonly bridge: GptLiveBridge;
  readonly media: MediaServerLeg;

  private frontend: WebSocket | null = null;
  private ready = false;
  private stopping = false;
  private lastAudioAt: number | null = null;
  private lastInterruptAt: number | null = null;
  private interruptWatch: NodeJS.Timeout | null = null;
  /** Bumped by mic frames — the idle watchdog's only liveness signal. */
  lastActivityAt = Date.now();
  readonly startedAt = Date.now();

  // ── Silence watchdog state ──
  private lastUserActivityAt: number | null = null;
  private lastCheckinAt: number | null = null;
  private readyAt: number | null = null;
  private silencePoll: NodeJS.Timeout | null = null;

  // ── Time limit state ──
  private wrapUpTimer: NodeJS.Timeout | null = null;
  private limitTimer: NodeJS.Timeout | null = null;

  constructor(
    readonly sessionId: string,
    mediaWsUrl: string,
    private readonly options: SessionOptions,
    /**
     * Ask the registry to end this session (it calls stop()). Reasons used
     * here: "leg_died", "time_limit".
     */
    private readonly onEnd: (sessionId: string, reason: string) => void,
  ) {
    this.media = new MediaServerLeg(mediaWsUrl, (msg) => this.log(msg));
    this.bridge = new GptLiveBridge(this, options.persona, (msg) => this.log(msg));
  }

  get personaId(): string {
    return this.options.persona.id;
  }

  /** Spawn both legs; whichever exits first ends the session. */
  start(): void {
    this.armTimeLimit();
    void this.runLeg("media server", () => this.media.run());
    void this.runLeg("GPT-Live", async () => {
      if (!(await this.media.waitUntilReady(MEDIA_READY_TIMEOUT_MS))) {
        // Said out loud: without this the avatar simply never speaks while the
        // transcript keeps printing, and the only person who can't tell why is
        // the user.
        this.log("media server never became ready");
        this.emit({ type: "error", message: STUDENT_DROPPED });
        return;
      }
      await this.bridge.run();
    });
  }

  async stop(): Promise<void> {
    this.stopping = true;
    if (this.interruptWatch) clearTimeout(this.interruptWatch);
    if (this.silencePoll) clearInterval(this.silencePoll);
    if (this.wrapUpTimer) clearTimeout(this.wrapUpTimer);
    if (this.limitTimer) clearTimeout(this.limitTimer);
    // Bridge first: GPT-Live gets its session.close and the drain still
    // delivers the last audio into the media leg before it closes.
    await this.bridge.close();
    this.media.close();
    // Close the browser socket too, so a server-side end (time limit, idle,
    // dead leg) reaches the page instead of leaving it on a frozen face. The
    // socket's close handler calls stopSession again, which is a no-op.
    const ws = this.frontend;
    this.frontend = null;
    if (ws && (ws.readyState === ws.OPEN || ws.readyState === ws.CONNECTING)) {
      ws.close(1000, "session ended");
    }
  }

  /**
   * Claim the single frontend slot; false if one is already attached.
   * Synchronous check-and-set, so two upgrades racing for the same session
   * cannot both win.
   */
  tryAttachFrontend(ws: WebSocket): boolean {
    if (this.frontend) return false;
    this.frontend = ws;
    // Replay `ready` to a browser that attached after the bridge came up, so a
    // slow tab doesn't miss the one `ready` that ever fires.
    if (this.ready) this.emit({ type: "ready" });
    return true;
  }

  detachFrontend(ws: WebSocket): void {
    // Identity-keyed so a stale socket cannot clear a newer connection.
    if (this.frontend === ws) this.frontend = null;
  }

  sendMicAudio(audioB64: string): void {
    this.lastActivityAt = Date.now();
    this.bridge.sendMicAudio(audioB64);
  }

  // ── GptLiveEvents ──────────────────────────────────────────────────────────

  onReady(): void {
    this.ready = true;
    this.readyAt = Date.now();
    const readySec = (this.readyAt - this.options.requestStartedAt) / 1000;
    metrics.sessionReady(readySec);
    this.log(`GPT-Live ready (persona ${this.options.persona.id}, ${readySec.toFixed(1)}s after start)`);
    this.emit({ type: "ready" });
    this.silencePoll = setInterval(() => this.checkSilence(), SILENCE_POLL_MS);
  }

  onAudio(audioB64: string): void {
    this.lastAudioAt = Date.now();
    this.media.speak(audioB64);
  }

  onTurn(turn: Turn): void {
    // AV4 hooks transcript capture in here: finished turns are what gets
    // written back to Open WebUI.
    this.emit({ type: "turn", ...turn });
    if (turn.role === "user") this.lastUserActivityAt = Date.now();
  }

  onUserTurnStarted(): void {
    this.lastUserActivityAt = Date.now();
    if (this.interruptWatch) return; // already watching this interruption
    this.interruptWatch = setTimeout(() => {
      this.interruptWatch = null;
      this.stopStaleAudio();
    }, YIELD_GRACE_MS);
  }

  onToolCall(
    name: string | null,
    args: Record<string, unknown>,
  ): Record<string, unknown> {
    // No tools are registered (AV1). Still a result — an unanswered call stays
    // pending upstream and blocks every later delegation.
    this.log(
      `unexpected tool call ignored: ${name ?? "(unnamed)"} ${JSON.stringify(args).slice(0, 120)}`,
    );
    return { shown: false, error: "no tools are available in this session" };
  }

  onError(message: string): void {
    // The detail is for us (log); the student gets plain words. Raw upstream
    // errors can name providers, models or account state.
    this.log(`error: ${message}`);
    // Errors during teardown are expected (e.g. GPT-Live rejecting appends it
    // had not injected yet when the session closed) and must not overwrite
    // the reason the session ended ("time limit reached") on the student's
    // screen.
    if (this.stopping) return;
    this.emit({ type: "error", message: STUDENT_ERROR });
  }

  // ── internals ──────────────────────────────────────────────────────────────

  /**
   * AV5a hard cap: a nudge to wrap up in character, then the stop. Timed from
   * here — right after the LiveAvatar session started, when billing began.
   */
  private armTimeLimit(): void {
    const max = this.options.maxSessionMs;
    const lead = Math.min(WRAP_UP_LEAD_MS, max / 2);
    this.wrapUpTimer = setTimeout(() => {
      this.wrapUpTimer = null;
      if (this.stopping || !this.ready) return;
      this.log("time limit approaching — asking the client to wrap up");
      this.bridge.append("instructions", WRAP_UP_PROMPT);
    }, max - lead);
    this.limitTimer = setTimeout(() => {
      this.limitTimer = null;
      if (this.stopping) return;
      const minutes = Math.round((max / 60_000) * 10) / 10;
      this.log(`time limit reached (${minutes} min) — ending session`);
      this.emit({
        type: "ended",
        reason: "time_limit",
        message: `This practice session reached its ${minutes}-minute limit.`,
      });
      this.onEnd(this.sessionId, "time_limit");
    }, max);
  }

  /**
   * The dead-air killer. If neither side has produced anything for a while —
   * no avatar audio generated, no student turn heard — prod the avatar to
   * check in, in character. Its own check-in speech bumps lastAudioAt, which
   * re-arms the watchdog naturally.
   */
  private checkSilence(): void {
    if (this.stopping) return;
    const now = Date.now();
    const lastSignal = Math.max(
      this.readyAt ?? this.startedAt,
      this.lastAudioAt ?? 0,
      this.lastUserActivityAt ?? 0,
    );
    if (now - lastSignal < SILENCE_CHECKIN_MS) return;
    if (
      this.lastCheckinAt !== null &&
      now - this.lastCheckinAt < CHECKIN_COOLDOWN_MS
    )
      return;
    this.lastCheckinAt = now;
    const quiet = (at: number | null) =>
      at === null ? "never" : `${Math.round((now - at) / 1000)}s`;
    this.log(
      `silence — prompting a check-in (avatar quiet ${quiet(this.lastAudioAt)}, student quiet ${quiet(this.lastUserActivityAt)})`,
    );
    this.bridge.append("instructions", SILENCE_CHECKIN);
  }

  /**
   * Clear the avatar's buffer, but only once the model has given up on it.
   * Clearing is destructive and unilateral — the media server drops audio
   * GPT-Live thinks it delivered, and nothing will produce it again — so it is
   * worth doing in exactly one state: the model has stopped speaking and the
   * avatar has not caught up yet.
   */
  private stopStaleAudio(): void {
    const last = this.lastAudioAt;
    if (last === null || this.stopping) return;
    if (
      this.lastInterruptAt !== null &&
      Date.now() - this.lastInterruptAt < INTERRUPT_COOLDOWN_MS
    ) {
      return;
    }
    const quietFor = Date.now() - last;
    // Still talking over the user → the model kept the floor (backchannel).
    if (quietFor < YIELD_QUIET_MS) return;
    // Long quiet → nothing left playing; clearing would empty an empty buffer.
    if (quietFor > YIELD_STALE_MS) return;
    this.lastInterruptAt = Date.now();
    this.log("user barge-in — clearing avatar audio buffer");
    this.media.interrupt();
    // The rest of that sentence is never coming; tell the browser so it can
    // reflect it (e.g. mark the transcript line). AV4 marks it in the saved
    // transcript too.
    this.emit({ type: "interrupted" });
  }

  private async runLeg(name: string, run: () => Promise<void>): Promise<void> {
    try {
      await run();
    } catch (err) {
      if (!this.stopping)
        this.log(
          `${name} leg failed: ${err instanceof Error ? err.message : err}`,
        );
    }
    if (this.stopping) return;
    // Neither leg reconnects forever, and a session with one live leg is
    // useless — GPT-Live would stream into a closed socket, or the avatar
    // would sit mute — so the survivor is not left running.
    this.log(`${name} leg exited — ending session`);
    this.emit({ type: "error", message: STUDENT_DROPPED });
    this.onEnd(this.sessionId, "leg_died");
  }

  private emit(message: ServerMessage): void {
    const ws = this.frontend;
    if (!ws || ws.readyState !== ws.OPEN) return;
    try {
      ws.send(JSON.stringify(message));
    } catch {
      this.frontend = null;
    }
  }

  private log(msg: string): void {
    console.log(`[session ${this.sessionId.slice(0, 8)}] ${msg}`);
  }
}
