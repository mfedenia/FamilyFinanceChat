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
 */

import type WebSocket from "ws";
import type { ServerMessage, Turn } from "../../shared/messages";
import { GptLiveBridge, type GptLiveEvents } from "./gptlive";
import { MediaServerLeg } from "./mediaServer";
import { SILENCE_CHECKIN } from "./prompts";

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

  constructor(
    readonly sessionId: string,
    mediaWsUrl: string,
    /** Called when a leg dies and the session can no longer work. */
    private readonly onDead: (sessionId: string) => void,
  ) {
    this.media = new MediaServerLeg(mediaWsUrl, (msg) => this.log(msg));
    this.bridge = new GptLiveBridge(this, (msg) => this.log(msg));
  }

  /** Spawn both legs; whichever exits first ends the session. */
  start(): void {
    void this.runLeg("media server", () => this.media.run());
    void this.runLeg("GPT-Live", async () => {
      if (!(await this.media.waitUntilReady(MEDIA_READY_TIMEOUT_MS))) {
        // Said out loud: without this the avatar simply never speaks while the
        // transcript keeps printing, and the only person who can't tell why is
        // the user.
        this.emit({
          type: "error",
          message: "the avatar could not be reached — start a new session",
        });
        return;
      }
      await this.bridge.run();
    });
  }

  async stop(): Promise<void> {
    this.stopping = true;
    if (this.interruptWatch) clearTimeout(this.interruptWatch);
    if (this.silencePoll) clearInterval(this.silencePoll);
    // Bridge first: GPT-Live gets its session.close and the drain still
    // delivers the last audio into the media leg before it closes.
    await this.bridge.close();
    this.media.close();
    this.frontend = null;
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
    this.log("GPT-Live ready");
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
    this.log(`error: ${message}`);
    this.emit({ type: "error", message });
  }

  // ── internals ──────────────────────────────────────────────────────────────

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
    this.emit({ type: "error", message: `the ${name} connection dropped` });
    this.onDead(this.sessionId);
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
