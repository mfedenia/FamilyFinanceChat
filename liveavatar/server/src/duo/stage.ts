/**
 * The duo director: two avatars, one conversation.
 *
 *   mic ─► Listener (transcribe-only) ─► final utterance
 *            └─ barge-in ─► cut the persona that is audible
 *   utterance ─► pickSpeaker (gemini-decision-1) ─► chosen.say("Adviser: …")
 *                                                   other.addContext("Adviser: …")
 *   persona reply ─► its avatar's media leg; on end ─► other.addContext("Name: …")
 *                                                  └─ handoff check ─► other.say
 *
 * Both personas carry the same conversation: every line reaches both, as a
 * turn for the one with the floor and as silent context for the other. Who
 * they are and what they know comes from the scenario (scenario.ts); the
 * human's label ("Adviser", "User") comes from it too.
 *
 * This departs, deliberately, from the single-avatar invariants (AGENTS.md):
 * there the live model hears the mic and owns turn-taking. Here no persona
 * hears audio — routing has to happen BEFORE anyone answers, so the listener's
 * voice activity owns turn-taking and the director owns the floor.
 */

import { randomUUID } from "node:crypto";
import type WebSocket from "ws";
import type { DuoFloorReason, DuoServerMessage } from "../../../shared/messages";
import { MediaServerLeg } from "../mediaServer";
import type { HostedSession } from "../registry";
import { handoffProbability, pickSpeaker } from "./decision";
import { Listener } from "./listener";
import { GptLivePersona } from "./gptLivePersona";
import { excerptsBlock, searchKnowledge, type Excerpt } from "./openwebui";
import { PersonaBridge, type PersonaEvents, type PersonaVoice } from "./personaBridge";
import { config } from "../config";
import type { Persona, Scenario } from "./scenario";

const MEDIA_READY_TIMEOUT_MS = 15_000;
/** Lines of transcript the decision model sees. */
const DECISION_CONTEXT_LINES = 8;
/**
 * Barge-in: the user has said this many words over an audible persona. Fewer
 * is a backchannel ("mm-hmm", "yeah") — talked through, never routed.
 */
const BARGE_IN_WORDS = 3;
/**
 * A persona counts as audible until its queued audio has played out + this —
 * generous, because the avatar renders and ships over LiveKit on top of the
 * playback clock this estimate runs on.
 */
const AUDIBLE_TAIL_MS = 1_000;
/** PCM16 mono @ 24kHz. */
const BYTES_PER_MS = 48;
/** Handoff: decision-model probability that the speaker passed the floor. */
const HANDOFF_THRESHOLD = 0.6;
/** Persona→persona exchanges in a row before the floor returns to the user. */
const MAX_HANDOFFS = 2;
const INTERRUPT_COOLDOWN_MS = 1_000;

interface Cast {
  persona: Persona;
  /** The LiveAvatar session this persona's avatar runs in. */
  upstreamId: string;
  media: MediaServerLeg;
  bridge: PersonaVoice;
  /** When the audio already handed to the avatar finishes playing (estimate). */
  playEndsAt: number;
  /** The reply being transcribed right now. */
  turn: { id: string; text: string } | null;
  ready: boolean;
  /** Already cut off since it last got the floor — never cut the same reply twice. */
  cut: boolean;
}

export class DuoStage implements HostedSession {
  readonly sessionId = randomUUID();
  lastActivityAt = Date.now();

  private readonly cast: Cast[];
  private readonly listener: Listener;
  private frontend: WebSocket | null = null;
  private stopping = false;
  private greeted = false;
  private listenerReady = false;

  /** Every spoken line, "Name: text", in order — the decision model's input. */
  private readonly history: string[] = [];
  private lastSpeaker: Cast | null = null;
  /** Bumped whenever the user takes the floor; stale async routing checks it. */
  private floorSeq = 0;
  private handoffs = 0;
  private lastInterruptAt = 0;

  // The current user utterance, as the UI sees it.
  private userSeq = 0;
  private userTurnId: string | null = null;
  // Utterances waiting to be routed (merged if the user keeps talking).
  private readonly pendingUser: string[] = [];
  private routing = false;
  private replySeq = 0;

  constructor(
    private readonly scenario: Scenario,
    upstreams: { sessionId: string; wsUrl: string }[],
    private readonly onDead: (sessionId: string) => void,
  ) {
    this.cast = scenario.personas.map((persona, i) => {
      const log = (msg: string) => this.log(`${persona.name}: ${msg}`);
      const c: Cast = {
        persona,
        upstreamId: upstreams[i]!.sessionId,
        media: new MediaServerLeg(upstreams[i]!.wsUrl, (msg) => this.log(`${persona.name} media: ${msg}`)),
        bridge: null as unknown as PersonaVoice,
        playEndsAt: 0,
        turn: null,
        ready: false,
        cut: false,
      };
      const events: PersonaEvents = {
        onReady: () => {
          c.ready = true;
          this.maybeGreet();
        },
        onAudio: (b64) => this.onPersonaAudio(c, b64),
        onText: (delta) => this.onPersonaText(c, delta),
        onTurnEnd: (cut) => this.onPersonaTurnEnd(c, cut),
        onError: (message) => this.fail(message),
      };
      // The live voice model, per DUO_PERSONA_ENGINE. Same interface either way.
      c.bridge =
        config.duo.personaEngine === "gptlive"
          ? new GptLivePersona(persona, persona.instructions, events, log)
          : new PersonaBridge(persona, persona.instructions, events, log);
      return c;
    });
    this.listener = new Listener(
      {
        onReady: () => {
          this.listenerReady = true;
          this.maybeGreet();
        },
        onSpeechStart: () => {},
        onInterim: (text) => this.onUserInterim(text),
        onFinal: (text) => this.onUserFinal(text),
        onError: (message) => this.fail(message),
      },
      (msg) => this.log(msg),
    );
  }

  get upstreamSessionIds(): readonly string[] {
    return this.cast.map((c) => c.upstreamId);
  }

  start(): void {
    for (const c of this.cast) void this.runLeg(`${c.persona.name} media`, () => c.media.run());
    void (async () => {
      const ready = await Promise.all(this.cast.map((c) => c.media.waitUntilReady(MEDIA_READY_TIMEOUT_MS)));
      if (this.stopping) return;
      if (!ready.every(Boolean)) {
        this.emit({ type: "error", message: "an avatar could not be reached — start a new session" });
        this.onDead(this.sessionId);
        return;
      }
      // Media first, then the models: audio sent before a media leg is up is
      // dropped with no replay, and the greeting would be lost.
      for (const c of this.cast) void this.runLeg(c.persona.name, () => c.bridge.run());
      void this.runLeg("listener", () => this.listener.run());
    })();
  }

  async stop(): Promise<void> {
    this.stopping = true;
    this.listener.close();
    await Promise.all(this.cast.map((c) => c.bridge.close()));
    for (const c of this.cast) c.media.close();
    this.frontend = null;
  }

  tryAttachFrontend(ws: WebSocket): boolean {
    if (this.frontend) return false;
    this.frontend = ws;
    if (this.greeted) this.emit({ type: "ready" });
    return true;
  }

  detachFrontend(ws: WebSocket): void {
    if (this.frontend === ws) this.frontend = null;
  }

  sendMicAudio(audioB64: string): void {
    this.lastActivityAt = Date.now();
    this.listener.sendMicAudio(audioB64);
  }

  // ── startup ────────────────────────────────────────────────────────────────

  private maybeGreet(): void {
    if (this.greeted || !this.listenerReady || !this.cast.every((c) => c.ready)) return;
    this.greeted = true;
    const g = this.scenario.greeting;
    const host = g && this.cast.find((c) => c.persona.id === g.persona);
    this.log(`all legs up — ${host ? `${host.persona.name} opens` : "waiting for the first words"}`);
    if (host && g) this.giveFloor(host, g.directive, "greeting", {}, 0);
    this.emit({ type: "ready" });
  }

  // ── the user ───────────────────────────────────────────────────────────────

  private onUserInterim(text: string): void {
    this.userTurnId ??= `user_${++this.userSeq}`;
    this.emit({ type: "turn", id: this.userTurnId, role: "user", text, done: false });
    if (wordCount(text) >= BARGE_IN_WORDS) this.bargeIn();
  }

  private onUserFinal(text: string): void {
    const id = this.userTurnId ?? `user_${++this.userSeq}`;
    this.userTurnId = null;
    this.emit({ type: "turn", id, role: "user", text, done: true });

    const audible = this.cast.find((c) => this.isAudible(c));
    if (audible && wordCount(text) < BARGE_IN_WORDS && !text.includes("?")) {
      // A backchannel over a speaking persona: both should know it was said,
      // nobody should answer it.
      this.log(`backchannel over ${audible.persona.name}: "${text}"`);
      this.history.push(`${this.who}: ${text}`);
      for (const c of this.cast) c.bridge.addContext(`${this.who} (while ${audible.persona.name} was talking): ${text}`);
      return;
    }
    this.bargeIn();
    this.floorSeq += 1; // the user has the floor now: cancel pending handoffs
    this.handoffs = 0;
    this.pendingUser.push(text);
    void this.route();
  }

  /** Route queued utterances, one decision at a time; merge what piles up. */
  private async route(): Promise<void> {
    if (this.routing) return;
    this.routing = true;
    try {
      while (this.pendingUser.length > 0 && !this.stopping) {
        const text = this.pendingUser.splice(0).join(" ");
        const line = `${this.who}: ${text}`;
        const recent = [...this.history, line].slice(-DECISION_CONTEXT_LINES);
        let chosen: Cast;
        let scores: Record<string, number> = {};
        let ms = 0;
        // FamilyFinanceChat retrieval, in parallel with the decision: search
        // every persona's knowledge now, keep the chosen one's afterwards.
        const searches = this.searchAll(text);
        try {
          const d = await pickSpeaker(this.scenario, recent, this.lastSpeaker?.persona ?? null);
          chosen = this.cast.find((c) => c.persona.id === d.personaId)!;
          scores = d.scores;
          ms = d.ms;
        } catch (err) {
          // The show must go on: whoever spoke last keeps the floor.
          chosen = this.lastSpeaker ?? this.cast[0]!;
          this.log(`decision failed (${err instanceof Error ? err.message : err}) — ${chosen.persona.name} keeps the floor`);
        }
        if (this.pendingUser.length > 0) {
          // The user kept talking while we decided — decide again on all of it.
          this.pendingUser.unshift(text);
          continue;
        }
        this.history.push(line);
        const pct = Object.entries(scores).map(([k, v]) => `${k}=${v.toFixed(2)}`).join(" ");
        this.log(`floor → ${chosen.persona.name} (${ms}ms${pct ? `, ${pct}` : ""}) for "${text.slice(0, 80)}"`);
        const t0 = Date.now();
        const excerpts = (await searches.get(chosen.persona.id)) ?? [];
        if (chosen.persona.collections.length > 0) {
          this.log(`documents for ${chosen.persona.name}: ${excerpts.length} excerpt(s) (+${Date.now() - t0}ms after the decision)`);
        }
        if (this.stopping) return;
        this.giveFloor(chosen, line, "user", scores, ms, line, excerptsBlock(excerpts));
      }
    } finally {
      this.routing = false;
    }
  }

  // ── the floor ──────────────────────────────────────────────────────────────

  /**
   * `turn` goes to `to` as its turn; `shared` (if any) goes to the other
   * persona as silent context — the same line, so both contexts match.
   */
  private giveFloor(
    to: Cast,
    turn: string,
    reason: DuoFloorReason,
    scores: Record<string, number>,
    ms: number,
    shared?: string,
    documents?: string,
  ): void {
    // Only the spoken line is shared; retrieved documents are the speaker's own.
    if (shared) for (const c of this.cast) if (c !== to) c.bridge.addContext(shared);
    to.cut = false;
    to.bridge.say(turn, documents || undefined);
    this.lastSpeaker = to;
    this.emit({ type: "floor", persona: to.persona.id, reason, scores, ms });
  }

  /** Cut off whichever persona is audible (the user is talking over it). */
  private bargeIn(): void {
    const now = Date.now();
    if (now - this.lastInterruptAt < INTERRUPT_COOLDOWN_MS) return;
    for (const c of this.cast) {
      if (c.cut || (!this.isAudible(c) && !c.bridge.busy)) continue;
      c.cut = true;
      this.lastInterruptAt = now;
      this.log(`barge-in — cutting ${c.persona.name}`);
      c.media.interrupt();
      c.bridge.silence();
      c.playEndsAt = 0;
      this.emit({ type: "interrupted", persona: c.persona.id });
    }
  }

  private isAudible(c: Cast): boolean {
    return Date.now() < c.playEndsAt + AUDIBLE_TAIL_MS;
  }

  // ── persona output ─────────────────────────────────────────────────────────

  private onPersonaAudio(c: Cast, b64: string): void {
    const ms = (b64.length * 3) / 4 / BYTES_PER_MS;
    c.playEndsAt = Math.max(Date.now(), c.playEndsAt) + ms;
    c.media.speak(b64);
  }

  private onPersonaText(c: Cast, delta: string): void {
    c.turn ??= { id: `${c.persona.id}_${++this.replySeq}`, text: "" };
    c.turn.text += delta;
    this.emit({ type: "turn", id: c.turn.id, role: "assistant", speaker: c.persona.id, text: c.turn.text.trim(), done: false });
  }

  private onPersonaTurnEnd(c: Cast, cut: boolean): void {
    const turn = c.turn;
    c.turn = null;
    const text = turn?.text.trim() ?? "";
    if (!turn || !text) return;
    this.emit({ type: "turn", id: turn.id, role: "assistant", speaker: c.persona.id, text, done: true });

    const line = `${c.persona.name}: ${text}${cut ? ` — (cut off by the ${this.who.toLowerCase()})` : ""}`;
    this.history.push(line);
    for (const o of this.cast) if (o !== c) o.bridge.addContext(line);
    if (cut) {
      // It believes it said all of that; the user heard only part.
      c.bridge.addContext(`[director] The ${this.who.toLowerCase()} cut you off before you finished that reply.`);
      return;
    }
    void this.maybeHandoff(c, text);
  }

  /** Did the persona that just spoke pass the floor to its co-host? */
  private async maybeHandoff(from: Cast, text: string): Promise<void> {
    const to = this.cast.find((c) => c !== from)!;
    // Cheap gate before the API call: a handoff names the co-host.
    if (this.handoffs >= MAX_HANDOFFS || !text.toLowerCase().includes(to.persona.name.toLowerCase())) return;
    const seq = this.floorSeq;
    const t0 = Date.now();
    let p = 0;
    try {
      p = await handoffProbability(from.persona, to.persona, text, this.who);
    } catch (err) {
      this.log(`handoff check failed: ${err instanceof Error ? err.message : err}`);
      return;
    }
    // The user spoke meanwhile — their turn wins.
    if (seq !== this.floorSeq || this.stopping || this.userTurnId) return;
    this.log(`handoff ${from.persona.name} → ${to.persona.name}? p=${p.toFixed(2)}`);
    if (p < HANDOFF_THRESHOLD) return;
    this.handoffs += 1;
    this.giveFloor(
      to,
      `[director] ${from.persona.name} just spoke to you — respond to them now, briefly.`,
      "handoff",
      { [to.persona.id]: p },
      Date.now() - t0,
    );
  }

  /**
   * Start a knowledge search for every persona that has knowledge, one search
   * per distinct set of collections (both spouses usually share one base).
   */
  private searchAll(query: string): Map<string, Promise<Excerpt[]>> {
    const byKey = new Map<string, Promise<Excerpt[]>>();
    const out = new Map<string, Promise<Excerpt[]>>();
    for (const c of this.cast) {
      const cols = c.persona.collections;
      if (cols.length === 0) continue;
      const key = [...cols].sort().join("|");
      if (!byKey.has(key)) byKey.set(key, searchKnowledge(cols, query, (m) => this.log(m)));
      out.set(c.persona.id, byKey.get(key)!);
    }
    return out;
  }

  // ── plumbing ───────────────────────────────────────────────────────────────

  /** What the human is called in transcript lines: "Adviser", "User". */
  private get who(): string {
    return this.scenario.userLabel;
  }

  private fail(message: string): void {
    this.log(`error: ${message}`);
    this.emit({ type: "error", message });
    if (!this.stopping) this.onDead(this.sessionId);
  }

  private async runLeg(name: string, run: () => Promise<void>): Promise<void> {
    try {
      await run();
    } catch (err) {
      if (!this.stopping) this.log(`${name} leg failed: ${err instanceof Error ? err.message : err}`);
    }
    if (!this.stopping) {
      this.log(`${name} leg ended — ending session`);
      this.onDead(this.sessionId);
    }
  }

  private emit(msg: DuoServerMessage): void {
    if (this.frontend?.readyState === 1) this.frontend.send(JSON.stringify(msg));
  }

  private log(msg: string): void {
    console.log(`[duo ${this.sessionId.slice(0, 8)}] ${msg}`);
  }
}

function wordCount(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}
