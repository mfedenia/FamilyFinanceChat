/**
 * The couple meeting: two avatars (Tony and Antonella), one student adviser.
 *
 * Each spouse is a full GPT-Live session driving its own LiveAvatar LITE
 * session — the same two legs the single-avatar `Session` runs, twice. Both
 * hear the adviser (mic audio goes to both), because GPT-Live needs continuous
 * input and because a spouse who did not hear the question cannot follow the
 * conversation.
 *
 * GPT-Live has no "don't speak" switch, so who talks is enforced here, at the
 * audio gate, by a FLOOR:
 *
 *   - every new adviser turn opens a "round" with the floor undecided;
 *   - a fast decision model (DECISION_MODEL) reads the adviser's words — while
 *     they are still being spoken, debounced — and picks who answers. Naming a
 *     spouse ("Antonella, ...") skips the model entirely;
 *   - only the floor holder's audio reaches its avatar. Audio produced while
 *     the floor is undecided is held (up to DECISION_WAIT_MS) and flushed to the
 *     winner; the other spouse's output is dropped and that spouse is told,
 *     silently, that their partner answered;
 *   - what the floor holder says is fed to the other spouse as silent context,
 *     so each can refer to what the other said;
 *   - the decision can ask the other spouse to chime in with one sentence after
 *     the answer (e.g. when they would disagree about risk).
 *
 * Personas also carry a "respond only when it is for you" rule (OpenAI's
 * selected-requests prompting pattern), which keeps the dropped audio rare.
 */

import { randomUUID } from "node:crypto";
import type WebSocket from "ws";
import type { ServerMessage, Speaker, Turn } from "../../shared/messages";
import { config } from "./config";
import { GptLiveBridge, type GptLiveEvents } from "./gptlive";
import type { StartedSession } from "./liveavatar";
import { MediaServerLeg } from "./mediaServer";
import type { Personas } from "./openwebui";
import { GREETING_PREAMBLE } from "./prompts";
import type { LiveSession } from "./registry";
import { TranscriptRecorder } from "./transcript";

const MEDIA_READY_TIMEOUT_MS = 15_000;

// Floor timing.
/**
 * While the adviser is talking, re-run the decision at most this often on the
 * words so far — so a decision is usually ready the moment they stop. (A
 * debounce would wait for a pause that continuous speech never has.)
 */
const DECISION_THROTTLE_MS = 500;
/** Don't decide on fewer words than this mid-sentence ("How do" says nothing). */
const DECISION_MIN_WORDS = 3;
/** Max time a spouse's reply is held back waiting for the decision. */
const DECISION_WAIT_MS = 1_200;
/** A speaker still producing audio this recently is "talking through" a new adviser turn. */
const CONTINUING_MS = 800;
const DECISION_TIMEOUT_MS = 4_000;

// Barge-in (same reasoning and numbers as session.ts).
const YIELD_GRACE_MS = 600;
const YIELD_QUIET_MS = 350;
const YIELD_STALE_MS = 6_000;
const INTERRUPT_COOLDOWN_MS = 1_000;

const HISTORY_LINES = 12;

// From Park's duo (feat/liveavatar-duo, stage.ts):
/** Fewer adviser words than this over a speaking spouse is a backchannel. */
const BARGE_IN_WORDS = 3;
/** A spouse counts as still talking this long after its last voiced audio. */
const BACKCHANNEL_WINDOW_MS = 1_500;
/** Spouse-to-spouse handoffs in a row before the floor returns to the adviser. */
const MAX_HANDOFFS = 2;

/**
 * GPT-Live streams output audio continuously — silence included — so "audio
 * arrived" does not mean "talking". RMS of the PCM16 chunk above this (0..1
 * scale) counts as speech. Room-quiet output measures ~0; speech ~0.05–0.3.
 */
const VOICED_RMS = 0.01;

function isVoiced(audioB64: string): boolean {
  const buf = Buffer.from(audioB64, "base64");
  const n = Math.floor(buf.length / 2);
  if (!n) return false;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = buf.readInt16LE(i * 2) / 32768;
    sum += v * v;
  }
  return Math.sqrt(sum / n) > VOICED_RMS;
}

interface Profile {
  who: Speaker;
  name: string;
  other: Speaker;
  otherName: string;
  /** How this spouse refers to the other: "wife" / "husband". */
  otherRelation: string;
  avatarId: string;
  voice: string;
  /** Speak-first opening; empty = wait. */
  greeting: string;
}

export const PROFILES: Record<Speaker, Profile> = {
  tony: {
    who: "tony",
    name: "Tony",
    other: "antonella",
    otherName: "Antonella",
    otherRelation: "wife",
    avatarId: config.couple.tony.avatarId,
    voice: config.couple.tony.voice,
    greeting:
      "Say a short, slightly nervous hello to the adviser, introduce yourself as Tony and your wife " +
      "Antonella sitting next to you, and say you're here because you finally want to get your " +
      "finances sorted out. One or two sentences, then stop and let the adviser lead.",
  },
  antonella: {
    who: "antonella",
    name: "Antonella",
    other: "tony",
    otherName: "Tony",
    otherRelation: "husband",
    avatarId: config.couple.antonella.avatarId,
    voice: config.couple.antonella.voice,
    greeting: "",
  },
};

function meetingRules(p: Profile): string {
  return `

You are in this meeting together with your ${p.otherRelation} ${p.otherName}, who is sitting next to you and is a separate person with their own voice. Rules for a three-way conversation:
- Answer when the adviser speaks to you, says your name, or asks about your own job, accounts, health, family role or feelings.
- If the question is clearly meant for ${p.otherName}, or is about ${p.otherName}'s job or accounts, stay quiet and keep listening.
- If a question is for both of you, answer briefly for yourself only.
- Do not produce backchannel interjections (like "mm-hmm" or "right") while anyone is speaking.
- You will receive notes in parentheses telling you what ${p.otherName} said. Treat them as things you heard ${p.otherName} say in the room. You can agree, add to, or gently disagree with ${p.otherName}, but never repeat what they already said.
- If a note says ${p.otherName} answered instead of you, your own reply was not heard: do not continue or repeat it.`;
}

function responsesInstructions(p: Profile, persona: string): string {
  return (
    `You are the background reasoning for ${p.name} Beckham, a simulated financial-advising client ` +
    `in a live spoken role-play (the person speaking is a student adviser). When the live voice ` +
    `hands you a turn, reply ONLY with what ${p.name} would say aloud: one to three short, natural ` +
    `spoken sentences, in character, no lists or formatting. Never give financial advice, never ` +
    `coach the student, never invent facts beyond the persona below.\n\n` +
    persona
  );
}

/**
 * Did a spouse just hand the floor to the other (Park's handoff)? True when
 * the last sentence names the other spouse and asks them something or invites
 * them in: "Antonella, what do you think?", "...right, Tony?".
 */
export function handsOffTo(text: string, otherName: string): boolean {
  const sentences = text.trim().split(/(?<=[.!?])\s+/).filter(Boolean);
  const last = (sentences.at(-1) ?? "").toLowerCase();
  if (!new RegExp(`\\b${otherName.toLowerCase()}\\b`).test(last)) return false;
  return /\?\s*$/.test(last) || /\b(what do you think|tell (them|him|her)|you (tell|explain)|go ahead|your turn|back me up)\b/.test(last);
}

/**
 * A short acknowledgement over a speaking spouse ("mm-hmm", "okay", "right"):
 * the adviser is listening, not asking. Park's rule: under three words and no
 * question mark.
 */
export function isBackchannel(text: string): boolean {
  const t = text.trim();
  return t.length > 0 && !t.includes("?") && t.split(/\s+/).length < BARGE_IN_WORDS;
}

/** Names the adviser used, latest wins: "Tony, ... Antonella?" → antonella. */
function namedSpeaker(text: string): Speaker | null {
  let best: { who: Speaker; at: number } | null = null;
  const patterns: [Speaker, RegExp][] = [
    ["tony", /\btony\b/gi],
    ["antonella", /\b(antonella|nella)\b/gi],
  ];
  for (const [who, re] of patterns) {
    for (const m of text.matchAll(re)) {
      if (!best || (m.index ?? 0) > best.at) best = { who, at: m.index ?? 0 };
    }
  }
  return best?.who ?? null;
}

const DECISION_SYSTEM = `You run floor control in a spoken role-play: a student adviser is meeting a married couple, Tony Beckham (54, the husband) and Antonella Beckham (52, the wife). Decide which spouse should answer the adviser's latest words.

Tony's topics: his UW-Milwaukee job and salary, his freelance website business, his Wisconsin Retirement System pension, his unused 403(b) and 457, his brokerage account of green energy funds, his hobbies (gambling, cars, hunting, skiing), his high risk tolerance, his parents' heart disease, the Vail timeshare, the homeowners insurance.
Antonella's topics: her Rockwell job, salary and bonus, her 401(k) and catch-up contributions, her variable annuity, her low risk tolerance, her father Giacomo, stepmother Pamela and the foundation, any inheritance from her family, her feelings about Rita.
Shared topics (either can answer; prefer the one who has spoken less recently, or who would feel most strongly): retirement goals, Bradley and college, Alana, Max, wills and estate planning, life and long-term care insurance, budget, debts, the house.

If the adviser addresses a spouse by name, or says "you" right after that spouse spoke, give it to that spouse. Short acknowledgements and follow-ups ("okay", "and how much is that?") go to whoever spoke last.
Set chime_in to true only when the other spouse would naturally add a brief reaction (they clearly disagree, e.g. about risk, or the topic is emotional and shared). Usually false.
Reply with JSON only: {"speaker":"tony"|"antonella","chime_in":true|false}`;

async function decideSpeaker(
  history: string[],
  latest: string,
  lastSpeaker: Speaker | null,
): Promise<{ speaker: Speaker; chimeIn: boolean } | null> {
  const model = config.couple.decisionModel;
  const effort =
    process.env.DECISION_REASONING ??
    (model.startsWith("gpt-5.4") ? "none" : model.startsWith("gpt-6") ? "low" : "");
  const response = await fetch("https://api.openai.com/v1/responses", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${config.gptlive.apiKey}`,
      "Content-Type": "application/json",
    },
    signal: AbortSignal.timeout(DECISION_TIMEOUT_MS),
    body: JSON.stringify({
      model,
      input: [
        { role: "system", content: DECISION_SYSTEM },
        {
          role: "user",
          content:
            `Recent conversation:\n${history.join("\n") || "(just started)"}\n\n` +
            `Last spouse to speak: ${lastSpeaker ? PROFILES[lastSpeaker].name : "nobody yet"}\n` +
            `Adviser's latest words: "${latest}"`,
        },
      ],
      text: { format: { type: "json_object" } },
      max_output_tokens: 300,
      ...(effort ? { reasoning: { effort } } : {}),
    }),
  });
  if (!response.ok) {
    throw new Error(`decision model HTTP ${response.status}: ${(await response.text()).slice(0, 200)}`);
  }
  const body = (await response.json()) as {
    output?: { type?: string; content?: { type?: string; text?: string }[] }[];
  };
  const text = body.output
    ?.filter((o) => o.type === "message")
    .flatMap((o) => o.content ?? [])
    .find((c) => c.type === "output_text")?.text;
  if (!text) return null;
  const parsed = JSON.parse(text) as { speaker?: string; chime_in?: boolean };
  if (parsed.speaker !== "tony" && parsed.speaker !== "antonella") return null;
  return { speaker: parsed.speaker, chimeIn: parsed.chime_in === true };
}

/** One spouse: a GPT-Live bridge and the avatar it drives. */
class Spouse implements GptLiveEvents {
  readonly bridge: GptLiveBridge;
  readonly media: MediaServerLeg;
  ready = false;
  /** Last time this spouse's SPEECH (not silence) was forwarded to its avatar. */
  lastVoicedAt: number | null = null;

  constructor(
    readonly profile: Profile,
    readonly upstream: StartedSession,
    private readonly meeting: CoupleSession,
    persona: string,
  ) {
    const log = (msg: string) => meeting.log(`${profile.name}: ${msg}`);
    this.media = new MediaServerLeg(upstream.wsUrl, log);
    this.bridge = new GptLiveBridge(this, log, {
      instructions: persona + meetingRules(profile),
      greeting: profile.greeting,
      voice: profile.voice,
      responsesInstructions: responsesInstructions(profile, persona),
      tools: false,
    });
  }

  get who(): Speaker {
    return this.profile.who;
  }

  /** Send audio to the avatar; returns whether the chunk was speech. */
  forward(audioB64: string, voiced = isVoiced(audioB64)): boolean {
    if (voiced) this.lastVoicedAt = Date.now();
    this.media.speak(audioB64);
    return voiced;
  }

  onReady(): void {
    this.ready = true;
    this.meeting.onSpouseReady();
  }
  onAudio(audioB64: string): void {
    this.meeting.onSpouseAudio(this, audioB64);
  }
  onTurn(turn: Turn): void {
    this.meeting.onSpouseTurn(this, turn);
  }
  onUserTurnStarted(): void {
    this.meeting.onAdviserStarted(this);
  }
  onToolCall(): Record<string, unknown> {
    return { shown: false, error: "no tools in this session" };
  }
  onError(message: string): void {
    this.meeting.log(`${this.profile.name}: error: ${message}`);
    this.meeting.emit({ type: "error", message: `${this.profile.name}: ${message}` });
  }
}

export class CoupleSession implements LiveSession {
  readonly sessionId = randomUUID();
  lastActivityAt = Date.now();
  readonly spouses: Record<Speaker, Spouse>;

  private frontend: WebSocket | null = null;
  private stopping = false;
  private readyEmitted = false;

  /** The adviser transcript is taken from ONE spouse's input transcript (both hear the same mic). */
  private readonly ear: Speaker = "tony";
  private readonly history: string[] = [];

  // ── floor state ──
  private floor: Speaker | null = "tony"; // Tony greets first
  private floorSource = "greeting";
  private lastSpeaker: Speaker | null = null;
  private round = 0;
  private held: Record<Speaker, string[]> = { tony: [], antonella: [] };
  private firstHeld: Speaker | null = null;
  private holdTimer: NodeJS.Timeout | null = null;
  private decideTimer: NodeJS.Timeout | null = null;
  private decisionSeq = 0;
  private decidedText = "";
  private latestAdviserText = "";
  private lastDecisionAt = 0;
  private chimeIn = false;
  private chimedRound = -1;
  private readonly spokeThisRound = new Set<Speaker>();
  private readonly suppressedThisRound = new Set<Speaker>();

  // ── barge-in ──
  private interruptWatch: NodeJS.Timeout | null = null;
  private lastInterruptAt: number | null = null;

  // ── handoffs (Park) ──
  /** Spouse-to-spouse handoffs since the adviser last spoke. */
  private handoffs = 0;

  // ── transcript ──
  readonly transcript = new TranscriptRecorder();
  private readonly startedAt = new Date();

  constructor(
    tony: StartedSession,
    antonella: StartedSession,
    private readonly personas: Personas,
    private readonly onDead: (sessionId: string) => void,
  ) {
    this.spouses = {
      tony: new Spouse(PROFILES.tony, tony, this, personas.text.tony),
      antonella: new Spouse(PROFILES.antonella, antonella, this, personas.text.antonella),
    };
  }

  get upstreamIds(): string[] {
    return [this.spouses.tony.upstream.sessionId, this.spouses.antonella.upstream.sessionId];
  }

  start(): void {
    for (const spouse of Object.values(this.spouses)) {
      void this.runLeg(`${spouse.profile.name} media server`, () => spouse.media.run());
      void this.runLeg(`${spouse.profile.name} GPT-Live`, async () => {
        if (!(await spouse.media.waitUntilReady(MEDIA_READY_TIMEOUT_MS))) {
          this.emit({ type: "error", message: `${spouse.profile.name}'s avatar could not be reached — start again` });
          return;
        }
        await spouse.bridge.run();
      });
    }
  }

  async stop(reason = "stopped"): Promise<void> {
    this.stopping = true;
    for (const t of [this.holdTimer, this.decideTimer, this.interruptWatch]) if (t) clearTimeout(t);
    await Promise.all(Object.values(this.spouses).map((s) => s.bridge.close()));
    for (const s of Object.values(this.spouses)) s.media.close();
    this.frontend = null;
    this.saveTranscript(reason);
  }

  /** Write the meeting's transcript (transcript.ts). Never throws into teardown. */
  private saveTranscript(reason: string): void {
    const t = config.transcripts;
    try {
      const file = this.transcript.save(t.dir, {
        sessionId: this.sessionId,
        model: config.gptlive.model,
        decisionModel: config.couple.decisionModel,
        personaSource: this.personas.source,
        student: { id: t.studentId, name: t.studentName, email: t.studentEmail },
        startedAt: this.startedAt,
        endedAt: new Date(),
        endReason: reason,
      });
      this.log(file ? `transcript saved: ${file}` : "transcript empty — nothing saved");
    } catch (err) {
      this.log(`transcript save failed: ${err instanceof Error ? err.message : err}`);
    }
  }

  tryAttachFrontend(ws: WebSocket): boolean {
    if (this.frontend) return false;
    this.frontend = ws;
    if (this.readyEmitted) this.emit({ type: "ready" });
    return true;
  }

  detachFrontend(ws: WebSocket): void {
    if (this.frontend === ws) this.frontend = null;
  }

  /** The adviser's mic goes to both spouses. */
  sendMicAudio(audioB64: string): void {
    this.lastActivityAt = Date.now();
    this.spouses.tony.bridge.sendMicAudio(audioB64);
    this.spouses.antonella.bridge.sendMicAudio(audioB64);
  }

  // ── events from the spouses ───────────────────────────────────────────────

  onSpouseReady(): void {
    if (this.readyEmitted || !this.spouses.tony.ready || !this.spouses.antonella.ready) return;
    this.readyEmitted = true;
    this.log(`both spouses ready (personas: ${this.personas.source})`);
    this.emit({ type: "ready" });
    this.emit({ type: "floor", speaker: "tony", why: "greeting" });
    this.transcript.floorChange("tony", "greeting");
  }

  onSpouseAudio(spouse: Spouse, audioB64: string): void {
    const who = spouse.who;
    const voiced = isVoiced(audioB64);
    if (this.floor === who) {
      // Silence is forwarded too: it keeps the avatar's timing natural.
      spouse.forward(audioB64, voiced);
      if (voiced) {
        this.lastSpeaker = who;
        this.spokeThisRound.add(who);
      }
      return;
    }
    if (this.floor !== null) {
      if (voiced) this.markSuppressed(who);
      return;
    }
    // Undecided: hold a reply (from its first spoken chunk on, pauses
    // included) until the decision lands or the wait runs out. Pure silence
    // before anyone speaks is just dropped.
    if (!voiced && this.held[who].length === 0) return;
    this.held[who].push(audioB64);
    this.firstHeld ??= who;
    this.holdTimer ??= setTimeout(() => {
      this.holdTimer = null;
      if (this.floor !== null) return;
      const fallback =
        namedSpeaker(this.decidedText) ??
        (this.held.tony.length && this.held.antonella.length ? this.firstHeld : null) ??
        (this.held.tony.length ? "tony" : "antonella");
      this.setFloor(fallback ?? "tony", "decision too slow — first to answer");
    }, DECISION_WAIT_MS);
  }

  onSpouseTurn(spouse: Spouse, turn: Turn): void {
    if (turn.role === "user") {
      if (spouse.who === this.ear) this.onAdviserTurn(turn);
      return;
    }
    const who = spouse.who;
    // Only what was actually heard goes to the transcript and the partner.
    if (this.floor !== who && !this.spokeThisRound.has(who)) return;
    const id = `${who}_${turn.id}`;
    this.emit({ type: "turn", ...turn, id, speaker: who });
    this.transcript.record(id, who, turn.text, turn.done);
    if (!turn.done || !turn.text.trim()) return;

    const p = spouse.profile;
    const said = turn.text.trim().slice(0, 600);
    this.pushHistory(`${p.name}: ${said}`);
    this.spouses[p.other].bridge.append("thinking", `(${p.name} just said to the adviser: "${said}")`);

    if (this.chimeIn && this.chimedRound !== this.round && this.floor === who) {
      this.chimedRound = this.round;
      this.log(`${PROFILES[p.other].name} chimes in after ${p.name}`);
      this.setFloor(p.other, "chime-in");
      this.spouses[p.other].bridge.append(
        "instructions",
        `Your ${PROFILES[p.other].otherRelation} ${p.name} just answered the adviser. Add one short, natural sentence from your own point of view, then stop and let the adviser continue.`,
      );
      return;
    }

    // Park's handoff: the spouse who just spoke asked the other directly.
    if (this.floor === who && this.handoffs < MAX_HANDOFFS && handsOffTo(said, p.otherName)) {
      this.handoffs += 1;
      this.log(`handoff ${p.name} → ${p.otherName} (${this.handoffs}/${MAX_HANDOFFS})`);
      this.setFloor(p.other, "handoff");
      this.spouses[p.other].bridge.append(
        "instructions",
        `Your ${PROFILES[p.other].otherRelation} ${p.name} just spoke to you directly: "${said}". Answer ${p.name} now, briefly and in your own words, then let the adviser continue.`,
      );
    }
  }

  /** A new adviser turn opened (from the ear's transcript): new round. */
  onAdviserStarted(spouse: Spouse): void {
    if (spouse.who !== this.ear) return;
    this.round += 1;
    this.handoffs = 0; // the adviser has the floor again
    if (this.decideTimer) clearTimeout(this.decideTimer);
    this.decideTimer = null;
    this.latestAdviserText = "";
    this.decidedText = "";
    this.chimeIn = false;
    this.spokeThisRound.clear();
    this.suppressedThisRound.clear();
    this.dropHeld();
    // Someone still talking through the adviser's "mm-hmm" keeps the floor
    // until the decision says otherwise; otherwise the floor is open.
    const s = this.lastSpeaker ? this.spouses[this.lastSpeaker] : null;
    const continuing = s?.lastVoicedAt != null && Date.now() - s.lastVoicedAt < CONTINUING_MS;
    this.floor = continuing ? this.lastSpeaker : null;
    this.floorSource = continuing ? "continuing" : "open";
    this.watchBargeIn();
  }

  // ── floor ─────────────────────────────────────────────────────────────────

  private onAdviserTurn(turn: Turn): void {
    const id = `adv_${turn.id}`;
    this.emit({ type: "turn", ...turn, id });
    this.transcript.record(id, "adviser", turn.text, turn.done);

    // Park's backchannel rule: "mm-hmm" over a spouse who is still talking is
    // listening, not a question. The speaker keeps the floor; nobody routes it.
    const talking = this.lastSpeaker ? this.spouses[this.lastSpeaker] : null;
    const stillTalking =
      talking?.lastVoicedAt != null && Date.now() - talking.lastVoicedAt < BACKCHANNEL_WINDOW_MS;
    if (turn.done && stillTalking && this.lastSpeaker && isBackchannel(turn.text) && !namedSpeaker(turn.text)) {
      this.pushHistory(`Adviser (while ${PROFILES[this.lastSpeaker].name} was talking): ${turn.text.trim()}`);
      if (this.floor !== this.lastSpeaker) this.setFloor(this.lastSpeaker, "backchannel");
      return;
    }
    if (turn.done && turn.text.trim()) this.pushHistory(`Adviser: ${turn.text.trim()}`);

    const named = namedSpeaker(turn.text);
    if (named) {
      // Latest name wins ("Tony — actually, Antonella?"), unless the first
      // named spouse has already started answering.
      const current = this.floor;
      const answering = current !== null && current !== named && this.spokeThisRound.has(current);
      if (!answering && (current !== named || this.floorSource !== "named")) this.setFloor(named, "named");
      return;
    }
    if (this.floorSource === "named") return;
    this.latestAdviserText = turn.text.trim();
    const round = this.round;
    if (turn.done) {
      if (this.decideTimer) clearTimeout(this.decideTimer);
      this.decideTimer = null;
      void this.decide(round, this.latestAdviserText);
      return;
    }
    if (this.latestAdviserText.split(/\s+/).length < DECISION_MIN_WORDS || this.decideTimer) return;
    const wait = Math.max(0, this.lastDecisionAt + DECISION_THROTTLE_MS - Date.now());
    this.decideTimer = setTimeout(() => {
      this.decideTimer = null;
      if (round === this.round) void this.decide(round, this.latestAdviserText);
    }, wait);
  }

  private async decide(round: number, text: string): Promise<void> {
    if (!text || text === this.decidedText || this.stopping) return;
    this.decidedText = text;
    const seq = ++this.decisionSeq;
    const started = Date.now();
    this.lastDecisionAt = started;
    let result: { speaker: Speaker; chimeIn: boolean } | null = null;
    try {
      result = await decideSpeaker(this.history.slice(-HISTORY_LINES), text, this.lastSpeaker);
    } catch (err) {
      this.log(`decision failed: ${err instanceof Error ? err.message : err}`);
    }
    const ms = Date.now() - started;
    if (!result) {
      this.log(`decision returned nothing usable (${ms}ms)`);
      return;
    }
    // A new adviser turn began: this answer is about the old one.
    if (round !== this.round || this.stopping) return;
    // An older (shorter-text) decision is still useful while the floor is
    // open; once someone holds it, only the newest decision may change it.
    if (seq !== this.decisionSeq && this.floor !== null) return;
    this.log(`decision: ${result.speaker}${result.chimeIn ? " +chime-in" : ""} in ${ms}ms for "${text.slice(0, 60)}"`);
    if (this.floorSource === "named") return;
    this.chimeIn = result.chimeIn;
    // Don't yank the floor from someone already answering this round.
    if (this.floor && this.floor !== result.speaker && this.spokeThisRound.has(this.floor)) {
      this.log(`decision ignored: ${PROFILES[this.floor].name} is already answering`);
      return;
    }
    this.setFloor(
      result.speaker,
      `${config.couple.decisionModel} in ${Date.now() - started}ms${result.chimeIn ? " (+chime-in)" : ""}`,
    );
  }

  private setFloor(who: Speaker, why: string): void {
    if (this.holdTimer) {
      clearTimeout(this.holdTimer);
      this.holdTimer = null;
    }
    const other: Speaker = who === "tony" ? "antonella" : "tony";
    const changed = this.floor !== who;
    this.floor = who;
    this.floorSource = why === "named" ? "named" : why;
    // Shown in the page and kept in the transcript, changed or confirmed.
    this.emit({ type: "floor", speaker: who, why });
    this.transcript.floorChange(who, why);
    if (changed) {
      this.log(`floor → ${PROFILES[who].name} (${why})`);
      // The other avatar may still be playing queued speech: cut it.
      const o = this.spouses[other];
      if (o.lastVoicedAt !== null && Date.now() - o.lastVoicedAt < YIELD_STALE_MS) o.media.interrupt();
    }
    // Flush the winner's held reply; the loser's is dropped. (Held audio
    // always starts with speech, so non-empty means they really answered.)
    const winner = this.spouses[who];
    for (const chunk of this.held[who]) winner.forward(chunk);
    if (this.held[who].length) {
      this.lastSpeaker = who;
      this.spokeThisRound.add(who);
    }
    if (this.held[other].length) this.markSuppressed(other);
    this.held = { tony: [], antonella: [] };
    this.firstHeld = null;
  }

  private dropHeld(): void {
    if (this.holdTimer) {
      clearTimeout(this.holdTimer);
      this.holdTimer = null;
    }
    this.held = { tony: [], antonella: [] };
    this.firstHeld = null;
  }

  /** Tell a spouse (once per round) that their reply was not heard. */
  private markSuppressed(who: Speaker): void {
    if (this.suppressedThisRound.has(who)) return;
    this.suppressedThisRound.add(who);
    const p = PROFILES[who];
    this.spouses[who].bridge.append(
      "thinking",
      `(${p.otherName} is answering the adviser this time, not you. Your own reply was not heard by anyone, so do not continue or repeat it. Listen to ${p.otherName}.)`,
    );
  }

  // ── barge-in ──────────────────────────────────────────────────────────────

  private watchBargeIn(): void {
    if (this.interruptWatch) return;
    this.interruptWatch = setTimeout(() => {
      this.interruptWatch = null;
      const who = this.lastSpeaker;
      if (!who || this.stopping) return;
      const s = this.spouses[who];
      if (s.lastVoicedAt === null) return;
      if (this.lastInterruptAt !== null && Date.now() - this.lastInterruptAt < INTERRUPT_COOLDOWN_MS) return;
      const quietFor = Date.now() - s.lastVoicedAt;
      if (quietFor < YIELD_QUIET_MS || quietFor > YIELD_STALE_MS) return;
      this.lastInterruptAt = Date.now();
      this.log(`adviser barge-in — clearing ${s.profile.name}'s audio`);
      s.media.interrupt();
      this.emit({ type: "interrupted", speaker: who });
      this.transcript.markInterrupted(who);
      // Park's cut-off notice: GPT-Live believes it said everything it
      // generated; tell it the adviser did not hear the end.
      s.bridge.append(
        "thinking",
        "(The adviser cut you off before you finished that reply; they did not hear the rest. Do not repeat it unless they ask.)",
      );
    }, YIELD_GRACE_MS);
  }

  // ── plumbing ──────────────────────────────────────────────────────────────

  private pushHistory(line: string): void {
    this.history.push(line);
    if (this.history.length > HISTORY_LINES * 2) this.history.splice(0, this.history.length - HISTORY_LINES * 2);
  }

  private async runLeg(name: string, run: () => Promise<void>): Promise<void> {
    try {
      await run();
    } catch (err) {
      if (!this.stopping) this.log(`${name} leg failed: ${err instanceof Error ? err.message : err}`);
    }
    if (this.stopping) return;
    this.log(`${name} leg exited — ending meeting`);
    this.emit({ type: "error", message: `the ${name} connection dropped` });
    this.onDead(this.sessionId);
  }

  emit(message: ServerMessage): void {
    const ws = this.frontend;
    if (!ws || ws.readyState !== ws.OPEN) return;
    try {
      ws.send(JSON.stringify(message));
    } catch {
      this.frontend = null;
    }
  }

  log(msg: string): void {
    console.log(`[couple ${this.sessionId.slice(0, 8)}] ${msg}`);
  }
}
