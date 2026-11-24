/**
 * Couple-meeting transcripts, saved automatically when a meeting ends.
 *
 * Every turn shown on screen is recorded as it streams: the adviser's (the
 * student's) words, and each spouse's words that were actually heard. When the
 * meeting ends — End, closing the tab, the idle watchdog, Ctrl+C — one JSON file
 * is written to TRANSCRIPTS_DIR (default: <app>/transcripts, gitignored):
 *
 *   { session, student, persona_source, started_at, ended_at, end_reason,
 *     metrics, turns: [...], floor: [...], retrievals: [...], openwebui_chat_id }
 *
 *  - turns: who spoke (adviser / tony / antonella), the text, times, whether a
 *    spouse was cut off, whether the turn finished, and for an adviser turn the
 *    spouse who answered it;
 *  - floor: every floor decision (who, why: named / model / chime-in /
 *    handoff / backchannel / ...), for checking the routing afterwards;
 *  - metrics: turns and words per speaker, interruptions, how often each
 *    spouse answered.
 *
 * Only text is kept; audio is never stored (PROJECT_PLAN B2's default rule:
 * keep the transcript, discard the audio).
 */

import { randomUUID } from "node:crypto";
import { mkdirSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { Speaker } from "../../shared/messages";

export type TranscriptSpeaker = "adviser" | Speaker;

export interface RecordedTurn {
  id: string;
  speaker: TranscriptSpeaker;
  /** Adviser turns only: the spouse who answered it (filled in at save). */
  answered_by?: Speaker;
  text: string;
  started_at: string;
  ended_at: string;
  /** A spouse was cut off: words after the cut may not have been heard. */
  interrupted: boolean;
  /** False when the meeting ended before the turn finished. */
  complete: boolean;
}

export interface FloorEvent {
  at: string;
  speaker: Speaker;
  why: string;
}

/** One delegated turn: what was asked, what Open WebUI's knowledge returned, how long it took. */
export interface RetrievalEvent {
  at: string;
  speaker: Speaker;
  question: string;
  sources: string[];
  ms: number;
  outcome: "answered" | "stale" | "failed";
}

export interface MeetingMeta {
  sessionId: string;
  model: string;
  decisionModel: string;
  personaSource: string;
  student: { id: string; name: string; email: string };
  startedAt: Date;
  endedAt: Date;
  endReason: string;
  /** Set when the meeting was also saved as an Open WebUI chat. */
  openWebUIChatId?: string | null;
}

const words = (s: string) => s.split(/\s+/).filter(Boolean).length;

export class TranscriptRecorder {
  private readonly turns = new Map<string, RecordedTurn>();
  private readonly floor: FloorEvent[] = [];
  private interruptions = 0;
  private readonly retrievals: RetrievalEvent[] = [];

  /** Every streaming update and the final one of each turn. */
  record(id: string, speaker: TranscriptSpeaker, text: string, done: boolean): void {
    const now = new Date().toISOString();
    const clean = text.trim();
    const existing = this.turns.get(id);
    if (existing) {
      if (clean) existing.text = clean;
      existing.ended_at = now;
      existing.complete ||= done;
      return;
    }
    if (!clean) return;
    this.turns.set(id, {
      id,
      speaker,
      text: clean,
      started_at: now,
      ended_at: now,
      interrupted: false,
      complete: done,
    });
  }

  /** The adviser cut this spouse off: mark its latest turn. */
  markInterrupted(who: Speaker): void {
    this.interruptions += 1;
    const last = this.list().reverse().find((t) => t.speaker === who);
    if (last) last.interrupted = true;
  }

  floorChange(speaker: Speaker, why: string): void {
    this.floor.push({ at: new Date().toISOString(), speaker, why });
  }

  list(): RecordedTurn[] {
    return [...this.turns.values()].sort((a, b) => a.started_at.localeCompare(b.started_at));
  }

  /** Turns with each adviser turn linked to the spouse who spoke next. */
  private linked(): RecordedTurn[] {
    const turns = this.list().map((t) => ({ ...t }));
    for (let i = 0; i < turns.length; i++) {
      const t = turns[i]!;
      if (t.speaker !== "adviser") continue;
      const reply = turns.slice(i + 1).find((n) => n.speaker !== "adviser");
      if (reply && reply.speaker !== "adviser") t.answered_by = reply.speaker;
    }
    return turns;
  }

  metrics() {
    const turns = this.linked();
    const of = (s: TranscriptSpeaker) => turns.filter((t) => t.speaker === s);
    const wordsOf = (list: RecordedTurn[]) => list.reduce((n, t) => n + words(t.text), 0);
    const adviserWords = wordsOf(of("adviser"));
    const clientWords = wordsOf(of("tony")) + wordsOf(of("antonella"));
    const spouse = (s: Speaker) => ({
      turns: of(s).length,
      words: wordsOf(of(s)),
      answered_adviser: of("adviser").filter((t) => t.answered_by === s).length,
      interrupted: of(s).filter((t) => t.interrupted).length,
    });
    return {
      adviser_turns: of("adviser").length,
      adviser_words: adviserWords,
      client_words: clientWords,
      // Share of words spoken by the student; a rough talk-time ratio.
      adviser_word_share: adviserWords + clientWords ? +(adviserWords / (adviserWords + clientWords)).toFixed(3) : null,
      interruptions_by_adviser: this.interruptions,
      tony: spouse("tony"),
      antonella: spouse("antonella"),
    };
  }

  /** A delegated turn answered from Open WebUI's knowledge (kept for checking afterwards). */
  retrieval(entry: Omit<RetrievalEvent, "at">): void {
    this.retrievals.push({ at: new Date().toISOString(), ...entry });
  }

  /**
   * The meeting as an Open WebUI chat document (POST /api/v1/chats/new), in
   * the shape Open WebUI's own UI saves: a `history` tree for the sidebar
   * view and a flat `messages` list, which is what the grading extraction
   * (grading_feature/backend/extract_chats.py) reads.
   *
   * The student is `user`; the spouses are `assistant`, each line prefixed
   * with who spoke. The grader pairs strictly user → assistant and drops a
   * second assistant in a row, so back-to-back turns of the same role are
   * merged into one message (Tony then Antonella → one assistant message).
   * Returns null when nothing was said.
   */
  toOpenWebUIChat(opts: {
    title: string;
    models: Record<Speaker, string>;
    names: Record<Speaker, string>;
  }): Record<string, unknown> | null {
    const turns = this.list().filter((t) => t.text.trim());
    if (turns.length === 0) return null;
    type Msg = {
      id: string;
      parentId: string | null;
      childrenIds: string[];
      role: "user" | "assistant";
      content: string;
      timestamp: number;
      model?: string;
      modelName?: string;
      models?: string[];
      done?: boolean;
    };
    const models = [...new Set(Object.values(opts.models))];
    const merged: Msg[] = [];
    for (const t of turns) {
      const role = t.speaker === "adviser" ? "user" : "assistant";
      const line =
        t.speaker === "adviser"
          ? t.text
          : `${opts.names[t.speaker]}: ${t.text}${t.interrupted ? " [interrupted]" : ""}`;
      const last = merged.at(-1);
      if (last && last.role === role) {
        last.content += `\n\n${line}`;
        continue;
      }
      const msg: Msg = {
        id: randomUUID(),
        parentId: last?.id ?? null,
        childrenIds: [],
        role,
        content: line,
        timestamp: Math.floor(Date.parse(t.started_at) / 1000),
      };
      if (role === "assistant") {
        const who = t.speaker as Speaker;
        Object.assign(msg, { model: opts.models[who], modelName: opts.names[who], done: true });
      } else {
        msg.models = models;
      }
      if (last) last.childrenIds.push(msg.id);
      merged.push(msg);
    }
    return {
      title: opts.title,
      models,
      params: {},
      history: {
        messages: Object.fromEntries(merged.map((m) => [m.id, m])),
        currentId: merged.at(-1)!.id,
      },
      messages: merged,
      tags: [],
      timestamp: Date.parse(turns[0]!.started_at),
    };
  }

  /** Write the meeting file; returns its path, or null when nothing was said. */
  save(dir: string, meta: MeetingMeta): string | null {
    if (this.turns.size === 0) return null;
    mkdirSync(dir, { recursive: true });
    const record = {
      session_id: meta.sessionId,
      mode: "couple-avatar-voice",
      model: meta.model,
      decision_model: meta.decisionModel,
      persona_source: meta.personaSource,
      student: meta.student,
      started_at: meta.startedAt.toISOString(),
      ended_at: meta.endedAt.toISOString(),
      end_reason: meta.endReason,
      metrics: this.metrics(),
      turns: this.linked(),
      floor: this.floor,
      retrievals: this.retrievals,
      openwebui_chat_id: meta.openWebUIChatId ?? null,
    };
    const stamp = meta.startedAt.toISOString().replace(/[:.]/g, "-");
    const file = join(dir, `${stamp}_couple_${meta.sessionId.slice(0, 8)}.json`);
    const tmp = `${file}.tmp`;
    writeFileSync(tmp, JSON.stringify(record, null, 2), "utf8");
    renameSync(tmp, file);
    return file;
  }
}
