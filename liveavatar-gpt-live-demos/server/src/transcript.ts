/**
 * Couple-meeting transcripts, saved automatically when a meeting ends.
 *
 * Every turn shown on screen is recorded as it streams: the adviser's (the
 * student's) words, and each spouse's words that were actually heard. When the
 * meeting ends — End, closing the tab, the idle watchdog, Ctrl+C — one JSON file
 * is written to TRANSCRIPTS_DIR (default: <app>/transcripts, gitignored):
 *
 *   { session, student, persona_source, started_at, ended_at, end_reason,
 *     metrics, turns: [...], floor: [...] }
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

export interface MeetingMeta {
  sessionId: string;
  model: string;
  decisionModel: string;
  personaSource: string;
  student: { id: string; name: string; email: string };
  startedAt: Date;
  endedAt: Date;
  endReason: string;
}

const words = (s: string) => s.split(/\s+/).filter(Boolean).length;

export class TranscriptRecorder {
  private readonly turns = new Map<string, RecordedTurn>();
  private readonly floor: FloorEvent[] = [];
  private interruptions = 0;

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
    };
    const stamp = meta.startedAt.toISOString().replace(/[:.]/g, "-");
    const file = join(dir, `${stamp}_couple_${meta.sessionId.slice(0, 8)}.json`);
    const tmp = `${file}.tmp`;
    writeFileSync(tmp, JSON.stringify(record, null, 2), "utf8");
    renameSync(tmp, file);
    return file;
  }
}
