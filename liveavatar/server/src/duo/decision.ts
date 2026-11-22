/**
 * The router: gemini-decision-1 over plain REST (POST /v1beta/decisions).
 *
 * Measured at 0.26–0.46s per call — cheap enough to sit on the critical path
 * between "the user stopped talking" and "a persona starts".
 *
 * The response shape (verified against the live API):
 *   { answers: { <key>: { type: "choice", choice, confidence, probabilities } |
 *                       { type: "noul", noul, confidence } }, usage }
 */

import { config } from "../config";
import type { Persona, Scenario } from "./scenario";

const URL = "https://generativelanguage.googleapis.com/v1beta/decisions";
const TIMEOUT_MS = 4_000;

type Question =
  | { type: "noul"; instructions: string }
  | { type: "choice"; instructions: string; criteria: Record<string, string> };

interface Answer {
  type?: string;
  choice?: string;
  noul?: number;
  confidence?: number;
  probabilities?: Record<string, number>;
}

async function decide(input: string, questions: Record<string, Question>): Promise<Record<string, Answer>> {
  const response = await fetch(URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": config.gemini.apiKey },
    body: JSON.stringify({ model: config.duo.decisionModel, input, questions }),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`decision API ${response.status}: ${text.slice(0, 300)}`);
  const body = JSON.parse(text) as { answers?: Record<string, Answer> };
  return body.answers ?? {};
}

export interface SpeakerDecision {
  personaId: string;
  scores: Record<string, number>;
  ms: number;
}

/**
 * Which persona answers the human's latest message. `recent` is the last few
 * transcript lines ("Name: text"), oldest first, ending with the human's line.
 */
export async function pickSpeaker(
  scenario: Scenario,
  recent: readonly string[],
  lastSpeaker: Persona | null,
): Promise<SpeakerDecision> {
  const t0 = Date.now();
  const { personas, userLabel } = scenario;
  const input = [
    `A live video call: ${personas.map((p) => p.name).join(" and ")}, and the ${userLabel.toLowerCase()}.`,
    lastSpeaker ? `Spoke last: ${lastSpeaker.name}.` : "",
    `Recent conversation (oldest first; the last line is the ${userLabel.toLowerCase()}'s latest message):`,
    ...recent,
  ]
    .filter(Boolean)
    .join("\n");
  const criteria = Object.fromEntries(personas.map((p) => [p.id, p.role]));
  const answers = await decide(input, {
    speaker: { type: "choice", instructions: scenario.routing, criteria },
  });
  const a = answers.speaker;
  const personaId = personas.some((p) => p.id === a?.choice) ? a!.choice! : null;
  if (!personaId) throw new Error(`decision returned no usable choice: ${JSON.stringify(a)}`);
  return { personaId, scores: a?.probabilities ?? {}, ms: Date.now() - t0 };
}

/**
 * Did `speaker` just pass the floor to `other` (asked them a direct question,
 * invited their take)? Returns the probability, 0–1.
 */
export async function handoffProbability(
  speaker: Persona,
  other: Persona,
  line: string,
  userLabel: string,
): Promise<number> {
  const answers = await decide(`${speaker.name} said, on a live call: "${line}"`, {
    handoff: {
      type: "noul",
      instructions:
        `Does ${speaker.name} end by asking ${other.name} a direct question or ` +
        `explicitly inviting ${other.name} to speak next, so that ${other.name} ` +
        `should answer now rather than the ${userLabel.toLowerCase()}?`,
    },
  });
  return typeof answers.handoff?.noul === "number" ? answers.handoff.noul : 0;
}
