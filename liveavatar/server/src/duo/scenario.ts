/**
 * Duo scenarios: one folder per scenario under server/scenarios/<name>/,
 * picked with DUO_SCENARIO (default "beckham"). A course adds a family by
 * adding a folder — no code.
 *
 *   scenario.json   who is on the call, how to route, the opening line
 *   <persona>.md    each persona's own system prompt ({{name}}, {{other}},
 *                   {{user}} are filled in)
 *   <knowledge>.md  shared background (e.g. the client data sheet), given to
 *                   BOTH personas in full
 *
 * Knowledge is loaded whole into the system prompt, not retrieved per turn: a
 * case file of a few thousand tokens costs nothing per turn, where a vector
 * search would sit on the critical path of every reply (FamilyFinanceChat's
 * AVATAR_TRACK §5 makes the same call). Revisit only if a scenario's documents
 * outgrow the context window.
 */

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const SCENARIOS_DIR = fileURLToPath(new URL("../../scenarios/", import.meta.url));

interface PersonaSpec {
  id: string;
  /** Omitted → the persona takes its (random) avatar's first name. */
  name?: string;
  tagline: string;
  /** Routing criterion: what the decision model reads to pick this persona. */
  role: string;
  /** Prompt file, relative to the scenario folder. */
  prompt: string;
  /** Gemini prebuilt voice; env DUO_VOICE_<ID> overrides. */
  voice: string;
  /** Omitted → a random public avatar (DUO_AVATAR_IDS overrides both). */
  avatarId?: string;
}

interface ScenarioSpec {
  title: string;
  subtitle?: string;
  /** What the human is called in transcripts and prompts: "Adviser", "User". */
  userLabel: string;
  knowledge: string[];
  /** The decision question's instructions. */
  routing: string;
  greeting: { persona: string; directive: string } | null;
  personas: PersonaSpec[];
}

export interface Persona {
  id: string;
  name: string;
  tagline: string;
  role: string;
  voice: string;
  avatarId: string;
  avatarName: string;
  /** The complete Gemini Live system instruction. */
  instructions: string;
}

export interface Scenario {
  id: string;
  title: string;
  subtitle: string;
  userLabel: string;
  routing: string;
  greeting: { persona: string; directive: string } | null;
  personas: [Persona, Persona];
}

export function loadScenarioSpec(id: string): ScenarioSpec {
  const dir = scenarioDir(id);
  const spec = JSON.parse(readFileSync(`${dir}/scenario.json`, "utf8")) as ScenarioSpec;
  if (spec.personas?.length !== 2) throw new Error(`scenario "${id}" needs exactly two personas`);
  for (const p of spec.personas) {
    if (!existsSync(`${dir}/${p.prompt}`)) throw new Error(`scenario "${id}": missing ${p.prompt}`);
  }
  for (const k of spec.knowledge ?? []) {
    if (!existsSync(`${dir}/${k}`)) throw new Error(`scenario "${id}": missing ${k}`);
  }
  if (spec.greeting && !spec.personas.some((p) => p.id === spec.greeting!.persona)) {
    throw new Error(`scenario "${id}": greeting persona "${spec.greeting.persona}" is not on the call`);
  }
  return spec;
}

/** Avatar ids the spec pins, or null for a persona that wants a random one. */
export function pinnedAvatarIds(spec: ScenarioSpec): (string | null)[] {
  return spec.personas.map((p) => p.avatarId ?? null);
}

/** Bind a spec to the avatars actually started for it. */
export function castScenario(
  id: string,
  spec: ScenarioSpec,
  avatars: { id: string; name: string }[],
): Scenario {
  const dir = scenarioDir(id);
  const knowledge = (spec.knowledge ?? [])
    .map((file) => readFileSync(`${dir}/${file}`, "utf8").trim())
    .join("\n\n");
  const names = spec.personas.map((p, i) => p.name ?? avatars[i]!.name.split(/\s+/)[0] ?? p.id);
  const personas = spec.personas.map((p, i): Persona => {
    const name = names[i]!;
    const other = names[1 - i]!;
    const fill = (text: string) =>
      text.replaceAll("{{name}}", name).replaceAll("{{other}}", other).replaceAll("{{user}}", spec.userLabel);
    const own = fill(readFileSync(`${dir}/${p.prompt}`, "utf8").trim());
    return {
      id: p.id,
      name,
      tagline: p.tagline,
      role: p.role,
      voice: process.env[`DUO_VOICE_${p.id.toUpperCase()}`]?.trim() || p.voice,
      avatarId: avatars[i]!.id,
      avatarName: avatars[i]!.name,
      instructions: [own, callMechanics(name, other, spec.userLabel), knowledge && `# Case file\n\n${knowledge}`]
        .filter(Boolean)
        .join("\n\n"),
    };
  }) as [Persona, Persona];
  return {
    id,
    title: spec.title,
    subtitle: spec.subtitle ?? "",
    userLabel: spec.userLabel,
    routing: spec.routing,
    greeting: spec.greeting,
    personas,
  };
}

/**
 * How the two-avatar call works, from one persona's side. Same for every
 * scenario — it describes the director's protocol, not a character.
 */
function callMechanics(me: string, other: string, user: string): string {
  return [
    "# How this call works",
    `- ${other} is on this live call too, with their own camera and voice. A director decides who answers each thing the ${user.toLowerCase()} says.`,
    `- When it is your turn you receive the ${user.toLowerCase()}'s words as "${user}: ...". Lines starting with "${other}:" are what ${other} already said out loud since you last spoke — you heard them, so do not repeat them. "${user} (while ${other} was talking): ..." was said over ${other}.`,
    `- Speak only as ${me}. Never speak ${other}'s lines, never start your reply with your own name, and never narrate actions or stage directions.`,
    `- You may react to ${other}, or turn to them with a direct question by name when something is really theirs to answer.`,
    '- Messages starting with "[director]" come from the show, not from anyone on the call. Follow them, and never mention the director.',
  ].join("\n");
}

function scenarioDir(id: string): string {
  if (!/^[a-z0-9_-]+$/i.test(id)) throw new Error(`bad scenario name "${id}"`);
  const dir = `${SCENARIOS_DIR}${id}`;
  if (!existsSync(`${dir}/scenario.json`)) throw new Error(`no scenario "${id}" in server/scenarios/`);
  return dir;
}
