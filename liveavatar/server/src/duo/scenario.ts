/**
 * Duo scenarios: one folder per scenario under server/scenarios/<name>/,
 * picked with DUO_SCENARIO (default "beckham"). A course adds a family by
 * adding a folder — no code.
 *
 *   scenario.json   who is on the call, how to route, the opening line, and
 *                   per persona EITHER an Open WebUI model preset id (the
 *                   FamilyFinanceChat way: prompt + documents live in Open
 *                   WebUI) OR a local prompt file
 *   <persona>.md    a local persona prompt ({{name}}, {{other}}, {{user}} are
 *                   filled in) — local scenarios only
 *   <knowledge>.md  local background given to BOTH personas in full —
 *                   local scenarios only
 *
 * Two ways to give personas their documents, kept side by side on purpose
 * (FamilyFinanceChat task B12 asks for exactly this comparison):
 *  - preset-backed: Open WebUI's knowledge search on every turn, run in
 *    parallel with the routing decision (stage.ts) — what FFC's text chat does;
 *  - local: the whole file preloaded into the system prompt (beckham-local).
 */

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { config } from "../config";
import { fetchPreset, type Preset } from "./openwebui";

const SCENARIOS_DIR = fileURLToPath(new URL("../../scenarios/", import.meta.url));

type Engine = "gemini" | "gptlive";

interface PersonaSpec {
  id: string;
  /**
   * Open WebUI model preset id (FamilyFinanceChat's client model): its system
   * prompt is the persona, its attached knowledge is searched every turn.
   * Env DUO_OPENWEBUI_MODEL_<ID> overrides (ids differ per Open WebUI instance).
   */
  openwebuiModel?: string;
  /** …or a local prompt file, relative to the scenario folder. */
  prompt?: string;
  /** Omitted → the preset's name, else the (random) avatar's first name. */
  name?: string;
  tagline: string;
  /** Routing criterion for the decision model. Omitted → the preset's description. */
  role?: string;
  /** Per engine; a bare string is the Gemini voice. Env DUO_VOICE_<ID> overrides. */
  voice: string | Partial<Record<Engine, string>>;
  /** Omitted → a random public avatar (DUO_AVATAR_IDS overrides both). */
  avatarId?: string;
}

interface ScenarioSpec {
  title: string;
  subtitle?: string;
  /** What the human is called in transcripts and prompts: "Adviser", "User". */
  userLabel: string;
  /**
   * Local files given IN FULL to both personas (preloaded, not searched).
   * Preset-backed personas get their documents by per-turn search instead.
   */
  knowledge?: string[];
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
  /** Voice for the configured engine. */
  voice: string;
  avatarId: string;
  avatarName: string;
  /** The complete system instruction for the persona's live session. */
  instructions: string;
  /** Open WebUI preset this persona comes from, if any. */
  presetId: string | null;
  /** Knowledge collections searched for this persona every turn (may be empty). */
  collections: string[];
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
    const preset = presetIdOf(p);
    if (!preset && !p.prompt) {
      throw new Error(
        `scenario "${id}": persona "${p.id}" has no Open WebUI preset — set DUO_OPENWEBUI_MODEL_${p.id.toUpperCase()} ` +
          `to its model id (Workspace → Models), or use a local scenario (e.g. DUO_SCENARIO=beckham-local)`,
      );
    }
    if (!preset && p.prompt && !existsSync(`${dir}/${p.prompt}`)) throw new Error(`scenario "${id}": missing ${p.prompt}`);
    if (!p.role && !preset) throw new Error(`scenario "${id}": persona "${p.id}" needs a "role"`);
  }
  for (const k of spec.knowledge ?? []) {
    if (!existsSync(`${dir}/${k}`)) throw new Error(`scenario "${id}": missing ${k}`);
  }
  if (spec.greeting && !spec.personas.some((p) => p.id === spec.greeting!.persona)) {
    throw new Error(`scenario "${id}": greeting persona "${spec.greeting.persona}" is not on the call`);
  }
  return spec;
}

/** Does any persona come from an Open WebUI preset? (Decides which env is required.) */
export function usesOpenWebUI(spec: ScenarioSpec): boolean {
  return spec.personas.some((p) => presetIdOf(p) !== null);
}

/** Avatar ids the spec pins, or null for a persona that wants a random one. */
export function pinnedAvatarIds(spec: ScenarioSpec): (string | null)[] {
  return spec.personas.map((p) => p.avatarId ?? null);
}

/**
 * Bind a spec to the avatars actually started for it. Presets are fetched
 * here, at every call start, so a professor's edit in Open WebUI applies to
 * the next call.
 */
export async function castScenario(
  id: string,
  spec: ScenarioSpec,
  avatars: { id: string; name: string }[],
): Promise<Scenario> {
  const dir = scenarioDir(id);
  const presets: (Preset | null)[] = await Promise.all(
    spec.personas.map((p) => {
      const presetId = presetIdOf(p);
      return presetId ? fetchPreset(presetId) : Promise.resolve(null);
    }),
  );
  const knowledge = (spec.knowledge ?? [])
    .map((file) => readFileSync(`${dir}/${file}`, "utf8").trim())
    .join("\n\n");
  const names = spec.personas.map(
    (p, i) => p.name ?? presets[i]?.name ?? avatars[i]!.name.split(/\s+/)[0] ?? p.id,
  );
  const personas = spec.personas.map((p, i): Persona => {
    const name = names[i]!;
    const other = names[1 - i]!;
    const preset = presets[i];
    const fill = (text: string) =>
      text.replaceAll("{{name}}", name).replaceAll("{{other}}", other).replaceAll("{{user}}", spec.userLabel);
    const own = preset ? preset.system : fill(readFileSync(`${dir}/${p.prompt!}`, "utf8").trim());
    return {
      id: p.id,
      name,
      tagline: p.tagline,
      role: p.role || preset?.description || name,
      voice: voiceFor(p),
      avatarId: avatars[i]!.id,
      avatarName: avatars[i]!.name,
      instructions: [own, callMechanics(name, other, spec.userLabel), knowledge && `# Case file\n\n${knowledge}`]
        .filter(Boolean)
        .join("\n\n"),
      presetId: preset?.id ?? null,
      collections: preset?.collections ?? [],
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

/** A per-persona env var, with the .env.example placeholder counted as unset. */
function personaEnv(prefix: string, p: PersonaSpec): string {
  const v = process.env[`${prefix}${p.id.toUpperCase()}`]?.trim() ?? "";
  return v === "REPLACE_ME" ? "" : v;
}

function presetIdOf(p: PersonaSpec): string | null {
  return personaEnv("DUO_OPENWEBUI_MODEL_", p) || p.openwebuiModel?.trim() || null;
}

function voiceFor(p: PersonaSpec): string {
  const override = personaEnv("DUO_VOICE_", p);
  if (override) return override;
  const engine = config.duo.personaEngine;
  if (typeof p.voice === "string") {
    // A bare string is a Gemini voice; GPT-Live falls back to its default.
    return engine === "gemini" ? p.voice : "marin";
  }
  return p.voice[engine] ?? (engine === "gemini" ? "Kore" : "marin");
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
