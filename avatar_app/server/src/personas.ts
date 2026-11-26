/**
 * AV2 — the persona loader: which client the avatar plays.
 *
 * GPT-Live's startup instructions are IMMUTABLE once the session starts, and
 * mid-session appends are capped at 500 tokens and acted on immediately — so
 * the whole persona goes in at `session.start`. No per-turn retrieval sits in
 * the speech loop (AVATAR_TRACK.md §5, "the RAG trap").
 *
 * Two sources behind one interface, chosen by AVATAR_PERSONA_SOURCE:
 *
 *  - `file` (default): avatar_app/personas/*.md — for development, and a
 *    fallback that needs no Open WebUI at all.
 *  - `openwebui`: the professor's own role-play models. Every active
 *    workspace model tagged AVATAR_PERSONA_TAG ("avatar") is offered; its
 *    system prompt becomes the persona. Professor-editable from the Open WebUI
 *    UI, no redeploy. Read with an admin key (openwebui.ts explains why).
 *
 * Known limit: documents attached to an Open WebUI model (its knowledge) are
 * NOT loaded — only the system prompt. A warning is logged when a persona
 * model has knowledge attached.
 */

import { readdir, readFile } from "node:fs/promises";
import { basename, extname, join } from "node:path";
import { config } from "./config";
import { listAllModels, modelTags, type OpenWebUIModel } from "./openwebui";
import { DEFAULT_GREETING, GREETING_PREAMBLE, LIVE_DIRECTIVE } from "./prompts";

// ── limits (GPT-Live gpt-live-1) ─────────────────────────────────────────────

/** `session.start` instructions limit. */
export const INSTRUCTIONS_TOKEN_LIMIT = 16_384;
/** Any single `session.*.append` (the greeting is one). */
export const APPEND_TOKEN_LIMIT = 500;

/**
 * Deliberately pessimistic token estimate: English runs ~4 characters per
 * token, so chars/3 over-counts. A persona that passes this check fits; one
 * near the limit may be refused although it would have fit — the safe side,
 * since the failure upstream would be a session that never starts.
 */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3);
}

// GPT-Live's built-in voices. Custom voices are `voice_…` ids.
const BUILTIN_VOICES = new Set([
  "marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse",
]);

// ── types ────────────────────────────────────────────────────────────────────

export interface PersonaSummary {
  id: string;
  name: string;
}

export interface Persona extends PersonaSummary {
  /** Who the client is. The mechanics directive is appended by assemble(). */
  instructions: string;
  /** How the client opens the meeting. */
  greeting: string;
  /** GPT-Live voice; undefined = the GPT_LIVE_VOICE default. */
  voice?: string;
}

/** What a session is started with: everything GPT-Live needs, checked. */
export interface AssembledPersona {
  id: string;
  name: string;
  instructions: string;
  greeting: string;
  voice: string;
  instructionTokens: number;
}

export interface PersonaSource {
  list(): Promise<PersonaSummary[]>;
  get(id: string): Promise<Persona | null>;
}

export class PersonaError extends Error {
  constructor(
    readonly kind: "not_found" | "invalid" | "unavailable",
    message: string,
  ) {
    super(message);
  }
}

// ── assembly + budget check ──────────────────────────────────────────────────

/**
 * Build the startup instructions and check them against GPT-Live's limits.
 * Fails loudly rather than truncating: a silently clipped persona is a client
 * missing half their facts, and nobody would know why.
 */
export function assemble(persona: Persona): AssembledPersona {
  const instructions = persona.instructions.trim() + LIVE_DIRECTIVE;
  const instructionTokens = estimateTokens(instructions);
  if (instructionTokens > INSTRUCTIONS_TOKEN_LIMIT) {
    throw new PersonaError(
      "invalid",
      `persona "${persona.id}" is too long: ~${instructionTokens} tokens estimated, ` +
        `limit ${INSTRUCTIONS_TOKEN_LIMIT}`,
    );
  }
  const greeting = persona.greeting.trim() || DEFAULT_GREETING;
  const greetingTokens = estimateTokens(GREETING_PREAMBLE + greeting);
  if (greetingTokens > APPEND_TOKEN_LIMIT) {
    throw new PersonaError(
      "invalid",
      `persona "${persona.id}" greeting is too long: ~${greetingTokens} tokens estimated, ` +
        `limit ${APPEND_TOKEN_LIMIT}`,
    );
  }
  return {
    id: persona.id,
    name: persona.name,
    instructions,
    greeting,
    voice: resolveVoice(persona),
    instructionTokens,
  };
}

function resolveVoice(persona: Persona): string {
  const voice = persona.voice?.trim();
  if (!voice) return config.gptlive.voice;
  if (voice.startsWith("voice_") || BUILTIN_VOICES.has(voice.toLowerCase())) {
    return voice.startsWith("voice_") ? voice : voice.toLowerCase();
  }
  console.warn(
    `[personas] "${persona.id}": unknown voice "${voice}" — using default "${config.gptlive.voice}"`,
  );
  return config.gptlive.voice;
}

// ── file source ──────────────────────────────────────────────────────────────

/**
 * One markdown file per persona: `personas/<id>.md`. Optional front matter
 * (simple `key: value` lines) carries `name`, `voice` and `greeting`; the body
 * is the persona instructions. Files starting with `_` are ignored, so
 * templates can live alongside.
 */
export class FilePersonaSource implements PersonaSource {
  constructor(private readonly dir: string) {}

  async list(): Promise<PersonaSummary[]> {
    const personas = await this.loadAll();
    return personas.map(({ id, name }) => ({ id, name }));
  }

  async get(id: string): Promise<Persona | null> {
    // Looked up from the directory listing, never by joining `id` into a path,
    // so a crafted id cannot read outside the directory.
    return (await this.loadAll()).find((p) => p.id === id) ?? null;
  }

  private async loadAll(): Promise<Persona[]> {
    let files: string[];
    try {
      files = await readdir(this.dir);
    } catch {
      throw new PersonaError("unavailable", `persona directory not readable: ${this.dir}`);
    }
    const personas: Persona[] = [];
    for (const file of files.sort()) {
      if (extname(file) !== ".md" || file.startsWith("_")) continue;
      const id = basename(file, ".md");
      if (!/^[A-Za-z0-9_-]+$/.test(id)) continue;
      const raw = await readFile(join(this.dir, file), "utf8");
      personas.push(parsePersonaFile(id, raw));
    }
    return personas;
  }
}

export function parsePersonaFile(id: string, raw: string): Persona {
  const text = raw.replace(/^﻿/, "").replace(/\r\n/g, "\n");
  const meta: Record<string, string> = {};
  let body = text;
  const fm = /^---\n([\s\S]*?)\n---\n?/.exec(text);
  if (fm) {
    body = text.slice(fm[0].length);
    for (const line of (fm[1] ?? "").split("\n")) {
      const m = /^\s*([A-Za-z_]+)\s*:\s*(.*?)\s*$/.exec(line);
      if (m?.[1]) meta[m[1].toLowerCase()] = (m[2] ?? "").replace(/^(["'])(.*)\1$/, "$2");
    }
  }
  return {
    id,
    name: meta.name || id,
    instructions: body.trim(),
    greeting: meta.greeting ?? "",
    voice: meta.voice || undefined,
  };
}

// ── Open WebUI source ────────────────────────────────────────────────────────

// Personas change when a professor edits a model, not per request; a short
// cache keeps a class starting at once from hammering Open WebUI.
const OPENWEBUI_CACHE_MS = 30_000;

export class OpenWebUIPersonaSource implements PersonaSource {
  private cache: { at: number; personas: Persona[] } | null = null;
  private readonly warned = new Set<string>();

  constructor(private readonly tag: string) {}

  async list(): Promise<PersonaSummary[]> {
    return (await this.loadAll()).map(({ id, name }) => ({ id, name }));
  }

  async get(id: string): Promise<Persona | null> {
    return (await this.loadAll()).find((p) => p.id === id) ?? null;
  }

  private async loadAll(): Promise<Persona[]> {
    if (this.cache && Date.now() - this.cache.at < OPENWEBUI_CACHE_MS) {
      return this.cache.personas;
    }
    let models: OpenWebUIModel[];
    try {
      models = await listAllModels();
    } catch (err) {
      // Keep serving the last good list through a blip rather than failing
      // every start while Open WebUI restarts.
      if (this.cache) {
        console.warn(`[personas] Open WebUI unreachable, using cached list: ${err}`);
        return this.cache.personas;
      }
      throw new PersonaError(
        "unavailable",
        `could not load personas from Open WebUI: ${err instanceof Error ? err.message : err}`,
      );
    }
    const wanted = this.tag.toLowerCase();
    const personas = models
      .filter((m) => m.is_active !== false)
      .filter((m) => modelTags(m).some((t) => t.toLowerCase() === wanted))
      .map((m) => this.toPersona(m))
      .filter((p): p is Persona => p !== null);
    this.cache = { at: Date.now(), personas };
    return personas;
  }

  private toPersona(model: OpenWebUIModel): Persona | null {
    const system = typeof model.params?.system === "string" ? model.params.system.trim() : "";
    if (!system) {
      this.warnOnce(model.id, "has no system prompt — skipped");
      return null;
    }
    if (Array.isArray(model.meta?.knowledge) && model.meta.knowledge.length > 0) {
      this.warnOnce(
        model.id,
        `has ${model.meta.knowledge.length} knowledge item(s) attached — the avatar loads the ` +
          "system prompt only, so facts that live only in those documents are unknown to it",
      );
    }
    // Optional per-persona voice: a tag such as "voice:cedar".
    const voiceTag = modelTags(model).find((t) => /^voice:/i.test(t));
    return {
      id: model.id,
      name: typeof model.name === "string" && model.name.trim() ? model.name.trim() : model.id,
      instructions: substituteVariables(system, model.id, (msg) => this.warnOnce(model.id, msg)),
      greeting: "", // assemble() falls back to the persona-neutral DEFAULT_GREETING
      voice: voiceTag ? voiceTag.slice("voice:".length).trim() : undefined,
    };
  }

  private warnOnce(modelId: string, message: string): void {
    const key = `${modelId}:${message}`;
    if (this.warned.has(key)) return;
    this.warned.add(key);
    console.warn(`[personas] Open WebUI model "${modelId}" ${message}`);
  }
}

/**
 * Open WebUI fills `{{…}}` variables into system prompts at chat time; the
 * avatar never goes through that path, so the common ones are filled here.
 * Unknown ones are removed (and warned about) rather than read aloud.
 */
export function substituteVariables(
  text: string,
  modelId: string,
  warn: (msg: string) => void = () => {},
  now: Date = new Date(),
): string {
  const values: Record<string, string> = {
    CURRENT_DATE: now.toISOString().slice(0, 10),
    CURRENT_TIME: now.toTimeString().slice(0, 5),
    CURRENT_DATETIME: `${now.toISOString().slice(0, 10)} ${now.toTimeString().slice(0, 5)}`,
    CURRENT_WEEKDAY: now.toLocaleDateString("en-US", { weekday: "long" }),
    // AV3 knows the student; until then the persona addresses "the advisor".
    USER_NAME: "the advisor",
  };
  return text.replace(/\{\{\s*([A-Z_]+)\s*\}\}/g, (_match, name: string) => {
    if (name in values) return values[name] ?? "";
    warn(`uses unsupported variable {{${name}}} — removed for the avatar (${modelId})`);
    return "";
  });
}

// ── selection ────────────────────────────────────────────────────────────────

let source: PersonaSource | null = null;

export function personaSource(): PersonaSource {
  source ??=
    config.personas.source === "openwebui"
      ? new OpenWebUIPersonaSource(config.personas.tag)
      : new FilePersonaSource(config.personas.dir);
  return source;
}
