/**
 * Env-driven configuration. The browser never sees any of it.
 *
 * Inside FamilyFinanceChat, liveavatar shares FFC's main `.env` (the repo
 * root, one level up): its variables are appended there, and FFC's own
 * `OPENAI_API_KEY` is reused. Load order — the first value found wins:
 *   1. the real environment,
 *   2. FamilyFinanceChat/.env (when liveavatar sits inside FFC),
 *   3. liveavatar/.env (standalone use, or to fill gaps).
 *
 * FFC's main .env is ALSO the open-webui container's env_file, and Open WebUI
 * itself reads `GEMINI_API_KEY` (config.py) and `PORT` (its start script). So
 * liveavatar reads collision-safe names first — `LIVEAVATAR_GEMINI_API_KEY`,
 * `LIVEAVATAR_PORT` — and the plain names only as a fallback. Put the prefixed
 * names in the main .env, or Open WebUI picks the values up too.
 */

import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("../../", import.meta.url)); // liveavatar/
const FFC_ENV = fileURLToPath(new URL("../../../.env", import.meta.url));
const FFC_MARKER = fileURLToPath(new URL("../../../docker-compose.yml", import.meta.url));

/** Which env files were loaded, for the boot log. */
export const loadedEnvFiles: string[] = [];
// loadEnvFile never overrides a variable that is already set, so loading the
// main .env first makes it win over liveavatar/.env.
for (const file of [existsSync(FFC_MARKER) ? FFC_ENV : null, `${ROOT}.env`]) {
  if (!file || !existsSync(file)) continue;
  try {
    process.loadEnvFile(file);
    loadedEnvFiles.push(file);
  } catch {
    /* unreadable — ambient env only */
  }
  // An unfilled placeholder must not block a real value from the next file.
  for (const [k, v] of Object.entries(process.env)) if (v?.trim() === "REPLACE_ME") delete process.env[k];
}

// A value still reading REPLACE_ME (the .env.example placeholder) counts as unset.
const env = (name: string, fallback = ""): string => {
  const v = process.env[name];
  return v === undefined || v.trim() === "REPLACE_ME" ? fallback : v;
};

/** Which speech-to-speech model drives the avatar. */
export type LiveProvider = "gptlive" | "gemini";

const provider: LiveProvider =
  env("LIVE_PROVIDER", "gptlive").trim().toLowerCase() === "gemini"
    ? "gemini"
    : "gptlive";

export const config = {
  /** LIVEAVATAR_PORT first: Open WebUI's container also reads PORT from FFC's .env. */
  port: Number(env("LIVEAVATAR_PORT") || env("PORT", "8787")),

  /** LIVE_PROVIDER=gemini swaps GPT-Live for the Gemini Live API (geminiLive.ts). */
  provider,

  liveavatar: {
    apiUrl: env("LIVEAVATAR_API_URL", "https://api.liveavatar.com").replace(
      /\/+$/,
      "",
    ),
    apiKey: env("LIVEAVATAR_API_KEY"),
    /** Optional — empty means "pick the first public avatar" (liveavatar.ts). */
    avatarId: env("LIVEAVATAR_AVATAR_ID"),
  },

  gptlive: {
    apiKey: env("OPENAI_API_KEY"),
    model: env("GPT_LIVE_MODEL", "gpt-live-1"),
    voice: env("GPT_LIVE_VOICE", "marin"),
    responsesModel: env("GPT_LIVE_RESPONSES_MODEL", "gpt-5.4-nano"),
    /**
     * Set GPT_LIVE_DEBUG=1 to log every upstream event (audio payloads
     * elided). Loud — for answering "what did OpenAI actually send/reject",
     * not for leaving on.
     */
    debug: Boolean(env("GPT_LIVE_DEBUG")),
  },

  gemini: {
    /** LIVEAVATAR_GEMINI_API_KEY first: Open WebUI also reads GEMINI_API_KEY from FFC's .env. */
    apiKey: env("LIVEAVATAR_GEMINI_API_KEY") || env("GEMINI_API_KEY"),
    model: env("GEMINI_LIVE_MODEL", "gemini-3.8-live"),
    /** A prebuilt voice name (Kore, Puck, Charon, Aoede, ...). */
    voice: env("GEMINI_LIVE_VOICE", "Kore"),
    /** Same as GPT_LIVE_DEBUG, for the Gemini bridge. */
    debug: Boolean(env("GEMINI_LIVE_DEBUG")),
  },

  /**
   * The two-avatar demo (/duo.html). A transcription-only Gemini listener
   * hears the mic, gemini-decision-1 picks who answers, and each persona is a
   * live voice model: Gemini Live (default) or OpenAI GPT-Live. Personas,
   * prompts and documents come from Open WebUI model presets, as in
   * FamilyFinanceChat (scenario.ts, openwebui.ts).
   */
  duo: {
    /** Folder under server/scenarios/ — who is on the call. */
    scenario: env("DUO_SCENARIO", "beckham"),
    /** The personas' live voice model: "gemini" (Gemini Live) or "gptlive" (OpenAI GPT-Live). */
    personaEngine: (env("DUO_PERSONA_ENGINE", "gemini").trim().toLowerCase() === "gptlive"
      ? "gptlive"
      : "gemini") as "gemini" | "gptlive",
    /** GPT-Live model for the personas when DUO_PERSONA_ENGINE=gptlive. */
    gptLiveModel: env("DUO_GPT_LIVE_MODEL", "gpt-live-1.1"),
    listenerModel: env("DUO_LISTENER_MODEL", "gemini-3.5-transcribe-live"),
    decisionModel: env("DUO_DECISION_MODEL", "gemini-decision-1"),
    /**
     * Listener end-of-turn silence. Measured: 600 → turn closes ~1.0s after
     * speech; the default (unset) ~1.3s; below 500 the turn never closed.
     */
    silenceMs: Number(env("DUO_SILENCE_MS", "600")),
    /** Optional "id,id" — overrides the scenario's avatars (or its random picks). */
    avatarIds: env("DUO_AVATAR_IDS")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean),
  },

  /**
   * FamilyFinanceChat's Open WebUI — same variable names as tools/kb_sync and
   * grading_feature. The key must be the client presets' owner's or an
   * admin's: Open WebUI 0.11.x hides a preset's system prompt from read-only
   * users. Create it under Settings → Account → API keys (FFC SETUP.md §7).
   */
  openwebui: {
    baseUrl: env("OPENWEBUI_BASE_URL").replace(/\/+$/, ""),
    apiKey: env("OPENWEBUI_API_KEY"),
  },
};

/**
 * Required vars for the duo, given what the chosen scenario and engine use.
 * `usesOpenWebUI` comes from the scenario (any persona backed by a preset).
 */
export function missingDuoConfig(usesOpenWebUI: boolean): string[] {
  const required: [string, string][] = [
    ["LIVEAVATAR_API_KEY", config.liveavatar.apiKey],
    // The listener and the router are Gemini whatever the persona engine is.
    ["LIVEAVATAR_GEMINI_API_KEY", config.gemini.apiKey],
  ];
  if (config.duo.personaEngine === "gptlive") required.push(["OPENAI_API_KEY", config.gptlive.apiKey]);
  if (usesOpenWebUI) {
    required.push(["OPENWEBUI_BASE_URL", config.openwebui.baseUrl]);
    required.push(["OPENWEBUI_API_KEY", config.openwebui.apiKey]);
  }
  return required.filter(([, value]) => !value).map(([name]) => name);
}

/**
 * Names of required vars that are missing. Checked at boot (a warning) and at
 * `/api/session/start` (a 500 naming them) — not a crash, so `pnpm dev` works
 * before the `.env` is filled in and the failure explains itself.
 */
export function missingConfig(): string[] {
  const required: [string, string][] = [
    ["LIVEAVATAR_API_KEY", config.liveavatar.apiKey],
    config.provider === "gemini"
      ? ["LIVEAVATAR_GEMINI_API_KEY", config.gemini.apiKey]
      : ["OPENAI_API_KEY", config.gptlive.apiKey],
  ];
  return required.filter(([, value]) => !value).map(([name]) => name);
}
