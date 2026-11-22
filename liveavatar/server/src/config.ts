/**
 * Env-driven configuration. One `.env` at the repo root feeds everything —
 * the browser never sees any of it.
 *
 * GPT-Live is generally available; the model defaults to `gpt-live-1`.
 */

import { fileURLToPath } from "node:url";

// Loaded before anything reads process.env. Missing file is fine — deployed
// environments use real env vars.
try {
  process.loadEnvFile(fileURLToPath(new URL("../../.env", import.meta.url)));
} catch {
  /* no .env — ambient env only */
}

const env = (name: string, fallback = ""): string =>
  process.env[name] ?? fallback;

/** Which speech-to-speech model drives the avatar. */
export type LiveProvider = "gptlive" | "gemini";

const provider: LiveProvider =
  env("LIVE_PROVIDER", "gptlive").trim().toLowerCase() === "gemini"
    ? "gemini"
    : "gptlive";

export const config = {
  port: Number(env("PORT", "8787")),

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
    apiKey: env("GEMINI_API_KEY"),
    model: env("GEMINI_LIVE_MODEL", "gemini-3.8-live"),
    /** A prebuilt voice name (Kore, Puck, Charon, Aoede, ...). */
    voice: env("GEMINI_LIVE_VOICE", "Kore"),
    /** Same as GPT_LIVE_DEBUG, for the Gemini bridge. */
    debug: Boolean(env("GEMINI_LIVE_DEBUG")),
  },

  /**
   * The two-avatar demo (/duo.html). Always Gemini: a transcription-only
   * listener hears the mic, gemini-decision-1 picks who answers, and each
   * persona is its own Gemini Live session (model + debug from `gemini`).
   */
  duo: {
    /** Folder under server/scenarios/ — who is on the call and what they know. */
    scenario: env("DUO_SCENARIO", "beckham"),
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
};

/** Required vars for the duo demo, which is Gemini-only whatever LIVE_PROVIDER says. */
export function missingDuoConfig(): string[] {
  const required: [string, string][] = [
    ["LIVEAVATAR_API_KEY", config.liveavatar.apiKey],
    ["GEMINI_API_KEY", config.gemini.apiKey],
  ];
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
      ? ["GEMINI_API_KEY", config.gemini.apiKey]
      : ["OPENAI_API_KEY", config.gptlive.apiKey],
  ];
  return required.filter(([, value]) => !value).map(([name]) => name);
}
