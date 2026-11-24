/**
 * Env-driven configuration. One `.env` at the repo root feeds everything —
 * the browser never sees any of it.
 *
 * GPT-Live is generally available; the model defaults to `gpt-live-1.1`.
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

export const config = {
  port: Number(env("PORT", "8787")),

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
    model: env("GPT_LIVE_MODEL", "gpt-live-1.1"),
    voice: env("GPT_LIVE_VOICE", "marin"),
    responsesModel: env("GPT_LIVE_RESPONSES_MODEL", "gpt-5.4-nano"),
    /**
     * Set GPT_LIVE_DEBUG=1 to log every upstream event (audio payloads
     * elided). Loud — for answering "what did OpenAI actually send/reject",
     * not for leaving on.
     */
    debug: Boolean(env("GPT_LIVE_DEBUG")),
  },

  /** The couple meeting (couple.ts): two spouses, one floor-deciding model. */
  couple: {
    /** Picks which spouse answers each adviser turn. Speed matters more than depth. */
    /**
     * OpenAI Decisions API (POST /v1/decisions); gpt-6-luna is the only model
     * it supports. The key needs the api.decisions.write permission.
     */
    decisionModel: env("DECISION_MODEL", "gpt-6-luna"),
    /** Overridable only so tests can point at a local stand-in. */
    decisionsUrl: env("DECISIONS_URL", "https://api.openai.com/v1/decisions"),
    tony: {
      avatarId: env("TONY_AVATAR_ID", "a4991f42-d211-4581-9631-639f7fe7c635"), // William (Friendly)
      voice: env("TONY_VOICE", "cedar"),
    },
    antonella: {
      avatarId: env("ANTONELLA_AVATAR_ID", "0cbed480-e2e3-453e-a85d-3cd69b97e2b2"), // Helen (Friendly)
      voice: env("ANTONELLA_VOICE", "marin"),
    },
  },

  /**
   * Couple meeting: where each meeting's transcript is saved when it ends
   * (transcript.ts). Text only — no audio is ever written.
   */
  transcripts: {
    dir: env("TRANSCRIPTS_DIR", fileURLToPath(new URL("../../transcripts", import.meta.url))),
    // Locally there is no login, so the student is named by env. A deployment
    // would take these from the platform's authentication instead.
    studentId: env("STUDENT_ID", "local-test-student"),
    studentName: env("STUDENT_NAME", "Local Test Student"),
    studentEmail: env("STUDENT_EMAIL", "local-test@example.invalid"),
  },

  /**
   * Optional: read Tony's and Antonella's persona from FamilyFinanceChat's
   * Open WebUI client model presets (openwebui.ts), so a professor edits the
   * persona in Open WebUI. All four set → presets; otherwise the local files
   * in server/prompts/couple/ are used.
   */
  openwebui: {
    baseUrl: env("OPENWEBUI_BASE_URL").replace(/\/+$/, ""),
    apiKey: env("OPENWEBUI_API_KEY"),
    tonyModel: env("COUPLE_OPENWEBUI_MODEL_TONY"),
    antonellaModel: env("COUPLE_OPENWEBUI_MODEL_ANTONELLA"),
  },
};

/**
 * Names of required vars that are missing. Checked at boot (a warning) and at
 * `/api/session/start` (a 500 naming them) — not a crash, so `pnpm dev` works
 * before the `.env` is filled in and the failure explains itself.
 */
export function missingConfig(): string[] {
  const required: [string, string][] = [
    ["LIVEAVATAR_API_KEY", config.liveavatar.apiKey],
    ["OPENAI_API_KEY", config.gptlive.apiKey],
  ];
  return required.filter(([, value]) => !value).map(([name]) => name);
}
