/**
 * Env-driven configuration. One `.env` at the repo root feeds everything —
 * the browser never sees any of it.
 *
 * GPT-Live is generally available; the model defaults to `gpt-live-1.1`.
 */

import { fileURLToPath } from "node:url";

// Loaded before anything reads process.env. Two files, both optional:
// this app's own .env, then the FamilyFinanceChat repo's main .env (the one
// Open WebUI's compose stack uses), so OPENAI_API_KEY, LIVEAVATAR_API_KEY etc.
// can live in one place. A variable already set — in the shell, or by the
// first file — is never overridden. Deployed environments use real env vars.
for (const file of ["../../.env", "../../../.env"]) {
  try {
    process.loadEnvFile(fileURLToPath(new URL(file, import.meta.url)));
  } catch {
    /* missing file — fine */
  }
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
    /** Overridable only so tests can point at a local stand-in. */
    responsesUrl: env("RESPONSES_URL", "https://api.openai.com/v1/responses"),
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
    /**
     * Budget for the presets' knowledge text in each spouse's startup
     * instructions (GPT-Live allows 16,384 tokens in all; ~4 chars a token).
     * Anything past it is still found by retrieval on a delegated turn.
     */
    knowledgeMaxChars: Number(env("OPENWEBUI_KNOWLEDGE_MAX_CHARS", "32000")),
    /** How long a delegated turn waits for Open WebUI's retrieval. */
    retrievalTimeoutMs: Number(env("OPENWEBUI_RETRIEVAL_TIMEOUT_MS", "4000")),
    /**
     * Local testing only: the student token to save meetings under when the
     * page was not opened from Open WebUI's "Talk to the avatar" button (which
     * hands the signed-in student's token to the page). A student's API key
     * works. Unset = meetings are saved locally only.
     */
    studentToken: env("OPENWEBUI_STUDENT_TOKEN"),
    /**
     * The family model preset the meeting is saved under in Open WebUI — the
     * one "Talk to the avatar" is pressed from (scripts/openwebui-setup.mjs
     * creates family-beckham). The button's ?family=<id> overrides it.
     */
    familyModel: env("COUPLE_OPENWEBUI_FAMILY_MODEL", "family-beckham"),
  },

  /**
   * GPT-Live's spoken acknowledgement on a delegated turn ("hmm, let me
   * think…"): how long it waits for our answer before saying one, 0–800 ms.
   * Unset = the API default (0: acknowledge at once).
   */
  ackDelayMs: env("GPT_LIVE_ACK_DELAY_MS") === "" ? null : Number(env("GPT_LIVE_ACK_DELAY_MS")),
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
