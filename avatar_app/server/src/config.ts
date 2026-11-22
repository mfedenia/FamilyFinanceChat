/**
 * Env-driven configuration. One `.env` at `avatar_app/` (or the compose
 * stack's environment) feeds everything — the browser never sees any of it.
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

/** Positive number, or the fallback when unset/invalid. Fractions allowed. */
const num = (name: string, fallback: number): number => {
  const value = Number(env(name, ""));
  return env(name, "") !== "" && Number.isFinite(value) && value > 0 ? value : fallback;
};

/** "false", "0", "no", "off" → false; anything else (incl. unset) → fallback/true. */
const bool = (name: string, fallback: boolean): boolean => {
  const raw = env(name, "").trim().toLowerCase();
  if (!raw) return fallback;
  return !["false", "0", "no", "off"].includes(raw);
};

export type PersonaSourceKind = "file" | "openwebui";

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
    model: env("GPT_LIVE_MODEL", "gpt-live-1"),
    /** Default voice; a persona may override it (personas.ts). */
    voice: env("GPT_LIVE_VOICE", "marin"),
    responsesModel: env("GPT_LIVE_RESPONSES_MODEL", "gpt-5.4-nano"),
    /**
     * Set GPT_LIVE_DEBUG=1 to log every upstream event (audio payloads
     * elided). Loud — for answering "what did OpenAI actually send/reject",
     * not for leaving on.
     */
    debug: Boolean(env("GPT_LIVE_DEBUG")),
  },

  // ── AV5a guardrails ─────────────────────────────────────────────────────────
  limits: {
    /** Kill switch. AVATAR_ENABLED=false refuses every new session. */
    enabled: bool("AVATAR_ENABLED", true),
    /** Hard cap per session, in minutes. Fractions allowed (useful in tests). */
    maxSessionMin: num("AVATAR_MAX_SESSION_MIN", 15),
  },

  /**
   * Prometheus metrics, served on their OWN port so the public /practice/
   * route (Caddy → PORT) can never expose them. 0 disables.
   */
  metricsPort: Number(env("AVATAR_METRICS_PORT", "9464")),

  // ── AV2 persona loader ──────────────────────────────────────────────────────
  personas: {
    /** "file" (avatar_app/personas/*.md) or "openwebui" (tagged workspace models). */
    source: (env("AVATAR_PERSONA_SOURCE", "file") === "openwebui"
      ? "openwebui"
      : "file") as PersonaSourceKind,
    /** Directory for the file source. Default: avatar_app/personas. */
    dir: env(
      "AVATAR_PERSONA_DIR",
      fileURLToPath(new URL("../../personas", import.meta.url)),
    ),
    /** Open WebUI workspace models carrying this tag are offered as personas. */
    tag: env("AVATAR_PERSONA_TAG", "avatar"),
  },

  // Same variable names the grading service uses (grading_feature/backend).
  openwebui: {
    baseUrl: env("OPENWEBUI_BASE_URL").replace(/\/+$/, ""),
    /**
     * An ADMIN user's API key. Open WebUI strips a model's system prompt from
     * every caller without write access to it — students included — so only
     * an admin key can read personas. Server-side only; never sent anywhere
     * but Open WebUI.
     */
    apiKey: env("OPENWEBUI_API_KEY"),
  },
};

/**
 * Names of required vars that are missing. Checked at boot (a warning) and at
 * `/api/session/start` (refused, with the names logged server-side) — not a
 * crash, so `pnpm dev` works before the `.env` is filled in.
 */
export function missingConfig(): string[] {
  const required: [string, string][] = [
    ["LIVEAVATAR_API_KEY", config.liveavatar.apiKey],
    ["OPENAI_API_KEY", config.gptlive.apiKey],
  ];
  if (config.personas.source === "openwebui") {
    required.push(
      ["OPENWEBUI_BASE_URL", config.openwebui.baseUrl],
      ["OPENWEBUI_API_KEY", config.openwebui.apiKey],
    );
  }
  return required.filter(([, value]) => !value).map(([name]) => name);
}
