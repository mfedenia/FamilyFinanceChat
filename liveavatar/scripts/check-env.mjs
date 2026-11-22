#!/usr/bin/env node
/**
 * Preflight for `pnpm dev` / `pnpm start`: refuse to boot with required env
 * missing, naming exactly what's absent and how to fix it — instead of a
 * server that starts fine and then 500s on the first session.
 *
 * Only the duo's baseline is required here; the per-page / per-scenario
 * requirements live in `missingDuoConfig()` and `missingConfig()` in
 * server/src/config.ts and are only warned about — keep the warnings in step.
 */
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Same files, same order as server/src/config.ts: the real environment, then
// FamilyFinanceChat's main .env (when inside FFC), then liveavatar/.env.
const ffcEnv = fileURLToPath(new URL("../../.env", import.meta.url));
const ffcMarker = fileURLToPath(new URL("../../docker-compose.yml", import.meta.url));
for (const file of [existsSync(ffcMarker) ? ffcEnv : null, fileURLToPath(new URL("../.env", import.meta.url))]) {
  if (file && existsSync(file)) {
    try {
      process.loadEnvFile(file);
    } catch {
      /* unreadable */
    }
  }
  // An unfilled placeholder must not block a real value from the next file.
  for (const [k, v] of Object.entries(process.env)) if (v?.trim() === "REPLACE_ME") delete process.env[k];
}

const has = (name) => {
  const v = (process.env[name] ?? "").trim();
  return Boolean(v) && v !== "REPLACE_ME";
};

// Hard requirement: what every duo call needs (LiveAvatar + the Gemini
// listener and router). Everything else depends on which page, scenario and
// engine are used, so it is only warned about here — each start endpoint
// (/api/duo/start, /api/session/start) names exactly what it is missing.
// Gemini: the collision-safe name (Open WebUI also reads GEMINI_API_KEY from
// FFC's .env), with the plain name as a fallback.
const missing = [];
if (!has("LIVEAVATAR_API_KEY")) missing.push("LIVEAVATAR_API_KEY");
if (!has("LIVEAVATAR_GEMINI_API_KEY") && !has("GEMINI_API_KEY")) missing.push("LIVEAVATAR_GEMINI_API_KEY");

if (missing.length) {
  console.error(
    `\n\x1b[31mMissing required env:\x1b[0m ${missing.join(", ")}\n\n` +
      "Append liveavatar/.env.example to FamilyFinanceChat's .env (or copy it to liveavatar/.env when\n" +
      "running standalone), fill it in, and keep the file chmod 600.\n",
  );
  process.exit(1);
}

const warnings = [];
const scenario = (process.env.DUO_SCENARIO ?? "beckham").trim();
if (scenario === "beckham" && (!has("OPENWEBUI_BASE_URL") || !has("OPENWEBUI_API_KEY"))) {
  warnings.push("duo scenario 'beckham' reads Open WebUI presets: set OPENWEBUI_BASE_URL and OPENWEBUI_API_KEY (or DUO_SCENARIO=beckham-local)");
}
if (scenario === "beckham" && (!has("DUO_OPENWEBUI_MODEL_TONY") || !has("DUO_OPENWEBUI_MODEL_ANTONELLA"))) {
  warnings.push("duo scenario 'beckham' needs the presets' ids: DUO_OPENWEBUI_MODEL_TONY and DUO_OPENWEBUI_MODEL_ANTONELLA");
}
if ((process.env.DUO_PERSONA_ENGINE ?? "").trim().toLowerCase() === "gptlive" && !has("OPENAI_API_KEY")) {
  warnings.push("DUO_PERSONA_ENGINE=gptlive needs OPENAI_API_KEY");
}
const singleGemini = (process.env.LIVE_PROVIDER ?? "").trim().toLowerCase() === "gemini";
if (singleGemini ? !(has("LIVEAVATAR_GEMINI_API_KEY") || has("GEMINI_API_KEY")) : !has("OPENAI_API_KEY")) {
  warnings.push(`the single-avatar page (/) needs ${singleGemini ? "LIVEAVATAR_GEMINI_API_KEY" : "OPENAI_API_KEY"}`);
}
for (const w of warnings) console.warn(`\x1b[33mnote:\x1b[0m ${w}`);
