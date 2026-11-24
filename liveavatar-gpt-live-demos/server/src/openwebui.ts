/**
 * Optional: Tony's and Antonella's personas from FamilyFinanceChat's Open WebUI.
 *
 * Adapted from Taehyun Park's duo (feat/liveavatar-duo, liveavatar/server/src/
 * duo/openwebui.ts). A professor creates one client model preset per spouse in
 * Open WebUI (Workspace → Models: system prompt + knowledge); this reads each
 * preset's system prompt through the public REST API and uses it as that
 * spouse's persona. Compatibility rule: public REST API only, nothing in Open
 * WebUI is changed.
 *
 * From Park's notes, verified against Open WebUI 0.8.12 and 0.11.4 source:
 * `GET /api/v1/models/model?id=` returns `params.system`; on 0.11.x a
 * read-only user's key gets `params: {}`, so use the preset owner's or an
 * admin's key.
 *
 * Presets are read BEFORE any LiveAvatar session starts, so a wrong id or key
 * costs nothing. (Park's per-turn knowledge search is not used here: GPT-Live
 * starts answering as soon as it hears the question, before a search could
 * return.)
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { Speaker } from "../../shared/messages";
import { config } from "./config";

const PRESET_TIMEOUT_MS = 8_000;

export interface Personas {
  text: Record<Speaker, string>;
  /** "local files" or "Open WebUI presets <tony-id>, <antonella-id>". */
  source: string;
}

export function openWebUIConfigured(): boolean {
  const o = config.openwebui;
  return Boolean(o.baseUrl && o.apiKey && o.tonyModel && o.antonellaModel);
}

/** One preset's system prompt. Throws with a message a person can act on. */
export async function fetchPresetSystem(modelId: string): Promise<string> {
  const { baseUrl, apiKey } = config.openwebui;
  const response = await fetch(`${baseUrl}/api/v1/models/model?id=${encodeURIComponent(modelId)}`, {
    headers: { Authorization: `Bearer ${apiKey}` },
    signal: AbortSignal.timeout(PRESET_TIMEOUT_MS),
  });
  const body = await response.text();
  if (!response.ok) throw new Error(`Open WebUI preset "${modelId}" → HTTP ${response.status}: ${body.slice(0, 200)}`);
  // An unknown id comes back as `null`, not as an error status.
  const m = ((body ? JSON.parse(body) : null) ?? {}) as { id?: string; params?: { system?: unknown } };
  if (!m.id) throw new Error(`Open WebUI has no model preset "${modelId}"`);
  const system = typeof m.params?.system === "string" ? m.params.system.trim() : "";
  if (!system) {
    throw new Error(
      `Open WebUI preset "${modelId}" returned no system prompt: it has none, or OPENWEBUI_API_KEY ` +
        "cannot read it (Open WebUI 0.11.x hides it from read-only users; use the owner's or an admin's key).",
    );
  }
  return system;
}

function localPersona(who: Speaker): string {
  return readFileSync(fileURLToPath(new URL(`../prompts/couple/${who}.md`, import.meta.url)), "utf8").trim();
}

/** Both personas: from Open WebUI when configured, else from the local files. */
export async function loadPersonas(): Promise<Personas> {
  if (!openWebUIConfigured()) {
    return { text: { tony: localPersona("tony"), antonella: localPersona("antonella") }, source: "local files" };
  }
  const { tonyModel, antonellaModel } = config.openwebui;
  const [tony, antonella] = await Promise.all([fetchPresetSystem(tonyModel), fetchPresetSystem(antonellaModel)]);
  return { text: { tony, antonella }, source: `Open WebUI presets ${tonyModel}, ${antonellaModel}` };
}
