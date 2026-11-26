/**
 * Every call this bridge makes to Open WebUI, in one module.
 *
 * Compatibility policy: Open WebUI is reached ONLY through its public REST API
 * (surface A7), and only from this file, so an upstream change breaks in one
 * place and AV7's contract test covers all of it. Never read Open WebUI's
 * database or call undocumented internals from anywhere else.
 *
 * Endpoints used, checked against Open WebUI v0.11.4 (2026-11-22):
 *
 *   GET /api/v1/models/all   admin only. Every workspace model record with
 *                            `params` (incl. `params.system`, the system
 *                            prompt) and `meta` (incl. `meta.tags`,
 *                            `meta.knowledge`). Non-admin endpoints strip
 *                            `params` from read-only callers by design.
 *
 * AV3 (login check) and AV4 (transcript write-back) add their calls here.
 */

import { config } from "./config";

const REQUEST_TIMEOUT_MS = 10_000;

export class OpenWebUIError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

/** The subset of Open WebUI's ModelModel this bridge reads. */
export interface OpenWebUIModel {
  id: string;
  name: string;
  is_active?: boolean;
  params?: { system?: unknown; [key: string]: unknown };
  meta?: {
    description?: unknown;
    tags?: unknown;
    knowledge?: unknown;
    [key: string]: unknown;
  };
}

async function get(path: string, bearer: string): Promise<unknown> {
  if (!config.openwebui.baseUrl) {
    throw new OpenWebUIError(0, "OPENWEBUI_BASE_URL is not set");
  }
  const response = await fetch(`${config.openwebui.baseUrl}${path}`, {
    headers: { Authorization: `Bearer ${bearer}`, Accept: "application/json" },
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (!response.ok) {
    const body = (await response.text().catch(() => "")).slice(0, 300);
    throw new OpenWebUIError(
      response.status,
      `Open WebUI GET ${path} returned ${response.status}: ${body}`,
    );
  }
  return response.json();
}

/** All workspace models, system prompts included. Requires an admin key. */
export async function listAllModels(): Promise<OpenWebUIModel[]> {
  const body = await get("/api/v1/models/all", config.openwebui.apiKey);
  if (!Array.isArray(body)) {
    throw new OpenWebUIError(0, "Open WebUI /api/v1/models/all did not return a list");
  }
  return body.filter(
    (m): m is OpenWebUIModel =>
      !!m && typeof m === "object" && typeof (m as OpenWebUIModel).id === "string",
  );
}

/** Tag names on a model, as Open WebUI normalises them: [{name}] → ["name"]. */
export function modelTags(model: OpenWebUIModel): string[] {
  const tags = model.meta?.tags;
  if (!Array.isArray(tags)) return [];
  return tags
    .map((t) => (t && typeof t === "object" ? (t as { name?: unknown }).name : t))
    .filter((n): n is string => typeof n === "string" && n.trim() !== "")
    .map((n) => n.trim());
}
