/**
 * FamilyFinanceChat's Open WebUI, through its public REST API only (the
 * compatibility rule: never touch Open WebUI's core). Two reads:
 *
 *  - a client **model preset** — what a professor creates under Workspace →
 *    Models: system prompt + attached knowledge. That IS a persona here.
 *  - a **knowledge search** — the same Qdrant search a typed chat gets.
 *
 * Verified against Open WebUI source at v0.8.12 (production) and v0.11.4
 * (pinned in FFC's docker-compose.yml):
 *  - `GET /api/v1/models/model?id=` returns `params.system`, `meta.knowledge`,
 *    `meta.description`, `name`. v0.11.4 returns `params: {}` to callers with
 *    read-only access, so the key must be the preset owner's or an admin's.
 *  - `POST /api/v1/retrieval/query/collection` `{collection_names, query, k?}`
 *    returns `{documents: [[…]], metadatas: [[…]], distances: [[…]]}`. Leaving
 *    `k` out uses the server's configured top-k (FFC: RAG_TOP_K=5).
 *  - Knowledge items map to collections as in retrieval/utils.py
 *    get_sources_from_items: a knowledge base (`type: "collection"`) searches
 *    collection `id`; a single file (`type: "file"`) searches `file-<id>`;
 *    legacy items carry `collection_name` / `collection_names`.
 *
 * Production (0.8.12) searches a preset's attached knowledge automatically on
 * every turn; 0.11.4 hands the model a search tool instead unless the preset
 * uses `function_calling: legacy`. The duo mirrors production: every turn.
 */

import { config } from "../config";

const PRESET_TIMEOUT_MS = 8_000;
/**
 * A spoken turn cannot wait on a slow search: past this the reply goes out
 * without documents. Runs in parallel with the ~0.3s decision call, so it
 * normally adds nothing.
 */
const SEARCH_TIMEOUT_MS = 1_500;

type Json = Record<string, any>;

export interface Preset {
  id: string;
  name: string;
  description: string;
  /** The preset's system prompt — the persona. */
  system: string;
  /** Collections its attached knowledge searches. */
  collections: string[];
}

export interface Excerpt {
  text: string;
  /** File name, when Open WebUI recorded one. */
  source: string;
}

async function call(path: string, init: RequestInit, timeoutMs: number): Promise<Json> {
  if (!config.openwebui.baseUrl || !config.openwebui.apiKey) {
    throw new Error("OPENWEBUI_BASE_URL and OPENWEBUI_API_KEY must be set");
  }
  const response = await fetch(`${config.openwebui.baseUrl}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${config.openwebui.apiKey}`,
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
    signal: AbortSignal.timeout(timeoutMs),
  });
  const text = await response.text();
  if (!response.ok) throw new Error(`Open WebUI ${path} → ${response.status}: ${text.slice(0, 200)}`);
  return text ? (JSON.parse(text) as Json) : {};
}

/** One client model preset, as a persona needs it. */
export async function fetchPreset(modelId: string): Promise<Preset> {
  const m = await call(`/api/v1/models/model?id=${encodeURIComponent(modelId)}`, { method: "GET" }, PRESET_TIMEOUT_MS);
  if (!m || !m.id) throw new Error(`Open WebUI has no model preset "${modelId}"`);
  const system = typeof m.params?.system === "string" ? m.params.system.trim() : "";
  if (!system) {
    throw new Error(
      `Open WebUI preset "${modelId}" returned no system prompt. Either it has none, or ` +
        "OPENWEBUI_API_KEY cannot read it — Open WebUI 0.11.x hides a preset's system prompt " +
        "from read-only users; use the preset owner's or an admin's key.",
    );
  }
  return {
    id: String(m.id),
    name: String(m.name ?? m.id),
    description: String(m.meta?.description ?? ""),
    system,
    collections: knowledgeCollections(Array.isArray(m.meta?.knowledge) ? m.meta.knowledge : []),
  };
}

/** Collection names an attached-knowledge list searches (retrieval/utils.py). */
export function knowledgeCollections(items: Json[]): string[] {
  const out: string[] = [];
  for (const item of items) {
    if (!item || typeof item !== "object") continue;
    if (typeof item.collection_name === "string") out.push(item.collection_name);
    else if (Array.isArray(item.collection_names)) out.push(...item.collection_names.map(String));
    else if (item.type === "file" && item.id) out.push(item.legacy ? String(item.id) : `file-${item.id}`);
    else if (item.id) out.push(String(item.id)); // a knowledge base ("collection"), the usual case
  }
  return [...new Set(out)];
}

/**
 * Search the given collections with the human's words, as a typed FFC chat
 * would. Never throws: on error or timeout it logs and returns nothing — a
 * reply without documents beats a call that stalls.
 */
export async function searchKnowledge(
  collections: string[],
  query: string,
  log: (msg: string) => void,
): Promise<Excerpt[]> {
  if (collections.length === 0 || !query.trim()) return [];
  try {
    const r = await call(
      "/api/v1/retrieval/query/collection",
      { method: "POST", body: JSON.stringify({ collection_names: collections, query }) },
      SEARCH_TIMEOUT_MS,
    );
    const docs: unknown[] = Array.isArray(r.documents?.[0]) ? r.documents[0] : [];
    const metas: Json[] = Array.isArray(r.metadatas?.[0]) ? r.metadatas[0] : [];
    return docs
      .map((d, i) => ({
        text: typeof d === "string" ? d.trim() : "",
        source: String(metas[i]?.name ?? metas[i]?.source ?? ""),
      }))
      .filter((e) => e.text);
  } catch (err) {
    log(`knowledge search failed (${err instanceof Error ? err.message : err}) — replying without documents`);
    return [];
  }
}

/**
 * How retrieved excerpts reach a speaking persona. Not Open WebUI's RAG
 * template: that one asks for `[1]`-style citations, which a client on a call
 * must never read aloud.
 */
export function excerptsBlock(excerpts: Excerpt[], maxChars = 4_000): string {
  if (excerpts.length === 0) return "";
  const parts: string[] = [];
  let used = 0;
  for (const e of excerpts) {
    const piece = e.text.length > 1_500 ? `${e.text.slice(0, 1_500)}…` : e.text;
    if (used + piece.length > maxChars) break;
    parts.push(piece);
    used += piece.length;
  }
  return (
    "[documents] Excerpts from your file that may be relevant to what was just asked. This is " +
    "background you know about your own life — answer as yourself, never read it out, never " +
    "mention documents or sources, and ignore anything not relevant:\n" +
    parts.map((p) => `---\n${p}`).join("\n")
  );
}
