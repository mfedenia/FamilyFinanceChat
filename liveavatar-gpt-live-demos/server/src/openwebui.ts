/**
 * Everything the avatar takes from, or gives back to, FamilyFinanceChat's
 * Open WebUI. Public REST API only: nothing in Open WebUI is changed
 * (COMPATIBILITY_POLICY). Endpoints checked against Open WebUI v0.13.4 source.
 *
 *  1. Personas: each spouse's model preset (Workspace → Models). Its system
 *     prompt (`params.system`) is the spouse's persona.
 *  2. Knowledge (step 1): the knowledge bases and files attached to that
 *     preset (`meta.knowledge`, the same list the chat pipeline reads). Their
 *     text is loaded once, at meeting start, into the spouse's startup
 *     instructions, so most answers need no lookup at all.
 *  3. Retrieval (step 2): when GPT-Live delegates a turn ("hmm, let me
 *     think…"), the question goes to Open WebUI's own retrieval endpoint over
 *     the same collections. Open WebUI runs the search with ITS settings:
 *     embedding model, vector DB, top-k, relevance threshold, hybrid search.
 *  4. Chat write-back: when the meeting ends, the transcript is saved as an
 *     ordinary Open WebUI chat owned by the student, so it shows in their
 *     sidebar and in the grading dashboard's extraction.
 *
 * Keys: personas, knowledge and retrieval use OPENWEBUI_API_KEY (the preset
 * owner's or an admin's: Open WebUI strips `params` for read-only users). The
 * chat write-back uses the STUDENT's token, because Open WebUI files a new
 * chat under whoever calls it; an admin key would file every meeting under
 * the admin.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import type { Speaker } from "../../shared/messages";
import { config } from "./config";

const PRESET_TIMEOUT_MS = 8_000;
const KNOWLEDGE_TIMEOUT_MS = 15_000;
/** Open WebUI pages knowledge file lists; it caps non-admin pages at 30. */
const KNOWLEDGE_PAGE_LIMIT = 30;

/** One spouse's knowledge, as loaded at meeting start. */
export interface SpouseKnowledge {
  /** Full text of the attached files, for the startup instructions ("" = none). */
  text: string;
  /** Vector collections to search on a delegated turn (empty = no retrieval). */
  collections: string[];
  /** File names loaded, for logs and the transcript. */
  files: string[];
  /** True when the text was cut to fit the instructions budget. */
  truncated: boolean;
}

export interface Personas {
  text: Record<Speaker, string>;
  knowledge: Record<Speaker, SpouseKnowledge>;
  /** "local files" or "Open WebUI presets <tony-id>, <antonella-id>". */
  source: string;
}

export const NO_KNOWLEDGE: SpouseKnowledge = { text: "", collections: [], files: [], truncated: false };

export function openWebUIConfigured(): boolean {
  const o = config.openwebui;
  return Boolean(o.baseUrl && o.apiKey && o.tonyModel && o.antonellaModel);
}

async function owui(
  path: string,
  init: { token?: string; method?: string; body?: unknown; timeoutMs?: number } = {},
): Promise<{ status: number; ok: boolean; data: any; text: string }> {
  const response = await fetch(`${config.openwebui.baseUrl}${path}`, {
    method: init.method ?? "GET",
    headers: {
      Authorization: `Bearer ${init.token ?? config.openwebui.apiKey}`,
      ...(init.body !== undefined ? { "Content-Type": "application/json" } : {}),
    },
    body: init.body !== undefined ? JSON.stringify(init.body) : undefined,
    signal: AbortSignal.timeout(init.timeoutMs ?? PRESET_TIMEOUT_MS),
  });
  const text = await response.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  return { status: response.status, ok: response.ok, data, text };
}

// ── 1. Personas ─────────────────────────────────────────────────────────────

/** A preset's `meta.knowledge` entry, as the chat pipeline reads it. */
interface KnowledgeRef {
  id?: string;
  type?: string;
  name?: string;
  /** Legacy entries name their vector collections directly. */
  collection_name?: string;
  collection_names?: string[];
}

interface Preset {
  system: string;
  knowledge: KnowledgeRef[];
}

/** One preset: system prompt + attached knowledge. Throws with a message a person can act on. */
export async function fetchPreset(modelId: string): Promise<Preset> {
  const r = await owui(`/api/v1/models/model?id=${encodeURIComponent(modelId)}`);
  // v0.13 answers an unknown id with 404; older versions with 200 + `null`.
  if (r.status === 404 || (r.ok && !r.data?.id)) throw new Error(`Open WebUI has no model preset "${modelId}"`);
  if (!r.ok) throw new Error(`Open WebUI preset "${modelId}" → HTTP ${r.status}: ${r.text.slice(0, 200)}`);
  const system = typeof r.data.params?.system === "string" ? r.data.params.system.trim() : "";
  if (!system) {
    throw new Error(
      `Open WebUI preset "${modelId}" returned no system prompt: it has none, or OPENWEBUI_API_KEY ` +
        "cannot read it (Open WebUI hides it from read-only users; use the owner's or an admin's key).",
    );
  }
  const knowledge = Array.isArray(r.data.meta?.knowledge) ? (r.data.meta.knowledge as KnowledgeRef[]) : [];
  return { system, knowledge };
}

/** Kept for callers that only need the prompt. */
export async function fetchPresetSystem(modelId: string): Promise<string> {
  return (await fetchPreset(modelId)).system;
}

function localPersona(who: Speaker): string {
  return readFileSync(fileURLToPath(new URL(`../prompts/couple/${who}.md`, import.meta.url)), "utf8").trim();
}

/** Both personas and their knowledge: from Open WebUI when configured, else the local files. */
export async function loadPersonas(): Promise<Personas> {
  if (!openWebUIConfigured()) {
    return {
      text: { tony: localPersona("tony"), antonella: localPersona("antonella") },
      knowledge: { tony: NO_KNOWLEDGE, antonella: NO_KNOWLEDGE },
      source: "local files",
    };
  }
  const { tonyModel, antonellaModel } = config.openwebui;
  const [tony, antonella] = await Promise.all([fetchPreset(tonyModel), fetchPreset(antonellaModel)]);
  const [tonyKnowledge, antonellaKnowledge] = await Promise.all([
    loadKnowledge(tony.knowledge),
    loadKnowledge(antonella.knowledge),
  ]);
  return {
    text: { tony: tony.system, antonella: antonella.system },
    knowledge: { tony: tonyKnowledge, antonella: antonellaKnowledge },
    source: `Open WebUI presets ${tonyModel}, ${antonellaModel}`,
  };
}

// ── 2. Knowledge (loaded once, into the instructions) ───────────────────────

interface KnowledgeFile {
  id: string;
  name: string;
}

async function knowledgeBaseFiles(kbId: string): Promise<KnowledgeFile[]> {
  const files: KnowledgeFile[] = [];
  for (let page = 1; page < 100; page++) {
    const r = await owui(
      `/api/v1/knowledge/${encodeURIComponent(kbId)}/files?page=${page}&limit=${KNOWLEDGE_PAGE_LIMIT}`,
      { timeoutMs: KNOWLEDGE_TIMEOUT_MS },
    );
    if (!r.ok) throw new Error(`Open WebUI knowledge base "${kbId}" → HTTP ${r.status}: ${r.text.slice(0, 200)}`);
    const items: any[] = Array.isArray(r.data?.items) ? r.data.items : [];
    for (const f of items) {
      if (typeof f?.id === "string") files.push({ id: f.id, name: String(f.filename ?? f.meta?.name ?? f.id) });
    }
    const total = typeof r.data?.total === "number" ? r.data.total : files.length;
    if (items.length === 0 || files.length >= total) break;
  }
  return files;
}

async function fileText(fileId: string): Promise<string> {
  const r = await owui(`/api/v1/files/${encodeURIComponent(fileId)}/data/content`, { timeoutMs: KNOWLEDGE_TIMEOUT_MS });
  if (!r.ok) throw new Error(`Open WebUI file "${fileId}" → HTTP ${r.status}: ${r.text.slice(0, 200)}`);
  return typeof r.data?.content === "string" ? r.data.content.trim() : "";
}

/**
 * The preset's knowledge: the collections retrieval will search, and the
 * files' text for the instructions, cut to OPENWEBUI_KNOWLEDGE_MAX_CHARS
 * (GPT-Live caps startup instructions at 16,384 tokens, persona included).
 * Whatever is cut is still reachable through retrieval.
 */
export async function loadKnowledge(refs: KnowledgeRef[]): Promise<SpouseKnowledge> {
  if (refs.length === 0) return NO_KNOWLEDGE;
  const collections = new Set<string>();
  const files = new Map<string, KnowledgeFile>();
  for (const ref of refs) {
    // Same mapping as Open WebUI's own retrieval (retrieval/utils.py):
    // a knowledge base's collection is its id; a single file's is file-<id>.
    if (ref.collection_name) collections.add(ref.collection_name);
    for (const c of ref.collection_names ?? []) collections.add(c);
    if (!ref.id) continue;
    if (ref.type === "file") {
      collections.add(`file-${ref.id}`);
      files.set(ref.id, { id: ref.id, name: ref.name ?? ref.id });
    } else if (ref.type === "collection" || !ref.type) {
      collections.add(ref.id);
      for (const f of await knowledgeBaseFiles(ref.id)) files.set(f.id, f);
    }
  }

  const max = config.openwebui.knowledgeMaxChars;
  const parts: string[] = [];
  let used = 0;
  let truncated = false;
  for (const f of files.values()) {
    const body = await fileText(f.id);
    if (!body) continue;
    const block = `### ${f.name}\n${body}`;
    const room = max - used;
    if (room <= 200) {
      truncated = true;
      break;
    }
    if (block.length > room) {
      parts.push(`${block.slice(0, room)}\n[…cut; ask the backend for the rest]`);
      used = max;
      truncated = true;
      break;
    }
    parts.push(block);
    used += block.length + 2;
  }
  return { text: parts.join("\n\n"), collections: [...collections], files: [...files.values()].map((f) => f.name), truncated };
}

// ── 3. Retrieval (per delegated turn) ───────────────────────────────────────

export interface Chunk {
  text: string;
  source: string;
  score: number | null;
}

/**
 * Open WebUI's own search over the collections: POST /api/v1/retrieval/
 * query/collection. Only the query and collections are sent, so top-k, the
 * relevance threshold, hybrid search and the embedding model all come from
 * Open WebUI's configuration, exactly as for a text chat.
 */
export async function queryKnowledge(collections: string[], query: string, timeoutMs: number): Promise<Chunk[]> {
  if (collections.length === 0 || !query.trim()) return [];
  const r = await owui("/api/v1/retrieval/query/collection", {
    method: "POST",
    body: { collection_names: collections, query },
    timeoutMs,
  });
  if (!r.ok) throw new Error(`Open WebUI retrieval → HTTP ${r.status}: ${r.text.slice(0, 200)}`);
  const docs: unknown[] = r.data?.documents?.[0] ?? [];
  const metas: any[] = r.data?.metadatas?.[0] ?? [];
  const scores: unknown[] = r.data?.distances?.[0] ?? [];
  return docs
    .map((d, i) => ({
      text: typeof d === "string" ? d.trim() : "",
      source: String(metas[i]?.name ?? metas[i]?.source ?? "knowledge"),
      score: typeof scores[i] === "number" ? (scores[i] as number) : null,
    }))
    .filter((c) => c.text);
}

// ── 4. Chat write-back ──────────────────────────────────────────────────────

export interface OpenWebUIUser {
  id: string;
  name: string;
  email: string;
}

/**
 * Who a student token belongs to. GET /api/v1/auths/ accepts only a browser
 * session token (v0.13 refuses API keys there with 403), so an API key —
 * fine for chats/new — comes back as `null` identity rather than an error.
 * Throws when the token is rejected outright.
 */
export async function whoAmI(token: string): Promise<OpenWebUIUser | null> {
  const r = await owui("/api/v1/auths/", { token });
  if (r.status === 403) return null;
  if (!r.ok || typeof r.data?.id !== "string") {
    throw new Error(`Open WebUI did not accept the student's sign-in (HTTP ${r.status}) — sign in to Open WebUI again`);
  }
  return { id: r.data.id, name: String(r.data.name ?? ""), email: String(r.data.email ?? "") };
}

/** Save a chat document under the token's owner; returns the new chat id. */
export async function createChat(token: string, chat: Record<string, unknown>, tags: string[] = []): Promise<string> {
  const r = await owui("/api/v1/chats/new", { token, method: "POST", body: { chat }, timeoutMs: KNOWLEDGE_TIMEOUT_MS });
  if (!r.ok || typeof r.data?.id !== "string") {
    throw new Error(`Open WebUI chats/new → HTTP ${r.status}: ${r.text.slice(0, 200)}`);
  }
  const id = r.data.id as string;
  for (const name of tags) {
    // Best effort: a tag is a convenience for finding avatar meetings.
    await owui(`/api/v1/chats/${encodeURIComponent(id)}/tags`, { token, method: "POST", body: { name } }).catch(() => {});
  }
  return id;
}
