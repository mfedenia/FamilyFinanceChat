/**
 * Spoken duo session → Open WebUI chat, in the shape Open WebUI stores in its
 * `chat` table and accepts on import (verified against open-webui v0.8.12 and
 * v0.11.4 source: routers/chats.py `POST /import`, models/chats.py
 * `ChatImportForm` / `_chat_import_form_to_chat_model`, and the web UI's
 * Settings → Data Controls → Import Chats handler).
 *
 * `toOpenWebUIExport` returns what the UI's "Export chats" produces and its
 * "Import Chats" reads: a JSON ARRAY of chat rows. For the REST API, wrap the
 * same rows: `POST /api/v1/chats/import` with `{ "chats": rows }` — the row
 * keys (chat, meta, pinned, archived, folder_id, created_at, updated_at) are
 * exactly ChatImportForm; `id`/`user_id`/`title` are ignored there (the server
 * assigns a new id and the CALLER as owner, so it must be the student's token).
 *
 * The `chat` object carries both message forms Open WebUI uses:
 *  - `history.messages` (id-keyed tree, parentId/childrenIds, currentId) — what
 *    the web UI renders and what import copies into the chat_message table;
 *  - `messages` (flat list) — what FamilyFinanceChat's grading extraction
 *    (grading_feature/backend/extract_chats.py) reads.
 *
 * Two speakers vs. grading: extract_chats.py pairs each user message with the
 * NEXT assistant message and drops a second assistant message in a row. So
 * consecutive turns of the same role are merged into one message: persona
 * lines keep a "Name: …" label each (also when alone — grading sees only
 * `content`), user fragments (barge-ins, backchannels) are joined.
 */

/** One line of the spoken session, as the duo page saw it. */
export interface SpokenTurn {
  role: "user" | "assistant";
  /** Persona name for assistant turns. */
  speaker?: string;
  text: string;
  /** When the line started (epoch ms). */
  at: number;
  /** The speaker was cut off; the line was not heard to its end. */
  interrupted?: boolean;
}

export interface SpokenSession {
  title: string;
  /** Scenario id + persona names — kept on the chat for provenance. */
  scenario: string;
  personas: string[];
  startedAt: number;
  endedAt: number;
  turns: SpokenTurn[];
}

interface OwuiMessage {
  id: string;
  parentId: string | null;
  childrenIds: string[];
  role: "user" | "assistant";
  content: string;
  /** Epoch SECONDS (Open WebUI convention for messages). */
  timestamp: number;
  models?: string[];
  model?: string;
  modelName?: string;
  done?: boolean;
}

/** The export file: Open WebUI chat rows (the UI's export/import format). */
export function toOpenWebUIExport(session: SpokenSession): Record<string, unknown>[] {
  return [toOpenWebUIRow(session)];
}

/** One chat row as Open WebUI stores it. */
export function toOpenWebUIRow(session: SpokenSession): Record<string, unknown> {
  const messages = toMessages(session);
  const models = session.personas.length ? session.personas : ["avatar"];
  const chat = {
    id: "",
    title: session.title,
    models,
    params: {},
    history: {
      currentId: messages.at(-1)?.id ?? null,
      messages: Object.fromEntries(messages.map((m) => [m.id, m])),
    },
    messages,
    tags: [],
    // Epoch MILLISECONDS on the chat object (Open WebUI convention).
    timestamp: session.startedAt,
    // Ours, not Open WebUI's: stored verbatim in the chat JSON, harmless to it.
    avatarSession: {
      scenario: session.scenario,
      personas: session.personas,
      startedAt: session.startedAt,
      endedAt: session.endedAt,
    },
  };
  return {
    id: uuid(),
    user_id: "",
    title: session.title,
    chat,
    meta: {},
    pinned: false,
    archived: false,
    folder_id: null,
    // Epoch SECONDS on the row.
    created_at: Math.floor(session.startedAt / 1000),
    updated_at: Math.floor(session.endedAt / 1000),
  };
}

function toMessages(session: SpokenSession): OwuiMessage[] {
  const turns = session.turns
    .filter((t) => t.text.trim())
    .slice()
    .sort((a, b) => a.at - b.at);

  // Merge runs of the same role (see header: grading pairs user→next assistant).
  const groups: { role: SpokenTurn["role"]; turns: SpokenTurn[] }[] = [];
  for (const t of turns) {
    const last = groups.at(-1);
    if (last && last.role === t.role) last.turns.push(t);
    else groups.push({ role: t.role, turns: [t] });
  }

  const messages: OwuiMessage[] = [];
  for (const g of groups) {
    const parent = messages.at(-1) ?? null;
    const id = uuid();
    const timestamp = Math.floor(g.turns[0]!.at / 1000);
    let msg: OwuiMessage;
    if (g.role === "user") {
      msg = {
        id,
        parentId: parent?.id ?? null,
        childrenIds: [],
        role: "user",
        content: g.turns.map((t) => t.text.trim()).join(" "),
        timestamp,
        models: session.personas,
      };
    } else {
      const speakers = [...new Set(g.turns.map((t) => t.speaker ?? "Assistant"))];
      const name = speakers.join(" & ");
      msg = {
        id,
        parentId: parent?.id ?? null,
        childrenIds: [],
        role: "assistant",
        content: g.turns
          .map((t) => `${t.speaker ?? "Assistant"}: ${t.text.trim()}${t.interrupted ? " — (cut off)" : ""}`)
          .join("\n\n"),
        timestamp,
        model: name,
        modelName: name,
        done: true,
      };
    }
    if (parent) parent.childrenIds.push(id);
    messages.push(msg);
  }
  return messages;
}

function uuid(): string {
  return globalThis.crypto.randomUUID();
}
