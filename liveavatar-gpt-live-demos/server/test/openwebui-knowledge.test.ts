/**
 * Open WebUI knowledge for the couple meeting: loading the presets' knowledge
 * (step 1), answering a delegated turn from Open WebUI's retrieval (step 2),
 * and saving the meeting as the student's Open WebUI chat. No network: Open
 * WebUI and the OpenAI answer model are local stand-in servers, shaped after
 * Open WebUI v0.13.4's routers.
 *
 *   cd server && npx tsx --test test/openwebui-knowledge.test.ts
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer, type IncomingMessage } from "node:http";
import { once } from "node:events";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

process.env.OPENAI_API_KEY ??= "test-only";
process.env.LIVEAVATAR_API_KEY ??= "test-only";

const { config } = await import("../src/config");
const { loadPersonas, queryKnowledge, whoAmI, createChat } = await import("../src/openwebui");
const { TranscriptRecorder } = await import("../src/transcript");
const { CoupleSession } = await import("../src/couple");

const body = async (req: IncomingMessage) => {
  let s = "";
  for await (const c of req) s += c;
  return s ? JSON.parse(s) : null;
};

interface Seen {
  auth: string;
  method: string;
  path: string;
  body: any;
}

/** A stand-in Open WebUI with two presets, a knowledge base (2 pages) and a single file. */
async function fakeOpenWebUI() {
  const seen: Seen[] = [];
  const chats: any[] = [];
  const server = createServer(async (req, res) => {
    const url = new URL(req.url!, "http://x");
    const b = req.method === "POST" ? await body(req) : null;
    seen.push({ auth: String(req.headers.authorization), method: req.method!, path: url.pathname + url.search, body: b });
    const send = (status: number, data: unknown) => {
      res.writeHead(status, { "content-type": "application/json" });
      res.end(JSON.stringify(data));
    };
    const p = url.pathname;
    if (p === "/api/v1/models/model") {
      const id = url.searchParams.get("id");
      if (id === "tony-preset")
        return send(200, {
          id,
          params: { system: "You are Tony." },
          meta: { knowledge: [{ id: "kb-case", type: "collection", name: "Beckham case" }, { id: "f-will", type: "file", name: "will.pdf" }] },
        });
      if (id === "antonella-preset") return send(200, { id, params: { system: "You are Antonella." }, meta: {} });
      return send(404, { detail: "We could not find what you're looking for :/" });
    }
    if (p === "/api/v1/knowledge/kb-case/files") {
      const page = Number(url.searchParams.get("page"));
      const items = page === 1 ? [{ id: "f-sheet", filename: "data-sheet.md" }] : page === 2 ? [{ id: "f-ins", filename: "insurance.md" }] : [];
      return send(200, { items, total: 2 });
    }
    const content = /^\/api\/v1\/files\/([^/]+)\/data\/content$/.exec(p);
    if (content) {
      const text: Record<string, string> = {
        "f-sheet": "Tony's 403(b) balance: $0. Antonella's 401(k): $212,000.",
        "f-ins": "Term life: $250,000 on Tony through UWM.",
        "f-will": "x".repeat(5_000),
      };
      return send(200, { content: text[content[1]!] ?? "" });
    }
    if (p === "/api/v1/retrieval/query/collection") {
      return send(200, {
        distances: [[0.82, 0.41]],
        documents: [["Term life: $250,000 on Tony through UWM.", "Tony's 403(b) balance: $0."]],
        metadatas: [[{ name: "insurance.md" }, { source: "data-sheet.md" }]],
      });
    }
    if (p === "/api/v1/auths/") {
      const auth = String(req.headers.authorization);
      if (auth === "Bearer student-session") return send(200, { id: "u-1", name: "Sam Student", email: "sam@wisc.edu", role: "user" });
      if (auth === "Bearer student-api-key") return send(403, { detail: "A human session is required." });
      return send(401, { detail: "Not authenticated" });
    }
    if (p === "/api/v1/chats/new") {
      chats.push(b.chat);
      return send(200, { id: `chat-${chats.length}`, title: b.chat.title });
    }
    if (/^\/api\/v1\/chats\/[^/]+\/tags$/.test(p)) return send(200, []);
    return send(404, { detail: "not found" });
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return { server, seen, chats, url: `http://127.0.0.1:${(server.address() as any).port}` };
}

/** A stand-in OpenAI Responses endpoint for the answer model. */
async function fakeResponses(reply: string) {
  const requests: any[] = [];
  const server = createServer(async (req, res) => {
    requests.push(await body(req));
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ output: [{ type: "message", content: [{ type: "output_text", text: reply }] }] }));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  return { server, requests, url: `http://127.0.0.1:${(server.address() as any).port}/v1/responses` };
}

function useOpenWebUI(url: string) {
  Object.assign(config.openwebui, {
    baseUrl: url,
    apiKey: "admin-key",
    tonyModel: "tony-preset",
    antonellaModel: "antonella-preset",
    knowledgeMaxChars: 2_000,
  });
}

test("step 1: the presets' knowledge is loaded into the personas, with the collections to search", async () => {
  const owui = await fakeOpenWebUI();
  try {
    useOpenWebUI(owui.url);
    const p = await loadPersonas();
    const k = p.knowledge.tony;
    // Same collection naming as Open WebUI's retrieval: a KB is its id, a file is file-<id>.
    assert.deepEqual(k.collections.sort(), ["file-f-will", "kb-case"]);
    // Both pages of the knowledge base, plus the single file.
    assert.deepEqual(k.files.sort(), ["data-sheet.md", "insurance.md", "will.pdf"]);
    assert.match(k.text, /### data-sheet\.md\nTony's 403\(b\) balance: \$0/);
    assert.match(k.text, /### insurance\.md\nTerm life/);
    // The 5,000-char file does not fit the 2,000-char budget: cut, and flagged.
    assert.equal(k.truncated, true);
    assert.ok(k.text.length <= 2_100);
    // No knowledge on Antonella's preset: no retrieval for her.
    assert.deepEqual(p.knowledge.antonella, { text: "", collections: [], files: [], truncated: false });
    // Everything read with the admin key.
    assert.ok(owui.seen.every((s) => s.auth === "Bearer admin-key"));
    // An unknown preset (v0.13: 404) still fails before any avatar starts.
    config.openwebui.antonellaModel = "missing-preset";
    await assert.rejects(loadPersonas(), /no model preset "missing-preset"/);
  } finally {
    owui.server.close();
  }
});

test("step 2: retrieval sends only query + collections, so Open WebUI's own RAG settings apply", async () => {
  const owui = await fakeOpenWebUI();
  try {
    useOpenWebUI(owui.url);
    const chunks = await queryKnowledge(["kb-case"], "What life insurance does Tony have?", 2_000);
    assert.deepEqual(chunks, [
      { text: "Term life: $250,000 on Tony through UWM.", source: "insurance.md", score: 0.82 },
      { text: "Tony's 403(b) balance: $0.", source: "data-sheet.md", score: 0.41 },
    ]);
    const call = owui.seen.find((s) => s.path === "/api/v1/retrieval/query/collection")!;
    assert.deepEqual(call.body, { collection_names: ["kb-case"], query: "What life insurance does Tony have?" });
    assert.deepEqual(await queryKnowledge([], "anything", 2_000), []);
  } finally {
    owui.server.close();
  }
});

test("step 2: a delegated turn is answered from Open WebUI's hits, as commentary on that delegation", async () => {
  const owui = await fakeOpenWebUI();
  const answers = await fakeResponses("Uh, I think it's a term policy, two-fifty, through work.");
  const dir = mkdtempSync(join(tmpdir(), "couple-rag-"));
  try {
    useOpenWebUI(owui.url);
    config.gptlive.responsesUrl = answers.url;
    config.transcripts.dir = dir;
    const personas = await loadPersonas();
    const fake = (id: string) => ({ sessionId: id, livekitUrl: "", livekitClientToken: "", wsUrl: "ws://127.0.0.1:9" });
    const meeting = new CoupleSession(fake("t"), fake("a"), personas, () => {}, {
      token: "student-session",
      user: { id: "u-1", name: "Sam Student", email: "sam@wisc.edu" },
    });
    meeting.log = () => {};
    meeting.emit = () => {};
    const tony = meeting.spouses.tony;
    // Tony's preset has knowledge, so his GPT-Live delegates to this app; Antonella's does not.
    assert.equal(tony.retrieval, true);
    assert.equal(meeting.spouses.antonella.retrieval, false);

    const appended: [string, string, string | null][] = [];
    tony.bridge.append = (kind, text, id = null) => void appended.push([kind, text, id]);

    // The adviser asks; Tony's GPT-Live says "hmm…" and delegates.
    meeting.onAdviserStarted(tony);
    meeting.onSpouseTurn(tony, { id: "1", role: "user", text: "What life insurance do you have, Tony?", done: true } as any);
    await meeting.onDelegation(tony, "item_deleg_1");

    assert.deepEqual(appended, [["commentary", "Uh, I think it's a term policy, two-fifty, through work.", "item_deleg_1"]]);
    // The answer model saw the retrieved passage and the question.
    const prompt = answers.requests[0].input[0].content as string;
    assert.match(prompt, /Term life: \$250,000 on Tony through UWM\./);
    assert.match(prompt, /The adviser just asked: "What life insurance do you have, Tony\?"/);
    assert.match(answers.requests[0].instructions, /You are Tony\./);

    // A delegation from the spouse who does NOT hold the floor is released silently.
    const ant = meeting.spouses.antonella;
    const antAppends: [string, string, string | null][] = [];
    ant.bridge.append = (kind, text, id = null) => void antAppends.push([kind, text, id]);
    await meeting.onDelegation(ant, "item_deleg_2");
    assert.equal(antAppends[0]![0], "thinking");
    assert.equal(antAppends[0]![2], "item_deleg_2");

    // Tony's answer is heard; the meeting ends → saved to Open WebUI under the student.
    meeting.onSpouseTurn(tony, { id: "2", role: "assistant", text: "Uh, I think it's a term policy, two-fifty, through work.", done: true } as any);
    await meeting.stop("client_stop");
    assert.equal(meeting.openWebUIChatId, "chat-1");
    const save = owui.seen.find((s) => s.path === "/api/v1/chats/new")!;
    assert.equal(save.auth, "Bearer student-session", "the chat must be created with the STUDENT's token");
    const chat = owui.chats[0];
    assert.match(chat.title, /^\[Avatar\] The Beckhams — /);
    assert.deepEqual(chat.models, ["family-beckham"]);
    assert.deepEqual(
      chat.messages.map((m: any) => [m.role, m.content]),
      [
        ["user", "What life insurance do you have, Tony?"],
        ["assistant", "Tony: Uh, I think it's a term policy, two-fifty, through work."],
      ],
    );
    assert.ok(owui.seen.some((s) => /\/api\/v1\/chats\/chat-1\/tags$/.test(s.path) && s.body.name === "avatar"));

    // The local transcript keeps the retrieval log and the chat id.
    const file = readdirSync(dir).find((f) => f.endsWith(".json"))!;
    const saved = JSON.parse(readFileSync(join(dir, file), "utf8"));
    assert.equal(saved.openwebui_chat_id, "chat-1");
    assert.equal(saved.student.email, "sam@wisc.edu");
    assert.deepEqual(
      saved.retrievals.map((r: any) => [r.speaker, r.outcome, r.sources]),
      [
        ["tony", "answered", ["insurance.md", "data-sheet.md"]],
        ["antonella", "stale", []],
      ],
    );
  } finally {
    owui.server.close();
    answers.server.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("chat document: grader-safe user → assistant alternation, Open WebUI history tree", () => {
  const t = new TranscriptRecorder();
  t.record("g", "tony", "Hi, I'm Tony, and this is Antonella.", true);
  t.record("a1", "adviser", "Nice to meet you both. What brings you in?", true);
  t.record("t1", "tony", "We want to retire someday.", true);
  t.record("n1", "antonella", "Safely, I'd add.", true);
  t.record("a2", "adviser", "How much do you save each month?", true);
  t.record("t2", "tony", "Maybe fifteen hundred", true);
  t.markInterrupted("tony");
  t.record("a3", "adviser", "Sorry, go on.", true);
  const chat = t.toOpenWebUIChat({
    title: "x",
    models: { tony: "family-beckham", antonella: "family-beckham" },
    names: { tony: "Tony", antonella: "Antonella" },
  }) as any;
  const roles = chat.messages.map((m: any) => m.role);
  assert.deepEqual(roles, ["assistant", "user", "assistant", "user", "assistant", "user"]);
  // Back-to-back spouses merge into ONE assistant message (the grader drops a second one).
  assert.equal(chat.messages[2].content, "Tony: We want to retire someday.\n\nAntonella: Safely, I'd add.");
  assert.equal(chat.messages[4].content, "Tony: Maybe fifteen hundred [interrupted]");
  // History tree: parent/child links and the current leaf, as Open WebUI's UI saves it.
  const ids = chat.messages.map((m: any) => m.id);
  assert.equal(chat.history.currentId, ids.at(-1));
  for (let i = 1; i < ids.length; i++) {
    assert.equal(chat.history.messages[ids[i]].parentId, ids[i - 1]);
    assert.deepEqual(chat.history.messages[ids[i - 1]].childrenIds, [ids[i]]);
  }
  assert.ok(chat.messages.every((m: any) => Number.isInteger(m.timestamp)));
  assert.equal(new TranscriptRecorder().toOpenWebUIChat({ title: "", models: {} as any, names: {} as any }), null);
});

test("student sign-in: a browser session names the student; an API key is accepted unnamed; a bad token fails", async () => {
  const owui = await fakeOpenWebUI();
  try {
    useOpenWebUI(owui.url);
    assert.deepEqual(await whoAmI("student-session"), { id: "u-1", name: "Sam Student", email: "sam@wisc.edu" });
    assert.equal(await whoAmI("student-api-key"), null);
    await assert.rejects(whoAmI("expired"), /did not accept the student's sign-in \(HTTP 401\)/);
    assert.equal(await createChat("student-session", { title: "t", messages: [] }), "chat-1");
  } finally {
    owui.server.close();
  }
});
