#!/usr/bin/env node
/**
 * Puts a "Talk to the avatar" button into Open WebUI for the couple meeting,
 * over Open WebUI's public REST API only (no Open WebUI source changes):
 *
 *   1. installs/updates two Functions and sets their AVATAR_APP_URL valve:
 *      - "Avatar banner" Filter (openwebui/avatar_banner.py): a large
 *        "Talk to Tony & Antonella face to face" banner on each reply;
 *      - "Talk with avatar" Action (openwebui/talk_with_avatar.py): the same
 *        link as a small button in the reply's toolbar;
 *   2. creates/updates the model preset `family-beckham` ("The Beckhams"):
 *      a text-chat role-play of both spouses, built from
 *      server/prompts/couple/{tony,antonella}.md, readable by every user,
 *      with both functions attached. Chatting with it shows the banner.
 *
 * Safe to re-run. Settings (this folder's .env or the environment):
 *   OPENWEBUI_BASE_URL    default http://localhost:3000
 *   OPENWEBUI_API_KEY     an admin's API key (needs API keys enabled), or:
 *   OPENWEBUI_EMAIL       admin login; on a brand-new Open WebUI with no users,
 *   OPENWEBUI_PASSWORD    this account is created and becomes the admin
 *   OPENWEBUI_BASE_MODEL  model the preset chats with, default gpt-5.4-mini
 *   AVATAR_APP_URL        where students' browsers reach this app,
 *                         default http://localhost:5173
 *
 * Note: OPENWEBUI_BASE_URL + OPENWEBUI_API_KEY alone do not switch the
 * avatar's personas to Open WebUI; that needs COUPLE_OPENWEBUI_MODEL_* too.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
try {
  process.loadEnvFile(join(ROOT, ".env"));
} catch {
  /* ambient env only */
}

const env = (k, d = "") => (process.env[k] ?? "").trim() || d;
const BASE = env("OPENWEBUI_BASE_URL", "http://localhost:3000").replace(/\/+$/, "");
const BASE_MODEL = env("OPENWEBUI_BASE_MODEL", "gpt-5.4-mini");
const AVATAR_APP_URL = env("AVATAR_APP_URL", "http://localhost:5173");
const ACTION_ID = "talk_with_avatar";
const BANNER_FILTER_ID = "avatar_banner";
const MODEL_ID = "family-beckham";
const EVERYONE_READS = [{ principal_type: "user", principal_id: "*", permission: "read" }];

const COUPLE = {
  name: "The Beckhams",
  description:
    "Tony (54) and Antonella (52), Milwaukee: the FIN 602 course case. " +
    'Use the "Talk to the avatar" banner on any reply to talk to them face to face.',
  members: [
    { name: "Tony", fullName: "Tony Beckham", age: 54, role: "husband", file: "tony.md" },
    { name: "Antonella", fullName: "Antonella Beckham", age: 52, role: "wife", file: "antonella.md" },
  ],
};

let token = "";

async function api(method, path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  return { ok: res.ok, status: res.status, data };
}

function fail(msg) {
  console.error(`✗ ${msg}`);
  process.exit(1);
}

async function login() {
  const ping = await fetch(`${BASE}/api/config`).catch(() => null);
  if (!ping?.ok) fail(`no Open WebUI at ${BASE} — start it first, or set OPENWEBUI_BASE_URL`);

  if (env("OPENWEBUI_API_KEY")) {
    token = env("OPENWEBUI_API_KEY");
    // v0.13 serves /api/v1/auths/ to browser sessions only (403 for API
    // keys), so check admin rights on an admin-only route instead.
    const me = await api("GET", "/api/v1/auths/");
    const admin = me.ok ? me.data?.role === "admin" : (await api("GET", "/api/v1/functions/list")).ok;
    if (!admin) fail("OPENWEBUI_API_KEY must belong to an admin");
    console.log(`✓ using OPENWEBUI_API_KEY on ${BASE}`);
    return;
  }
  const email = env("OPENWEBUI_EMAIL");
  const password = env("OPENWEBUI_PASSWORD");
  if (!email || !password) fail("set OPENWEBUI_API_KEY, or OPENWEBUI_EMAIL and OPENWEBUI_PASSWORD, in .env");

  let r = await api("POST", "/api/v1/auths/signin", { email, password });
  if (!r.ok) {
    // A brand-new instance has no users: the first signup becomes the admin.
    const signup = await api("POST", "/api/v1/auths/signup", { name: "Admin", email, password });
    if (!signup.ok) fail(`could not sign in as ${email} (${r.status}: ${JSON.stringify(r.data)})`);
    console.log(`✓ created admin account ${email}`);
    r = signup;
  }
  if (r.data?.role !== "admin") fail(`${email} is not an admin in this Open WebUI`);
  token = r.data.token;
  console.log(`✓ signed in to ${BASE} as ${email}`);
}

async function installFunction(id, name, description) {
  const content = readFileSync(join(ROOT, "openwebui", `${id}.py`), "utf8");
  const form = { id, name, content, meta: { description, manifest: {} } };
  const existing = await api("GET", `/api/v1/functions/id/${id}`);
  const r = existing.ok
    ? await api("POST", `/api/v1/functions/id/${id}/update`, form)
    : await api("POST", "/api/v1/functions/create", form);
  if (!r.ok) fail(`installing function ${id} failed (${r.status}): ${JSON.stringify(r.data)}`);

  const now = await api("GET", `/api/v1/functions/id/${id}`);
  if (!now.data?.is_active) {
    const t = await api("POST", `/api/v1/functions/id/${id}/toggle`);
    if (!t.ok) fail(`enabling function ${id} failed (${t.status})`);
  }
  const v = await api("POST", `/api/v1/functions/id/${id}/valves/update`, {
    AVATAR_APP_URL,
    MODEL_PREFIX: "family-",
  });
  if (!v.ok) fail(`setting function ${id}'s valves failed (${v.status}): ${JSON.stringify(v.data)}`);
  console.log(`✓ function "${id}" ${existing.ok ? "updated" : "installed"}, enabled, AVATAR_APP_URL=${AVATAR_APP_URL}`);
}

function textPrompt() {
  const [a, b] = COUPLE.members.map((m) => ({
    ...m,
    persona: readFileSync(join(ROOT, "server", "prompts", "couple", m.file), "utf8").trim(),
  }));
  return [
    `You are role-playing ${COUPLE.name}: ${a.fullName} (${a.age}, ${a.role}) and ${b.fullName} (${b.age}, ${b.role}), ` +
      `simulated clients meeting a financial adviser for the first time. The user is a student practising client ` +
      `advising; they are the adviser. Stay in character as the clients. Never give financial advice, never coach ` +
      `the student, never step out of character, and never invent facts beyond the personas below.`,
    `Answer as whichever of them the question is for (both if it is for both), prefixing each line with the ` +
      `speaker's first name, e.g. "${a.name}: ...". Keep replies short and conversational. Do not volunteer ` +
      `everything at once; a good adviser has to ask.`,
    `The personas below were written for a spoken meeting. Ignore their instructions about speaking aloud; ` +
      `everything else applies.`,
    `## ${a.fullName}\n\n${a.persona}`,
    `## ${b.fullName}\n\n${b.persona}`,
  ].join("\n\n");
}

async function installModel() {
  const models = await api("GET", "/api/models");
  const ids = new Set((models.data?.data ?? []).map((m) => m.id));
  if (!ids.has(BASE_MODEL)) {
    console.warn(`! base model "${BASE_MODEL}" is not in this Open WebUI's model list — set OPENWEBUI_BASE_MODEL`);
  }
  const form = {
    id: MODEL_ID,
    base_model_id: BASE_MODEL,
    name: COUPLE.name,
    meta: {
      description: COUPLE.description,
      profile_image_url: "/static/favicon.png",
      actionIds: [ACTION_ID],
      filterIds: [BANNER_FILTER_ID],
      avatar_members: COUPLE.members.map((m) => m.name).join(" & "),
      tags: [{ name: "family" }],
      suggestion_prompts: [
        { content: "Hi, thanks for coming in. Could you each tell me a little about yourselves?" },
      ],
    },
    params: { system: textPrompt() },
    access_grants: EVERYONE_READS,
    is_active: true,
  };
  const existing = await api("GET", `/api/v1/models/model?id=${encodeURIComponent(MODEL_ID)}`);
  const exists = existing.ok && existing.data;
  const r = exists
    ? await api("POST", "/api/v1/models/model/update", form)
    : await api("POST", "/api/v1/models/create", form);
  if (!r.ok) fail(`saving model ${MODEL_ID} failed (${r.status}): ${JSON.stringify(r.data)}`);
  console.log(`✓ model "${COUPLE.name}" (${MODEL_ID}) ${exists ? "updated" : "created"} on ${BASE_MODEL}`);
}

await login();
await installFunction(BANNER_FILTER_ID, "Avatar banner", "Large 'Talk to them face to face' banner on each family reply.");
await installFunction(ACTION_ID, "Talk with avatar", "Opens the spoken avatar meeting for the family you are chatting with.");
await installModel();
// Open WebUI caches the model list in memory; re-list so the cache picks up
// the new filterIds/actionIds now rather than on the next browser load.
await api("GET", "/api/models");
console.log(`\nDone. Open ${BASE}, pick "${COUPLE.name}" in the model menu, say hello, then press "Talk to the avatar" on the reply.`);
