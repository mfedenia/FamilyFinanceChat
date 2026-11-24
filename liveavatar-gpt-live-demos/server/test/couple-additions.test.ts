/**
 * The additions on top of the couple meeting: transcript saving, Park's
 * handoff and backchannel rules, and persona loading from Open WebUI.
 * No network: Open WebUI is a local stand-in server.
 *
 *   cd server && npx tsx --test test/couple-additions.test.ts
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { once } from "node:events";
import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// config.ts reads the environment once, at first import.
process.env.OPENAI_API_KEY ??= "test-only";
process.env.LIVEAVATAR_API_KEY ??= "test-only";

const { TranscriptRecorder } = await import("../src/transcript");
const { handsOffTo, isBackchannel, decisionRequest, parseDecision } = await import("../src/couple");

test("decisions: request follows the Decisions API guide (gpt-6-luna, choice + predicate)", () => {
  const body = decisionRequest(["Tony: I work at UW-Milwaukee."], "And how much is in your 401(k)?", "tony");
  assert.equal(body.model, "gpt-6-luna");
  assert.equal(body.input[0]!.role, "developer");
  assert.equal(body.input[1]!.role, "user");
  assert.match(body.input[1]!.content[0]!.text, /Adviser's latest words: "And how much is in your 401\(k\)\?"/);
  assert.match(body.input[1]!.content[0]!.text, /Last spouse to speak: Tony/);
  const [speaker, chime] = body.questions;
  assert.equal(speaker!.type, "choice");
  assert.equal(speaker!.name, "speaker");
  assert.deepEqual(speaker!.choices!.map((c) => c.value), ["tony", "antonella"]);
  assert.ok(speaker!.choices!.every((c) => c.description.length > 50));
  assert.equal(chime!.type, "predicate");
  assert.equal(chime!.name, "chime_in");
});

test("decisions: answers, probabilities, chime-in threshold, refusals", () => {
  const ok = parseDecision({
    answers: [
      { type: "choice", name: "speaker", choice: "antonella",
        probabilities: [{ value: "tony", probability: 0.07 }, { value: "antonella", probability: 0.93 }] },
      { type: "predicate", name: "chime_in", probability: 0.72 },
    ],
  });
  assert.deepEqual(ok, { speaker: "antonella", chimeIn: true, probabilities: { tony: 0.07, antonella: 0.93 } });
  const quiet = parseDecision({
    answers: [
      { type: "choice", name: "speaker", choice: "tony" },
      { type: "predicate", name: "chime_in", probability: 0.4 },
    ],
  });
  assert.deepEqual(quiet, { speaker: "tony", chimeIn: false, probabilities: {} });
  assert.equal(parseDecision({ answers: [{ type: "refusal", name: "speaker" }] }), null);
  assert.equal(parseDecision({ answers: [{ type: "choice", name: "speaker", choice: "rita" }] }), null);
  assert.equal(parseDecision({}), null);
});

const tick = () => new Promise((r) => setTimeout(r, 5));

test("transcript: turns, who answered, interruptions, floor log, saved file", async () => {
  const r = new TranscriptRecorder();
  r.record("tony_assistant_1", "tony", "Hi, I'm Tony, and this is my wife Antonella.", true);
  r.floorChange("tony", "greeting");
  await tick();
  r.record("adv_user_1", "adviser", "How much is in your 401", false);
  r.record("adv_user_1", "adviser", "How much is in your 401(k)?", true);
  r.floorChange("antonella", "gpt-5.4-mini in 410ms");
  await tick();
  r.record("antonella_assistant_2", "antonella", "About two hundred ninety thousand, I", true);
  r.markInterrupted("antonella");
  await tick();
  r.record("adv_user_2", "adviser", "Sorry, Tony, and your pension", false);

  const dir = mkdtempSync(join(tmpdir(), "couple-transcripts-"));
  try {
    const file = r.save(dir, {
      sessionId: "abcdef12-0000",
      model: "gpt-live-1",
      decisionModel: "gpt-5.4-mini",
      personaSource: "local files",
      student: { id: "s-1", name: "Test", email: "t@example.invalid" },
      startedAt: new Date("2026-11-23T10:00:00Z"),
      endedAt: new Date("2026-11-23T10:05:00Z"),
      endReason: "client_stop",
    });
    assert.ok(file && file.endsWith("_couple_abcdef12.json"));
    const saved = JSON.parse(readFileSync(file!, "utf8"));
    assert.equal(saved.end_reason, "client_stop");
    assert.equal(saved.persona_source, "local files");
    assert.deepEqual(saved.turns.map((t: any) => t.speaker), ["tony", "adviser", "antonella", "adviser"]);
    assert.equal(saved.turns[1].text, "How much is in your 401(k)?");
    assert.equal(saved.turns[1].answered_by, "antonella");
    assert.equal(saved.turns[2].interrupted, true);
    assert.equal(saved.turns[3].complete, false);
    assert.deepEqual(saved.floor.map((f: any) => f.why), ["greeting", "gpt-5.4-mini in 410ms"]);
    assert.equal(saved.metrics.antonella.answered_adviser, 1);
    assert.equal(saved.metrics.interruptions_by_adviser, 1);
    assert.equal(readdirSync(dir).filter((f) => f.endsWith(".tmp")).length, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
  assert.equal(new TranscriptRecorder().save(tmpdir(), {} as any), null);
});

test("handoff: the last sentence names the other spouse and asks or invites", () => {
  assert.equal(handsOffTo("We like to travel. Antonella, what do you think?", "Antonella"), true);
  assert.equal(handsOffTo("I'd retire tomorrow. Right, Tony?", "Tony"), true);
  assert.equal(handsOffTo("Honestly I'm not sure. Antonella, you tell them about the annuity.", "Antonella"), true);
  // Naming the spouse is not handing over.
  assert.equal(handsOffTo("Antonella handles the 401(k), not me.", "Antonella"), false);
  assert.equal(handsOffTo("Antonella, what do you think? Anyway, my pension is with WRS.", "Antonella"), false);
  assert.equal(handsOffTo("What do you think about that?", "Antonella"), false);
});

test("backchannel: short and not a question", () => {
  for (const t of ["Mm-hmm.", "Okay", "Right, right."]) assert.equal(isBackchannel(t), true, t);
  for (const t of ["Okay, and your pension?", "Why?", "How much do you save?", ""]) assert.equal(isBackchannel(t), false, t);
});

test("personas: Open WebUI presets when configured, else local files", async () => {
  const seen: string[] = [];
  const owui = createServer((req, res) => {
    seen.push(`${req.headers.authorization} ${req.url}`);
    const id = new URL(req.url!, "http://x").searchParams.get("id");
    res.writeHead(200, { "content-type": "application/json" });
    if (id === "tony-preset") res.end(JSON.stringify({ id, params: { system: "You are Tony (from Open WebUI)." } }));
    else if (id === "antonella-preset") res.end(JSON.stringify({ id, params: { system: "You are Antonella (from Open WebUI)." } }));
    else if (id === "hidden-preset") res.end(JSON.stringify({ id, params: {} }));
    else res.end("null");
  });
  owui.listen(0, "127.0.0.1");
  await once(owui, "listening");
  const { config } = await import("../src/config");
  const { loadPersonas } = await import("../src/openwebui");
  try {
    // Not configured: local files.
    Object.assign(config.openwebui, { baseUrl: "", apiKey: "", tonyModel: "", antonellaModel: "" });
    const local = await loadPersonas();
    assert.equal(local.source, "local files");
    assert.ok(local.text.tony.length > 100 && local.text.antonella.length > 100);

    // Configured: presets, with the owner's key.
    Object.assign(config.openwebui, {
      baseUrl: `http://127.0.0.1:${(owui.address() as any).port}`,
      apiKey: "owui-test-key",
      tonyModel: "tony-preset",
      antonellaModel: "antonella-preset",
    });
    const remote = await loadPersonas();
    assert.equal(remote.text.tony, "You are Tony (from Open WebUI).");
    assert.equal(remote.text.antonella, "You are Antonella (from Open WebUI).");
    assert.match(remote.source, /Open WebUI presets tony-preset, antonella-preset/);
    assert.ok(seen.every((s) => s.startsWith("Bearer owui-test-key /api/v1/models/model?id=")));

    // A preset whose prompt the key cannot read fails before any avatar starts.
    config.openwebui.antonellaModel = "hidden-preset";
    await assert.rejects(loadPersonas(), /returned no system prompt/);
    config.openwebui.antonellaModel = "missing-preset";
    await assert.rejects(loadPersonas(), /no model preset "missing-preset"/);
  } finally {
    owui.close();
  }
});
