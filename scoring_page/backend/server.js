// server.js — Question Quality + ABI scorer backend
// --------------------------------------------------
import express from "express";
import cors from "cors";
import { config as dotenv } from "dotenv";
import OpenAI from "openai";
import path from "path";
import { fileURLToPath } from "url";

dotenv(); // load .env

// ==== ENV ==========================
const PORT = Number(process.env.PORT || 8787);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_BASE_URL =
  process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const MOCK = process.env.MOCK_SCORER === "1";

// ==== Paths ========================
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const FRONTEND_DIR = path.join(__dirname, "..", "frontend");

// ==== OpenAI client ================
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY || undefined,
  baseURL: OPENAI_BASE_URL,
});

// ==== Express app ===================
const app = express();
app.use(
  cors({
    origin: "*",
  })
);
app.use(express.json({ limit: "4mb" }));
app.use(express.static(FRONTEND_DIR)); // serve index.html + script.js

// ---- Helpers ----------------------
function round(x, digits = 2) {
  return Number.isFinite(x)
    ? Math.round(x * 10 ** digits) / 10 ** digits
    : 0;
}

function safeNumber(x, fallback = 0) {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
}

// ==== Scoring schema ===============
// rubric: each dim 0–2
const SCORE_SCHEMA = {
  type: "object",
  properties: {
    relevance: { type: "number", minimum: 0, maximum: 2 },
    politeness: { type: "number", minimum: 0, maximum: 2 },
    on_topic: { type: "number", minimum: 0, maximum: 2 },
    neutrality: { type: "number", minimum: 0, maximum: 2 },
    non_imperative: { type: "number", minimum: 0, maximum: 2 },
    clarity_optional: { type: "number", minimum: 0, maximum: 2 },
    privacy_minimization_optional: { type: "number", minimum: 0, maximum: 2 },
    notes: { type: "string" },
  },
    required: [
    "relevance",
    "politeness",
    "on_topic",
    "neutrality",
    "non_imperative",
    "clarity_optional",
    "privacy_minimization_optional",
    "notes"
  ],
  additionalProperties: false,
};

// ---- mock scorer (offline) --------
function mockScore(question) {
  const len = question.text.length;
  const base = len > 80 ? 2 : len > 40 ? 1.5 : 1;
  const rel = base;
  const pol = base;
  const on = base;
  const neu = 1;
  const nonImp = 1;
  const clarity = len > 100 ? 1.5 : 1;
  const privacy = question.text.match(/password|account|id|ssn|social/i)
    ? 0
    : 1;

  const dims = {
    relevance: rel,
    politeness: pol,
    on_topic: on,
    neutrality: neu,
    non_imperative: nonImp,
    clarity_optional: clarity,
    privacy_minimization_optional: privacy,
  };
  return decorateScore(question, dims);
}

// ---- OpenAI scorer -----------------
async function scoreWithOpenAI(question) {
  const messages = [
    {
      role: "system",
      content:
        "You are a careful rater for student questions in a financial-planning interview practice.\n" +
        "Given ONE question from the student, score each dimension from 0 to 2 (0=poor,1=mixed,2=good).\n" +
        "Dimensions:\n" +
        "- relevance: is it relevant to the client's financial situation and this exercise?\n" +
        "- politeness: polite, respectful, no blame.\n" +
        "- on_topic: stays on finance / planning, not random life topics.\n" +
        "- neutrality: non-leading, non-judgmental phrasing.\n" +
        "- non_imperative: avoids ordering the client to do things; questions instead of commands.\n" +
        "- clarity_optional: clear wording and enough context.\n" +
        "- privacy_minimization_optional: avoids asking for unnecessary sensitive identifiers (account numbers, passwords, full SSN, etc.).\n" +
        "Return STRICT JSON only.",
    },
    {
      role: "user",
      content: question.text,
    },
  ];

  const completion = await openai.chat.completions.create({
    model: OPENAI_MODEL,
    messages,
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "question_score",
        strict: true,
        schema: SCORE_SCHEMA,
      },
    },
    temperature: 0,
  });

  const raw = JSON.parse(completion.choices[0].message.content);
  return decorateScore(question, raw);
}

// ---- normalize + derived fields ----
function decorateScore(question, raw) {
  const r = {
    relevance: safeNumber(raw.relevance),
    politeness: safeNumber(raw.politeness),
    on_topic: safeNumber(raw.on_topic),
    neutrality: safeNumber(raw.neutrality),
    non_imperative: safeNumber(raw.non_imperative),
    clarity_optional: safeNumber(raw.clarity_optional),
    privacy_minimization_optional: safeNumber(
      raw.privacy_minimization_optional
    ),
  };

  const score_total =
    r.relevance +
    r.politeness +
    r.on_topic +
    r.neutrality +
    r.non_imperative +
    r.clarity_optional +
    r.privacy_minimization_optional;

  let verdict = "ok";
  if (score_total >= 11) verdict = "good";
  else if (score_total <= 5) verdict = "needs_work";

  return {
    id: question.id,
    question: question.text,
    studentId: question.studentId,
    studentName: question.studentName,
    score_total: round(score_total, 2),
    verdict,
    ...r,
    notes: raw.notes || "",
  };
}

// ==== ABI helpers ===================
// Map rubric into ABI 3+12 scores (all 0–1)
function computeAbiForQuestion(scored) {
  const rel = safeNumber(scored.relevance, 0);
  const onTopic = safeNumber(scored.on_topic, 0);
  const clarity = safeNumber(scored.clarity_optional, 0);
  const polite = safeNumber(scored.politeness, 0);
  const neutral = safeNumber(scored.neutrality, 0);
  const nonImp = safeNumber(scored.non_imperative, 0);
  const privacy = safeNumber(scored.privacy_minimization_optional, 0);

  // map 0–2 rubric to 0–1
  const norm = (v) => v / 2;

  const subs = {
    // Ability
    knowledge_consistency: norm(rel),
    professional_tone: norm(clarity),
    rationality: norm(neutral),
    calibrated_confidence: 0.6, // not directly from rubric, keep mid
    // Benevolence
    politeness: norm(polite),
    human_care: norm(polite),
    care_my_interest: norm(onTopic),
    shared_interest: 0.5,
    // Integrity
    legality: norm(privacy),
    morality: norm(neutral),
    contract: norm(onTopic),
    inducement: 1 - norm(nonImp), // more imperative -> more inducing
  };

  const ability =
    0.35 * subs.knowledge_consistency +
    0.25 * subs.professional_tone +
    0.2 * subs.rationality +
    0.2 * subs.calibrated_confidence;

  const benevolence =
    0.25 * subs.politeness +
    0.3 * subs.human_care +
    0.25 * subs.care_my_interest +
    0.2 * subs.shared_interest;

  const integrity =
    0.3 * subs.legality +
    0.25 * subs.morality +
    0.3 * subs.contract +
    0.15 * (1 - subs.inducement);

  return {
    ability: round(ability),
    benevolence: round(benevolence),
    integrity: round(integrity),
    abi_total: round((ability + benevolence + integrity) / 3),
    subs,
  };
}

function aggregateAbi(list) {
  if (!list.length) {
    return null;
  }
  const sum = (getter) =>
    list.reduce((acc, x) => acc + safeNumber(getter(x)), 0);

  const ability = sum((x) => x.ability) / list.length;
  const benevolence = sum((x) => x.benevolence) / list.length;
  const integrity = sum((x) => x.integrity) / list.length;

  // average all subs
  const subKeys = Object.keys(list[0].subs);
  const subsAvg = {};
  for (const k of subKeys) {
    subsAvg[k] =
      sum((x) => safeNumber(x.subs[k])) / list.length;
  }

  return {
    ability: round(ability),
    benevolence: round(benevolence),
    integrity: round(integrity),
    abi_total: round((ability + benevolence + integrity) / 3),
    subs: Object.fromEntries(
      Object.entries(subsAvg).map(([k, v]) => [k, round(v)])
    ),
  };
}

// ==== Aggregation over questions =====
function buildAggregate(results, abiEnabled) {
  const count = results.length;
  const sum = (key) =>
    results.reduce((acc, r) => acc + safeNumber(r[key]), 0);

  const avg_total_0_14 = count ? sum("score_total") / count : 0;
  const overall_0_100 = (avg_total_0_14 / 14) * 100;

  const dimsList = [
    "relevance",
    "politeness",
    "on_topic",
    "neutrality",
    "non_imperative",
    "clarity_optional",
    "privacy_minimization_optional",
  ];
  const dimsAvg = {};
  for (const d of dimsList) {
    dimsAvg[d] = count ? round(sum(d) / count) : 0;
  }

  // histogram 0–3,4–6,7–10,11–14
  const bins = [0, 0, 0, 0];
  for (const r of results) {
    const t = safeNumber(r.score_total);
    if (t <= 3) bins[0]++;
    else if (t <= 6) bins[1]++;
    else if (t <= 10) bins[2]++;
    else bins[3]++;
  }

  // simple habit feedback bullets
  const habits = [];
  if (dimsAvg.relevance < 1.3) {
    habits.push("Stay closer to the client scenario and be more specific.");
  }
  if (dimsAvg.politeness < 1.3) {
    habits.push("Use more polite, tentative phrasing instead of direct blame.");
  }
  if (dimsAvg.on_topic < 1.3) {
    habits.push("Keep questions focused on finances and planning, not side topics.");
  }
  if (dimsAvg.privacy_minimization_optional < 1) {
    habits.push("Avoid asking for detailed IDs, passwords, or account numbers unless strictly necessary.");
  }
  if (habits.length === 0) {
    habits.push("Good habits overall — keep asking clear, polite, and focused questions!");
  }

  // per-student
  const perStudent = {};
  for (const r of results) {
    const sid = r.studentId || "unknown";
    if (!perStudent[sid]) {
      perStudent[sid] = {
        studentId: sid,
        studentName: r.studentName || sid,
        count: 0,
        sum_total: 0,
        dims_sum: Object.fromEntries(dimsList.map((d) => [d, 0])),
        abi_list: [],
      };
    }
    const s = perStudent[sid];
    s.count += 1;
    s.sum_total += safeNumber(r.score_total);
    for (const d of dimsList) {
      s.dims_sum[d] += safeNumber(r[d]);
    }
    if (abiEnabled && r.abi) s.abi_list.push(r.abi);
  }

  const perStudentOut = {};
  for (const [sid, s] of Object.entries(perStudent)) {
    const avg_total = s.count ? s.sum_total / s.count : 0;
    const dimsAvgStu = {};
    for (const d of dimsList) {
      dimsAvgStu[d] = s.count ? round(s.dims_sum[d] / s.count) : 0;
    }
    perStudentOut[sid] = {
      studentId: sid,
      studentName: s.studentName,
      count: s.count,
      avg_total_0_14: round(avg_total),
      overall_0_100: round((avg_total / 14) * 100),
      dims: dimsAvgStu,
      abi_avg: abiEnabled ? aggregateAbi(s.abi_list) : null,
    };
  }

  const abiAll = abiEnabled
    ? aggregateAbi(
        results
          .map((r) => r.abi)
          .filter((x) => x)
      )
    : null;

  return {
    count,
    avg_total_0_14: round(avg_total_0_14),
    overall_0_100: round(overall_0_100),
    dims: dimsAvg,
    distribution: {
      labels: ["0–3", "4–6", "7–10", "11–14"],
      counts: bins,
    },
    habits,
    perStudent: perStudentOut,
    abi_global: abiAll,
  };
}

// ==== API routes =====================

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    model: OPENAI_MODEL,
    mock: MOCK,
  });
});

app.post("/api/score", async (req, res) => {
  try {
    const { questions, useAbi } = req.body || {};
    if (!Array.isArray(questions) || !questions.length) {
      return res.status(400).json({ error: "questions must be a non-empty array" });
    }

    const abiEnabled = !!useAbi;

    const normalized = questions.map((q, idx) => {
      if (typeof q === "string") {
        return {
          id: idx,
          text: q,
          studentId: "unknown",
          studentName: "Unknown",
        };
      }
      return {
        id: q.id ?? idx,
        text: String(q.text ?? ""),
        studentId: q.studentId || "unknown",
        studentName: q.studentName || q.studentId || "Unknown",
      };
    });

    let scoredPromises;
    if (MOCK || !OPENAI_API_KEY) {
      scoredPromises = normalized.map((q) => mockScore(q));
    } else {
      scoredPromises = normalized.map((q) => scoreWithOpenAI(q));
    }

    const resultsRaw = await Promise.all(scoredPromises);

    // attach ABI if enabled
    const results = resultsRaw.map((r) => {
      if (!abiEnabled) return r;
      const abi = computeAbiForQuestion(r);
      return { ...r, abi };
    });

    const aggregate = buildAggregate(results, abiEnabled);

    res.json({
      ok: true,
      results,
      aggregate,
    });
  } catch (err) {
    console.error("Error in /api/score:", err);
    res.status(500).json({ error: String(err.message || err) });
  }
});

// ==== Start ==========================
app.listen(PORT, "0.0.0.0", () => {
  console.log(
    `Server listening on http://0.0.0.0:${PORT} | OpenAI base=${OPENAI_BASE_URL} | model=${OPENAI_MODEL} | mock=${MOCK}`
  );
});
