// server.js — OpenAI JSON-Schema enforced scorer
// --------------------------------------------------
import express from "express";
import cors from "cors";
import { config as dotenv } from "dotenv";
import OpenAI from "openai";

dotenv(); // load .env

// ==== ENV ====
const PORT = Number(process.env.PORT || 8787);
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;                // 必填（用你自己的 OpenAI key）
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";   // 推荐
const MOCK = process.env.MOCK_SCORER === "1";                     // 可选：本地演示模式

// ==== OpenAI Client ====
const oa = new OpenAI({ apiKey: OPENAI_API_KEY, baseURL: OPENAI_BASE_URL });

// ==== App ====
const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    provider: "openai",
    model: OPENAI_MODEL,
    baseURL: OPENAI_BASE_URL,
    mock: MOCK,
  });
});

// ==== JSON Schema（强约束返回结构）====
const SCORE_SCHEMA = {
  type: "object",
  properties: {
    results: {
      type: "array",
      items: {
        type: "object",
        // ✅ 把 "notes" 加进来
        required: [
          "question",
          "relevance",
          "politeness",
          "on_topic",
          "neutrality",
          "non_imperative",
          "clarity_optional",
          "privacy_minimization_optional",
          "score_total",
          "verdict",
          "notes"               // <—
        ],
        properties: {
          question: { type: "string" },
          relevance: { type: "integer", minimum: 0, maximum: 2 },
          politeness: { type: "integer", minimum: 0, maximum: 2 },
          on_topic: { type: "integer", minimum: 0, maximum: 2 },
          neutrality: { type: "integer", minimum: 0, maximum: 2 },
          non_imperative: { type: "integer", minimum: 0, maximum: 2 },
          clarity_optional: { type: "integer", minimum: 0, maximum: 2 },
          privacy_minimization_optional: { type: "integer", minimum: 0, maximum: 2 },
          score_total: { type: "integer", minimum: 0, maximum: 14 },
          verdict: { type: "string", enum: ["good","okay","rewrite"] },
          notes: { type: "string" }          // 仍保留
        },
        additionalProperties: false
      }
    }
  },
  required: ["results"],
  additionalProperties: false
};


// ==== Helpers ====
function buildPrompt(questions) {
  const rubric = `You are a careful dialogue question-quality rater.
Score EACH question independently on these integer dimensions (0–2 each):
- relevance
- politeness
- on_topic
- neutrality
- non_imperative
Optional (0–2 each):
- clarity_optional
- privacy_minimization_optional

Total score per item: max 14.
Verdict rule: >=11 "good"; 7–10 "okay"; <7 "rewrite".
If an item is not a question or language is unknown, use zeros and verdict "rewrite" with a short reason in notes.
Return STRICT JSON only. Keep answers concise.
For each item set "notes" to "" unless there is a concrete reason (e.g., grammar error or sensitive PII).
Do not include explanations outside JSON.`;

  const list = questions.map((q, i) => `${i + 1}. ${q}`).join("\n");
  return `${rubric}\n\nRate the following questions:\n${list}`;
}

// 兼容常见 LLM 输出格式：正常 JSON / 双重编码 / 代码块 / 截取外层花括号
function safeParseJSON(txt) {
  // 1) 正常 JSON
  try {
    return JSON.parse(txt);
  } catch {}

  // 2) 双重编码：外层是字符串，内部才是 JSON（含 \" 与 \n）
  try {
    if ((txt.startsWith('"') && txt.endsWith('"')) || txt.includes('\\"')) {
      const innerText = JSON.parse(txt);      // 先还原字符串
      return JSON.parse(innerText);           // 再解析成对象
    }
  } catch {}

  // 3) 代码块 ```json ... ``` 或普通 ```
  const code = txt.match(/```json([\s\S]*?)```/i) || txt.match(/```([\s\S]*?)```/);
  if (code) {
    try { return JSON.parse(code[1]); } catch {}
  }

  // 4) 截取第一个外层大括号块
  const brace = txt.match(/\{[\s\S]*\}/);
  if (brace) {
    try { return JSON.parse(brace[0]); } catch {}
  }
  return null;
}

function aggregate(results) {
  const totals = results.map((r) => Number(r.score_total || 0));
  const count = results.length || 1;
  const avg = totals.reduce((a, b) => a + b, 0) / count;
  const norm100 = Math.round((avg / 14) * 100);

  const bins = { "0-3": 0, "4-6": 0, "7-10": 0, "11-14": 0 };
  for (const t of totals) {
    if (t <= 3) bins["0-3"]++;
    else if (t <= 6) bins["4-6"]++;
    else if (t <= 10) bins["7-10"]++;
    else bins["11-14"]++;
  }

  const dims = [
    "relevance",
    "politeness",
    "on_topic",
    "neutrality",
    "non_imperative",
    "clarity_optional",
    "privacy_minimization_optional",
  ];
  const sums = Object.fromEntries(dims.map((d) => [d, 0]));
  for (const r of results) for (const d of dims) sums[d] += Number(r[d] || 0);
  const perDimAvg = Object.fromEntries(
    dims.map((d) => [d, +(sums[d] / (results.length || 1)).toFixed(2)])
  );

  const habit_feedback = buildHabit(perDimAvg);

  return {
    count: results.length,
    avg_total_0_14: +avg.toFixed(2),
    overall_0_100: norm100,
    distribution: bins,
    per_dimension_avg: perDimAvg,
    habit_feedback,
  };
}

function buildHabit(avg) {
  const out = [];
  const add = (cond, msg) => cond && out.push(msg);

  add(avg.politeness >= 1.7, "Polite tone is consistent.");
  add(avg.politeness < 1.0, "Tone is blunt; add softeners like “please / could you”.");
  add(avg.non_imperative >= 1.7, "Requests avoid commands—good style.");
  add(avg.non_imperative < 1.0, "Prefer requests (“Would you…”) over commands.");
  add(avg.neutrality >= 1.7, "Neutral wording avoids presuppositions.");
  add(avg.neutrality < 1.0, "Reduce bias/leading phrasing.");
  add(avg.on_topic >= 1.7, "Focus is tight and on-topic.");
  add(avg.on_topic < 1.0, "Keep to a single objective per question.");
  add(avg.relevance >= 1.7, "Good linkage to prior context.");
  add(avg.relevance < 1.0, "Explain “why” when asking for private data.");
  add(avg.clarity_optional >= 1.7, "Specific scope/time/format improves answerability.");
  add(avg.clarity_optional < 1.0, "Be more specific (time window, units, examples).");
  add(avg.privacy_minimization_optional >= 1.7, "Minimal personal data requested.");
  add(avg.privacy_minimization_optional < 1.0, "Avoid unnecessary PII; state the reason if needed.");
  return out;
}

// ==== 可选：离线 Mock 打分（演示用）====
function mockScoreOne(q) {
  const t = (q || "").toLowerCase();
  const polite = /(please|could you|would you|能否|请)/.test(t) ? 2 : /(now|asap|马上)/.test(t) ? 0 : 1;
  const nonImp = /(please|could you|would you|能否|请)/.test(t) ? 2 : /(do|给我|必须|需要你)\b/.test(t) ? 0 : 1;
  const relevance = /\b(why|what|how|when|where)\b|\?/.test(t) ? 2 : 1;
  const onTopic = t.length < 140 ? 2 : 1;
  const neutrality = /(should|must|blame|最好)/.test(t) ? 1 : 2;
  const clarity = t.length > 10 ? 1 : 0;
  const privacy = /(phone|email|id|身份证|住址)/.test(t) ? 0 : 1;
  const total = Math.max(0, Math.min(14, relevance + polite + onTopic + neutrality + nonImp + clarity + privacy));
  const verdict = total >= 11 ? "good" : total >= 7 ? "okay" : "rewrite";
  return {
    relevance,
    politeness: polite,
    on_topic: onTopic,
    neutrality,
    non_imperative: nonImp,
    clarity_optional: clarity,
    privacy_minimization_optional: privacy,
    score_total: total,
    verdict,
    notes: "mock",
  };
}
function mockScoreBatch(questions) {
  return questions.map((q) => ({ question: q, ...mockScoreOne(q) }));
}

// ==== API ====
app.post("/api/score", async (req, res) => {
  try {
    const questions = Array.isArray(req.body?.questions) ? req.body.questions : [];
    if (!questions.length) return res.status(400).json({ error: "questions must be a non-empty array" });

    if (MOCK) {
      const results = mockScoreBatch(questions);
      return res.json({ results, aggregate: aggregate(results) });
    }
    if (!OPENAI_API_KEY) return res.status(401).json({ error: "Missing OPENAI_API_KEY" });

    const sys = "You are a strict grader. Respond with JSON only.";
    const batches = chunk(questions, Number(process.env.BATCH_SIZE || 15));
    const allResults = [];

    for (const part of batches) {
      const usr = buildPrompt(part); // 只给这一批的问题
      const completion = await oa.chat.completions.create({
        model: OPENAI_MODEL,
        messages: [{ role: "system", content: sys }, { role: "user", content: usr }],
        temperature: 0,
        // 提高输出上限，避免被截断
        max_tokens: Number(process.env.MAX_TOKENS || 4000),
        response_format: {
          type: "json_schema",
          json_schema: { name: "score_payload", schema: SCORE_SCHEMA, strict: true }
        }
      });

      const text = completion.choices?.[0]?.message?.content?.trim() ?? "{}";
      const parsed = safeParseJSON(text);
      if (!parsed || !Array.isArray(parsed.results)) {
        return res.status(502).json({ error: "Upstream returned non-JSON or unexpected shape", raw: text });
      }
      allResults.push(...parsed.results);
    }

    return res.json({ results: allResults, aggregate: aggregate(allResults) });
  } catch (err) {
    return res.status(500).json({ error: String(err) });
  }
});


function chunk(arr, size) {
  const out = [];
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size));
  return out;
}


// ==== Start ====
app.listen(PORT, () => {
  console.log(
    `Server listening on http://localhost:${PORT} | OpenAI base=${OPENAI_BASE_URL} | model=${OPENAI_MODEL} | mock=${MOCK}`
  );
});
