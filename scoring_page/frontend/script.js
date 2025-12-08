const state = {
  raw: null,
  questions: [],
  results: null,
  aggregate: null,
  chartOverall: null,
  chartTrust: null,
  chartStudentDist: null,
  endpoint: "/api/score",
};

function setStatus(msg) {
  document.getElementById("status").textContent = msg || "";
}

// ---------------- Question extraction ----------------

function isQuestionLike(text) {
  if (!text || typeof text !== "string") return false;
  const t = text.trim();
  if (!t) return false;
  const hasQM = t.endsWith("?") || t.includes("?");
  const starters = [
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "would ",
    "will ",
    "should ",
  ];
  const lower = t.toLowerCase();
  const startsLike = starters.some((s) => lower.startsWith(s));
  return hasQM || startsLike;
}

function parseQuestionsFromExport(raw) {
  const out = [];

  if (!raw || typeof raw !== "object") return out;

  if (Array.isArray(raw.chats)) {
    // 主路径：all-chats-export 风格
    raw.chats.forEach((chat, idxChat) => {
      const meta = chat.metadata || {};
      const baseId =
        meta.studentId ||
        meta.studentName ||
        (typeof chat.title === "string" ? chat.title : `student_${idxChat + 1}`);
      const studentId = String(baseId).toLowerCase().replace(/\s+/g, "_");
      const studentName =
        meta.studentName ||
        meta.studentId ||
        (typeof chat.title === "string" ? chat.title : `Student ${idxChat + 1}`);

      const messages =
        chat.history && Array.isArray(chat.history.messages)
          ? chat.history.messages
          : [];

      messages.forEach((m, idx) => {
        if (m && m.role === "user" && typeof m.content === "string") {
          const text = m.content.trim();
          if (isQuestionLike(text)) {
            out.push({
              id: `${idxChat}_${idx}`,
              text,
              studentId,
              studentName,
            });
          }
        }
      });
    });
  } else {
    // 兜底：随便扫一遍 JSON 里的 user+content
    const visited = new Set();
    const stack = [raw];
    let idx = 0;
    while (stack.length) {
      const cur = stack.pop();
      if (!cur || typeof cur !== "object") continue;
      if (visited.has(cur)) continue;
      visited.add(cur);

      if (Array.isArray(cur)) {
        for (const v of cur) stack.push(v);
      } else {
        const role = cur.role;
        const content = cur.content;
        if (role === "user" && typeof content === "string") {
          const text = content.trim();
          if (isQuestionLike(text)) {
            out.push({
              id: `auto_${idx++}`,
              text,
              studentId: "unknown",
              studentName: "Unknown",
            });
          }
        }
        for (const k of Object.keys(cur)) {
          stack.push(cur[k]);
        }
      }
    }
  }

  // 去重：同一个学生 + 完全相同文本只保留一次
  const seen = new Set();
  const uniq = [];
  for (const q of out) {
    const key = `${q.studentId}:::${q.text}`;
    if (seen.has(key)) continue;
    seen.add(key);
    uniq.push(q);
  }
  return uniq;
}

// ---------------- Rendering: overview / habits ----------------

function renderOverview(agg) {
  const sec = document.getElementById("overview");
  sec.classList.remove("hidden");
  document.getElementById("qCount").textContent = String(agg.count ?? 0);
  document.getElementById("avgScore").textContent = String(
    agg.avg_total_0_14 ?? 0
  );
  document.getElementById("overallPct").textContent = String(
    agg.overall_0_100 ?? 0
  );
}

function renderDistribution(dist) {
  const sec = document.getElementById("chartSec");
  sec.classList.remove("hidden");
  const ctx = document.getElementById("scoreHistogram").getContext("2d");
  if (state.chartOverall) {
    state.chartOverall.destroy();
    state.chartOverall = null;
  }
  state.chartOverall = new Chart(ctx, {
    type: "bar",
    data: {
      labels: dist.labels,
      datasets: [
        {
          label: "Questions (all students)",
          data: dist.counts,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });
}

function renderHabit(agg) {
  const sec = document.getElementById("habitSec");
  sec.classList.remove("hidden");

  const ul = document.getElementById("habitList");
  ul.innerHTML = "";
  (agg.habits || []).forEach((h) => {
    const li = document.createElement("li");
    li.textContent = h;
    ul.appendChild(li);
  });

  const dimsDiv = document.getElementById("habitDims");
  dimsDiv.innerHTML = "";
  const dims = agg.dims || {};
  Object.keys(dims).forEach((k) => {
    const card = document.createElement("div");
    card.className =
      "border rounded-xl px-2 py-2 flex flex-col bg-gray-50";
    const name = document.createElement("div");
    name.className = "text-[11px] text-gray-500 mb-1";
    name.textContent = k;
    const val = document.createElement("div");
    val.className = "text-sm font-semibold";
    val.textContent = String(dims[k] ?? 0);
    card.appendChild(name);
    card.appendChild(val);
    dimsDiv.appendChild(card);
  });
}

// ---------------- Rendering: per-student summary ----------------

function renderStudentCharts(studentId) {
  const trustSec = document.getElementById("trustChartSec");
  const distSec = document.getElementById("studentDistSec");

  if (!state.results || !state.results.length) {
    trustSec.classList.add("hidden");
    distSec.classList.add("hidden");
    return;
  }

  const rows = state.results.filter(
    (r) => (r.studentId || "unknown") === studentId
  );
  if (!rows.length) {
    trustSec.classList.add("hidden");
    distSec.classList.add("hidden");
    return;
  }

  // --- 图1：ABI 总分折线（如果有 ABI） ---
  const hasAbi = rows[0].abi != null;
  if (hasAbi) {
    const labels = rows.map((_, i) => i + 1);
    const data = rows.map((r) => r.abi?.abi_total ?? 0);

    const ctx = document.getElementById("trustSeries").getContext("2d");
    if (state.chartTrust) {
      state.chartTrust.destroy();
      state.chartTrust = null;
    }
    state.chartTrust = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "ABI total (0–1)",
            data,
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            beginAtZero: true,
            suggestedMax: 1,
          },
          x: {
            title: { display: true, text: "Question index in dialogue" },
          },
        },
      },
    });
    trustSec.classList.remove("hidden");
  } else {
    if (state.chartTrust) {
      state.chartTrust.destroy();
      state.chartTrust = null;
    }
    trustSec.classList.add("hidden");
  }

  // --- 图2：该学生的分数分布直方图 ---
  const bins = [0, 0, 0, 0];
  rows.forEach((r) => {
    const t = Number(r.score_total) || 0;
    if (t <= 3) bins[0]++;
    else if (t <= 6) bins[1]++;
    else if (t <= 10) bins[2]++;
    else bins[3]++;
  });

  const ctx2 = document.getElementById("studentHistogram").getContext("2d");
  if (state.chartStudentDist) {
    state.chartStudentDist.destroy();
    state.chartStudentDist = null;
  }
  state.chartStudentDist = new Chart(ctx2, {
    type: "bar",
    data: {
      labels: ["0–3", "4–6", "7–10", "11–14"],
      datasets: [
        {
          label: "Questions (selected student)",
          data: bins,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });
  distSec.classList.remove("hidden");
}

function renderStudentSection(agg) {
  const sec = document.getElementById("studentSec");
  const per = agg.perStudent || {};
  const ids = Object.keys(per);
  if (!ids.length) {
    sec.classList.add("hidden");
    document.getElementById("trustChartSec").classList.add("hidden");
    document.getElementById("studentDistSec").classList.add("hidden");
    return;
  }
  sec.classList.remove("hidden");

  const select = document.getElementById("studentSelect");
  select.innerHTML = "";

  ids.forEach((sid, idx) => {
    const s = per[sid];
    const opt = document.createElement("option");
    opt.value = sid;
    opt.textContent = s.studentName || sid;
    if (idx === 0) opt.selected = true;
    select.appendChild(opt);
  });

  function renderFor(id) {
    const s = per[id];
    if (!s) return;
    const container = document.getElementById("studentSummary");
    container.innerHTML = "";

    const topRow = document.createElement("div");
    topRow.className =
      "grid grid-cols-2 md:grid-cols-4 gap-2 mb-2 text-xs md:text-sm";

    function smallCard(label, value) {
      const card = document.createElement("div");
      card.className =
        "border rounded-xl px-2 py-2 flex flex-col bg-gray-50";
      const l = document.createElement("div");
      l.className = "text-[11px] text-gray-500 mb-1";
      l.textContent = label;
      const v = document.createElement("div");
      v.className = "text-sm font-semibold";
      v.textContent = String(value);
      card.appendChild(l);
      card.appendChild(v);
      return card;
    }

    topRow.appendChild(
      smallCard("Questions", s.count ?? 0)
    );
    topRow.appendChild(
      smallCard("Avg Score (0–14)", s.avg_total_0_14 ?? 0)
    );
    topRow.appendChild(
      smallCard("Overall (0–100)", s.overall_0_100 ?? 0)
    );
    if (s.abi_avg) {
      topRow.appendChild(
        smallCard("ABI Total", s.abi_avg.abi_total ?? 0)
      );
    }
    container.appendChild(topRow);

    const dimsGrid = document.createElement("div");
    dimsGrid.className =
      "grid grid-cols-2 md:grid-cols-4 gap-2 text-xs md:text-sm mb-2";
    const dims = s.dims || {};
    Object.keys(dims).forEach((k) => {
      dimsGrid.appendChild(smallCard(k, dims[k]));
    });
    container.appendChild(dimsGrid);

    if (s.abi_avg) {
      const abiBlock = document.createElement("div");
      abiBlock.className = "mt-2 space-y-2";

      const title = document.createElement("div");
      title.className = "text-xs font-semibold";
      title.textContent =
        "ABI breakdown for this student (0–1) and its 12 sub-dimensions";
      abiBlock.appendChild(title);

      const abiRow = document.createElement("div");
      abiRow.className = "flex flex-wrap gap-2 text-xs";
      abiRow.appendChild(
        smallCard("Ability", s.abi_avg.ability ?? 0)
      );
      abiRow.appendChild(
        smallCard("Benevolence", s.abi_avg.benevolence ?? 0)
      );
      abiRow.appendChild(
        smallCard("Integrity", s.abi_avg.integrity ?? 0)
      );
      abiBlock.appendChild(abiRow);

      const subsGrid = document.createElement("div");
      subsGrid.className =
        "grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 text-xs";
      const subs = s.abi_avg.subs || {};
      Object.keys(subs).forEach((k) => {
        subsGrid.appendChild(smallCard(k, subs[k]));
      });
      abiBlock.appendChild(subsGrid);

      container.appendChild(abiBlock);
    }

    // 更新两张和学生相关的图
    renderStudentCharts(id);
  }

  renderFor(ids[0]);

  select.onchange = () => {
    renderFor(select.value);
  };
}

// ---------------- Per-question table ----------------

function renderTable(results) {
  const sec = document.getElementById("tableSec");
  sec.classList.remove("hidden");
  const tbody = document.getElementById("resultRows");
  tbody.innerHTML = "";

  results.forEach((r, idx) => {
    const tr = document.createElement("tr");
    tr.className = idx % 2 ? "bg-gray-50" : "";
    function td(text, alignRight = false) {
      const cell = document.createElement("td");
      cell.className =
        "px-2 py-1 whitespace-nowrap" +
        (alignRight ? " text-right" : " text-left");
      cell.textContent = text;
      return cell;
    }
    tr.appendChild(td(String(idx + 1), true));
    tr.appendChild(td(r.studentName || r.studentId || ""));
    tr.appendChild(td(r.question || ""));
    tr.appendChild(td(String(r.score_total ?? ""), true));
    tr.appendChild(td(r.verdict || ""));
    tr.appendChild(td(String(r.relevance ?? ""), true));
    tr.appendChild(td(String(r.politeness ?? ""), true));
    tr.appendChild(td(String(r.on_topic ?? ""), true));
    tr.appendChild(td(String(r.neutrality ?? ""), true));
    tr.appendChild(td(String(r.non_imperative ?? ""), true));
    tr.appendChild(td(String(r.clarity_optional ?? ""), true));
    tr.appendChild(td(String(r.privacy_minimization_optional ?? ""), true));
    tbody.appendChild(tr);
  });
}

// ---------------- Event wiring ----------------

document
  .getElementById("fileInput")
  .addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatus("Loading JSON…");
    try {
      const text = await file.text();
      state.raw = JSON.parse(text);
      state.questions = [];
      state.results = null;
      state.aggregate = null;
      setStatus("File loaded. Click Extract to find questions.");
    } catch (err) {
      console.error(err);
      setStatus("Failed to parse JSON: " + err.message);
    }
  });

document.getElementById("btnExtract").addEventListener("click", () => {
  if (!state.raw) {
    setStatus("Please upload a JSON file first.");
    return;
  }
  state.questions = parseQuestionsFromExport(state.raw);
  setStatus(
    `Extracted ${state.questions.length} question(s) from ${
      new Set(state.questions.map((q) => q.studentId)).size
    } student(s).`
  );
});

document.getElementById("btnScore").addEventListener("click", async () => {
  if (!state.questions.length) {
    setStatus("No questions found. Click Extract first.");
    return;
  }
  setStatus("Scoring via backend…");
  const useAbi = document.getElementById("applyAbi").checked;
  try {
    const resp = await fetch(state.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ questions: state.questions, useAbi }),
    });
    if (!resp.ok) {
      const t = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${t}`);
    }
    const data = await resp.json();
    state.results = data.results;
    state.aggregate = data.aggregate;

    renderOverview(data.aggregate);
    renderDistribution(data.aggregate.distribution);
    renderHabit(data.aggregate);
    renderStudentSection(data.aggregate);
    renderTable(data.results);

    setStatus("Done.");
  } catch (e) {
    console.error(e);
    setStatus("Error: " + e.message);
  }
});
