const state = {
  raw: null,
  questions: [],
  results: null,
  aggregate: null,
  chart: null,
  endpoint: "http://localhost:8787/api/score"
};

function setStatus(msg) {
  document.getElementById("status").textContent = msg || "";
}

function isQuestionLike(text) {
  if (!text || typeof text !== "string") return false;
  const t = text.trim();
  if (t.length < 2) return false;
  const qm = t.includes("?");
  const starts = /^(what|why|how|can|could|would|should|is|are|do|does|did|may|might|when|where|which|who|whom|是否|能否|可以|为什么)/i.test(t);
  return qm || starts;
}

function traverse(obj, collector) {
  const visited = new Set();
  const stack = [obj];
  while (stack.length) {
    const cur = stack.pop();
    if (cur && typeof cur === "object") {
      if (visited.has(cur)) continue;
      visited.add(cur);
      if (Array.isArray(cur)) {
        for (const v of cur) stack.push(v);
      } else {
        const role = cur.role;
        const content = cur.content;
        if (role === "user" && typeof content === "string" && isQuestionLike(content)) {
          collector.push(content.trim());
        }
        for (const k of Object.keys(cur)) stack.push(cur[k]);
      }
    }
  }
}

function extractQuestionsFromJSON(raw) {
  const list = [];
  traverse(raw, list);
  const uniq = Array.from(new Set(list));
  return uniq;
}

function renderOverview(agg) {
  document.getElementById("overview").classList.remove("hidden");
  document.getElementById("qCount").textContent = String(agg.count);
  document.getElementById("avgScore").textContent = String(agg.avg_total_0_14);
  document.getElementById("overallPct").textContent = String(agg.overall_0_100);
}

function renderDistribution(dist) {
  document.getElementById("chartSec").classList.remove("hidden");
  const ctx = document.getElementById("distChart");
  if (state.chart) { state.chart.destroy(); }
  state.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: Object.keys(dist),
      datasets: [{
        label: "Count",
        data: Object.values(dist)
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 } }
      }
    }
  });
}

function renderHabit(agg) {
  document.getElementById("habitSec").classList.remove("hidden");
  const ul = document.getElementById("habitList");
  ul.innerHTML = "";
  (agg.habit_feedback || []).forEach(t => {
    const li = document.createElement("li");
    li.textContent = t;
    ul.appendChild(li);
  });
  const dim = document.getElementById("dimAvgs");
  dim.innerHTML = "";
  for (const [k,v] of Object.entries(agg.per_dimension_avg || {})) {
    const card = document.createElement("div");
    card.className = "rounded-xl border p-3";
    card.innerHTML = `<div class="text-gray-500">${k}</div><div class="font-semibold">${v}</div>`;
    dim.appendChild(card);
  }
}

function renderTable(results) {
  document.getElementById("tableSec").classList.remove("hidden");
  const tbody = document.getElementById("resultRows");
  tbody.innerHTML = "";
  results.forEach((r, idx) => {
    const tr = document.createElement("tr");
    tr.className = "border-b align-top";
    function td(t) {
      const el = document.createElement("td");
      el.className = "py-2 pr-4";
      el.textContent = t;
      return el;
    }
    tr.appendChild(td(String(idx+1)));
    tr.appendChild(td(r.question || ""));
    tr.appendChild(td(String(r.score_total ?? "")));
    tr.appendChild(td(r.verdict || ""));
    tr.appendChild(td(String(r.relevance ?? "")));
    tr.appendChild(td(String(r.politeness ?? "")));
    tr.appendChild(td(String(r.on_topic ?? "")));
    tr.appendChild(td(String(r.neutrality ?? "")));
    tr.appendChild(td(String(r.non_imperative ?? "")));
    tr.appendChild(td(String(r.clarity_optional ?? "")));
    tr.appendChild(td(String(r.privacy_minimization_optional ?? "")));
    tbody.appendChild(tr);
  });
}

document.getElementById("btnExtract").addEventListener("click", async () => {
  const file = document.getElementById("fileInput").files[0];
  if (!file) { setStatus("Please choose a JSON file."); return; }
  try {
    const txt = await file.text();
    state.raw = JSON.parse(txt);
  } catch (e) {
    setStatus("Invalid JSON file.");
    return;
  }
  state.questions = extractQuestionsFromJSON(state.raw);
  setStatus(`Extracted ${state.questions.length} question(s).`);
});

document.getElementById("btnScore").addEventListener("click", async () => {
  if (!state.questions.length) { setStatus("No questions found. Click Extract first."); return; }
  setStatus("Scoring via backend…");
  try {
    const resp = await fetch(state.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ questions: state.questions })
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
    renderTable(data.results);
    setStatus("Done.");
  } catch (e) {
    console.error(e);
    setStatus("Error: " + e.message);
  }
});
