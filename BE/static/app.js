// Olympics Investigator dashboard — vanilla JS, talks to the FastAPI backend.

const API = ""; // same origin (served by FastAPI)

const PIPELINE_META = {
  rag: { label: "RAG", blurb: "Vector retrieval only" },
  graphrag: { label: "GraphRAG", blurb: "Vector + graph enrichment" },
  agentic: { label: "Agentic", blurb: "Plans a multi-step investigation" },
};

const EXAMPLES = [
  "According to the provided corpus, how many biathlon events at the 2018 Winter Olympics had more than 73 competitors?",
  "Who won the gold medal in the event held at Olympic Weightlifting Gymnasium on 20 September 1988?",
  "According to the provided corpus, which athletics event at the 2008 Summer Olympics had the highest number of competitors?",
  "Who won the gold medal in the men's 20 kilometres walk athletics event at the Summer Olympics held immediately before 2016?",
];

// ---- helpers -------------------------------------------------------------

async function apiGet(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const txt = await r.text();
    throw new Error(`${r.status}: ${txt.slice(0, 200)}`);
  }
  return r.json();
}
function el(html) {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstChild;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );
}

// ---- health --------------------------------------------------------------

async function loadHealth() {
  const box = document.getElementById("health");
  try {
    const h = await apiGet("/health");
    const tg = h.tigergraph_reachable;
    box.innerHTML = "";
    box.append(
      el(`<span class="pill ${tg ? "pill-ok" : "pill-bad"}">TigerGraph: ${tg ? "connected" : "unreachable"}</span>`),
      el(`<span class="pill">LLM: ${esc(h.active_llm)}</span>`),
      el(`<span class="pill">Embed: ${esc(h.embedding_provider)}</span>`),
      el(`<span class="pill">Agent: ${esc(h.agent_impl)}</span>`)
    );
    document.getElementById("footerInfo").textContent =
      `graph=${h.graph} · retriever=${h.retriever}`;
  } catch (e) {
    box.innerHTML = `<span class="pill pill-bad">backend unreachable</span>`;
  }
}

// ---- tabs ----------------------------------------------------------------

document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".tabpane").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    document.getElementById("tab-" + t.dataset.tab).classList.add("active");
    if (t.dataset.tab === "bench") loadBench();
  });
});

// ---- examples ------------------------------------------------------------

const exSel = document.getElementById("examples");
EXAMPLES.forEach((q) => exSel.append(el(`<option value="${esc(q)}">${esc(q.slice(0, 70))}…</option>`)));
exSel.addEventListener("change", () => {
  if (exSel.value) document.getElementById("question").value = exSel.value;
});

// ---- compare / investigate ----------------------------------------------

function selectedPipelines() {
  return [...document.querySelectorAll(".checks input:checked")].map((c) => c.value);
}

function pipelineCard(name, res) {
  const meta = PIPELINE_META[name] || { label: name, blurb: "" };
  const usage = res.usage || {};
  const am = res.agentic_meta;
  const cites = res.citations || [];

  const traceHtml = (res.trace || [])
    .map(
      (s) => `<div class="trace-step">
        <div class="act">${esc(s.action)}</div>
        ${s.detail ? `<div class="det">${esc(s.detail)}</div>` : ""}
        ${s.result ? `<div class="res">${esc(s.result)}</div>` : ""}
      </div>`
    )
    .join("");

  const agenticChips = am
    ? `<span class="chip">steps <b>${am.steps}</b></span>
       <span class="chip">tools <b>${esc((am.distinct_tools || []).join(", ") || "—")}</b></span>
       <span class="chip">stop <b>${esc(am.stop_reason || "")}</b></span>`
    : "";

  return el(`<div class="card ${name}">
    <h3>${meta.label}</h3>
    <p class="blurb">${meta.blurb}</p>
    <div class="answer">${esc(res.answer || "—")}</div>
    <div class="meta-row">
      <span class="chip">tokens <b>${usage.total_tokens ?? 0}</b></span>
      <span class="chip">prompt <b>${usage.prompt_tokens ?? 0}</b></span>
      <span class="chip">completion <b>${usage.completion_tokens ?? 0}</b></span>
      ${agenticChips}
    </div>
    ${cites.length ? `<div class="cites"><b>Citations (${cites.length}):</b> ${esc(cites.slice(0, 10).join(", "))}${cites.length > 10 ? " …" : ""}</div>` : ""}
    <details>
      <summary>Investigation trace (${(res.trace || []).length} steps)</summary>
      ${traceHtml || '<div class="det">no trace</div>'}
    </details>
  </div>`);
}

function renderTokenChart(results) {
  const chart = document.getElementById("tokenchart");
  const bars = document.getElementById("tokenbars");
  bars.innerHTML = "";
  const entries = Object.entries(results).map(([n, r]) => [n, (r.usage || {}).total_tokens || 0]);
  const max = Math.max(1, ...entries.map(([, v]) => v));
  entries.forEach(([n, v]) => {
    bars.append(
      el(`<div class="bar-row">
        <span class="name">${PIPELINE_META[n]?.label || n}</span>
        <span class="bar-track"><span class="bar-fill ${n}" style="width:${(v / max) * 100}%"></span></span>
        <span class="val">${v}</span>
      </div>`)
    );
  });
  chart.classList.remove("hidden");
}

document.getElementById("investigate").addEventListener("click", async () => {
  const question = document.getElementById("question").value.trim();
  const pipelines = selectedPipelines();
  const resultsEl = document.getElementById("results");
  const btn = document.getElementById("investigate");

  if (!question) return;
  if (!pipelines.length) {
    resultsEl.innerHTML = `<div class="error-box">Select at least one pipeline.</div>`;
    return;
  }

  document.getElementById("tokenchart").classList.add("hidden");
  resultsEl.innerHTML = `<div class="loading"><span class="spinner"></span> Running ${pipelines.join(", ")} against TigerGraph…</div>`;
  btn.disabled = true;

  try {
    const agentImpl = document.getElementById("agentImpl").value || undefined;
    const data = await apiPost("/compare", { question, pipelines, agent_impl: agentImpl });
    const results = data.results || {};
    resultsEl.innerHTML = "";
    // preserve pipeline order
    ["rag", "graphrag", "agentic"].forEach((n) => {
      if (results[n]) resultsEl.append(pipelineCard(n, results[n]));
    });
    renderTokenChart(results);
  } catch (e) {
    resultsEl.innerHTML = `<div class="error-box">Request failed: ${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false;
  }
});

// ---- benchmark -----------------------------------------------------------

function bestPerMetric(summary, metric, higherIsBetter = true) {
  let best = null, bestVal = higherIsBetter ? -Infinity : Infinity;
  for (const [p, s] of Object.entries(summary)) {
    const v = s[metric];
    if (higherIsBetter ? v > bestVal : v < bestVal) { bestVal = v; best = p; }
  }
  return best;
}

function renderBench(data) {
  const summaryEl = document.getElementById("benchSummary");
  const byTypeEl = document.getElementById("benchByType");
  const detailEl = document.getElementById("benchDetail");
  summaryEl.innerHTML = byTypeEl.innerHTML = detailEl.innerHTML = "";

  if (!data || !data.summary) {
    summaryEl.innerHTML = `<div class="error-box">No benchmark results yet. Run a benchmark.</div>`;
    return;
  }
  const summary = data.summary;

  const bestAcc = bestPerMetric(summary, "accuracy", true);
  const bestTok = bestPerMetric(summary, "avg_tokens", false);

  let rows = "";
  for (const [p, s] of Object.entries(summary)) {
    rows += `<tr>
      <td>${PIPELINE_META[p]?.label || p}</td>
      <td class="num ${p === bestAcc ? "cell-best" : ""}">${(s.accuracy * 100).toFixed(0)}%</td>
      <td class="num">${(s.completeness * 100).toFixed(0)}%</td>
      <td class="num ${p === bestTok ? "cell-best" : ""}">${s.avg_tokens}</td>
      <td class="num">${s.avg_seconds ?? "—"}</td>
    </tr>`;
  }
  summaryEl.append(el(`<div>
    <h3 class="section-title">Summary (n=${data.n_questions ?? "?"})</h3>
    <table>
      <thead><tr><th>Pipeline</th><th>Accuracy</th><th>Completeness</th><th>Avg tokens</th><th>Avg sec</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`));

  // accuracy by qtype
  const qtypes = [...new Set(Object.values(summary).flatMap((s) => Object.keys(s.by_qtype || {})))].sort();
  if (qtypes.length) {
    let head = `<th>Pipeline</th>` + qtypes.map((q) => `<th>${esc(q)}</th>`).join("");
    let body = "";
    for (const [p, s] of Object.entries(summary)) {
      body += `<tr><td>${PIPELINE_META[p]?.label || p}</td>` +
        qtypes.map((q) => {
          const v = (s.by_qtype || {})[q];
          return `<td class="num">${v == null ? "—" : (v * 100).toFixed(0) + "%"}</td>`;
        }).join("") + `</tr>`;
    }
    byTypeEl.append(el(`<div>
      <h3 class="section-title">Accuracy by question type</h3>
      <table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>
    </div>`));
  }

  // per-question detail
  const pq = data.per_question || [];
  if (pq.length) {
    const cols = Object.keys(pq[0]);
    const head = cols.map((c) => `<th>${esc(c)}</th>`).join("");
    const body = pq.slice(0, 100).map((row) =>
      `<tr>${cols.map((c) => `<td>${esc(typeof row[c] === "object" ? JSON.stringify(row[c]) : row[c])}</td>`).join("")}</tr>`
    ).join("");
    detailEl.append(el(`<details>
      <summary>Per-question detail (${pq.length})</summary>
      <table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>
    </details>`));
  }
}

async function loadBench() {
  const summaryEl = document.getElementById("benchSummary");
  summaryEl.innerHTML = `<div class="loading"><span class="spinner"></span> Loading latest results…</div>`;
  try {
    const data = await apiGet("/benchmark/latest");
    if (!data.available) {
      summaryEl.innerHTML = `<div class="error-box">${esc(data.message || "No results yet.")}</div>`;
      return;
    }
    renderBench(data);
  } catch (e) {
    summaryEl.innerHTML = `<div class="error-box">Could not load: ${esc(e.message)}</div>`;
  }
}

document.getElementById("loadBench").addEventListener("click", loadBench);
document.getElementById("runBench").addEventListener("click", async () => {
  const summaryEl = document.getElementById("benchSummary");
  const limit = parseInt(document.getElementById("benchLimit").value || "0", 10);
  summaryEl.innerHTML = `<div class="loading"><span class="spinner"></span> Running benchmark (limit=${limit})… this can take a while.</div>`;
  try {
    const data = await apiPost("/benchmark", { limit, pipelines: ["rag", "graphrag", "agentic"] });
    renderBench(data);
  } catch (e) {
    summaryEl.innerHTML = `<div class="error-box">Benchmark failed: ${esc(e.message)}</div>`;
  }
});

// ---- init ----------------------------------------------------------------
loadHealth();
