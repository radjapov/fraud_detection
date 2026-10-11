// static/compare.js — model & explanation comparison page (/compare)

document.addEventListener("DOMContentLoaded", () => {
  const $ = (id) => document.getElementById(id);
  const btnFraud = $("btn-rand-fraud"), btnNormal = $("btn-rand-normal");
  const btnCompare = $("btn-compare"), btnSbs = $("btn-sidebyside");
  const selMethod = $("sel-method"), selModel = $("sel-model");
  const status = $("cmp-status"), body = $("cmp-body");

  let row = null;      // current transaction (feature values)
  let truth = null;    // its true label, when it came from the test period
  let models = [];

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
  const ms = (v) => (v === null || v === undefined ? "–" : v < 1 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : Math.round(v).toLocaleString());

  async function getJSON(url, opts) {
    const r = await fetch(url, opts);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  }
  const post = (url, payload) => getJSON(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });

  function setBusy(on, text) {
    [btnFraud, btnNormal, btnCompare, btnSbs].forEach((b) => (b.disabled = on || ((b === btnCompare || b === btnSbs) && !row)));
    status.textContent = text || "";
  }

  // ---- transaction ----
  const SHOWN = [["amount", "Amount"], ["hour", "Hour"], ["card4", "Card network"], ["card6", "Card type"],
                 ["P_emaildomain", "Payer e-mail"], ["DeviceType", "Device"], ["sender_txn_24h", "Card txns, 24h"]];

  function showTransaction() {
    const box = $("txn-summary");
    box.classList.remove("muted");
    const chips = SHOWN.filter(([k]) => row[k] !== null && row[k] !== undefined)
      .map(([k, l]) => `<span class="txn-chip">${esc(l)}: <b>${esc(typeof row[k] === "number" ? +row[k].toFixed(2) : row[k])}</b></span>`);
    const t = truth === null ? "" : `<span class="truth ${truth ? "fraud" : "legit"}">True label: ${truth ? "fraud" : "legitimate"}</span>`;
    box.innerHTML = t + chips.join("");
  }

  async function pickRandom(fraud) {
    setBusy(true, "Loading a transaction from the test period…");
    try {
      const d = await getJSON(`/api/random?fraud=${fraud}`);
      row = d.row; truth = d.label ?? null;
      showTransaction();
      setBusy(false, "");
      compare();
    } catch (e) { setBusy(false, "Error: " + e.message); }
  }

  // ---- comparison table ----
  function probCell(r) {
    const p = r.probability, t = r.threshold;
    return `<div class="prob"><div class="prob-track">
        <div class="prob-fill ${r.fraud ? "fraud" : ""}" style="width:${(p * 100).toFixed(1)}%"></div>
        <div class="prob-thr" style="left:calc(${(t * 100).toFixed(1)}% - 1px)" title="threshold ${fmt(t)}"></div>
      </div><span class="prob-num">${fmt(p)}</span></div>`;
  }

  function decisionCell(r) {
    const pill = `<span class="pill ${r.fraud ? "fraud" : "legit"}">${r.fraud ? "Fraud" : "Legit"}</span>`;
    if (truth === null) return pill;
    const right = (r.fraud ? 1 : 0) === truth;
    return pill + `<span class="verdict">${right ? "✓ correct" : "✗ wrong"}</span>`;
  }

  function chips(list) {
    if (!list) return "–";
    return `<div class="chips">${list.map((c) => `<span class="chip ${c.contribution >= 0 ? "pos" : "neg"}" title="${esc(c.feature)} = ${esc(c.value ?? "missing")}; contribution ${c.contribution.toFixed(4)}">${esc(c.feature)}</span>`).join("")}</div>`;
  }

  function renderTable(results) {
    body.innerHTML = results.map((r) => {
      if (r.error) return `<tr><td>${esc(r.label)}</td><td colspan="6" class="err">${esc(r.error)}</td></tr>`;
      return `<tr>
        <td>${esc(r.label)}</td>
        <td class="num">${fmt(r.metrics && r.metrics.pr_auc)}</td>
        <td>${probCell(r)}</td>
        <td>${decisionCell(r)}</td>
        <td>${chips(r.explanation)}</td>
        <td class="num">${ms(r.predict_ms)}</td>
        <td class="num">${ms(r.explain_ms)}</td>
      </tr>`;
    }).join("");
  }

  async function compare() {
    if (!row) return;
    const method = selMethod.value;
    setBusy(true, method === "none" ? "Scoring with every model…" : `Scoring and explaining with ${method.toUpperCase()}…`);
    const t0 = performance.now();
    try {
      const d = await post("/api/compare", { row, explain: method, top: 5 });
      renderTable(d.results);
      setBusy(false, `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
    } catch (e) { setBusy(false, "Error: " + e.message); }
  }

  // ---- SHAP vs LIME ----
  function renderList(el, contribs, shared) {
    el.innerHTML = contribs.map((c) => `<li class="${shared.has(c.feature) ? "shared" : ""}">
      <span>${esc(c.feature)}</span><span class="${c.contribution >= 0 ? "c-pos" : "c-neg"}">${c.contribution >= 0 ? "+" : ""}${c.contribution.toFixed(4)}</span></li>`).join("");
  }

  async function sideBySide() {
    if (!row) return;
    const model = selModel.value;
    setBusy(true, `Explaining with SHAP and LIME (${model})…`);
    try {
      const [s, l] = await Promise.all([
        post("/api/explain", { row, model, method: "shap", top: 10 }),
        post("/api/explain", { row, model, method: "lime", top: 10 }),
      ]);
      const a = new Set(s.contributions.map((c) => c.feature)), b = new Set(l.contributions.map((c) => c.feature));
      const shared = new Set([...a].filter((f) => b.has(f)));
      renderList($("list-shap"), s.contributions, shared);
      renderList($("list-lime"), l.contributions, shared);
      $("t-shap").textContent = `${ms(s.ms)} ms · ${s.space}`;
      $("t-lime").textContent = `${ms(l.ms)} ms`;
      $("overlap").textContent = `Shared features in the two top-10 lists: ${shared.size} of 10`;
      setBusy(false, "");
    } catch (e) { setBusy(false, "Error: " + e.message); }
  }

  // ---- experiment tables ----
  const TABLES = [
    ["table1_models", "Model comparison (test period, chronological split, 30-day label delay)"],
    ["table2_protocols", "Same LightGBM under four evaluation protocols"],
    ["table4_latency", "Service response time"],
  ];

  function renderExperiments(d) {
    const box = $("exp-box");
    if (!d.available) { box.textContent = "No experiment results found (run experiments/make_figures.py)."; return; }
    box.classList.remove("muted");
    box.innerHTML = TABLES.filter(([k]) => d.tables[k]).map(([k, cap]) => {
      const { columns, rows } = d.tables[k];
      const cell = (v) => (v === "nan" ? "" : v);
      return `<div class="exp-caption">${esc(cap)}</div><div class="table-wrap"><table class="exp-table">
        <thead><tr>${columns.map((c) => `<th>${esc(c.replace(/_/g, " "))}</th>`).join("")}</tr></thead>
        <tbody>${rows.map((r) => `<tr>${r.map((v) => `<td>${esc(cell(v))}</td>`).join("")}</tr>`).join("")}</tbody>
      </table></div>`;
    }).join("");
  }

  // ---- init ----
  btnFraud.addEventListener("click", () => pickRandom(1));
  btnNormal.addEventListener("click", () => pickRandom(0));
  btnCompare.addEventListener("click", compare);
  btnSbs.addEventListener("click", sideBySide);
  selMethod.addEventListener("change", () => row && compare());

  getJSON("/api/models").then((d) => {
    models = d.models;
    selModel.innerHTML = models.map((m) => `<option value="${esc(m.name)}" ${m.name === "LightGBM" ? "selected" : ""}>${esc(m.label)}</option>`).join("");
  }).catch((e) => (status.textContent = "Error loading models: " + e.message));
  getJSON("/api/experiments").then(renderExperiments).catch(() => ($("exp-box").textContent = "Could not load experiment results."));
});
