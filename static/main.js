// static/main.js
// Frontend for IEEE fraud demo: form, predict, random, SHAP horizontal bar chart, history toggle + metrics

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("feature-form");
  const btnRandomNormal = document.getElementById("btn-random-normal");
  const btnRandomFraud = document.getElementById("btn-random-fraud");
  const btnClear = document.getElementById("btn-clear");
  const btnPredict = document.getElementById("btn-predict");
  const btnShap = document.getElementById("btn-shap");
  const btnToggleHistory = document.getElementById("btn-toggle-history");

  const predLabel = document.getElementById("pred-label");
  const predProb = document.getElementById("pred-prob");
  const predThreshold = document.getElementById("pred-threshold");

  // SHAP
  const shapBox = document.getElementById("shap-box"); // container div
  const shapMessage = document.getElementById("shap-message");
  const shapCanvas = document.getElementById("shap-chart");
  const shapList = document.getElementById("shap-list");

  // History
  const historyContainer = document.getElementById("history-container");
  const historyList = document.getElementById("history-list");

  // Metrics
  const metricsBox = document.getElementById("metrics-box");
  const metricsMessage = document.getElementById("metrics-message");
  const mSamples = document.getElementById("m-samples");
  const mFraudRate = document.getElementById("m-fraud-rate");
  const mRocAuc = document.getElementById("m-roc-auc");
  const mPrAuc = document.getElementById("m-pr-auc");
  const mAcc = document.getElementById("m-acc");
  const mF1 = document.getElementById("m-f1");
  const mTN = document.getElementById("m-tn");
  const mFP = document.getElementById("m-fp");
  const mFN = document.getElementById("m-fn");
  const mTP = document.getElementById("m-tp");
  const mThr = document.getElementById("m-threshold");

  let lastPayload = null;
  let lastPrediction = null;

  // ---- helpers ----
  function getFormPayload() {
    const ids = [
      "amount",
      "hour",
      "card_age_months",
      "sender_txn_24h",
      "sender_avg_amount",
      "distance_km",
      "ip_risk",
      "mcc",
      "country_risk",
      "receiver_new",
      "device_new",
      "is_foreign",
    ];
    const payload = {};
    ids.forEach((id) => {
      const el = document.getElementById(id);
      if (!el) return;
      const val = el.value;

      if (val === "" || val === null || typeof val === "undefined") {
        return; // skip empty
      }

      if (el.type === "number") {
        const num = Number(val);
        if (!Number.isNaN(num)) {
          payload[id] = num;
        }
      } else {
        // select
        const num = Number(val);
        payload[id] = Number.isNaN(num) ? val : num;
      }
    });
    return payload;
  }

  function fillFormFromRow(row) {
    if (!row || typeof row !== "object") return;
    Object.entries(row).forEach(([key, value]) => {
      const el = document.getElementById(key);
      if (!el) return;
      el.value = value;
    });
  }

  function clearForm() {
    Array.from(form.elements).forEach((el) => {
      if (el.tagName === "INPUT" || el.tagName === "SELECT") {
        el.value = "";
      }
    });
    predLabel.textContent = "No prediction yet";
    predLabel.className = "pred-label";
    predProb.textContent = "–";
    predThreshold.textContent = "–";
    shapMessage.textContent = "Press \"Explain (SHAP)\" after prediction.";
    shapBox?.classList.remove("loading");
    clearShapChart();
    clearShapList();
  }

  function setPrediction(prob, threshold) {
    const pct = (prob * 100).toFixed(1) + " %";
    predProb.textContent = pct;
    predThreshold.textContent = threshold.toFixed(3);

    const isFraud = prob >= threshold;
    predLabel.textContent = isFraud ? "High fraud risk" : "Low fraud risk";
    // ВАЖНО: классы синхронизированы с CSS (.pred-label.ok / .pred-label.fraud)
    predLabel.className = "pred-label " + (isFraud ? "fraud" : "ok");
  }

  function clearShapChart() {
    if (!shapCanvas) return;
    const ctx = shapCanvas.getContext("2d");
    ctx.clearRect(0, 0, shapCanvas.width, shapCanvas.height);
  }

  function clearShapList() {
    if (!shapList) return;
    shapList.innerHTML = "";
  }

  function renderHistory(items) {
    historyList.innerHTML = "";
    if (!Array.isArray(items) || items.length === 0) {
      const li = document.createElement("li");
      li.className = "history-item empty";
      li.textContent = "No history yet.";
      historyList.appendChild(li);
      return;
    }
    items.forEach((h) => {
      const li = document.createElement("li");
      li.className = "history-item";
      const ts = h.ts || "";
      const res = h.result || {};
      const p = Array.isArray(res.probs) ? res.probs[0] : null;
      const label = Array.isArray(res.labels) ? res.labels[0] : null;

      const line1 = document.createElement("div");
      line1.className = "history-main";
      line1.textContent = ts + " · " + (p != null ? (p * 100).toFixed(1) + "%" : "?");

      const line2 = document.createElement("div");
      line2.className = "history-sub";
      line2.textContent = label === 1 ? "fraud" : label === 0 ? "not fraud" : "";

      li.appendChild(line1);
      li.appendChild(line2);
      historyList.appendChild(li);
    });
  }

  function fetchHistory() {
    fetch("/api/history")
      .then((r) => r.json())
      .then((data) => {
        renderHistory(data);
      })
      .catch((err) => {
        console.error("history error:", err);
      });
  }

  // ---- SHAP horizontal bar chart ----
  function renderShapChart(shapData) {
    clearShapChart();
    clearShapList();
    shapBox?.classList.remove("loading");

    if (!shapData || !Array.isArray(shapData.shap) || shapData.shap.length === 0) {
      shapMessage.textContent = "No SHAP data.";
      return;
    }

    shapMessage.textContent = "";

    const items = shapData.shap
      .filter((it) => typeof it.shap === "number" && Number.isFinite(it.shap))
      .slice();

    if (items.length === 0) {
      shapMessage.textContent = "No SHAP data.";
      return;
    }

    // sort by |shap| desc and take top-8
    items.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
    const top = items.slice(0, 8);

    const ctx = shapCanvas.getContext("2d");
    const W = shapCanvas.width;
    const H = shapCanvas.height;

    ctx.clearRect(0, 0, W, H);

    const paddingLeft = 100;
    const paddingRight = 40;
    const paddingTop = 20;
    const paddingBottom = 20;

    const zeroX = Math.round((paddingLeft + (W - paddingRight)) / 2); // center line
    const maxBarWidth = (W - paddingLeft - paddingRight) / 2;

    const barAreaHeight = H - paddingTop - paddingBottom;
    const n = top.length;
    const rowHeight = barAreaHeight / n;
    const barThickness = Math.min(24, rowHeight * 0.6);

    const maxAbs = top.reduce(
      (acc, it) => Math.max(acc, Math.abs(it.shap)),
      1e-9
    );

    // background
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    ctx.fillRect(0, 0, W, H);

    // zero axis
    ctx.beginPath();
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 1;
    ctx.moveTo(zeroX, paddingTop - 4);
    ctx.lineTo(zeroX, H - paddingBottom + 4);
    ctx.stroke();

    ctx.font = "12px system-ui, -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textBaseline = "middle";

    top.forEach((item, idx) => {
      const value = item.shap;
      const feature = item.feature;
      const yCenter = paddingTop + rowHeight * idx + rowHeight / 2;

      const fraction = value / maxAbs;
      const barLen = fraction * maxBarWidth;

      const isPositive = value >= 0;
      const barColor = isPositive ? "#ff5370" : "#82aaff"; // red / blue neo-style

      const xStart = barLen >= 0 ? zeroX : zeroX + barLen;
      const width = Math.abs(barLen);
      const yTop = yCenter - barThickness / 2;

      // subtle line under bar
      ctx.beginPath();
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.moveTo(paddingLeft - 10, yCenter + barThickness / 2 + 3);
      ctx.lineTo(W - paddingRight + 10, yCenter + barThickness / 2 + 3);
      ctx.stroke();

      // bar
      ctx.fillStyle = barColor;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(xStart, yTop, width || 1, barThickness, 6);
      } else {
        ctx.rect(xStart, yTop, width || 1, barThickness);
      }
      ctx.fill();

      // feature name (left)
      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.textAlign = "right";
      ctx.fillText(feature, paddingLeft - 12, yCenter);

      // SHAP value near bar
      ctx.textAlign = isPositive ? "left" : "right";
      const textX = isPositive ? xStart + width + 6 : xStart - 6;
      ctx.fillStyle = "rgba(255,255,255,0.8)";
      ctx.fillText(value.toFixed(3), textX, yCenter);
    });

    // compact list under chart
    top.forEach((item) => {
      const li = document.createElement("li");
      li.className = "shap-item";

      const left = document.createElement("span");
      left.className = "shap-item-feature";
      left.textContent = item.feature;

      const mid = document.createElement("span");
      mid.className = "shap-item-value " + (item.shap >= 0 ? "pos" : "neg");
      mid.textContent = item.shap.toFixed(3);

      const right = document.createElement("span");
      right.className = "shap-item-input";
      right.textContent =
        item.value !== null && item.value !== undefined ? String(item.value) : "–";

      li.appendChild(left);
      li.appendChild(mid);
      li.appendChild(right);
      shapList.appendChild(li);
    });
  }

// ---- metrics ----
function renderMetrics(m) {
  if (!m) {
    metricsMessage.textContent = "Metrics not available.";
    return;
  }

  metricsMessage.textContent = "";

  const data = m.data || m || {};
  const global = m.global_metrics || m.global || {};
  const thr = m.threshold_metrics || {};
  const cmRaw = thr.confusion_matrix || null;

  const samples = data.samples ?? null;
  const fraudRate = data.fraud_rate ?? null;

  const rocAuc = global.roc_auc ?? null;
  const prAuc = global.pr_auc ?? null;

  const acc = thr.accuracy ?? null;
  const f1 = thr.f1 ?? null;
  const threshold = thr.threshold ?? null;

  let tn = null;
  let fp = null;
  let fn = null;
  let tp = null;

  // 1) [[TN, FP], [FN, TP]]
  // 2) {TN: ..., FP: ..., FN: ..., TP: ...}
  if (cmRaw) {
    if (Array.isArray(cmRaw)) {
      tn = cmRaw[0]?.[0] ?? null;
      fp = cmRaw[0]?.[1] ?? null;
      fn = cmRaw[1]?.[0] ?? null;
      tp = cmRaw[1]?.[1] ?? null;
    } else if (typeof cmRaw === "object") {
      tn = cmRaw.TN ?? cmRaw.tn ?? null;
      fp = cmRaw.FP ?? cmRaw.fp ?? null;
      fn = cmRaw.FN ?? cmRaw.fn ?? null;
      tp = cmRaw.TP ?? cmRaw.tp ?? null;
    }
  }

  mSamples.textContent =
    samples != null ? samples.toLocaleString("en-US") : "–";
  mFraudRate.textContent =
    fraudRate != null ? (fraudRate * 100).toFixed(2) + " %" : "–";

  mRocAuc.textContent = rocAuc != null ? rocAuc.toFixed(4) : "–";
  mPrAuc.textContent = prAuc != null ? prAuc.toFixed(4) : "–";

  mAcc.textContent = acc != null ? acc.toFixed(4) : "–";
  mF1.textContent = f1 != null ? f1.toFixed(4) : "–";

  mTN.textContent = "TN: " + (tn != null ? tn.toLocaleString("en-US") : "–");
  mFP.textContent = "FP: " + (fp != null ? fp.toLocaleString("en-US") : "–");
  mFN.textContent = "FN: " + (fn != null ? fn.toLocaleString("en-US") : "–");
  mTP.textContent = "TP: " + (tp != null ? tp.toLocaleString("en-US") : "–");

  mThr.textContent = threshold != null ? threshold.toFixed(3) : "–";
}

function fetchMetrics() {
  if (!metricsBox) return;
  fetch("/api/metrics")
    .then((r) => r.json())
    .then((data) => {
      // поддерживаем оба варианта:
      // 1) {metrics: {...}}
      // 2) {...} сразу
      const m = data && (data.metrics ?? data);
      renderMetrics(m);
    })
    .catch((err) => {
      console.error("metrics error:", err);
      metricsMessage.textContent = "Metrics load failed.";
    });
}
  // ---- events ----

  btnRandomNormal?.addEventListener("click", () => {
    fetch("/api/random?fraud=0")
      .then((r) => r.json())
      .then((data) => {
        if (data && data.row) {
          fillFormFromRow(data.row);
        }
      })
      .catch((err) => console.error("random normal error:", err));
  });

  btnRandomFraud?.addEventListener("click", () => {
    fetch("/api/random?fraud=1")
      .then((r) => r.json())
      .then((data) => {
        if (data && data.row) {
          fillFormFromRow(data.row);
        }
      })
      .catch((err) => console.error("random fraud error:", err));
  });

  btnClear?.addEventListener("click", () => {
    clearForm();
  });

  btnPredict?.addEventListener("click", () => {
    const payload = getFormPayload();
    lastPayload = payload;

    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          console.error("Predict error:", data);
          predLabel.textContent = "Predict error";
          predLabel.className = "pred-label error";
          predProb.textContent = "–";
          return;
        }
        const prob = Array.isArray(data.probs) ? data.probs[0] : null;
        const thr = data.threshold ?? 0.5;
        lastPrediction = { prob, thr };
        if (prob == null) {
          predLabel.textContent = "No probability returned";
          predLabel.className = "pred-label error";
          predProb.textContent = "–";
          return;
        }
        setPrediction(prob, thr);
        fetchHistory();
      })
      .catch((err) => {
        console.error("Predict failed:", err);
        predLabel.textContent = "Predict failed";
        predLabel.className = "pred-label error";
        predProb.textContent = "–";
      });
  });

  btnShap?.addEventListener("click", () => {
    const payload = lastPayload || getFormPayload();
    if (!payload || Object.keys(payload).length === 0) {
      shapMessage.textContent = "Fill the form or click Random first.";
      return;
    }

    shapMessage.textContent = "Calculating SHAP...";
    shapBox?.classList.add("loading");
    clearShapChart();
    clearShapList();

    fetch("/api/shap", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.error) {
          console.error("SHAP failed:", data);
          shapBox?.classList.remove("loading");
          shapMessage.textContent = "SHAP failed: " + (data.message || data.error);
          return;
        }
        renderShapChart(data);
      })
      .catch((err) => {
        console.error("SHAP failed:", err);
        shapBox?.classList.remove("loading");
        shapMessage.textContent = "SHAP failed: " + err;
      });
  });

  btnToggleHistory?.addEventListener("click", () => {
    historyContainer.classList.toggle("hidden");
    if (!historyContainer.classList.contains("hidden")) {
      fetchHistory();
    }
  });

  // initial loads
  fetchHistory();
  fetchMetrics();
});