/**
 * script.js — IPO Success Predictor (Full Upgrade)
 * Handles: IPO name lookup, auto-fill, prediction API, result rendering,
 *          donut chart, dual probability bars, insights, action buttons.
 */

// ── Chart instances ──────────────────────────────────────────────────────────
let subscriptionChart = null;
let donutChart = null;

// ── State ────────────────────────────────────────────────────────────────────
let currentIPOName = "";
let manualMode = false;
let suggestTimeout = null;

// ── IPO Name Lookup ──────────────────────────────────────────────────────────
const ipoNameInput = document.getElementById("ipo_name_input");
if (ipoNameInput) {
  ipoNameInput.addEventListener("input", () => {
    clearTimeout(suggestTimeout);
    suggestTimeout = setTimeout(fetchSuggestions, 300);
  });
  ipoNameInput.addEventListener("blur", () => {
    setTimeout(() => {
      const sug = document.getElementById("ipo-suggestions");
      if (sug) sug.classList.remove("open");
    }, 200);
  });
}

async function fetchSuggestions() {
  const input = document.getElementById("ipo_name_input");
  if (!input) return;
  const q = input.value.trim();
  if (q.length < 2) {
    const sug = document.getElementById("ipo-suggestions");
    if (sug) sug.classList.remove("open");
    return;
  }
  try {
    const res  = await fetch(`/ipo/lookup?name=${encodeURIComponent(q)}`);
    const data = await res.json();
    renderSuggestions(data);
  } catch (e) {
    // silent
  }
}

function renderSuggestions(items) {
  const box = document.getElementById("ipo-suggestions");
  if (!box) return;
  if (!items || items.length === 0) {
    box.classList.remove("open");
    showAutofillStatus("IPO not found in database. Please enter details manually below.", "not-found");
    return;
  }
  box.innerHTML = items.map(ipo => `
    <div class="suggestion-item" onclick="selectIPO(${ipo.id})">
      <div>
        <div class="sug-name">${ipo.name}</div>
        <div class="sug-meta">${ipo.sector} · ₹${ipo.offer_price} · Sub: ${ipo.total_sub}×</div>
      </div>
      <div class="prob-badge ${probClass(ipo.success_prob)}">${(ipo.success_prob*100).toFixed(1)}%</div>
    </div>
  `).join("");
  box.classList.add("open");
}

function probClass(p) {
  if (p >= 0.7) return "prob-high";
  if (p >= 0.45) return "prob-mid";
  return "prob-low";
}

async function selectIPO(id) {
  const box = document.getElementById("ipo-suggestions");
  if (box) box.classList.remove("open");
  try {
    const res  = await fetch(`/listings/${id}`);
    const ipo  = await res.json();
    autofillFields(ipo);
    currentIPOName = ipo.name;
  } catch (e) {
    showAutofillStatus("Could not load IPO data. Please enter manually.", "not-found");
  }
}

function autofillFields(ipo) {
  setField("Issue_Size",  ipo.issue_size);
  setField("Offer_Price", ipo.offer_price);
  setField("QIB",         ipo.qib);
  setField("HNI",         ipo.hni);
  setField("RII",         ipo.rii);
  setField("Total",       ipo.total_sub);
  setField("quick_price", ipo.offer_price);

  const nameEl = document.getElementById("ipo_name_input");
  if (nameEl) nameEl.value = ipo.name;

  showAutofillStatus(`✅ Auto-filled data for "${ipo.name}" (${ipo.sector}) — you can edit any field before predicting.`, "filled");
  showDetailsCard();
}

function setField(id, val) {
  const el = document.getElementById(id);
  if (el && val !== null && val !== undefined) el.value = val;
}

function showAutofillStatus(msg, type) {
  const el = document.getElementById("autofill-status");
  if (!el) return;
  el.textContent = msg;
  el.className = `autofill-msg ${type}`;
  el.style.display = "block";
}

function showDetailsCard() {
  const card = document.getElementById("details-card");
  if (card) card.style.display = "block";
}

function toggleManualEntry() {
  manualMode = !manualMode;
  const card = document.getElementById("details-card");
  const btn  = document.getElementById("manual-toggle-btn");
  if (!card) return;
  if (manualMode) {
    card.style.display = "block";
    if (btn) btn.textContent = "🔼 Hide Manual Entry";
  } else {
    card.style.display = "none";
    if (btn) btn.textContent = "✏️ Enter Details Manually";
  }
}

// ── Main predict function ────────────────────────────────────────────────────
async function predictIPO() {
  const btn      = document.getElementById("predict-btn");
  const errorBox = document.getElementById("error-msg");
  const resultCard = document.getElementById("result-card");

  if (errorBox) { errorBox.style.display = "none"; errorBox.textContent = ""; }
  if (resultCard) resultCard.style.display = "none";

  // Sync quick_price to Offer_Price if name was set
  const qp = document.getElementById("quick_price");
  const op = document.getElementById("Offer_Price");
  if (qp && op && qp.value && !op.value) op.value = qp.value;

  const fieldNames = ["Issue_Size", "Offer_Price", "QIB", "HNI", "RII", "Total"];
  const payload = {};

  for (const field of fieldNames) {
    const val = parseFloat(document.getElementById(field)?.value || "");
    if (isNaN(val)) {
      showError(`Please fill in a valid number for "${field.replace("_", " ")}".`);
      return;
    }
    payload[field] = val;
  }

  // Add IPO name for history saving
  const nameEl = document.getElementById("ipo_name_input");
  payload.ipo_name = (nameEl && nameEl.value.trim()) || currentIPOName || "Manual Entry";

  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="loader"></span>Predicting...'; }

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || data.error) throw new Error(data.error || "Server error");
    renderResult(data, payload);
  } catch (err) {
    showError("Prediction failed: " + err.message);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = "🔍 Predict IPO Success"; }
  }
}

// ── Render result ────────────────────────────────────────────────────────────
function renderResult(data, payload) {
  const isSuccess  = data.prediction === 1;
  const probPct    = (data.success_probability * 100).toFixed(1);
  const failPct    = (data.failure_probability * 100).toFixed(1);

  // Icon & label
  setElem("result-icon",  isSuccess ? "🚀" : "📉");
  setClass("result-icon", "result-icon " + (isSuccess ? "success" : "fail"));
  setElem("result-label", isSuccess ? "IPO SUCCESS" : "IPO LIKELY TO FAIL");
  setClass("result-label", "result-label " + (isSuccess ? "success" : "fail"));
  setElem("result-sub",
    isSuccess
      ? "The model predicts this IPO is likely to list at a gain."
      : "The model predicts this IPO may list below offer price.");

  // Probabilities
  setElem("prob-value", probPct + "%");
  setElem("fail-value", failPct + "%");

  const fill     = document.getElementById("progress-fill");
  const failFill = document.getElementById("progress-fail");
  if (fill)     { fill.style.width = "0%";    setTimeout(() => fill.style.width     = probPct + "%", 60); }
  if (failFill) { failFill.style.width = "0%"; setTimeout(() => failFill.style.width = failPct + "%", 80); }

  // IPO Score
  const score = Math.round(data.success_probability * 100);
  setElem("ipo-score", score);

  // Risk Level
  const badge = document.getElementById("risk-badge");
  if (badge) {
    const r = data.risk_level || "Medium";
    badge.textContent = r + " Risk";
    badge.className   = "risk-badge " + r.toLowerCase();
  }
  setElem("risk-desc", riskDesc(data.risk_level));

  // Confidence
  const conf = document.getElementById("conf-badge");
  if (conf) conf.textContent = data.confidence || "—";
  setElem("conf-sub", "Prediction confidence");

  // Recommendation
  setElem("rec-text", data.recommendation || "—");
  setElem("rec-icon",
    data.recommendation && data.recommendation.toLowerCase().includes("buy") ? "✅" :
    data.recommendation && data.recommendation.toLowerCase().includes("avoid") ? "🚫" : "📌");

  // Insights
  const ul = document.getElementById("insights-list");
  if (ul && data.insights) {
    ul.innerHTML = data.insights.map(ins =>
      `<li class="ins-${ins.type}">${ins.icon} ${ins.text}</li>`
    ).join("");
  }

  // Charts
  drawDonut(data.success_probability, data.failure_probability);
  drawSubChart(payload);

  // Show result
  const resultCard = document.getElementById("result-card");
  if (resultCard) {
    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

// ── Draw donut chart ─────────────────────────────────────────────────────────
function drawDonut(successProb, failProb) {
  const ctx = document.getElementById("donutChart");
  if (!ctx) return;
  if (donutChart) donutChart.destroy();
  donutChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Success", "Failure"],
      datasets: [{
        data: [
          parseFloat((successProb * 100).toFixed(1)),
          parseFloat((failProb * 100).toFixed(1))
        ],
        backgroundColor: ["rgba(0,212,170,0.8)", "rgba(255,77,109,0.8)"],
        borderColor:      ["#00d4aa", "#ff4d6d"],
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: { position: "bottom", labels: { color: "#8b90b8", font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.toFixed(1)}%` } },
      }
    }
  });
}

// ── Draw subscription bar chart ──────────────────────────────────────────────
function drawSubChart(payload) {
  const ctx = document.getElementById("subscriptionChart");
  if (!ctx) return;
  if (subscriptionChart) subscriptionChart.destroy();
  subscriptionChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["QIB", "HNI", "RII", "Total"],
      datasets: [{
        label: "Subscription (×)",
        data: [payload.QIB, payload.HNI, payload.RII, payload.Total],
        backgroundColor: [
          "rgba(108,99,255,0.75)", "rgba(0,212,170,0.75)",
          "rgba(255,180,0,0.75)", "rgba(255,77,109,0.75)",
        ],
        borderColor: ["#6c63ff", "#00d4aa", "#ffb400", "#ff4d6d"],
        borderWidth: 2, borderRadius: 8,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y}× subscribed` } },
      },
      scales: {
        x: { grid: { color: "rgba(255,255,255,.05)" }, ticks: { color: "#8b90b8", font: { weight: "bold" } } },
        y: { grid: { color: "rgba(255,255,255,.05)" }, ticks: { color: "#8b90b8" }, beginAtZero: true,
             title: { display: true, text: "Subscription (×)", color: "#8b90b8" } },
      },
    },
  });
}

// ── Reset predictor ──────────────────────────────────────────────────────────
function resetPredictor() {
  ["Issue_Size","Offer_Price","QIB","HNI","RII","Total","quick_price"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
  const ni = document.getElementById("ipo_name_input");
  if (ni) ni.value = "";
  const status = document.getElementById("autofill-status");
  if (status) { status.style.display = "none"; status.textContent = ""; }
  const rc = document.getElementById("result-card");
  if (rc) rc.style.display = "none";
  currentIPOName = "";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function setElem(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function setClass(id, cls) {
  const el = document.getElementById(id);
  if (el) el.className = cls;
}
function riskDesc(r) {
  if (r === "Low")    return "Strong indicators of successful listing.";
  if (r === "High")   return "Weak indicators — high chance of loss.";
  return "Mixed signals — proceed with caution.";
}

function showError(msg) {
  const box = document.getElementById("error-msg");
  if (box) { box.textContent = "⚠️  " + msg; box.style.display = "block"; }
}

// ── Allow Enter key ──────────────────────────────────────────────────────────
document.addEventListener("keydown", e => {
  if (e.key === "Enter" && document.getElementById("predict-btn")) predictIPO();
});

// ── Navbar mobile toggle ──────────────────────────────────────────────────────
function toggleNav() {
  const nl = document.getElementById("nav-links");
  if (nl) nl.classList.toggle("open");
}

// ── Auto-fill from history page re-predict ────────────────────────────────────
(function checkRepredictData() {
  const raw = sessionStorage.getItem("repredictData");
  if (!raw) return;
  try {
    const d = JSON.parse(raw);
    sessionStorage.removeItem("repredictData");
    if (document.getElementById("predict-btn")) {
      // We're on the predictor page
      if (d.ipo_name) {
        const ni = document.getElementById("ipo_name_input");
        if (ni) ni.value = d.ipo_name;
        currentIPOName = d.ipo_name;
      }
      ["Issue_Size","Offer_Price","QIB","HNI","RII","Total"].forEach(k => {
        if (d[k] !== undefined) setField(k, d[k]);
      });
      showDetailsCard();
      showAutofillStatus(`✅ Re-loaded data for "${d.ipo_name || 'previous prediction'}" — click Predict to run again.`, "filled");
    }
  } catch (e) {}
})();