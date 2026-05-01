/**
 * history.js — Prediction History Page
 * Fetches, renders, and manages prediction history for logged-in users.
 */

document.addEventListener("DOMContentLoaded", loadHistory);

let historyData = [];

async function loadHistory() {
  try {
    const res  = await fetch("/history/data");
    const data = await res.json();
    historyData = data;
    renderHistory(data);
  } catch (e) {
    console.error("History load error:", e);
    renderHistory([]);
  }
}

function renderHistory(items) {
  const loading = document.getElementById("history-loading");
  const empty   = document.getElementById("history-empty");
  const tableWrap = document.getElementById("history-table-wrap");
  const countEl  = document.getElementById("history-count");

  if (loading) loading.style.display = "none";

  if (!items || items.length === 0) {
    if (empty) empty.style.display = "block";
    if (tableWrap) tableWrap.style.display = "none";
    if (countEl) countEl.textContent = "No predictions yet.";
    return;
  }

  if (empty) empty.style.display = "none";
  if (tableWrap) tableWrap.style.display = "block";
  if (countEl) countEl.textContent = `${items.length} prediction(s) in your history`;

  const tbody = document.getElementById("history-tbody");
  if (!tbody) return;

  tbody.innerHTML = items.map((item, idx) => {
    const isSuccess = item.prediction === 1;
    const prob      = item.success_prob;
    const probPct   = prob !== null ? (prob * 100).toFixed(1) + "%" : "—";
    const risk      = item.risk_level || "—";
    const riskClass = risk.toLowerCase() === "low" ? "var(--accent2)"
                    : risk.toLowerCase() === "high" ? "var(--danger)"
                    : "var(--warning)";
    const conf = item.confidence || "—";
    const date = item.predicted_at
      ? new Date(item.predicted_at).toLocaleDateString("en-IN", { day:"2-digit", month:"short", year:"numeric" })
      : "—";

    return `
    <tr id="history-row-${item.id}">
      <td style="color:var(--muted)">${idx + 1}</td>
      <td><strong>${escH(item.ipo_name || "Manual Entry")}</strong></td>
      <td>₹${item.offer_price !== null ? item.offer_price : "—"}</td>
      <td class="${isSuccess ? 'pred-success' : 'pred-fail'}">
        ${isSuccess ? '✅ Success' : '❌ Fail'}
      </td>
      <td>
        <span style="font-weight:700;color:var(--accent2)">${probPct}</span>
      </td>
      <td>
        <span style="font-size:.78rem;font-weight:700;color:${riskClass}">${risk}</span>
      </td>
      <td style="font-size:.78rem;color:var(--muted)">${conf}</td>
      <td style="font-size:.78rem;color:var(--muted)">${date}</td>
      <td style="white-space:nowrap;">
        <button class="h-action-btn h-btn-repred" onclick="repredictEntry(${item.id})">🔄 Redo</button>
        <button class="h-action-btn h-btn-delete" onclick="deleteEntry(${item.id})">🗑️</button>
      </td>
    </tr>`;
  }).join("");
}

// ── Delete history entry ─────────────────────────────────────────────────────
async function deleteEntry(id) {
  if (!confirm("Delete this prediction from history?")) return;
  try {
    const res = await fetch(`/history/${id}`, { method: "DELETE" });
    const data = await res.json();
    if (data.status === "deleted") {
      const row = document.getElementById(`history-row-${id}`);
      if (row) {
        row.style.opacity = "0";
        row.style.transition = "opacity .3s";
        setTimeout(() => row.remove(), 300);
      }
      historyData = historyData.filter(h => h.id !== id);
      const countEl = document.getElementById("history-count");
      if (countEl) countEl.textContent = `${historyData.length} prediction(s) in your history`;
      if (historyData.length === 0) renderHistory([]);
    }
  } catch (e) {
    alert("Failed to delete entry. Please try again.");
  }
}

// ── Re-predict (navigate to predictor with params) ───────────────────────────
function repredictEntry(id) {
  const item = historyData.find(h => h.id === id);
  if (!item) return;

  // Store in sessionStorage for predictor to pick up
  sessionStorage.setItem("repredictData", JSON.stringify({
    ipo_name:   item.ipo_name,
    Issue_Size: item.issue_size,
    Offer_Price: item.offer_price,
    QIB:        item.qib,
    HNI:        item.hni,
    RII:        item.rii,
    Total:      item.total_sub,
  }));
  window.location.href = "/predictor";
}

// ── Clear all history ─────────────────────────────────────────────────────────
async function clearAllHistory() {
  if (!confirm("Delete ALL prediction history? This cannot be undone.")) return;
  const ids = historyData.map(h => h.id);
  for (const id of ids) {
    try {
      await fetch(`/history/${id}`, { method: "DELETE" });
    } catch (e) {}
  }
  historyData = [];
  renderHistory([]);
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function escH(str) {
  if (!str) return "";
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
