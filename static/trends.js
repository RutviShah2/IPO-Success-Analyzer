/**
 * trends.js — IPO Trends & Timeline Page
 * Fetches /trends/data and renders charts + timeline cards.
 */

const ACCENT  = "#6c63ff";
const ACCENT2 = "#00d4aa";
const DANGER  = "#ff4d6d";
const MUTED   = "#8b90b8";
const gridColor = "rgba(255,255,255,.05)";

document.addEventListener("DOMContentLoaded", loadTrends);

async function loadTrends() {
  try {
    const res  = await fetch("/trends/data");
    const data = await res.json();
    if (data.error) { console.error(data.error); return; }

    // Stats
    const total = data.success_count + data.fail_count;
    setText("s-total",     total);
    setText("s-success",   data.success_count);
    setText("s-fail",      data.fail_count);
    setText("s-avg-total", data.avg_total + "×");

    // Timeline Cards
    renderTimelineCards(data.timeline_ipos || []);

    // Charts
    buildSuccessRateChart(data.periods, data.success_rate);
    buildTotalSubChart(data.periods, data.avg_total_sub);
    buildSectorPerfChart(data.sector);
    buildAvgSubChart(data.avg_qib, data.avg_hni, data.avg_rii, data.avg_total);
    buildDonutChart(data.success_count, data.fail_count);

  } catch (e) {
    console.error("Trends load error:", e);
  }
}

// ── Timeline Cards ────────────────────────────────────────────────────────────
function renderTimelineCards(ipos) {
  const box = document.getElementById("timeline-cards");
  if (!box) return;

  if (!ipos || ipos.length === 0) {
    box.innerHTML = `<p style="color:var(--muted);text-align:center;padding:20px 0;">No timeline data available.</p>`;
    return;
  }

  box.innerHTML = ipos.map(ipo => {
    const prob    = ipo.success_prob;
    const probPct = (prob * 100).toFixed(1);
    const pClass  = prob >= 0.7 ? "prob-high" : prob >= 0.45 ? "prob-mid" : "prob-low";
    const isSucc  = ipo.predicted_success === 1;

    return `
    <div class="timeline-card">
      <div style="flex:1;min-width:150px;">
        <div class="tc-name">${escH(ipo.name)}</div>
        <div style="font-size:.75rem;color:var(--muted);margin-top:2px;">
          <span class="sector-tag">${escH(ipo.sector)}</span>
          &nbsp;₹${ipo.offer_price}
        </div>
      </div>

      <div class="tc-meta">
        <div class="tc-date">
          <span class="tc-label">📅 Open</span>
          <span class="tc-date-val">${ipo.open_date || '—'}</span>
        </div>
        <div class="tc-date">
          <span class="tc-label">🔒 Close</span>
          <span class="tc-date-val">${ipo.close_date || '—'}</span>
        </div>
        <div class="tc-date">
          <span class="tc-label">🎫 Allotment</span>
          <span class="tc-date-val">${ipo.allotment_date || '—'}</span>
        </div>
        <div class="tc-date">
          <span class="tc-label">📈 Listing</span>
          <span class="tc-date-val">${ipo.listing_date || '—'}</span>
        </div>
      </div>

      <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px;min-width:90px;">
        <span class="prob-badge ${pClass}">${probPct}%</span>
        <span style="font-size:.75rem;font-weight:700;color:${isSucc ? 'var(--accent2)' : 'var(--danger)'}">
          ${isSucc ? '✅ Success' : '❌ Fail'}
        </span>
      </div>
    </div>`;
  }).join("");
}

// ── Chart 1: Success Rate by Period ──────────────────────────────────────────
function buildSuccessRateChart(periods, rates) {
  const ctx = document.getElementById("successRateChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "line",
    data: {
      labels: periods,
      datasets: [{
        label: "Success Rate (%)",
        data: rates,
        borderColor: ACCENT2, backgroundColor: "rgba(0,212,170,0.1)",
        fill: true, tension: 0.4, pointBackgroundColor: ACCENT2,
        pointRadius: 5, borderWidth: 2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: MUTED } } },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: MUTED } },
        y: { grid: { color: gridColor }, ticks: { color: MUTED }, min: 0, max: 100,
             title: { display: true, text: "Success Rate (%)", color: MUTED } },
      }
    }
  });
}

// ── Chart 2: Avg Total Subscription Trend ────────────────────────────────────
function buildTotalSubChart(periods, subs) {
  const ctx = document.getElementById("totalSubChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: periods,
      datasets: [{
        label: "Avg Total Subscription (×)",
        data: subs,
        backgroundColor: "rgba(108,99,255,0.7)",
        borderColor: ACCENT, borderWidth: 1, borderRadius: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: MUTED } } },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: MUTED } },
        y: { grid: { color: gridColor }, ticks: { color: MUTED }, beginAtZero: true,
             title: { display: true, text: "Total Sub (×)", color: MUTED } },
      }
    }
  });
}

// ── Chart 3: Sector-wise Performance ─────────────────────────────────────────
function buildSectorPerfChart(sector) {
  const ctx = document.getElementById("sectorPerfChart");
  if (!ctx || !sector) return;

  const rates = sector.success.map((s, i) =>
    sector.total[i] > 0 ? parseFloat((s / sector.total[i] * 100).toFixed(1)) : 0
  );

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: sector.labels,
      datasets: [
        { label: "Success Rate (%)", data: rates, backgroundColor: "rgba(0,212,170,0.75)",
          borderColor: ACCENT2, borderWidth: 1, borderRadius: 5, yAxisID: "y" },
        { label: "Avg Subscription (×)", data: sector.avg_sub, backgroundColor: "rgba(108,99,255,0.5)",
          borderColor: ACCENT, borderWidth: 1, borderRadius: 5, yAxisID: "y1" },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: MUTED } } },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: MUTED, maxRotation: 30, font: { size: 10 } } },
        y: { grid: { color: gridColor }, ticks: { color: MUTED }, beginAtZero: true,
             title: { display: true, text: "Success Rate (%)", color: MUTED }, position: "left" },
        y1: { grid: { display: false }, ticks: { color: ACCENT },
              title: { display: true, text: "Avg Sub (×)", color: ACCENT }, position: "right" },
      }
    }
  });
}

// ── Chart 4: Avg Subscription by Category ────────────────────────────────────
function buildAvgSubChart(qib, hni, rii, total) {
  const ctx = document.getElementById("avgSubChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["QIB", "HNI", "RII", "Total"],
      datasets: [{
        label: "Average Subscription (×)",
        data: [qib, hni, rii, total],
        backgroundColor: [
          "rgba(108,99,255,0.75)", "rgba(0,212,170,0.75)",
          "rgba(255,180,0,0.75)", "rgba(255,77,109,0.75)",
        ],
        borderColor: [ACCENT, ACCENT2, "#ffb400", DANGER],
        borderWidth: 2, borderRadius: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: MUTED, font: { weight: "bold" } } },
        y: { grid: { color: gridColor }, ticks: { color: MUTED }, beginAtZero: true,
             title: { display: true, text: "Avg Subscription (×)", color: MUTED } },
      }
    }
  });
}

// ── Chart 5: Success vs Fail Donut ────────────────────────────────────────────
function buildDonutChart(success, fail) {
  const ctx = document.getElementById("donutChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Successful IPOs", "Failed IPOs"],
      datasets: [{
        data: [success, fail],
        backgroundColor: ["rgba(0,212,170,0.8)", "rgba(255,77,109,0.8)"],
        borderColor: [ACCENT2, DANGER],
        borderWidth: 2, hoverOffset: 10,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "60%",
      plugins: {
        legend: { position: "bottom", labels: { color: MUTED, font: { size: 11 } } },
        tooltip: { callbacks: { label: c => ` ${c.label}: ${c.parsed} IPOs` } },
      }
    }
  });
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function escH(str) {
  if (!str) return "";
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}