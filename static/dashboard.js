/**
 * dashboard.js — IPO Analytics Dashboard
 * Fetches all chart data from /dashboard/data and renders 12 Chart.js charts.
 */

const ACCENT   = "#6c63ff";
const ACCENT2  = "#00d4aa";
const DANGER   = "#ff4d6d";
const WARNING  = "#ffb400";
const MUTED    = "#8b90b8";

const SECTOR_COLORS = [
  "rgba(108,99,255,0.75)", "rgba(0,212,170,0.75)",
  "rgba(255,180,0,0.75)",  "rgba(255,77,109,0.75)",
  "rgba(100,180,255,0.75)","rgba(255,140,80,0.75)",
  "rgba(160,100,255,0.75)","rgba(60,220,140,0.75)",
  "rgba(255,200,80,0.75)", "rgba(200,100,180,0.75)",
];

const gridColor = "rgba(255,255,255,.05)";
const tickColor = MUTED;

const defaultScales = {
  x: { grid: { color: gridColor }, ticks: { color: tickColor } },
  y: { grid: { color: gridColor }, ticks: { color: tickColor }, beginAtZero: true },
};

async function loadDashboard() {
  try {
    const res  = await fetch("/dashboard/data");
    const data = await res.json();
    if (data.error) { console.error(data.error); return; }

    buildSummaryCards(data.summary);
    buildPieChart(data.summary);
    buildSectorBar(data.sector_bar);
    buildSubCompare(data.subscription_avg);
    buildMostSub(data.most_subscribed);
    buildHighProb(data.highest_prob);
    buildLGCDist(data.lgc_distribution);
    buildLGCSector(data.lgc_by_sector);
    buildRecentTrend(data.recent_trend);
    buildRiskChart(data.risk_distribution);
    buildConfChart(data.confidence_distribution);
    buildFeatureChart(data.feature_importance);

  } catch (e) {
    console.error("Dashboard load error:", e);
  }
}

function buildSummaryCards(s) {
  setText("d-total",    s.total);
  setText("d-avg-prob", s.avg_prob + "%");
  setText("d-best-sec", s.best_sector);
  setText("d-top-lgc",  s.top_lgc);
}

// ── 1. Success vs Failure Pie ─────────────────────────────────────────────────
function buildPieChart(s) {
  const ctx = document.getElementById("piChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Success", "Failure"],
      datasets: [{
        data: [s.success, s.fail],
        backgroundColor: ["rgba(0,212,170,0.8)", "rgba(255,77,109,0.8)"],
        borderColor:     [ACCENT2, DANGER],
        borderWidth: 2, hoverOffset: 10,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "62%",
      plugins: {
        legend: { position: "bottom", labels: { color: MUTED, font: { size: 11 } } },
        tooltip: { callbacks: { label: c => ` ${c.label}: ${c.parsed} IPOs` } },
      }
    }
  });
}

// ── 2. Sector-wise IPO Success Bar ───────────────────────────────────────────
function buildSectorBar(d) {
  const ctx = document.getElementById("sectorBarChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: d.labels,
      datasets: [
        { label: "Successful", data: d.success, backgroundColor: "rgba(0,212,170,0.75)", borderColor: ACCENT2, borderWidth: 1, borderRadius: 5 },
        { label: "Total",      data: d.total,   backgroundColor: "rgba(108,99,255,0.4)", borderColor: ACCENT,  borderWidth: 1, borderRadius: 5 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: MUTED } } },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: tickColor, maxRotation: 30, font: { size: 10 } } },
        y: { grid: { color: gridColor }, ticks: { color: tickColor }, beginAtZero: true,
             title: { display: true, text: "No. of IPOs", color: MUTED } },
      },
    }
  });
}

// ── 3. QIB vs HNI vs RII ────────────────────────────────────────────────────
function buildSubCompare(d) {
  const ctx = document.getElementById("subCompareChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["QIB", "HNI", "RII", "Total"],
      datasets: [{
        label: "Avg Subscription (×)",
        data: [d.qib, d.hni, d.rii, d.total],
        backgroundColor: [
          "rgba(108,99,255,0.75)", "rgba(0,212,170,0.75)",
          "rgba(255,180,0,0.75)",  "rgba(255,77,109,0.75)",
        ],
        borderColor: [ACCENT, ACCENT2, WARNING, DANGER],
        borderWidth: 2, borderRadius: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: defaultScales,
    }
  });
}

// ── 4. Most Subscribed IPOs ──────────────────────────────────────────────────
function buildMostSub(items) {
  const ctx = document.getElementById("mostSubChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: items.map(i => i.name.split(" ").slice(0,2).join(" ")),
      datasets: [{
        label: "Total Subscription (×)",
        data: items.map(i => i.total_sub),
        backgroundColor: items.map((_, idx) => SECTOR_COLORS[idx % SECTOR_COLORS.length]),
        borderRadius: 6, borderWidth: 0,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: c => ` ${c.parsed.x}× subscribed` } } },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: tickColor }, beginAtZero: true },
        y: { grid: { display: false },   ticks: { color: tickColor, font: { size: 10 } } },
      }
    }
  });
}

// ── 5. Highest Probability IPOs ──────────────────────────────────────────────
function buildHighProb(items) {
  const ctx = document.getElementById("highProbChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: items.map(i => i.name.split(" ").slice(0,2).join(" ")),
      datasets: [{
        label: "Success Probability (%)",
        data: items.map(i => parseFloat((i.success_prob * 100).toFixed(1))),
        backgroundColor: "rgba(0,212,170,0.7)",
        borderColor: ACCENT2, borderWidth: 1, borderRadius: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => ` ${c.parsed.y}% success probability` } },
      },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 10 }, maxRotation: 30 } },
        y: { grid: { color: gridColor }, ticks: { color: tickColor }, min: 80, max: 100,
             title: { display: true, text: "Success Prob (%)", color: MUTED } },
      }
    }
  });
}

// ── 6. Listing Gain Distribution ─────────────────────────────────────────────
function buildLGCDist(d) {
  const ctx = document.getElementById("lgcDistChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: d.labels,
      datasets: [{
        data: d.counts,
        backgroundColor: [
          "rgba(0,212,170,0.8)", "rgba(108,99,255,0.8)",
          "rgba(255,180,0,0.8)", "rgba(255,77,109,0.8)", "rgba(160,100,255,0.8)"
        ],
        borderWidth: 2, hoverOffset: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "55%",
      plugins: {
        legend: { position: "bottom", labels: { color: MUTED, font: { size: 9 }, boxWidth: 12 } },
      }
    }
  });
}

// ── 7. Avg Listing Gain Proxy by Sector ──────────────────────────────────────
function buildLGCSector(d) {
  const ctx = document.getElementById("lgcSectorChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "radar",
    data: {
      labels: d.labels,
      datasets: [{
        label: "Avg Success Prob (%)",
        data: d.values,
        borderColor: ACCENT,
        backgroundColor: "rgba(108,99,255,0.15)",
        pointBackgroundColor: ACCENT,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: MUTED } } },
      scales: {
        r: {
          grid: { color: gridColor }, pointLabels: { color: MUTED, font: { size: 9 } },
          ticks: { color: MUTED, backdropColor: "transparent", stepSize: 20 },
          min: 0, max: 100,
        }
      }
    }
  });
}

// ── 8. Recent IPO Trends Line ─────────────────────────────────────────────────
function buildRecentTrend(items) {
  const ctx = document.getElementById("recentTrendChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "line",
    data: {
      labels: items.map(i => i.month),
      datasets: [
        {
          label: "IPO Count",
          data: items.map(i => i.cnt),
          borderColor: ACCENT, backgroundColor: "rgba(108,99,255,0.1)",
          fill: true, tension: 0.4, pointBackgroundColor: ACCENT, borderWidth: 2,
          yAxisID: "y",
        },
        {
          label: "Avg Prob (%)",
          data: items.map(i => parseFloat(i.avg_prob.toFixed(1))),
          borderColor: ACCENT2, backgroundColor: "rgba(0,212,170,0.08)",
          fill: true, tension: 0.4, pointBackgroundColor: ACCENT2, borderWidth: 2,
          yAxisID: "y1",
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: MUTED } } },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: tickColor, font: { size: 9 } } },
        y: { grid: { color: gridColor }, ticks: { color: tickColor }, beginAtZero: true,
             title: { display: true, text: "Count", color: MUTED }, position: "left" },
        y1: { grid: { display: false }, ticks: { color: ACCENT2 },
              title: { display: true, text: "Avg Prob (%)", color: ACCENT2 }, position: "right" },
      }
    }
  });
}

// ── 9. Risk Distribution ─────────────────────────────────────────────────────
function buildRiskChart(d) {
  const ctx = document.getElementById("riskChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Low Risk", "Medium Risk", "High Risk"],
      datasets: [{
        data: [d.low, d.medium, d.high],
        backgroundColor: [
          "rgba(0,212,170,0.8)", "rgba(255,180,0,0.8)", "rgba(255,77,109,0.8)"
        ],
        borderWidth:2, hoverOffset: 6,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "60%",
      plugins: { legend: { position: "bottom", labels: { color: MUTED, font: { size: 10 } } } }
    }
  });
}

// ── 10. Confidence Distribution ──────────────────────────────────────────────
function buildConfChart(d) {
  const ctx = document.getElementById("confChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Very High", "High", "Medium", "Low"],
      datasets: [{
        label: "IPO Count",
        data: [d.very_high, d.high, d.medium, d.low],
        backgroundColor: [
          "rgba(0,212,170,0.8)", "rgba(108,99,255,0.8)",
          "rgba(255,180,0,0.8)", "rgba(255,77,109,0.8)",
        ],
        borderRadius: 6, borderWidth: 0,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: defaultScales,
    }
  });
}

// ── 11. Feature Importance ───────────────────────────────────────────────────
function buildFeatureChart(items) {
  const ctx = document.getElementById("featureChart");
  if (!ctx) return;
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: items.map(i => i.name.replace("_", " ")),
      datasets: [{
        label: "Coefficient",
        data: items.map(i => i.coef),
        backgroundColor: items.map(i => i.coef >= 0
          ? "rgba(0,212,170,0.75)" : "rgba(255,77,109,0.75)"),
        borderColor: items.map(i => i.coef >= 0 ? ACCENT2 : DANGER),
        borderWidth: 1, borderRadius: 5,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => ` Coeff: ${c.parsed.x > 0 ? "+" : ""}${c.parsed.x}` } },
      },
      scales: {
        x: { grid: { color: gridColor }, ticks: { color: tickColor } },
        y: { grid: { display: false },   ticks: { color: tickColor } },
      }
    }
  });
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

// ── Init ─────────────────────────────────────────────────────────────────────
loadDashboard();
