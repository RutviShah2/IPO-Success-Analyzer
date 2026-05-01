/**
 * listings.js — IPO Listings Page
 * Handles: fetch/render IPO cards, search, filter, sort, pagination, modal detail
 */

let currentPage = 1;
let currentTab  = "all";
let searchTimer = null;

// ── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadListings();
});

function debounceLoad() {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { currentPage = 1; loadListings(); }, 350);
}

// ── Tab control ───────────────────────────────────────────────────────────────
function setTab(tab) {
  currentTab  = tab;
  currentPage = 1;
  document.querySelectorAll(".status-tab").forEach(el => el.classList.remove("active"));
  document.getElementById(`tab-${tab}`)?.classList.add("active");
  loadListings();
}

// ── Fetch & Render Listings ──────────────────────────────────────────────────
async function loadListings(page) {
  if (page !== undefined) currentPage = page;

  const search = document.getElementById("search-input")?.value.trim() || "";
  const sector = document.getElementById("sector-filter")?.value || "";
  const lgc    = document.getElementById("lgc-filter")?.value || "";
  const prob   = document.getElementById("prob-filter")?.value || "";
  const sort   = document.getElementById("sort-select")?.value || "prob_desc";

  const params = new URLSearchParams({
    search, sector, lgc, prob, sort,
    status: currentTab === "all" ? "" : currentTab,
    page: currentPage, per_page: 12,
  });

  try {
    const res  = await fetch(`/listings/data?${params}`);
    const data = await res.json();

    if (data.error) { console.error(data.error); return; }

    // Populate filters if first load
    if (currentPage === 1 && search === "" && sector === "" && lgc === "" && prob === "") {
      populateFilterOptions(data.sectors, data.lgc_opts);
    }

    // Update tab counts
    if (data.status_counts) {
      const m = data.status_counts;
      ["all","upcoming","open","closed","listed"].forEach(k => {
        const el = document.getElementById(`cnt-${k}`);
        if (el) el.textContent = m[k] ?? "—";
      });
    }

    updateMeta(data.total, data.page, data.pages);
    renderGrid(data.items);
    renderPagination(data.page, data.pages);

  } catch (e) {
    console.error("Listings load error:", e);
  }
}

function populateFilterOptions(sectors, lgcOpts) {
  const secSel = document.getElementById("sector-filter");
  if (secSel && secSel.options.length <= 1) {
    sectors.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s; opt.textContent = s;
      secSel.appendChild(opt);
    });
  }

  const lgcSel = document.getElementById("lgc-filter");
  if (lgcSel && lgcSel.options.length <= 1) {
    lgcOpts.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l; opt.textContent = l;
      lgcSel.appendChild(opt);
    });
  }
}

function updateMeta(total, page, pages) {
  const el = document.getElementById("listings-meta");
  if (el) el.textContent = `Showing page ${page} of ${pages} · ${total} IPOs found`;
}

// ── Render IPO Card Grid ──────────────────────────────────────────────────────
function renderGrid(items) {
  const grid = document.getElementById("listings-grid");
  if (!grid) return;

  if (!items || items.length === 0) {
    grid.innerHTML = `<div style="text-align:center;padding:40px 20px;color:var(--muted);">
      <div style="font-size:2.5rem;margin-bottom:10px;">🔍</div>
      <p>No IPOs found matching your search.</p>
    </div>`;
    return;
  }

  grid.innerHTML = items.map(ipo => {
    const prob     = ipo.success_prob;
    const probPct  = (prob * 100).toFixed(1);
    const pClass   = prob >= 0.7 ? "prob-high" : prob >= 0.45 ? "prob-mid" : "prob-low";
    const isSuccess = ipo.predicted_success === 1;

    return `
    <div class="listing-card">
      <div class="lc-header">
        <div>
          <div class="lc-name">${escH(ipo.name)}</div>
          <div class="lc-sector">${escH(ipo.sector)}</div>
        </div>
        <span class="prob-badge ${pClass}">${probPct}%</span>
      </div>

      <div class="lc-price-row">
        <span>Offer Price</span>
        <span class="lc-price">₹${ipo.offer_price}</span>
      </div>

      <div class="lc-sub-row">
        <div class="lc-sub-item">
          <div class="lc-sub-label">QIB</div>
          <div class="lc-sub-val">${ipo.qib}×</div>
        </div>
        <div class="lc-sub-item">
          <div class="lc-sub-label">HNI</div>
          <div class="lc-sub-val">${ipo.hni}×</div>
        </div>
        <div class="lc-sub-item">
          <div class="lc-sub-label">RII</div>
          <div class="lc-sub-val">${ipo.rii}×</div>
        </div>
      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;font-size:.78rem;">
        <span class="lc-lgc">${escH(ipo.listing_gain_cat)}</span>
        <span class="lc-pred ${isSuccess ? 'success' : 'fail'}">
          ${isSuccess ? '✅ Success' : '❌ Fail'}
        </span>
      </div>

      <div style="font-size:.75rem;color:var(--muted);">
        📅 Listing: ${ipo.listing_date || '—'}
      </div>

      <div class="lc-actions">
        <button class="lc-btn lc-btn-primary" onclick="openModal(${ipo.id})">View Details</button>
        <a href="/predictor" class="lc-btn lc-btn-sec">Predict Again</a>
      </div>
    </div>`;
  }).join("");
}

// ── Pagination ────────────────────────────────────────────────────────────────
function renderPagination(page, pages) {
  const pg = document.getElementById("pagination");
  if (!pg) return;

  if (pages <= 1) { pg.innerHTML = ""; return; }

  let btns = "";
  btns += `<button class="page-btn" onclick="loadListings(${page - 1})" ${page <= 1 ? "disabled" : ""}>← Prev</button>`;

  const start = Math.max(1, page - 2);
  const end   = Math.min(pages, page + 2);

  if (start > 1) btns += `<button class="page-btn" onclick="loadListings(1)">1</button><span style="color:var(--muted);padding:0 4px;">…</span>`;

  for (let i = start; i <= end; i++) {
    btns += `<button class="page-btn ${i === page ? 'active' : ''}" onclick="loadListings(${i})">${i}</button>`;
  }

  if (end < pages) btns += `<span style="color:var(--muted);padding:0 4px;">…</span><button class="page-btn" onclick="loadListings(${pages})">${pages}</button>`;

  btns += `<button class="page-btn" onclick="loadListings(${page + 1})" ${page >= pages ? "disabled" : ""}>Next →</button>`;

  pg.innerHTML = btns;
}

// ── Modal ─────────────────────────────────────────────────────────────────────
async function openModal(id) {
  const overlay = document.getElementById("detail-modal");
  const content = document.getElementById("modal-content");
  if (!overlay || !content) return;

  overlay.classList.add("open");
  content.innerHTML = `<div class="listings-loading"><div class="loader"></div><span>Loading IPO details...</span></div>`;

  try {
    const res = await fetch(`/listings/${id}`);
    const ipo = await res.json();
    renderModal(ipo);
  } catch (e) {
    content.innerHTML = `<p style="color:var(--danger);">Failed to load IPO details.</p>`;
  }
}

function renderModal(ipo) {
  const content = document.getElementById("modal-content");
  if (!content) return;

  const prob      = ipo.success_prob;
  const probPct   = (prob * 100).toFixed(1);
  const pClass    = prob >= 0.7 ? "prob-high" : prob >= 0.45 ? "prob-mid" : "prob-low";
  const isSuccess = ipo.predicted_success === 1;

  content.innerHTML = `
    <div class="modal-title">${escH(ipo.name)}</div>

    <div style="display:flex;gap:10px;align-items:center;margin-bottom:20px;flex-wrap:wrap;">
      <span class="sector-tag">${escH(ipo.sector)}</span>
      <span class="prob-badge ${pClass}">${probPct}% Success Prob</span>
      <span class="lc-pred ${isSuccess ? 'success' : 'fail'}" style="font-size:.88rem;">
        ${isSuccess ? '✅ Predicted Success' : '❌ Predicted Fail'}
      </span>
    </div>

    <div class="modal-grid">
      ${modalField("Issue Size", "₹" + ipo.issue_size + " Cr")}
      ${modalField("Offer Price", "₹" + ipo.offer_price)}
      ${modalField("QIB Subscription", ipo.qib + "×")}
      ${modalField("HNI Subscription", ipo.hni + "×")}
      ${modalField("RII Subscription", ipo.rii + "×")}
      ${modalField("Total Subscription", ipo.total_sub + "×")}
      ${modalField("Past Performance", ipo.past_performance)}
      ${modalField("Sector Performance", ipo.sector_performance)}
      ${modalField("Listing Gain Category", ipo.listing_gain_cat)}
      ${modalField("Listing Date", ipo.listing_date || "—")}
      ${modalField("Open Date", ipo.open_date || "—")}
      ${modalField("Close Date", ipo.close_date || "—")}
      ${modalField("Allotment Date", ipo.allotment_date || "—")}
      ${modalField("Success Probability", probPct + "%")}
    </div>

    <div style="margin-top:8px;">
      <div class="prob-header" style="margin-bottom:8px;">
        <span class="prob-title">Success Probability</span>
        <span style="font-size:1.2rem;font-weight:800;color:var(--accent2);">${probPct}%</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill" id="modal-prob-bar" style="width:0%"></div>
      </div>
    </div>

    <div class="action-btns" style="margin-top:20px;">
      <a href="/predictor" class="btn-action btn-action-primary">🔍 Predict Again</a>
      <button class="btn-action btn-action-sec" onclick="closeModal()">✕ Close</button>
    </div>
  `;

  // Animate bar
  setTimeout(() => {
    const bar = document.getElementById("modal-prob-bar");
    if (bar) bar.style.width = probPct + "%";
  }, 100);
}

function modalField(label, val) {
  return `
    <div class="modal-field">
      <span class="modal-field-label">${label}</span>
      <span class="modal-field-val">${val ?? "—"}</span>
    </div>`;
}

function closeModal(event) {
  if (event && event.target !== event.currentTarget) return;
  const overlay = document.getElementById("detail-modal");
  if (overlay) overlay.classList.remove("open");
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function escH(str) {
  if (!str) return "";
  return String(str).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
