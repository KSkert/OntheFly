/**
 * =====================================================================
 * dashboard.js (refactor-ready edition)
 * =====================================================================
 * TABLE OF CONTENTS
 * -----------------
 *  1) Utilities & Debug
 *  2) DOM References
 *  3) Sessions/Compare (state, helpers, cache keys)
 *  4) Autofork Defaults & UI (prefill, read, plan rendering, tabs)
 *  5) Run/Training State (lineage maps, status flags, nav helpers)
 *  6) Charting (init, plugins, rendering, export)
 *  7) Streamed Data Model (STATE/PEND, redraw pipeline)
 *  8) Log Ring Buffer
 *  9) IPC Commands & Buttons (send helpers + button handlers)
 * 10) Compare Columns UI (layout, columns, summaries, copy)
 * 11) Window/Message Handlers (all postMessage routes)
 * 12) Select Populators (runs dropdown)
 * 13) Status Gating (running/paused UI)
 * 14) Report Histogram + Manual Fork Overlay (drag/select UX)
 * 15) DAG (graph layout + rendering + helpers)


/* ====================================================================
 * 1) Utilities & Debug
 * --------------------------------------------------------------------
 * Small helpers used across the file + debug shim for navigation logs.
 * ==================================================================== */

/* global acquireVsCodeApi, Chart */
const vscode = acquireVsCodeApi();

const byId = (id) => document.getElementById(id);
const on = (el, evt, fn) => { if (el) el.addEventListener(evt, fn); };
const nf = v => (Number.isFinite(v) ? v : NaN);

// ===== DEBUG SHIM =====
window.DEBUG_NAV = false;
const logNav = (...args) => { if (window.DEBUG_NAV) console.log('[NAV]', ...args); };
const groupNav = (label, fn) => {
  if (!window.DEBUG_NAV) return fn();
  console.groupCollapsed(`%c[NAV] ${label}`, 'color:#0ea5e9');
  try { fn(); } finally { console.groupEnd(); }
};


/* ====================================================================
 * 2) DOM References
 * --------------------------------------------------------------------
 * Cache relevant DOM elements once. These are used widely across the UI.
 * ==================================================================== */

/* -------- DOM refs -------- */
const btnChoose = byId('btnChoose');
const btnSetPy = byId('btnSetPy');
const pyPath = byId('pyPath');


const btnPause = byId('btnPause');
const btnResume = byId('btnResume');
const rewindSteps = byId('rewindSteps');

const btnSaveCkpt = byId('btnSaveCkpt');
const btnAutoSave = byId('btnAutoSave');
const btnLoad = byId('btnLoad');

const btnTestNow = byId('btnTestNow');
const btnReport = byId('btnGenerateReport') || byId('btnReport');
const reportNote = byId('reportNote');
const btnManualFork = byId('btnManualFork');

const btnPrevModel = byId('btnPrevModel');
const btnNextModel = byId('btnNextModel');

const reportMeta = byId('reportMeta');

const runSel = byId('runSel');
const btnRefreshRuns = byId('btnRefreshRuns');
const scriptName = byId('scriptName');
const logDiv = byId('log');

const btnOpenDag = byId('btnOpenDag');
const btnExportLoss = byId('exportLossBtn');
const btnExportLossDist = byId('exportLossDistBtn');
const btnExportValLoss = byId('exportValLossBtn');

const dagOverlay = byId('dagOverlay');
const dagSvg     = byId('dagSvg');
const dagClose   = byId('dagClose');
const dagMergeBtn= byId('dagMergeBtn');
const dagStrategy= byId('dagStrategy');


const btnAutoForkOn     = byId('btnAutoForkOn');
const btnAutoForkOff    = byId('btnAutoForkOff');
const btnAutoForkApply  = byId('btnAutoForkApply');
const btnAutoForkExec   = byId('btnAutoForkExec');
const afVariantIndex    = byId('afVariantIndex');
const afPlanBox         = byId('afPlan'); // <pre> or <div>

const afEnabled                = byId('afEnabled');
const afLossPlateauPatience    = byId('afLossPlateauPatience');
const afLossPlateauDelta       = byId('afLossPlateauDelta');
const afPerSampleWindow        = byId('afPerSampleWindow');
const afKmeansK                = byId('afKmeansK');
const afDeadClusterZ           = byId('afDeadClusterZ');
const afHighLossQuantile       = byId('afHighLossQuantile');
const afSpikeSigma             = byId('afSpikeSigma');
const afEmaDecay               = byId('afEmaDecay');
const afMaxParallelChildren    = byId('afMaxParallelChildren');
const afForkCooldownSteps      = byId('afForkCooldownSteps');
const afGateEpochs             = byId('afGateEpochs');

const afPslEvery               = byId('afPslEvery');
const afPslBudget              = byId('afPslBudget');
const afMirrorTrain            = byId('afMirrorTrain');
const afAmpForPsl              = byId('afAmpForPsl');
// feature sampling toggles
const afComputeMargins         = byId('afComputeMargins');
const afComputeEmbeddings      = byId('afComputeEmbeddings');
const afEmbedMaxDim            = byId('afEmbedMaxDim');

const afMergeOnPlateau          = byId('afMergeOnPlateau');
const afReunifyEverySteps       = byId('afReunifyEverySteps');
const afMinChildStepsBeforeMerge= byId('afMinChildStepsBeforeMerge');
const afMergeUpliftThreshold    = byId('afMergeUpliftThreshold');
const afInterForkImprovement    = byId('afInterForkImprovement');
const afForkBackpressureAlpha   = byId('afForkBackpressureAlpha');
const afMergeMethodSelector    = byId('afMergeMethodSelector');

// ===== Model Comparison (Columns) refs =====
const tabCreate       = byId('tabCreate');
const tabCompare      = byId('tabCompare');
const wCompare        = byId('w-compare');
const columnsWrapEl   = byId('columnsWrap');
const compareRail     = byId('compareRail');
const columnsGrid     = byId('columnsGrid');
const btnAddColumn    = byId('btnAddColumn');
const btnClearColumns = byId('btnClearColumns');
const btnUploadSession= byId('btnUploadSession');
const sessionFilePicker = byId('sessionFilePicker');


/* ====================================================================
 * 3) Sessions/Compare: state, helpers, caches
 * --------------------------------------------------------------------
 * Compare view is session-first. Cache keys reflect session+run+view.
 * ==================================================================== */

// ==== NEW: Sessions-first Compare state ====
const COL_STATE = new Map(); // colEl -> { sessionId, runId }
const __AGG = '__aggregate__';

// cache now keyed by session + run + view
function sumKey(sessionId, runId, view){ return `${String(sessionId)}|${String(runId)}:${String(view)}`; }

// requests
function requestSessionRuns(sessionId) { send('fs.session.runs', { sessionId }); }
function requestSessionSummary(sessionId, runId, view) {
  send('fs.session.summary.get', { sessionId, runId, view });
}
function getSessionIdForRun(runId) {
  const row = runsIndex.get(String(runId)) || {};
  return row.session_id ?? row.sessionId ?? row.session ?? null;
}

function ensureSessionColumnForRun(runId) {
  const sid = getSessionIdForRun(runId);
  if (!sid || !columnsGrid || !btnAddColumn) return false;

  // reuse existing column for this session if present; otherwise create it
  let col = columnsGrid.querySelector(`.modelCol[data-session-id="${cssAttr(sid)}"]`);
  if (!col) col = makeColumnForSession({ id: sid, label: sid });

  // lock the column to the session aggregate and (re)fill it
  COL_STATE.set(col, { sessionId: sid, runId: __AGG });
  setColumnSubtitle(col, _compareView);
  fillSummary(col, sid, __AGG, _compareView);
  const other = _compareView === 'train' ? 'test' : 'train';
  requestSessionSummary(sid, __AGG, other);  // warm the other view
  return true;
}


/* ====================================================================
 * 4) Autofork Defaults & UI
 * --------------------------------------------------------------------
 * Defaults, UI prefill/reading, rendering suggested plans, tab wiring.
 * No runtime semantics changed.
 * ==================================================================== */

const AF_DEFAULTS = {
  rules: {
    enabled: false,
    loss_plateau_patience: 200,
    loss_plateau_delta: 1e-4,
    per_sample_window: 5000,
    kmeans_k: 5,
    dead_cluster_z: 1.0,
    high_loss_quantile: 0.85,
    spike_sigma: 3.0,
    ema_decay: 0.98,
    max_parallel_children: 2,
    fork_cooldown_steps: 1000,
    gate_epochs: 30,

    merge_on_plateau: true,
    reunify_every_steps: 0,
    min_child_steps_before_merge: 0,
    merge_uplift_threshold: 0.0,
    inter_fork_improvement: 0.0,
    fork_backpressure_alpha: 0.0,
    merge_method: 'swa',
  },
  sampling: {
    psl_every: 200,
    psl_budget: 4000,
    mirror_train: true,
    amp_for_psl: true,
    compute_margins: true,
    compute_embeddings: false,
    embed_max_dim: 256,
  },
};

// --- subset export UI ---
const btnExportSubset = byId('btnExportSubset');
const exportSubsetFmt = byId('exportSubsetFmt');

on(btnExportSubset, 'click', () => {
  const runId = currentPageRunId();
  if (!runId) { notify('No run selected.', 'warn'); return; }
  const format = (exportSubsetFmt && exportSubsetFmt.value) || 'parquet';
  // Ask the extension host to open Save dialog + call Python
  vscode.postMessage({ command: 'exportSubset', runId, format });
});

// Optional: toast when the extension finishes
window.addEventListener('message', (e) => {
  const m = e.data;
  if (m?.type === 'subsetExported') {
    const rows = Number(m.rows || 0);
    const fmt  = (m.format || '').toUpperCase();
    notify(`Subset exported (${rows} rows, ${fmt}).`);
  }
});

function _set(el, v) { if (el) { if (el.type === 'checkbox') el.checked = !!v; else el.value = String(v); } }
function _num(el)    { const n = el ? Number(el.value) : NaN; return Number.isFinite(n) ? n : NaN; }
function _bool(el)   { return !!(el && el.checked); }
function setAutoModeUI(isAuto) {
  // Hide any manual-only actions in auto mode
  [btnManualFork, btnAutoForkExec].forEach(b => { if (!b) return; b.style.display = isAuto ? 'none' : ''; b.disabled = !!isAuto; });
  // (Optional) make the plan <pre> read-only or relabel
  if (afPlanBox) afPlanBox.closest('.card')?.classList.toggle('readonly', isAuto);
}
function reflectAutoForkBtns() {
  const onActive = !!afEnabled?.checked;           // true = ON selected
  btnAutoForkOn?.classList.toggle('is-active', onActive);
  btnAutoForkOn?.classList.toggle('is-inactive', !onActive);
  btnAutoForkOn?.setAttribute('aria-pressed', String(onActive));

  btnAutoForkOff?.classList.toggle('is-active', !onActive);
  btnAutoForkOff?.classList.toggle('is-inactive', onActive);
  btnAutoForkOff?.setAttribute('aria-pressed', String(!onActive));
}

function prefillAutoforkUi() {
  if (!afEnabled) return;
  _set(afEnabled,             AF_DEFAULTS.rules.enabled);
  _set(afLossPlateauPatience, AF_DEFAULTS.rules.loss_plateau_patience);
  _set(afLossPlateauDelta,    AF_DEFAULTS.rules.loss_plateau_delta);
  _set(afPerSampleWindow,     AF_DEFAULTS.rules.per_sample_window);
  _set(afKmeansK,             AF_DEFAULTS.rules.kmeans_k);
  _set(afDeadClusterZ,        AF_DEFAULTS.rules.dead_cluster_z);
  _set(afHighLossQuantile,    AF_DEFAULTS.rules.high_loss_quantile);
  _set(afSpikeSigma,          AF_DEFAULTS.rules.spike_sigma);
  _set(afEmaDecay,            AF_DEFAULTS.rules.ema_decay);
  _set(afMaxParallelChildren, AF_DEFAULTS.rules.max_parallel_children);
  _set(afForkCooldownSteps,   AF_DEFAULTS.rules.fork_cooldown_steps);
  _set(afGateEpochs,          AF_DEFAULTS.rules.gate_epochs);

  _set(afMergeOnPlateau,          AF_DEFAULTS.rules.merge_on_plateau);
  _set(afReunifyEverySteps,       AF_DEFAULTS.rules.reunify_every_steps);
  _set(afMinChildStepsBeforeMerge,AF_DEFAULTS.rules.min_child_steps_before_merge);
  _set(afMergeUpliftThreshold,    AF_DEFAULTS.rules.merge_uplift_threshold);
  _set(afInterForkImprovement,    AF_DEFAULTS.rules.inter_fork_improvement);
  _set(afForkBackpressureAlpha,   AF_DEFAULTS.rules.fork_backpressure_alpha);

  _set(afPslEvery,            AF_DEFAULTS.sampling.psl_every);
  _set(afPslBudget,           AF_DEFAULTS.sampling.psl_budget);
  _set(afMirrorTrain,         AF_DEFAULTS.sampling.mirror_train);
  _set(afAmpForPsl,           AF_DEFAULTS.sampling.amp_for_psl);
  _set(afComputeMargins,      AF_DEFAULTS.sampling.compute_margins);
  _set(afComputeEmbeddings,   AF_DEFAULTS.sampling.compute_embeddings);
  _set(afEmbedMaxDim,         AF_DEFAULTS.sampling.embed_max_dim);

  _set(afMergeMethodSelector,     AF_DEFAULTS.rules.merge_method);

}


function readAutoForkConfig() {
  return {
    rules: {
      enabled:               _bool(afEnabled),
      loss_plateau_patience: _num(afLossPlateauPatience),
      loss_plateau_delta:    _num(afLossPlateauDelta),
      per_sample_window:     _num(afPerSampleWindow),
      kmeans_k:              _num(afKmeansK),
      dead_cluster_z:        _num(afDeadClusterZ),
      high_loss_quantile:    _num(afHighLossQuantile),
      spike_sigma:           _num(afSpikeSigma),
      ema_decay:             _num(afEmaDecay),
      max_parallel_children: _num(afMaxParallelChildren),
      fork_cooldown_steps:   _num(afForkCooldownSteps),
      gate_epochs:           _num(afGateEpochs),

      merge_on_plateau:           _bool(afMergeOnPlateau),
      reunify_every_steps:        _num(afReunifyEverySteps),
      min_child_steps_before_merge:_num(afMinChildStepsBeforeMerge),
      merge_uplift_threshold:     _num(afMergeUpliftThreshold),
      inter_fork_improvement:     _num(afInterForkImprovement),
      fork_backpressure_alpha:    _num(afForkBackpressureAlpha),
      merge: {
        method: (afMergeMethodSelector && afMergeMethodSelector.value) || 'swa',
      },
    },
    sampling: {
      psl_every:        _num(afPslEvery),
      psl_budget:       _num(afPslBudget),
      mirror_train:     _bool(afMirrorTrain),
      amp_for_psl:      _bool(afAmpForPsl),
      compute_margins:  _bool(afComputeMargins),
      compute_embeddings:_bool(afComputeEmbeddings),
      embed_max_dim:    _num(afEmbedMaxDim),
    },
  };
}

let _lastAutoForkPlan = null;

function renderAutoForkPlan(plan) {
  if (!afPlanBox) return;
  if (!plan) { afPlanBox.textContent = '(no plan yet)'; return; }

  const variants = Array.isArray(plan?.training_recipe?.variants) ? plan.training_recipe.variants : [];
  const summary = {
    reason: plan.reason,
    priority: plan.priority,
    selection: plan.selection,
    variants: variants.map((v, i) => ({ i, ...v })),
    cooldown_steps: plan.cooldown_steps,
    analyzed_at_step: plan.at_step ?? plan.step ?? null,
  };
  afPlanBox.textContent = JSON.stringify(summary, null, 2);
}

function wireAfTabs() {
  const LS_KEY = 'fs.autofork.tab.v1';
  const btns = document.querySelectorAll('#afTabButtons .afTabBtn');
  const panels = document.querySelectorAll('.afTabPanel');

  const setActive = (tab) => {
    btns.forEach(b => {
      const is = b.dataset.tab === tab;
      b.setAttribute('aria-selected', is ? 'true' : 'false');
    });
    panels.forEach(p => {
      const show = p.getAttribute('data-tab') === tab;
      if (show) p.removeAttribute('hidden'); else p.setAttribute('hidden','');
    });
  };

  let initial = localStorage.getItem(LS_KEY) || 'forking';
  const known = new Set(Array.from(btns).map(b => b.dataset.tab));
  if (!known.has(initial)) initial = 'forking';
  setActive(initial);

  btns.forEach(b => b.addEventListener('click', () => {
    const t = b.dataset.tab;
    setActive(t);
    localStorage.setItem(LS_KEY, t);
  }));
}
// prefill once on load if the fields are present
prefillAutoforkUi();
reflectAutoForkBtns();
wireAfTabs();


/* ====================================================================
 * 5) Run/Training State & Navigation
 * --------------------------------------------------------------------
 * Lineage maps, run status/flags, follow behavior, model nav arrows.
 * ==================================================================== */

let manualForkMode = false;
let forkSel = { active: false, aVal: null, bVal: null, dragging: null, activeHandle: 'a' };
let forkOverlay;


let _compareView = localStorage.getItem('fs.compare.view') || 'train';

// lineage maps
// multi-parent lineage
const parentsOf  = new Map();       // child -> Set(parents)
const parentOf   = new Map();       // child -> primary parent (for nav arrows)
const childrenOf = new Map();       // parent -> Set(children)
const runsIndex  = new Map();       // run_id -> row
const extraParents = new Map(); // childId -> Set(parents)
const AF_MARKERS = new Map(); // runKey -> [{ step, kind: 'suggested'|'executed' }]

const RUN_STATE = new Map(); // runKey -> { running:boolean, paused:boolean }
let FOLLOW_ACTIVE = true;
let _followResetTimer = null;


const LAST_PAUSED_STEP = new Map();   // runId -> step
let IS_RUNNING = false;
let IS_PAUSED  = false;
function curRunKey() { return keyOf(currentPageRunId()); }


function reportMatchesPause(runKey = curRunKey()) {
  const cache = REPORT_CACHE.get(cacheKeyFor(runKey));
  const pausedAt = LAST_PAUSED_STEP.get(runKey);
  const atStep = cache?.at_step;
  return Number.isFinite(atStep) && Number.isFinite(pausedAt) && atStep === pausedAt;
}

function updateManualForkGate() {
  if (!btnManualFork) return;
  const hasReport = !!REPORT_CACHE.get(cacheKeyFor(curRunKey()));
  const st = RUN_STATE.get(curRunKey()) || {};
  const allowed = !!st.paused && hasReport && reportMatchesPause();


  btnManualFork.disabled = !allowed;

  if (!hasReport) {
    btnManualFork.title = 'Generate a report (while paused) to enable manual fork.';
  } else if (!reportMatchesPause()) {
    const cache = REPORT_CACHE.get(cacheKeyFor(curRunKey()));
    const paused = LAST_PAUSED_STEP.get(curRunKey());
    btnManualFork.title = `Report is from step ${cache?.at_step ?? '—'}, current pause is step ${paused ?? '—'}. Re-run report to fork.`;
  } else {
    btnManualFork.title = '';
  }
}

const selectedForMerge = new Set();
function pickPrimaryParent(childId) {
  const ps = Array.from(parentsOf.get(childId) || []);
  if (!ps.length) return null;
  // choose newest-by-created_at if available, else first
  ps.sort((a, b) => (runsIndex.get(b)?.created_at || 0) - (runsIndex.get(a)?.created_at || 0));
  return ps[0];
}

/* ========= MODEL NAV ========= */

let NAV_LIST = []; // ["run3", "run2", "run1", "run0"] newest -> oldest

// === PAGE INDEX (newest-left list) ===
let PAGE_INDEX = 0;  // 0 = newest (left-most)

function clampPage(i) {
  return Math.max(0, Math.min(NAV_LIST.length - 1, Number(i) || 0));
}
function currentPageRunId() {
  if (!NAV_LIST.length) return null;
  return NAV_LIST[clampPage(PAGE_INDEX)] || null;
}
function notifyModelNavSelected(runId) {
  vscode.postMessage({ command: 'modelNav.select', runId: String(runId || '') });
}
/** The single entry-point to switch pages */
function gotoPageByIndex(i) {
  PAGE_INDEX = clampPage(i);
  const runId = currentPageRunId();
  if (!runId) {
    updateModelNav();
    return;
  }

  // keep dropdown in sync
  if (runSel) runSel.value = runId;
  notifyModelNavSelected(runId);

  // switch charts/data to that run and (re)paint cached report
  activeRunId = runId;     // tie streaming to viewed page
  clearCharts();
  send('requestRows',   { runId });
  send('requestReport', { runId });   // if cached, showReportFor will display immediately
  showReportFor(runId);

  updateManualForkGate();
  updateModelNav();
}

/** Convenience: jump to whatever index a runId currently occupies */
function gotoPageByRunId(runId) {
  const idx = NAV_LIST.indexOf(String(runId));
  if (idx >= 0) gotoPageByIndex(idx);
}


function rebuildNavListFromRows(rows) {
  // Normalize to {id, created_at}, newest first (left)
  const items = (rows || []).map(r => {
    const id = runIdOf(r);
    const created = r.created_at ?? r.createdAt ?? r.created ?? r.timestamp ?? r.ts ?? 0;
    return id ? { id: String(id), created_at: Number(created) || 0 } : null;
  }).filter(Boolean);

  items.sort((a, b) => b.created_at - a.created_at); // newest first
  NAV_LIST = items.map(x => x.id);
}


function updateModelNav() {
  const { prev, next } = navEls();
  if (!prev || !next) return;

  const idx = clampPage(PAGE_INDEX);
  const hasPrev = idx > 0;
  const hasNext = idx < NAV_LIST.length - 1;

  const prevVal = hasPrev ? NAV_LIST[idx - 1] : '';
  const nextVal = hasNext ? NAV_LIST[idx + 1] : '';

  setArrow(prev, hasPrev, prevVal);
  setArrow(next, hasNext, nextVal);
}



function wireModelNavClicks() {
  const { prev, next } = navEls();

  if (prev && !prev._wired) {
    prev.addEventListener('click', () => {
      followTemporarilyOff();
      if (PAGE_INDEX > 0) gotoPageByIndex(PAGE_INDEX - 1);
    });
    prev._wired = true;
  }
  if (next && !next._wired) {
    next.addEventListener('click', () => {
      followTemporarilyOff();
      if (PAGE_INDEX < NAV_LIST.length - 1) gotoPageByIndex(PAGE_INDEX + 1);
    });
    next._wired = true;
  }
}


function formatWhen(step, epoch) {
  const stepTxt = (Number.isFinite(step) ? step : '—');
  const epTxt   = (Number.isFinite(epoch) ? epoch : '—');
  return `Analyzed at step ${stepTxt} (epoch ${epTxt})`;
}

// --- Fork plugins ---

// --- Fork selection render plugin ---
const forkSelectionPlugin = {
  id: 'forkSelection',
  afterDraw(chart) {
    if (!chart || chart.canvas.id !== 'lossDistChart') return;
    if (!manualForkMode || !forkSel.active) return;

    const scaleX = chart.scales?.x;
    const area = chart.chartArea;
    if (!scaleX || !area) return;

    if (forkSel.aVal == null || forkSel.bVal == null) return;
    const aXraw = scaleX.getPixelForValue(forkSel.aVal);
    const bXraw = scaleX.getPixelForValue(forkSel.bVal);
    const clamp = (x) => Math.max(area.left, Math.min(area.right, x));
    const aX = clamp(aXraw);
    const bX = clamp(bXraw);

    const left = Math.min(aX, bX);
    const right = Math.max(aX, bX);

    const ctx = chart.ctx;
    ctx.save();

    ctx.fillStyle = 'rgba(100, 149, 237, 0.18)';
    ctx.fillRect(left, area.top, right - left, area.bottom - area.top);

    ctx.lineWidth = 2;

    ctx.strokeStyle = '#efe444ff';
    ctx.beginPath();
    ctx.moveTo(aX, area.top);
    ctx.lineTo(aX, area.bottom);
    ctx.stroke();

    ctx.strokeStyle = '#efe444ff';
    ctx.beginPath();
    ctx.moveTo(bX, area.top);
    ctx.lineTo(bX, area.bottom);
    ctx.stroke();

    const cap = 6;
    ctx.beginPath();
    ctx.fillStyle = '#efe444ff';
    ctx.fillRect(aX - 3, area.top - cap, 6, cap);
    ctx.fillStyle = '#efe444ff';
    ctx.fillRect(bX - 3, area.top - cap, 6, cap);

    ctx.restore();

    updateSelectedCountPill();
  }
};
// --- Vertical dotted line markers for AutoFork provenance ---
const autoForkMarkerPlugin = {
  id: 'autoForkMarkers',
  afterDatasetsDraw(chart) {
    const id = chart?.canvas?.id;
    if (!id || (id !== 'lossChart' && id !== 'valLossChart')) return;

    const runKey = keyOf(currentPageRunId());
    const marks = AF_MARKERS.get(runKey);
    if (!marks || !marks.length) return;

    const scaleX = chart.scales?.x;
    const area = chart.chartArea;
    if (!scaleX || !area) return;

    const labels = chart.data?.labels || [];
    const pixelForStep = (step) => {
      let idx = labels.findIndex(v => Number(v) === Number(step));
      if (idx >= 0) return scaleX.getPixelForTick(idx);
      // fallback: nearest existing tick (handles slight mismatches)
      let best = -1, dBest = Infinity;
      for (let i = 0; i < labels.length; i++) {
        const d = Math.abs(Number(labels[i]) - Number(step));
        if (d < dBest) { dBest = d; best = i; }
      }
      return (best >= 0) ? scaleX.getPixelForTick(best) : NaN;
    };

    const ctx = chart.ctx;
    ctx.save();
    ctx.setLineDash([6, 5]);  // dotted
    ctx.lineWidth = 1;

    for (const m of marks) {
      const x = pixelForStep(m.step);
      if (!Number.isFinite(x)) continue;

      ctx.strokeStyle = (m.kind === 'executed') ? '#22c55e' : '#94a3b8' ;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, area.top);
      ctx.lineTo(x + 0.5, area.bottom);
      ctx.stroke();
    }

    ctx.restore();
  }
};


/* ====================================================================
 * 6) Charting
 * --------------------------------------------------------------------
 * Canvas sizing, PNG export via extension, chart init with plugins,
 * y-scale sync. All behavior preserved.
 * ==================================================================== */

function prepareCanvasSizes() {
  ['lossChart','valLossChart','lossDistChart'].forEach(id => {
    const c = byId(id);
    if (!c) return;
    c.style.width = '100%';
    c.style.height = '100%';
    c.removeAttribute('width');
    c.removeAttribute('height');
  });
}
function chartToPngDataURL(chart) {
  if (!chart || !chart.canvas) return null;
  const src = chart.canvas;
  const w = src.width, h = src.height;

  // Compose onto a background that matches the pane for “what you see is what you get”
  const bg = getComputedStyle(src.closest('.chartWrap') || document.body).backgroundColor || '#ffffff';
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  const ctx = out.getContext('2d');
  ctx.save();
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(src, 0, 0);
  ctx.restore();
  return out.toDataURL('image/png');
}

function exportChartViaExtension(chart, suggestedName) {
  const dataUrl = chartToPngDataURL(chart);
  if (!dataUrl) { notify('Chart not ready to export.', 'warn'); return; }
  // Ask the extension to save the PNG (reliable in VS Code webviews)
  vscode.postMessage({ command: 'exportChart', filename: suggestedName, dataUrl });
}
on(btnExportLoss, 'click', () => {
  const fname = `loss_chart_${Date.now()}.png`;
  exportChartViaExtension(lossChart, fname);
});
on(btnExportLossDist, 'click', () => {
  const fname = `loss_distribution_${Date.now()}.png`;
  exportChartViaExtension(lossDistChart, fname);
});
on(btnExportValLoss, 'click', () => {
  const fname = `val_loss_${Date.now()}.png`;
  exportChartViaExtension(val_lossChart, fname);
});


/* -------- charts -------- */
let lossChart, val_lossChart, lossDistChart;
// --- Report request tracking (prevents cross-run bleed) ---
let reportSeq = 0;
const reportReqToRun = new Map();          // reqId -> runId
const latestReqForRun = new Map();         // runId -> reqId
const keyOf = id => (id == null ? '' : String(id));
const runIdOf = (r) => keyOf(r?.run_id ?? r?.id ?? r?.runId ?? r?.uuid ?? r?.uid);

function computeLossHistogram(values, numBins = 30) {
  if (!Array.isArray(values) || values.length === 0) {
    return { bars: [], line: [], xmin: undefined, xmax: undefined };
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const width = (max - min) || 1e-9;
  const step = width / numBins;

  const edges = Array.from({ length: numBins + 1 }, (_, i) => min + i * step);
  const counts = new Array(numBins).fill(0);
  for (const v of values) {
    let idx = Math.floor((v - min) / step);
    if (idx >= numBins) idx = numBins - 1;
    if (idx < 0) idx = 0;
    counts[idx]++;
  }
  const centers = counts.map((_, i) => edges[i] + step * 0.5);

  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, (n - 1));
  const std = Math.sqrt(Math.max(variance, 0));
  let h = 1.06 * (std || (step / 1.06)) * Math.pow(n, -1 / 5);
  if (!Number.isFinite(h) || h <= 1e-12) h = step;
  h = Math.max(0.6 * step, Math.min(3 * step, h));

  const sampleCount = 160;
  const xs = Array.from({ length: sampleCount }, (_, i) => min + (i / (sampleCount - 1)) * (max - min));
  const invNorm = 1 / (Math.sqrt(2 * Math.PI) * h);
  const ySmooth = xs.map(x => {
    let dens = 0;
    for (let i = 0; i < centers.length; i++) {
      const u = (x - centers[i]) / h;
      dens += counts[i] * invNorm * Math.exp(-0.5 * u * u);
    }
    return dens * step;
  });

  return {
    bars: centers.map((x, i) => ({ x, y: counts[i] })),
    line: xs.map((x, i) => ({ x, y: ySmooth[i] })),
    xmin: min,
    xmax: max,
    edges
  };
}

// Cache: one report per runId
const REPORT_CACHE = (window.__REPORT_CACHE__ ||= new Map());
const SUMMARY_CACHE = (window.__SUMMARY_CACHE__ ||= new Map());


const cacheKeyFor = (runId) => `${keyOf(runId)}`;
const cloneXY = (arr) => (Array.isArray(arr) ? arr.map(p => ({ x: +p.x, y: +p.y })) : []);

function showReportFor(runId) {
  if (!lossDistChart) return;
  const cache = REPORT_CACHE.get(cacheKeyFor(runId));

  if (!cache) { clearReportChart(); return; }

  const bars = lossDistChart.data?.datasets?.[0];
  const line = lossDistChart.data?.datasets?.[1];

  if (bars) bars.data = cache.bars.slice();
  if (line) line.data = cache.line.slice();

  // lastLossEdges = (cache.edges || []).slice();
  if (lossDistChart.options?.scales?.x) {
    lossDistChart.options.scales.x.min = cache.xmin;
    lossDistChart.options.scales.x.max = cache.xmax;
  }
  if (reportNote) reportNote.textContent = cache.note || '';
  if (reportMeta)  reportMeta.textContent  = formatWhen(cache.at_step, cache.at_epoch);

  manualForkMode = false;
  forkSel = { active: false, aVal: null, bVal: null, dragging: null, activeHandle: 'a' };
  if (forkOverlay) forkOverlay.style.display = 'none';
  lossDistChart.$ownerKey = cacheKeyFor(runId);
  lossDistChart.update('none');
}


/* ====================================================================
 * 7) Streamed Data Model (STATE/PEND) + redraw pipeline
 * --------------------------------------------------------------------
 * Batched push into PEND, periodic flush to STATE, and apply to charts.
 * ==================================================================== */

/* -------- state (no feature sets) -------- */
let activeRunId = null;
let pendingSelectRunId = null;

/* ========= STREAMING DATA MODEL ========= */
const STATE = {
  labels: [],
  loss: [],
  val_loss: [],
};


const PEND = {
  labels: [],
  loss: [],
  val_loss: [],
};

function pendPush(step, loss, val_loss) {
  if (Number.isFinite(step) && Number.isFinite(loss)) {
    PEND.labels.push(step);
    PEND.loss.push(loss);
    PEND.val_loss.push(Number.isFinite(val_loss) ? val_loss : NaN);
  }
  
}

/* ========= FRAME-PACED REDRAW ========= */
const MIN_DRAW_INTERVAL_MS = 120;
let lastDrawTs = 0;
let rafPending = false;

function scheduleChartsRedraw() {
  const now = performance.now();
  if (now - lastDrawTs < MIN_DRAW_INTERVAL_MS) return;
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => {
    rafPending = false;
    flushPendingToState();
    applyStateToCharts();
    lastDrawTs = performance.now();
  });
}

function flushPendingToState() {
  if (!PEND.labels.length) return;

  if (PEND.labels.length) {
    STATE.labels.push(...PEND.labels);
    STATE.loss.push(...PEND.loss);
    STATE.val_loss.push(...PEND.val_loss);
  }


  PEND.labels.length = 0;
  PEND.loss.length = 0;
  PEND.val_loss.length = 0;

}

function applyStateToCharts() {
  if (!lossChart || !val_lossChart) return;

  lossChart.data.labels = STATE.labels.map(Number);

  const L = STATE.labels.length;
  const loss = STATE.loss.slice(0, L);

  lossChart.data.datasets[0].data = loss;

  val_lossChart.data.labels = STATE.labels.map(Number);
  val_lossChart.data.datasets[0].data = STATE.val_loss.slice(0, L);

  syncLossYScale(loss);

  lossChart.update('none');
  val_lossChart.update('none');
}


/* ====================================================================
 * 8) Log Ring Buffer
 * --------------------------------------------------------------------
 * Efficient log text area updates with minimal DOM churn.
 * ==================================================================== */

const MAX_LOG_LINES = 2000;
const _logRing = new Array(MAX_LOG_LINES);
let _logStart = 0, _logLen = 0;
let _logFlushTimer = null;

// Clear logs (ring buffer + UI) in one place
function clearLogs() {
  _logStart = 0;
  _logLen = 0;
  _logRing.fill(undefined);
  if (_logFlushTimer) {
    clearTimeout(_logFlushTimer);
    _logFlushTimer = null;
  }
  if (logDiv) {
    logDiv.value = '';
    logDiv.scrollTop = 0;
  }
}

function _pushLogLine(s) {
  const idx = (_logStart + _logLen) % MAX_LOG_LINES;
  _logRing[idx] = s;
  if (_logLen < MAX_LOG_LINES) _logLen++;
  else _logStart = (_logStart + 1) % MAX_LOG_LINES;
}

function _flushLog() {
  if (!logDiv) return;
  const out = new Array(_logLen);
  for (let i = 0; i < _logLen; i++) out[i] = _logRing[(_logStart + i) % MAX_LOG_LINES];
  const atBottom = (logDiv.scrollTop + logDiv.clientHeight) >= (logDiv.scrollHeight - 4);
  logDiv.value = out.join('\n');
  if (atBottom) logDiv.scrollTop = logDiv.scrollHeight;
}

function log(t) {
  if (!logDiv) return;
  _pushLogLine(String(t));
  if (!_logFlushTimer) {
    _logFlushTimer = setTimeout(() => {
      _logFlushTimer = null;
      _flushLog();
    }, 200);
  }
}


/* ====================================================================
 * 9) IPC Commands & Buttons
 * --------------------------------------------------------------------
 * `send()` helper + all button click handlers that post messages to host.
 * ==================================================================== */

function send(command, extra = {}) { vscode.postMessage({ command, ...extra }); }

on(btnChoose, 'click', () => send('chooseScript'));
on(btnSetPy, 'click', () => send('setPython', { path: (pyPath && pyPath.value) || 'python' }));


on(btnPause,  'click', () => send('pause',  { runId: currentPageRunId() }));
on(btnTestNow,'click', () => send('testNow', { runId: currentPageRunId() }));
on(btnResume, 'click', () => {
  const rk = keyOf(currentPageRunId());
  // setRunningFor(rk, true);
  // setPausedFor(rk, false);
  send('resume', { runId: rk });
});

on(btnSaveCkpt, 'click', () => send('saveCkpt'));
on(btnAutoSave, 'click', () => send('exportSession'));
on(btnLoad, 'click', () => send('loadSession'));
on(btnReport, 'click', () => {
  const runId = currentPageRunId();
  if (!runId) { notify('No run selected.', 'warn'); return; }

  const reqId  = ++reportSeq;
  const runKey = keyOf(runId);
  reportReqToRun.set(reqId, runKey);
  latestReqForRun.set(runKey, reqId);
  send('generateReport', { runId, reqId });
  notify('Generating fresh report…');
});


// One-click: turn on with defaults and apply
on(btnAutoForkOn, 'click', () => {
  if (!afEnabled) return;
  prefillAutoforkUi();
  afEnabled.checked = true;
  const cfg = readAutoForkConfig();
  cfg.runtime = { ...(cfg.runtime||{}), auto_execute: true };  // ← ensure auto mode
  vscode.postMessage({ command: 'applyAutoForkRules', config: cfg });
  setAutoModeUI(true);  // ← hide manual controls immediately
  reflectAutoForkBtns();
  notify('AutoFork enabled with defaults.');
});

// Turn AutoFork OFF: disable rules + exit auto mode (show manual controls)
on(btnAutoForkOff, 'click', () => {
  if (!afEnabled) return;
  // reflect OFF in the UI model
  afEnabled.checked = false;
  // read current fields, then force-disable and turn off auto_execute
  const cfg = readAutoForkConfig();
  cfg.rules.enabled = false;
  cfg.runtime = { ...(cfg.runtime || {}), auto_execute: false };
  vscode.postMessage({ command: 'applyAutoForkRules', config: cfg });
  // bring back manual controls immediately
  setAutoModeUI(false);
  reflectAutoForkBtns();
  notify('AutoFork disabled. Manual controls restored.');
});


// Apply whatever is currently typed in the fields
on(btnAutoForkApply, 'click', () => {
  if (!afEnabled) return;
  vscode.postMessage({ command: 'applyAutoForkRules', config: readAutoForkConfig() });
  notify('AutoFork rules applied.');
});

// Execute latest suggested plan (optional variant index)
on(btnAutoForkExec, 'click', () => {
  if (!_lastAutoForkPlan) { notify('No AutoFork plan yet.', 'warn'); return; }
  const variantIndex = Number(afVariantIndex?.value ?? 0);
  vscode.postMessage({
    command: 'executeAutoForkPlan',
    plan: _lastAutoForkPlan,
    variantIndex: Number.isFinite(variantIndex) ? variantIndex : 0,
    runId: currentPageRunId(),
  });
  notify('Requested AutoFork execution.');
});






if (btnRefreshRuns) on(btnRefreshRuns, 'click', () => {
 // Alerts are blocked in webviews; ask the extension host to show a native modal.
 send('resetAll');
  if (columnsGrid) {
    columnsGrid.querySelectorAll('.modelCol').forEach(n => n.remove());
    ensureAddTile();
  }
  const rk = currentPageRunId();
  if (rk) ensureSessionColumnForRun(rk);

});

on(btnOpenDag, 'click', () => { openDag(); });
on(dagClose,   'click', () => { closeDag(); });
on(dagMergeBtn,'click', () => {
  if (selectedForMerge.size !== 2) return;
   const parents = Array.from(selectedForMerge); // ← run IDs, not file paths
   const strategy = (dagStrategy && dagStrategy.value) || 'swa';
   vscode.postMessage({ command: 'merge', payload: { parents, strategy } });
   notify(`Requested merge: ${parents.join(' + ')} (${strategy})`);
});


/* ====================================================================
 * 10) Compare Columns UI
 * --------------------------------------------------------------------
 * Compare mode rendering: grid, tabs, columns, summary cache, copy btn.
 * ==================================================================== */

function openDag(){
  if (!dagOverlay) return;
  dagOverlay.classList.add('show');
  renderDag(); // draw current snapshot
}
function closeDag(){
  if (!dagOverlay) return;
  dagOverlay.classList.remove('show');
  selectedForMerge.clear();
  updateMergeUi();
}
function updateMergeUi(){
  if (!dagMergeBtn) return;
  const n = selectedForMerge.size;
  dagMergeBtn.disabled = (n !== 2);
  dagMergeBtn.textContent = (n === 2) ? 'Merge selected' : 'Pick 2 to merge';
}


/* ========= RUN SELECTOR ========= */
if (runSel) {
  runSel.onchange = () => {
    followTemporarilyOff();
    const id  = String(runSel.value);
    const idx = NAV_LIST.indexOf(id);
    if (idx >= 0) gotoPageByIndex(idx);
  };
}


/* ========= CHART INIT ========= */
(function waitForChart() {
  if (typeof Chart !== 'undefined') {
    log(`✅ Chart.js ${Chart.version || ''} loaded`);
    initCharts();

    wireModelNavClicks();
    updateModelNav();

    return;
  }
  const started = Date.now();
  const tick = () => {
    if (typeof Chart !== 'undefined') {
      log(`✅ Chart.js ${Chart.version || ''} loaded`);
      initCharts();

      wireModelNavClicks();
      updateModelNav();
    } else if (Date.now() - started > 3000) {
      log('❌ Chart.js not loaded. Ensure chart.js is resolved and injected.');
      ['lossChart','val_lossChart','lossDistChart'].forEach(id => {
        const el = byId(id);
        if (el && el.parentElement) el.parentElement.style.display = 'none';
      });
    } else {
      setTimeout(tick, 50);
    }
  };
  tick();
})();

function initCharts() {
  if (typeof Chart !== 'undefined') {
    if (!Chart.registry.plugins.get('forkSelection'))   Chart.register(forkSelectionPlugin);
    if (!Chart.registry.plugins.get('autoForkMarkers')) Chart.register(autoForkMarkerPlugin);
  }
  prepareCanvasSizes();
  const defer = (chart) => chart && _deferUntilLayout(chart, () => { chart.resize(); chart.update('none'); });
  const commonOpts = {
    animation: false,
    animations: { colors: false, x: false, y: false },
    parsing: true,
    normalized: true,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { intersect: false},
    devicePixelRatio: 1,
    plugins: {
      legend: { display: false },
      decimation: { enabled: true, algorithm: 'lttb', samples: 500 },
    },
    scales: { x: { type: 'category', ticks: { maxTicksLimit: 8 } } },
    elements: { point: { radius: 0, hitRadius: 0 } }
  };

  const lossCtx = byId('lossChart')?.getContext('2d');
  if (lossCtx) {
    lossChart = new Chart(lossCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { label: 'Loss', data: [], borderColor: 'blue', fill: false, yAxisID: 'y' }
        ]
      },
      options: {
        ...commonOpts,
        scales: {
          ...commonOpts.scales,
          y: { type: 'linear', position: 'left', title: { display: true, text: 'Train Loss' } }
        }
      }
    });
    defer(lossChart);
  }

  const val_lossCtx = byId('valLossChart')?.getContext('2d');
  if (val_lossCtx) {
    val_lossChart = new Chart(val_lossCtx, {
      type: 'line',
      data: { labels: [], datasets: [{ label: 'VAL', data: [], borderColor: 'orange', fill: false }] },
            options: {
        ...commonOpts,
        scales: {
          ...commonOpts.scales,
          y: { type: 'linear', position: 'left', title: { display: true, text: 'Validation Loss' } }
        }
      }
    });
    defer(val_lossChart);
  }


  const rdc = byId('lossDistChart')?.getContext('2d');
  if (rdc) {
   lossDistChart = new Chart(rdc, {
     type: 'bar',
     data: {
       datasets: [
         { type: 'bar',  label: 'Loss frequency', data: [], parsing: false, borderWidth: 1, backgroundColor: 'rgba(128, 0, 128, 0.15)' },
         { type: 'line', label: 'Loss density (smooth)', data: [], parsing: false, pointRadius: 0, tension: 0.3, borderWidth: 2, borderColor: 'purple' }
       ]
     },
     options: {
       ...commonOpts,
       plugins: { ...commonOpts.plugins, decimation: { enabled: false } },
       scales: { x: { type: 'linear', ticks: { maxTicksLimit: 8 }, offset: false }, y: { type: 'linear', beginAtZero: true } }
     }
    });
    defer(lossDistChart);
    enableDragOnReportChart();
  }
}

function syncLossYScale(lossData) {
  if (!lossChart?.options?.scales) return;
  const ls = (lossData || []).filter(Number.isFinite);
  if (ls.length) {
    const minL = Math.min(...ls), maxL = Math.max(...ls);
    const padL = (maxL - minL) * 0.05 || 1e-6;
    lossChart.options.scales.y.min = minL - padL;
    lossChart.options.scales.y.max = maxL + padL;
  }
}


/* ====================================================================
 * 11) Clear/Reset helpers
 * --------------------------------------------------------------------
 * Destroy/re-init charts and clear report visuals safely.
 * ==================================================================== */

function resetPendingBuffers() {
  PEND.labels.length = 0;
  PEND.loss.length = 0;
  PEND.val_loss.length = 0;
}
function clearCharts() {
  lossChart?.destroy();
  val_lossChart?.destroy();
  lossDistChart?.destroy();
  lossDistChart = null;

  STATE.labels.length = 0;
  STATE.loss.length = 0;
  STATE.val_loss.length = 0;

  resetPendingBuffers();           // ← prevent single-point bleed across runs
  rafPending = false;              // ← drop any queued frame using stale PEND
  lastDrawTs = 0;

  prepareCanvasSizes();
  initCharts();
}

function clearReportChart() {
  if (!lossDistChart) return;
  const bars = lossDistChart.data?.datasets?.[0];
  const line = lossDistChart.data?.datasets?.[1];
  if (bars) bars.data = [];
  if (line) line.data = [];

  if (reportMeta) reportMeta.textContent = '';
  if (reportNote) reportNote.textContent = '';

  manualForkMode = false;
  forkSel = { active: false, aVal: null, bVal: null, dragging: null, activeHandle: 'a' };
  if (forkOverlay) forkOverlay.style.display = 'none';
  lossDistChart.update('none');
}



/* ====================================================================
 * 12) Nav helpers (arrows) & lineage rebuild
 * --------------------------------------------------------------------
 * Compute arrows’ targets, rebuild parent/child maps from rows.
 * ==================================================================== */

function setArrow(btn, enabled, targetId) {
  if (!btn) { logNav('setArrow: no btn'); return; }
  btn.dataset.target = enabled ? keyOf(targetId) : '';
  logNav('setArrow', { id: btn.id, enabled, targetId: btn.dataset.target });

  if (enabled) {
    btn.disabled = false;
    btn.removeAttribute('disabled');
    btn.removeAttribute('aria-disabled');
    btn.style.opacity = '';
    btn.style.pointerEvents = '';
  } else {
    btn.disabled = true;
    btn.setAttribute('disabled', '');
    btn.setAttribute('aria-disabled', 'true');
    btn.style.opacity = '0.4';
    btn.style.pointerEvents = 'none';
  }
}


function rebuildLineageFromRows(rows){
  parentsOf.clear(); parentOf.clear(); childrenOf.clear(); runsIndex.clear();
  if (!Array.isArray(rows) || rows.length === 0) return;

  for (const r of rows) {
    const id = runIdOf(r); if (!id) continue;
    const created = r.created_at ?? r.createdAt ?? r.created ?? r.timestamp ?? r.ts ?? 0;
    runsIndex.set(id, { ...r, created_at: created });

    // parents from rows (array preferred, otherwise legacy single-parent field)
    const rowParents =
      Array.isArray(r.parents) ? r.parents :
      [ r.parent ?? r.parent_run ?? r.parent_run_id ?? r.parentId ?? null ];

    // parents learned from runtime `newRun` events (may include multi-parent for merges)
    const learned = extraParents.get(id);
    const ps = new Set(
      rowParents
        .map(keyOf)
        .filter(p => p && p !== id)
    );
    if (learned) for (const p of learned) if (p && p !== id) ps.add(keyOf(p));

    if (ps.size) {
      parentsOf.set(id, ps);
      for (const p of ps) {
        if (!childrenOf.has(p)) childrenOf.set(p, new Set());
        childrenOf.get(p).add(id);
      }
    }
  }

  // derive a primary parent for prev/next arrows
  for (const id of runsIndex.keys()) {
    const p = pickPrimaryParent(id);
    if (p && p !== id) parentOf.set(id, p);
  }
}

function navEls() {
  const els = {
    prev: document.getElementById('btnPrevModel'),
    next: document.getElementById('btnNextModel'),
    sel:  document.getElementById('runSel'),
  };
  logNav('navEls()', {
    prev: !!els.prev, next: !!els.next, sel: !!els.sel,
    prevDisabled: els.prev?.disabled, nextDisabled: els.next?.disabled
  });
  return els;
}
// --- Merge gating banner (no HTML changes needed) ---
let _mergeBannerEl = null, _mergeBannerTimer = null;

function ensureMergeBanner() {
  if (_mergeBannerEl) return _mergeBannerEl;
  const el = document.createElement('div');
  el.id = 'mergeBanner';
  Object.assign(el.style, {
    position: 'fixed',
    right: '16px',
    bottom: '16px',
    zIndex: 9999,
    display: 'none',
    alignItems: 'center',
    gap: '8px',
    padding: '10px 12px',
    borderRadius: '10px',
    background: 'var(--btn-bg, #1f2937)',
    color: 'var(--btn-fg, #fff)',
    boxShadow: 'var(--btn-shadow, 0 6px 20px rgba(0,0,0,0.18))',
    fontWeight: 600,
    fontSize: '12px',
    maxWidth: '42ch',
    lineHeight: '1.25',
  });
  document.body.appendChild(el);
  _mergeBannerEl = el;
  return el;
}

function showMergeBanner(text, kind = 'info', { sticky = false } = {}) {
  const el = ensureMergeBanner();
  el.textContent = ''; // reset
  const icon = document.createElement('span');
  icon.textContent = kind === 'error' ? '⚠️' : (kind === 'busy' ? '⏳' : 'ℹ️');
  el.appendChild(icon);
  el.appendChild(document.createTextNode(text));
  el.style.display = 'flex';

  if (!sticky) {
    if (_mergeBannerTimer) clearTimeout(_mergeBannerTimer);
    _mergeBannerTimer = setTimeout(() => { el.style.display = 'none'; }, 3500);
  }
}

function formatParents(parents) {
  return (Array.isArray(parents) && parents.length) ? parents.map(String).join(' + ') : 'parents';
}

function humanizeMergeGating(m) {
  const r = String(m?.reason || 'unknown');
  switch (r) {
    case 'engine_error':           return 'Merge engine error. Check logs for details.';
    case 'awaiting_signal':        return 'Merge pending: waiting for a suggestion.';
    case 'auto_merge_disabled':    return 'Auto-merge disabled. Enable it to proceed automatically.';
    case 'awaiting_checkpoint': {
      const haveP = !!m.have_parent_ckpt;
      const haveC = !!m.have_child_ckpt;
      const pTxt = haveP ? 'parent ✓' : 'parent ✗';
      const cTxt = haveC ? 'child ✓'  : 'child ✗';
      return `Merge pending: checkpoints (${pTxt}, ${cTxt}).`;
    }
    case 'saving_child_checkpoint': return `Saving checkpoint for child ${m.child_id || ''}…`.trim();
    case 'merging':                 return `Merging ${formatParents(m.parents)}…`;
    default:
      return 'Paused';
  }
}
// Track last reason shown per run (prevents spam)
const LAST_MERGE_REASON = new Map(); // runKey -> reason


function shouldShowMergeGating(m) {
  const rk = String(m.run_id || curRunKey());
  if (rk !== String(curRunKey())) return false;    // ignore off-screen runs
  const prev = LAST_MERGE_REASON.get(rk);
  if (prev === m.reason) return false;                 // no repeat spam
  LAST_MERGE_REASON.set(rk, m.reason);
  return true;
}

function hideMergeBanner() {
  const el = document.getElementById('mergeBanner');
  if (el) el.style.display = 'none';
}

// ===================== Model Comparison (Columns) =====================
// Clean, text-first columns with a vertical TRAIN/TEST rail.
// IPC (host):
//   send('fs.summary.list')        -> type: 'fs.summary.list.result'  { runs:[{id,label}] }
//   send('fs.summary.get', {runId,view}) -> type: 'fs.summary.get.result' { runId, view, text }
//   send('fs.summary.pick', {current})    -> type: 'fs.summary.pick.result' { previous, chosen:{id,label} }

// --- elements ---
const grid       = byId('forgeGrid');
const compareEl  = (wCompare && (wCompare.closest('.widget') || wCompare)) || null;
const layoutBar  = byId('customBar') || document.querySelector('.customBar');

// mark the compare card once so CSS can target it
compareEl?.classList.add('is-compare-card');

// --- CSS installed once ---
(function installCompareCSS(){
  if (document.getElementById('compareCSS')) return;
  const style = document.createElement('style');
  style.id = 'compareCSS';
  style.textContent = `
    /* baseline: never show the compare card unless explicitly in compare mode */
    #forgeGrid .widget.is-compare-card { display: none !important; }

    /* compare mode: show ONLY the compare card */
    #forgeGrid[data-mode="compare"] .widget { display: none !important; }
    #forgeGrid[data-mode="compare"] .widget.is-compare-card { display: block !important; }
  `;
  document.head.appendChild(style);
})();


// --- the one setter: updates UI + persists ---
function setMode(mode) {
  const isCompare = mode === 'compare';
  grid?.setAttribute('data-mode', isCompare ? 'compare' : 'create');

  // keep the tabs mutually exclusive
  tabCreate?.setAttribute('aria-selected', String(!isCompare));
  tabCompare?.setAttribute('aria-selected', String(isCompare));

  // hide layout bar in compare
  if (layoutBar) layoutBar.style.display = isCompare ? 'none' : '';

  localStorage.setItem('fs.tab', isCompare ? 'compare' : 'create');
  if (isCompare) {
    ensureDefaultCompareColumn();
    refreshAllColumns();
  }
    
}

// --- wire clicks ---
on(tabCreate,  'click', () => setMode('create'));
on(tabCompare, 'click', () => setMode('compare'));

// --- boot (no first-boot dance) ---
setMode(localStorage.getItem('fs.tab') === 'compare' ? 'compare' : 'create');



// ---- rail + columns ----
function ensureAddTile() {
  if (!columnsGrid || !btnAddColumn) return;
  // keep the add-tile as the last child
  columnsGrid.appendChild(btnAddColumn);
}

// --- Create a run-based column (used for the default/current run) ---
function makeRunColumn(runId, label) {
  if (!columnsGrid || !btnAddColumn) return null;
  const el = document.createElement('article');
  el.className = 'modelCol';
  el.dataset.runId = String(runId);

  const head = document.createElement('header');
  head.className = 'modelCap';
  const cap = document.createElement('div');
  cap.className = 'capInner';
  cap.innerHTML = `
    <span class="capName">${escapeHtml(label || runId)}</span>
    <span class="capSub">${escapeHtml(_compareView)} summary</span>
  `;
  const copyBtn = makeCopyBtn();
  head.append(cap, copyBtn);

  const pre = document.createElement('pre');
  pre.className = 'summaryBox';
  pre.textContent = '(loading…)';

  el.append(head, pre);
  columnsGrid.insertBefore(el, btnAddColumn);
  ensureAddTile();
  return el;
}



function ensureDefaultCompareColumn() {
  if (!grid || grid.getAttribute('data-mode') !== 'compare') return;
  if (!columnsGrid) return;
  if (columnsGrid.querySelector('.modelCol')) return;

  const currentRun = (runSel && runSel.value) || activeRunId;
  if (!currentRun) return;

  // NEW: if the current run has a session_id, show that session’s aggregate by default
  if (ensureSessionColumnForRun(currentRun)) return;

  // Fallback: old behavior (run-based column)
  const label = (runsIndex.get(String(currentRun))?.name) || String(currentRun);
  makeRunColumn(currentRun, label);
  requestSummary(currentRun, 'train');
  requestSummary(currentRun, 'test');
}

function invalidateCompareSummariesForRun(runKey){
  if (!runKey) return;
  ['train','test'].forEach(v => SUMMARY_CACHE.delete(`${runKey}:${v}`));
}

function invalidateVisibleSessionAggregates(){
  // Drop cached aggregate for any session columns we currently show
  columnsGrid?.querySelectorAll('.modelCol[data-session-id]').forEach(col=>{
    const st = COL_STATE.get(col);
    if (!st) return;
    ['train','test'].forEach(v => SUMMARY_CACHE.delete(sumKey(st.sessionId, __AGG, v)));
  });
}


function requestRunList()   { send('fs.summary.list'); }
function requestSummary(id, view) { send('fs.summary.get', { runId: id, view }); }
function requestPick(current)     { send('fs.summary.pick', { current }); }

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }
function cssAttr(s)    { return String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"'); }

function makeColumnForSession(session) {
  const el = document.createElement('article');
  el.className = 'modelCol';
  el.dataset.sessionId = session.id;

  const head = document.createElement('header');
  head.className = 'modelCap';
  head.tabIndex = 0;
  head.title = 'Change session';


  const cap = document.createElement('div');
  cap.className = 'capInner';
  cap.innerHTML = `
    <span class="capName">${escapeHtml(session.label || session.id)}</span>
    <span class="capSub">${escapeHtml(_compareView)} summary</span>
  `;
  const copyBtn = makeCopyBtn();
  head.append(cap, copyBtn);

  const pre = document.createElement('pre');
  pre.className = 'summaryBox';
  pre.textContent = '(loading…)';

  el.append(head, pre);
  columnsGrid.insertBefore(el, btnAddColumn);
  ensureAddTile();

 // Track “aggregate” and fetch immediately
 COL_STATE.set(el, { sessionId: session.id, runId: __AGG });
 setColumnSubtitle(el, _compareView);
 fillSummary(el, session.id, __AGG, _compareView);
 const other = _compareView === 'train' ? 'test' : 'train';
 requestSessionSummary(session.id, __AGG, other);

  return el;
}

function setColumnSubtitle(colEl, view){
  const sub = colEl.querySelector('.capSub');
  if (sub) sub.textContent = `${view} summary`;
}

function fillSummary(colEl, sessionId, runId, view) {
  const key = sumKey(sessionId, runId, view);
  const cached = SUMMARY_CACHE.get(key);
  if (typeof cached === 'string' && cached.trim()) {
    const pre = colEl.querySelector('.summaryBox');
    if (pre) pre.textContent = cached;
  } else {
    const pre = colEl.querySelector('.summaryBox');
    pre.textContent = '(loading…)';
    requestSessionSummary(sessionId, runId, view);
  }
}

function setSummary(runId, view, text) {
  const key = `${runId}:${view}`;
  const clean = (typeof text === 'string') ? text : '';
  if (clean.trim()) SUMMARY_CACHE.set(key, clean);
  else               SUMMARY_CACHE.delete(key); // don't poison cache with empty
  const col = columnsGrid?.querySelector(`.modelCol[data-run-id="${cssAttr(runId)}"]`);
  if (!col) return;
  const pre = col.querySelector('.summaryBox');
  const sub = col.querySelector('.capSub');
  if (pre) pre.textContent = clean.trim() ? clean : '(loading…)'; // show loading until we have real data
  if (sub) sub.textContent = `${view} summary`;
}

function refreshAllColumns() {
  if (!columnsGrid) return;

  // 1) Update session-based columns (tracked in COL_STATE)
  columnsGrid.querySelectorAll('.modelCol[data-session-id]').forEach(col => {
    const st = COL_STATE.get(col);
    if (!st) return;
    setColumnSubtitle(col, _compareView);
    fillSummary(col, st.sessionId, __AGG, _compareView);
  });

  // 2) Update run-based columns (the default “current run” column)
  columnsGrid.querySelectorAll('.modelCol[data-run-id]').forEach(col => {
    const runId = col.dataset.runId;
    const key = `${runId}:${_compareView}`;
    const pre = col.querySelector('.summaryBox');
    const cached = SUMMARY_CACHE.get(key);
    const ready = (typeof cached === 'string') && cached.trim();
    if (pre) pre.textContent = ready ? cached : '(loading…)';
    if (!ready) requestSummary(runId, _compareView);
    const sub = col.querySelector('.capSub');
    if (sub) sub.textContent = `${_compareView} summary`;
  });
}

(function () {
  const PLACEHOLDER_RE = /^\s*(\(loading…\)|\(loading\.\.\.\)|loading|—|-|no data|n\/a)\s*$/i;

  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-copy-summary]');
    if (!btn) return;

    // nearest column (works for single-pane and compare columns)
    const col = btn.closest('.modelCol') || document.getElementById('singleSessionCol');
    const pre = col && col.querySelector('.summaryBox');
    const text = (pre && pre.textContent || '').trim();

    if (!text || PLACEHOLDER_RE.test(text)) return; // nothing to copy

    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // simple fallback
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed'; ta.style.top = '-10000px';
      document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
    }
    // quick visual feedback
    const orig = btn.innerHTML;
    btn.innerHTML = '<svg class="exportIcon" viewBox="0 0 24 24"><path d="M5 13l4 4L19 7" stroke="currentColor" stroke-width="2" fill="none"/></svg>';
    setTimeout(() => (btn.innerHTML = orig), 800);
  });
})();

function makeCopyBtn() {
  const b = document.createElement('button');
  b.type = 'button';
  b.title = 'Copy summary';
  b.className = 'exportBtn';
  b.setAttribute('data-copy-summary', '');   // <- picked up by the delegated handler
  b.innerHTML = `
    <svg class="exportIcon" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
      <path d="M16 3H5a2 2 0 0 0-2 2v11h2V5h11V3zm3 4H9a2 2 0 0 0-2 2v12h12a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2zm0 14H9V9h10v12z" fill="currentColor"/>
    </svg>`;
  return b;
}




// ---- events ----

on(compareRail, 'click', (e) => {
  const btn = e.target.closest('.railTab');
  if (!btn) return;
  const view = btn.dataset.view;
  if (!view || view === _compareView) return;
  compareRail.querySelectorAll('.railTab').forEach(b => b.setAttribute('aria-selected', b === btn ? 'true' : 'false'));
  _compareView = view;
  localStorage.setItem('fs.compare.view', _compareView);
  refreshAllColumns();
});

on(btnClearColumns, 'click', () => {
  if (!columnsGrid) return;
  columnsGrid.querySelectorAll('.modelCol').forEach(n => n.remove());
  ensureAddTile();
});


/* ====================================================================
 * 11.5) Window Events
 * --------------------------------------------------------------------
 * Global listeners (resize) that affect rendering cadence.
 * ==================================================================== */

window.addEventListener('resize', scheduleChartsRedraw);


/* ====================================================================
 * 11.6) Message Handler
 * --------------------------------------------------------------------
 * Single switchboard for messages from the extension host.
 * ==================================================================== */

window.addEventListener('message', (e) => {
  const m = e.data;

  switch (m.type) {
    case 'resetOk': {
      // Now it’s safe to clear the webview state because the user confirmed.
      try {
        clearLogs();
        parentsOf.clear(); parentOf.clear(); childrenOf.clear(); runsIndex.clear(); extraParents.clear();
        selectedForMerge.clear();
        REPORT_CACHE.clear();
        LAST_PAUSED_STEP.clear();
        AF_MARKERS.clear();

        clearCharts();
        clearReportChart();

        if (runSel) runSel.innerHTML = '';
        activeRunId = null;
        pendingSelectRunId = null;
        IS_RUNNING = false;
        IS_PAUSED = false;
        setRunning(false);
        updateModelNav();

        manualForkMode = false;
        forkSel = { active:false, aVal:null, bVal:null, dragging:null, activeHandle:'a' };
        if (forkOverlay) forkOverlay.style.display = 'none';

        log('Session reset.');
      } catch (e) {
        console.error('Reset UI error:', e);
      }
      break;
    }
    case 'scriptChosen':
      if (scriptName) scriptName.textContent = `Chosen Python Script: ${m.file}`;
      break;

    case 'session_started':
      if (m.run_id) pendingSelectRunId = m.run_id;
      send('requestRuns');
      break;

    case 'testNow':

      break;

    case 'newRun': {
      const child = keyOf(m.run_id || '');
      const mode = (m.meta && m.meta.mode) || null;
      if (mode) setAutoModeUI(mode === 'auto');
      if (child) {
        const ps = Array.isArray(m.parents) ? m.parents.map(keyOf).filter(Boolean) : [];
        if (ps.length) extraParents.set(child, new Set(ps));

        if (FOLLOW_ACTIVE) {
          pendingSelectRunId = child;   // let fillRunSel drive the page switch
        }
        send('requestRuns');            // always refresh the list
        } else {
          send('requestRuns');
          wireModelNavClicks();
          updateModelNav();
        }
      break;
    }

    case 'fs.session.runs.result': {
      const { sessionId, runs } = m;
      const col = columnsGrid?.querySelector(`.modelCol[data-session-id="${cssAttr(sessionId)}"]`);
      if (!col) break;
      const mini = col.querySelector('.runMiniSel');
      if (!mini) break;

      const prev = mini.value || __AGG;
      // const options = [`<option value="${__AGG}">Session overview</option>`]
      //   .concat((runs||[]).map(r => `<option value="${escapeHtml(r.id)}">${escapeHtml(r.label || r.id)}</option>`));
      // mini.innerHTML = options.join('');
      mini.disabled = false;

      // restore old selection if possible
      const want = Array.from(mini.options).some(o => o.value === prev) ? prev : __AGG;
      mini.value = want;
      const st = COL_STATE.get(col);
      if (st) { st.runId = want; fillSummary(col, st.sessionId, st.runId, _compareView); }
      break;
    }

    case 'fs.session.summary.get.result': {
      const sessionId = String(m.sessionId ?? m.session_id ?? '');
      const runId     = String(m.runId ?? m.run_id ?? __AGG);
      const view      = String(m.view ?? _compareView ?? 'train').toLowerCase();
      const text      = (typeof m.text === 'string') ? m.text : '';
      const k = sumKey(sessionId, runId, view);
      if (text.trim()) SUMMARY_CACHE.set(k, text); else SUMMARY_CACHE.delete(k);
      const col = columnsGrid?.querySelector(`.modelCol[data-session-id="${cssAttr(sessionId)}"]`);
      if (!col) break;

      const st = COL_STATE.get(col);
      // Only paint if the column is still showing this run+view
      if (st && st.runId === runId && _compareView === view) {
        const pre = col.querySelector('.summaryBox');
        if (pre) pre.textContent = text.trim() ? text : '(loading…)';
        setColumnSubtitle(col, view);
      }
      break;
    }



    case 'fs.summary.get.result': {
      const runId = String(m.runId ?? m.run_id ?? '');
      if (!runId) break;
      const v = String(m.view ?? _compareView ?? 'train').toLowerCase().trim();
      const txt = (typeof m.text === 'string') ? m.text : '';
      const key = `${runId}:${v}`;
      if (txt.trim()) SUMMARY_CACHE.set(key, txt); else SUMMARY_CACHE.delete(key); // cache, but don't poison

      // Only paint if incoming view is the one currently selected in the rail
      if (v === _compareView) {
        const col = columnsGrid?.querySelector(`.modelCol[data-run-id="${cssAttr(runId)}"]`);
        if (col) {
          const pre = col.querySelector('.summaryBox');
          if (pre) pre.textContent = txt.trim() ? txt : '(loading…)';
          setColumnSubtitle(col, v);
        }
      }
      break;
    }

    case 'fs.summary.pick.result': {
      // { previous, chosen:{id,label} }
      const { previous, chosen } = m;
      if (!columnsGrid || !previous || !chosen?.id) break;
      const col = columnsGrid.querySelector(`.modelCol[data-run-id="${cssAttr(previous)}"]`);
      if (col) {
        col.dataset.runId = chosen.id;
        const name = col.querySelector('.capName');
        const pre  = col.querySelector('.summaryBox');
        const sub  = col.querySelector('.capSub');
        if (name) name.textContent = chosen.label || chosen.id;
        if (sub)  sub.textContent  = `${_compareView} summary`;
        if (pre)  pre.textContent  = '(loading…)';
        requestSummary(chosen.id, _compareView);
      }
      break;
    }

    case 'auto_fork_suggested': {
      _lastAutoForkPlan = m.plan || null;
      renderAutoForkPlan(_lastAutoForkPlan);
      // record a marker at the plan's checkpoint
      const runKey = keyOf(m.run_id || currentPageRunId());
      const step   = Number(m.plan?.at_step ?? m.step);
      if (Number.isFinite(step)) addAutoForkMarker(runKey, step, 'suggested');
      notify(`AutoFork suggested: ${_lastAutoForkPlan?.reason || 'plan'}`);
      break;
    }


    case 'auto_fork_executed': {
      // Refresh runs so the new child appears in the selector/DAG immediately
      const child = (typeof m.child_run === 'string') ? m.child_run : '';
      const idx = Number.isFinite(m.variant_index) ? Number(m.variant_index) : null;
      if (child) {
        notify(`AutoFork executed → new run ${child}${idx != null ? ` (variant ${idx})` : ''}`);
      } else {
        notify(`AutoFork executed${idx != null ? ` (variant ${idx})` : ''}.`);
      }
      // Optionally reflect the executed plan in the plan box
      if (m.plan) { _lastAutoForkPlan = m.plan; renderAutoForkPlan(_lastAutoForkPlan); }

      // record an executed marker, too
      const runKey = keyOf(m.run_id || currentPageRunId());
      const step   = Number(m.plan?.at_step ?? m.step);
      if (Number.isFinite(step)) addAutoForkMarker(runKey, step, 'executed');

      send('requestRuns');
      break;
    }

    case 'auto_fork_rules_set':
      // If Python echoed runtime.auto_execute, gate UI accordingly
      if (m.config?.runtime && typeof m.config.runtime.auto_execute === 'boolean') {
        setAutoModeUI(!!m.config.runtime.auto_execute);
      }
    break;

    case 'merge_gating': {
      if (!shouldShowMergeGating(m)) break;

      if (m.reason === 'cleared') {
        LAST_MERGE_REASON.delete(curRunKey());
        hideMergeBanner();
        break;
      }

      const msg  = humanizeMergeGating(m);
      const kind = m.reason === 'engine_error' ? 'error' : (m.reason === 'merging' ? 'busy' : 'info');
      showMergeBanner(msg, kind, { sticky: m.reason === 'merging' });
      log(`[merge] ${msg}`);
      break;
    }

    case 'autoMergeExecuted':
    case 'auto_merge_executed': {
      // Clear any merge banner + de-dupe state for the active run
      const rk = keyOf(m.run_id || curRunKey());
      LAST_MERGE_REASON.delete(rk);
      hideMergeBanner();

      // Optional: reflect lineage so the DAG/selector knows this was a merge
      if (m.new_run && Array.isArray(m.merged) && m.merged.length) {
        extraParents.set(String(m.new_run), new Set(m.merged.map(keyOf)));
      }

      // Nudge the UI to refresh runs / rows
      send('requestRuns');

      if (FOLLOW_ACTIVE && m.new_run) {
        const newId = keyOf(m.new_run);
        activeRunId = newId;
        clearCharts();
        send('requestRows', { runId: newId });
        showReportFor(newId);
        updateModelNav();
      }

      // Nice to have: toast
      try {
        const parents = (m.merged || []).map(String).join(' + ');
        notify(`Merge completed: ${parents} → ${m.new_run}`);
      } catch {}

      break;
    }

    case 'status': {
      const rk = keyOf(m.run_id || curRunKey());
      setRunningFor(rk, !!m.running);
      log(`Status: ${m.running ? 'running' : 'idle'}${m.run_id ? ` (run ${rk})` : ''}`);
      break;
    }

    case 'runs': {
      fillRunSel(m.rows);

      const rk = keyOf(activeRunId || (runSel && runSel.value));
      const row = (m.rows || []).find(r => keyOf(runIdOf(r)) === rk);
      if (row && row.mode) setAutoModeUI(String(row.mode).toLowerCase()==='auto');
      wireModelNavClicks();     // ensure listeners exist if buttons were late
      updateModelNav();
      if (dagOverlay && dagOverlay.classList.contains('show')) renderDag();
      updateManualForkGate();
      // if user is on the Compare tab and no columns yet, add current run by default
      ensureDefaultCompareColumn();
      if (grid?.getAttribute('data-mode') === 'compare') {
        // If any session-aggregate columns are visible, ensure they re-fetch
        refreshAllColumns();
      }
      break;
    }

    case 'rows': {
      const rows = m.rows || [];
      resetPendingBuffers();
      STATE.labels = rows.map(r => r.step);
      STATE.loss   = rows.map(r => nf(r.loss));
      STATE.val_loss    = rows.map(r => nf(r.val_loss));
      scheduleChartsRedraw();
      break;
    }
    case 'trainStep': {
      if (!activeRunId) activeRunId = m.run_id;
      if (m.run_id && activeRunId && m.run_id !== activeRunId) break;
      pendPush(m.step, m.loss, m.val_loss);
      scheduleChartsRedraw();
      break;
    }
    case 'paused': {
      log('Training is paused.');
      const rk = keyOf(m.run_id || currentPageRunId());
      LAST_PAUSED_STEP.set(rk, Number(m.step) || 0);
      setRunningFor(rk, false);   // <- add this
      setPausedFor(rk, true);
      if (columnsGrid) {
        columnsGrid.querySelectorAll('.modelCol').forEach(n => n.remove());
        ensureAddTile();
      }
      ensureSessionColumnForRun(rk);
      if (rk === curRunKey() && btnReport) btnReport.disabled = false;
      break;
    }

    case 'resumed': {
      log('Training resumed...');
      const rk = keyOf(m.run_id || currentPageRunId());
      setRunningFor(rk, true);    // <- add this
      setPausedFor(rk, false);
      if (rk === curRunKey() && btnReport) btnReport.disabled = true;
      break;
    }


    case 'trainingFinished': {
      log('Training finished.');
      const rk = keyOf(m.run_id || currentPageRunId());
      setRunningFor(rk, false);
      setPausedFor(rk, false);

      // freshen run-level cache
      invalidateCompareSummariesForRun(rk);
      invalidateVisibleSessionAggregates();

      // If Compare is visible, kick off fresh fetches so panes populate right away
      if (grid?.getAttribute('data-mode') === 'compare' && columnsGrid) {
        columnsGrid.querySelectorAll('.modelCol').forEach(col => {
          const st = COL_STATE.get(col);
          const pre = col.querySelector('.summaryBox');
          if (pre) pre.textContent = '(loading…)';
          if (col.dataset.runId === rk) {               // run-based cols
            requestSummary(rk, 'train');
            requestSummary(rk, 'test');
          } else if (st?.sessionId) {                   // session aggregate cols
            requestSessionSummary(st.sessionId, __AGG, 'train');
            requestSessionSummary(st.sessionId, __AGG, 'test');
          }
        });
      }
      if (columnsGrid) {
        columnsGrid.querySelectorAll('.modelCol').forEach(n => n.remove());
        ensureAddTile();
      }
      ensureSessionColumnForRun(rk);
      break;
    }

    case 'sessionLoaded':
      log('Session loaded. Refreshing runs...');
      send('requestRuns');
      const cr = currentPageRunId();
      if (cr) notifyModelNavSelected(cr);
      break;
    case 'log':
      log(m.text);
      break;
    case 'error':
      log(`[stderr] ${m.text}`);
      break;

    case 'reportData': {
      const { losses, meta, reqId, owner_run_id } = m;

      // Require reqId to route the response; ignore anything else.
      if (reqId == null || !reportReqToRun.has(reqId)) {
        log('(ignored) Report without a known reqId');
        return;
      }

      const intendedRunKey = reportReqToRun.get(reqId);
      // Drop stale/out-of-order replies for that run.
      if (latestReqForRun.get(intendedRunKey) !== reqId) {
        log(`(ignored) Stale report for ${intendedRunKey} (reqId=${reqId})`);
        return;
      }

      // Enforce attribution: the payload MUST say which run it belongs to.
      if (keyOf(owner_run_id) !== intendedRunKey) {
        log(`(ignored) Report attributed to ${owner_run_id}, expected ${intendedRunKey} (reqId=${reqId})`);
        return;
      }

      const cacheKey = cacheKeyFor(intendedRunKey);

      const reportCache =
      (window.__REPORT_CACHE__ = window.__REPORT_CACHE__ || new Map()); // key: runId -> { ... }

      function writeCache(bars, line, xmin, xmax, note, meta) {
        REPORT_CACHE.set(cacheKey, {
          bars: cloneXY(bars),
          line: cloneXY(line),
          xmin,
          xmax,
          note: note || '',
          at_step: meta?.at_step ?? null,
          at_epoch: meta?.at_epoch ?? null,
        });
      }

      let bars = [], line = [], xmin, xmax, edges = [], note = '';
      if (!Array.isArray(losses) || !losses.length) {
        log('No losses returned for report.');
        note = (meta?.note) || 'No per-sample losses available.';
      } else {
        const ownerHint = ` (run: ${intendedRunKey})`;
        note = `Samples: ${losses.length}${ownerHint}` + (meta?.note ? `\n${meta.note}` : '');
        ({ bars, line, xmin, xmax, edges } = computeLossHistogram(losses, 30));
      }
      lossDistChart.$ownerKey = cacheKey;
      writeCache(bars, line, xmin, xmax, note, meta);

      const isVisible = (keyOf(currentPageRunId()) === intendedRunKey);
      if (isVisible && lossDistChart) {
        if (reportMeta) reportMeta.textContent = formatWhen(meta?.at_step, meta?.at_epoch);
        _deferUntilLayout(lossDistChart, () => {
          const incomingKey = cacheKey;
          if (lossDistChart.$ownerKey && lossDistChart.$ownerKey !== incomingKey) {
            if (lossDistChart.options?.scales?.x) {
              delete lossDistChart.options.scales.x.min;
              delete lossDistChart.options.scales.x.max;
            }
          }

          const ds0 = lossDistChart.data?.datasets?.[0];
          const ds1 = lossDistChart.data?.datasets?.[1];
          if (ds0) ds0.data = bars.slice();
          if (ds1) ds1.data = line.slice();

          // lastLossEdges = edges.slice();
          if (lossDistChart.options?.scales?.x) {
            lossDistChart.options.scales.x.min = xmin;
            lossDistChart.options.scales.x.max = xmax;
          }
          if (reportNote) reportNote.textContent = note;

          // lossDistChart.$ownerKey = incomingKey;
          lossDistChart.update('none');
        });
      }
      updateManualForkGate();
      break;
    }

    case 'reportFromDb': {
      const { losses, meta, owner_run_id } = m;
      const intendedRunKey = keyOf(owner_run_id || activeRunId);

      let bars = [], line = [], xmin, xmax, note = '';
      if (!Array.isArray(losses) || !losses.length) {
        note = (meta?.note) || 'No per-sample losses available.';
      } else {
        note = `Samples: ${losses.length} (run: ${intendedRunKey})` + (meta?.note ? `\n${meta.note}` : '');
        ({ bars, line, xmin, xmax } = computeLossHistogram(losses, 30));
      }


      REPORT_CACHE.set(cacheKeyFor(intendedRunKey), {
        bars: cloneXY(bars),
        line: cloneXY(line),
        xmin, xmax,
        note: note || '',
        at_step: meta?.at_step ?? null,
        at_epoch: meta?.at_epoch ?? null,
      });

      const isVisible = (keyOf(currentPageRunId()) === intendedRunKey);
      if (isVisible && lossDistChart) {
        if (reportMeta) reportMeta.textContent = formatWhen(meta?.at_step, meta?.at_epoch);
        _deferUntilLayout(lossDistChart, () => {
          const ds0 = lossDistChart.data?.datasets?.[0];
          const ds1 = lossDistChart.data?.datasets?.[1];
          if (ds0) ds0.data = bars.slice();
          if (ds1) ds1.data = line.slice();
          if (lossDistChart.options?.scales?.x) {
            lossDistChart.options.scales.x.min = xmin;
            lossDistChart.options.scales.x.max = xmax;
          }
          if (reportNote) reportNote.textContent = note;
          lossDistChart.update('none');
        });
      }
      updateManualForkGate();
      break;
    }

  }
});


/* ====================================================================
 * 13) Select Populators
 * --------------------------------------------------------------------
 * Populate the run <select> consistently and keep active run synced.
 * ==================================================================== */

function fillRunSel(rows) {
  if (!runSel) return;

  // Keep lineage + NAV_LIST in sync with incoming rows
  rebuildLineageFromRows(rows);
  rebuildNavListFromRows(rows);

  // Rebuild <select> in NAV_LIST order (newest-left)
  const prevViewed = currentPageRunId();   // runId the user was viewing before rebuild
  const prevSelect = String(runSel.value || '');

  runSel.innerHTML = '';
  for (const id of NAV_LIST) {
    const row = runsIndex.get(id) || {};
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = row?.name || id;
    runSel.appendChild(opt);
  }

  if (!NAV_LIST.length) {
    clearCharts();
    activeRunId = null;
    updateModelNav();
    return;
  }

  // Choose the page to show after rebuild:
  // 1) pendingSelectRunId (e.g., "follow newest") if present
  // 2) keep the page the user was viewing if still present
  // 3) keep the previous <select> value if still present
  // 4) fallback to newest (index 0)
  const exists = (id) => !!id && NAV_LIST.includes(String(id));
  let targetIdx = 0;

  if (exists(pendingSelectRunId)) {
    targetIdx = NAV_LIST.indexOf(String(pendingSelectRunId));
  } else if (exists(prevViewed)) {
    targetIdx = NAV_LIST.indexOf(String(prevViewed));
  } else if (exists(prevSelect)) {
    targetIdx = NAV_LIST.indexOf(String(prevSelect));
  }

  pendingSelectRunId = null;

  // This single call updates dropdown, charts, report view, gating, arrows
  gotoPageByIndex(targetIdx);
}



/* ====================================================================
 * 14) Status Gating
 * --------------------------------------------------------------------
 * Toggle enabled/disabled UI based on running/paused flags.
 * ==================================================================== */

function setRunning(running) {
  IS_RUNNING = !!running;

  // Primary controls
  if (btnPause)  btnPause.disabled  = !running;
  if (btnResume) btnResume.disabled = running;  // enabled when paused OR idle

  // Secondary gating
  if (btnSaveCkpt) btnSaveCkpt.disabled = !(running || IS_PAUSED);

  if (btnChoose) btnChoose.disabled = running;
  if (btnSetPy)  btnSetPy.disabled  = running;

  if (btnReport) btnReport.disabled = running;
  updateManualForkGate();
}


function followTemporarilyOff() {
  FOLLOW_ACTIVE = false;
  if (_followResetTimer) clearTimeout(_followResetTimer);
  // re-enable “follow live run” after a quiet period
  _followResetTimer = setTimeout(() => { FOLLOW_ACTIVE = true; }, 10000);
}

function setRunningFor(runKey, running) {
  const st = RUN_STATE.get(runKey) || {};
  st.running = !!running;
  RUN_STATE.set(runKey, st);
  if (runKey === keyOf(currentPageRunId())) {
    const effectiveRunning = !st.paused && st.running;
    setRunning(effectiveRunning);
  }
}


function setPausedFor(runKey, paused) {
  const st = RUN_STATE.get(runKey) || {};
  st.paused = !!paused;
  RUN_STATE.set(runKey, st);
  if (runKey === keyOf(currentPageRunId())) {
    IS_PAUSED = !!paused;
    // If paused, UI should look not-running; if unpaused, reflect per-run running state.
    const effectiveRunning = !paused && !!st.running;
    setRunning(effectiveRunning);
    updateManualForkGate();
  }
}



/* ====================================================================
 * 15) Report Histogram + Manual Fork Overlay
 * --------------------------------------------------------------------
 * Drag handles on the loss distribution to select a range and trigger
 * manual fork. Selection count pill, keyboard nudge, overlay form.
 * ==================================================================== */

let forkCountPill = null;

function computeSelectedSampleCount(minLoss, maxLoss) {
  if (!lossDistChart) return 0;
  const bars = lossDistChart.data?.datasets?.[0]?.data || [];
  if (!Array.isArray(bars) || bars.length === 0) return 0;
  const lo = Math.min(minLoss, maxLoss);
  const hi = Math.max(minLoss, maxLoss);
  let total = 0;
  for (const p of bars) {
    if (p && Number.isFinite(p.x) && p.x >= lo && p.x <= hi) {
      total += Number.isFinite(p.y) ? p.y : 0;
    }
  }
  return total;
}

function updateSelectedCountPill() {
  if (!forkCountPill || !manualForkMode || !forkSel.active) return;
  if (forkSel.aVal == null || forkSel.bVal == null) {
    forkCountPill.textContent = 'Total Samples Selected: 0';
    return;
  }
  const n = computeSelectedSampleCount(forkSel.aVal, forkSel.bVal);
  forkCountPill.textContent = `Total Samples Selected: ${n}`;
}

function notify(text, level = 'info') {
  vscode.postMessage({ command: 'notify', level, text });
}


if (btnManualFork) {
  on(btnManualFork, 'click', () => {
    if (!lossDistChart) { notify('Report chart not ready yet.', 'warn'); return; }
    if (!reportMatchesPause()) {
      notify('Report is stale vs current checkpoint. Pause and re-generate the report to fork.', 'warn');
      return;
    }
    const points = (lossDistChart.data?.datasets?.[0]?.data?.length) || 0;
    if (!points) { notify('Generate the report first.', 'warn'); return; }

    manualForkMode = true;
    forkSel.active = true;
    forkSel.dragging = null;
    const s = lossDistChart.scales.x;
    if (s && Number.isFinite(s.min) && Number.isFinite(s.max)) {
      const span = (s.max - s.min) || 1;
      if (forkSel.aVal == null) forkSel.aVal = s.min + 0.25 * span;
      if (forkSel.bVal == null) forkSel.bVal = s.min + 0.75 * span;
    }

    ensureForkOverlay();
    forkOverlay.style.display = 'block';
    if (btnManualFork) btnManualFork.style.display = 'none';
    lossDistChart.update('none');
  });
}

function enableDragOnReportChart() {
  const canvas = byId('lossDistChart');
  if (!canvas || !lossDistChart || canvas._forkDragWired) return;
  canvas._forkDragWired = true;

  const valFromEvent = (evt) => {
    const rect = canvas.getBoundingClientRect();
    const xPix = evt.clientX - rect.left;
    const scaleX = lossDistChart.scales.x;
    const v = scaleX.getValueForPixel(xPix);
    const min = scaleX.min, max = scaleX.max;
    return Math.max(min, Math.min(max, v));
  };

  const nearHandle = (evt) => {
    const rect = canvas.getBoundingClientRect();
    const xPix = evt.clientX - rect.left;
    const scaleX = lossDistChart.scales.x;
    if (forkSel.aVal == null || forkSel.bVal == null) return null;
    const aPix = scaleX.getPixelForValue(forkSel.aVal);
    const bPix = scaleX.getPixelForValue(forkSel.bVal);
    const hitRadius = 8;
    if (Math.abs(xPix - aPix) <= hitRadius) return 'a';
    if (Math.abs(xPix - bPix) <= hitRadius) return 'b';
    return null;
  };

  on(canvas, 'mousedown', (e) => {
    if (!manualForkMode || !forkSel.active) return;
    const h = nearHandle(e);
    if (h) { 
      forkSel.dragging = h;
      forkSel.activeHandle = h;
    }
  });

  on(canvas, 'click', (e) => {
    if (!manualForkMode || !forkSel.active) return;
    setActiveHandleFromEvent(e);
    const v = valFromEvent(e);
    if (forkSel.activeHandle === 'a') forkSel.aVal = v; else forkSel.bVal = v;
    lossDistChart.update('none');
  });

  on(window, 'mousemove', (e) => {
    if (!manualForkMode || !forkSel.active || !forkSel.dragging) return;
    const v = valFromEvent(e);
    if (forkSel.dragging === 'a') forkSel.aVal = v; else forkSel.bVal = v;
    lossDistChart.update('none');
  });

  on(window, 'mouseup', () => { forkSel.dragging = null; });
}


function bumpHandle(which, delta) {
  if (!lossDistChart || !manualForkMode || !forkSel.active) return;
  const scaleX = lossDistChart.scales.x;
  const step = (scaleX.max - scaleX.min) / 100;
  if (which === 'a') forkSel.aVal = clampValue((forkSel.aVal ?? scaleX.min) + delta * step);
  else               forkSel.bVal = clampValue((forkSel.bVal ?? scaleX.max) + delta * step);
  lossDistChart.update('none');
}

function setActiveHandleFromEvent(evt) {
  if (!lossDistChart) return;
  const canvas = byId('lossDistChart');
  const rect = canvas.getBoundingClientRect();
  const xPix = evt.clientX - rect.left;
  const scaleX = lossDistChart.scales.x;
  if (forkSel.aVal == null || forkSel.bVal == null) return;
  const aPix = scaleX.getPixelForValue(forkSel.aVal);
  const bPix = scaleX.getPixelForValue(forkSel.bVal);
  forkSel.activeHandle = Math.abs(xPix - aPix) <= Math.abs(xPix - bPix) ? 'a' : 'b';
}

window.addEventListener('keydown', (e) => {
  if (!manualForkMode || !forkSel.active) return;
  const which = forkSel.dragging || forkSel.activeHandle || 'a';
  if (e.key === 'ArrowLeft')  { bumpHandle(which, -1); e.preventDefault(); }
  if (e.key === 'ArrowRight') { bumpHandle(which, +1); e.preventDefault(); }
});

function clampValue(v) {
  const s = lossDistChart?.scales?.x;
  if (!s) return v;
  return Math.max(s.min, Math.min(s.max, v));
}

function ensureForkOverlay() {
  if (forkOverlay) return;
  forkOverlay = document.createElement('div');
  Object.assign(forkOverlay.style, {
    position: 'static',
    background: '#334155',
    border: '1px solid #ddd',
    borderRadius: '10px',
    padding: '10px',
    boxShadow: '0 6px 20px rgba(0,0,0,0.15)',
    marginTop: '12px',
    display: 'none'
  });

  const chartWrap = byId('lossDistChart')?.closest('.chartWrap');
  if (chartWrap && chartWrap.parentNode) {
    chartWrap.parentNode.insertBefore(forkOverlay, chartWrap.nextSibling);
  } else {
    document.body.appendChild(forkOverlay);
  }

  forkOverlay.innerHTML = `
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <span id="forkCountPill"
        style="
          padding:4px 10px;
          border-radius:9999px;
          background:var(--btn-bg);
          color:var(--btn-fg);
          font-weight:600;
          font-size:12px;
          box-shadow:var(--btn-shadow);
          user-select:none;
        "
      >Total Samples Selected: 0</span>

      <label>LR <input id="forkLR" type="number" step="0.0001" value="0.001" style="width:90px"></label>
      <label>Batch <input id="forkBS" type="number" step="1" value="32" style="width:80px"></label>
      <label>Patience <input id="forkPat" type="number" step="1" value="5" style="width:80px"></label>
      <button id="forkOk">Okay, fork</button>
      <button id="forkCancel">Cancel</button>
    </div>`;

  forkCountPill = byId('forkCountPill');
  updateSelectedCountPill();

  on(byId('forkOk'), 'click', () => {
    const lr  = parseFloat(byId('forkLR').value || '0.001');
    const bs  = parseInt(byId('forkBS').value || '32', 10);
    const pat = parseInt(byId('forkPat').value || '5', 10);

    if (forkSel.aVal == null || forkSel.bVal == null) {
      notify('Selection handles not set. Click/drag on the chart first.', 'warn');
      return;
    }
    const minLoss = Math.min(forkSel.aVal, forkSel.bVal);
    const maxLoss = Math.max(forkSel.aVal, forkSel.bVal);
    console.log('[UI] forkSel.aVal =', forkSel.aVal, 'forkSel.bVal =', forkSel.bVal);

    send('manualFork', {
      runId: currentPageRunId(),
      region: { minLoss, maxLoss },
      hparams: { lr, batch_size: bs, patience: pat }
    });
    if (minLoss == null || maxLoss == null) {
      notify('No loss region available. Generate the report first.', 'warn');
      return;
    }
    
    forkSel.aVal = null;
    forkSel.bVal = null;
    forkSel.activeHandle = 'a';
    manualForkMode = false;
    forkSel.active = false;
    forkOverlay.style.display = 'none';
    if (lossDistChart) lossDistChart.update('none');
  });

  on(byId('forkCancel'), 'click', () => {
    manualForkMode = false;
    forkSel.active = false;
    forkOverlay.style.display = 'none';
    if (btnManualFork) btnManualFork.style.display = '';
    if (lossDistChart) lossDistChart.update('none');
  });
}

on(byId('forkALeft'),  'click', () => { forkSel.dragging = 'a'; bumpHandle('a', -1); });
on(byId('forkARight'), 'click', () => { forkSel.dragging = 'a'; bumpHandle('a', +1); });
on(byId('forkBLeft'),  'click', () => { forkSel.dragging = 'b'; bumpHandle('b', -1); });
on(byId('forkBRight'), 'click', () => { forkSel.dragging = 'b'; bumpHandle('b', +1); });


/* ====================================================================
 * 16) DAG + Layout Helpers
 * --------------------------------------------------------------------
 * Compute layered layout, render SVG nodes/edges, and interactions.
 * ==================================================================== */

/* ===== DAG + layout helpers (unchanged) ===== */
function _canvasHasLayout(cnv) {
  if (!cnv) return false;
  const r = cnv.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}
function _deferUntilLayout(chart, cb, tries = 20) {
  if (!chart?.canvas) { cb(); return; }
  if (_canvasHasLayout(chart.canvas)) { chart.resize(); cb(); return; }
  if (tries <= 0) { cb(); return; }
  requestAnimationFrame(() => _deferUntilLayout(chart, cb, tries - 1));
}
function computeLayers(){
  const depth = new Map();
  const ids = Array.from(runsIndex.keys());

  const getDepth = (id, stack = new Set()) => {
    if (depth.has(id)) return depth.get(id);
    if (stack.has(id)) { depth.set(id, 0); return 0; } // cycle guard
    stack.add(id);
    const ps = Array.from(parentsOf.get(id) || []);
    const d = ps.length ? (Math.max(...ps.map(p => getDepth(p, stack))) + 1) : 0;
    depth.set(id, d);
    stack.delete(id);
    return d;
  };

  ids.forEach(id => getDepth(id));

  const byDepth = new Map();
  ids.forEach(id => {
    const d = depth.get(id) || 0;
    if (!byDepth.has(d)) byDepth.set(d, []);
    byDepth.get(d).push(id);
  });

  const depths = Array.from(byDepth.keys()).sort((a,b)=>a-b);
  depths.forEach(d => byDepth.get(d).sort((a,b)=> a.localeCompare(b)));
  return { depth, depths, byDepth };
}


const SVG_NS = 'http://www.w3.org/2000/svg';

// ensure <defs>, a persistent pan/zoom viewport <g>, and a root <g> for content
function ensureDagLayers(svg) {
  if (!svg) return null;

  // container object hung off the svg so we persist between re-renders
  if (!svg.__dagLayers) svg.__dagLayers = {};
  const layers = svg.__dagLayers;

  // 1) defs (arrows, etc.)
  if (!layers.defs) {
    const defs = document.createElementNS(SVG_NS, 'defs');
    defs.id = 'dag-defs';
    defs.innerHTML = `
      <marker id="arrowHead" markerWidth="12" markerHeight="10"
              viewBox="0 0 12 10" refX="12" refY="5" orient="auto"
              markerUnits="userSpaceOnUse">
        <path class="dagEdgeArrow" d="M0,0 L12,5 L0,10 Z"></path>
      </marker>`;
    svg.appendChild(defs);
    layers.defs = defs;
  }

  // 2) pan/zoom viewport
  if (!layers.viewport) {
    const viewport = document.createElementNS(SVG_NS, 'g');
    viewport.id = 'dagViewport';
    svg.appendChild(viewport);
    layers.viewport = viewport;

    // attach pan/zoom once
    if (window.enablePanZoom) {
      layers.panzoom = window.enablePanZoom(svg, viewport, {
        minScale: 0.2,
        maxScale: 5,
        step: 0.14,
      });
    }
  }

  // 3) root content group (cleared every render)
  if (!layers.root) {
    const root = document.createElementNS(SVG_NS, 'g');
    root.id = 'dagRoot';
    layers.viewport.appendChild(root);
    layers.root = root;
  }

  return layers;
}

// --- Robust SVG label wrapper (centers vertically, 2 lines + ellipsis) ---
function wrapSvgTextIntoTspans(textEl, raw, maxWidthPx, maxLines = 2) {
  const svg = textEl.ownerSVGElement;
  const label = String(raw || '').trim();
  if (!svg || !label) {
    textEl.textContent = label || '';
    return;
  }

  const cs = getComputedStyle(textEl);
  const fontSize = parseFloat(cs.fontSize) || 12;
  const lineHeight = Math.round(fontSize * 1.2);

  // hidden measurer (reused)
  let meas = svg.__measureTextEl;
  if (!meas) {
    meas = document.createElementNS(SVG_NS, 'text');
    meas.setAttribute('x', '-9999');
    meas.setAttribute('y', '-9999');
    meas.style.visibility = 'hidden';
    svg.appendChild(meas);
    svg.__measureTextEl = meas;
  }
  // copy relevant font properties so measurements match
  const mcs = meas.style;
  mcs.font = cs.font;                    // copies family/size/weight/variant/line-height
  meas.setAttribute('font-family', cs.fontFamily);
  meas.setAttribute('font-weight', cs.fontWeight);
  meas.setAttribute('font-size', cs.fontSize);

  const widthOf = (s) => {
    meas.textContent = s;
    return meas.getComputedTextLength();
  };

  const words = label.split(/\s+/).filter(Boolean);
  const lines = [];
  let cur = '';

  // Build up to maxLines lines by measuring word-by-word
  while (words.length && lines.length < maxLines) {
    if (!cur) {
      // start a line; if a single word is too wide, ellipsize that word itself
      if (widthOf(words[0]) > maxWidthPx) {
        let w = words.shift();
        // find largest prefix that fits with an ellipsis
        let lo = 1, hi = w.length, best = 1;
        while (lo <= hi) {
          const mid = (lo + hi) >> 1;
          const slice = w.slice(0, mid) + '…';
          if (widthOf(slice) <= maxWidthPx) { best = mid; lo = mid + 1; }
          else { hi = mid - 1; }
        }
        lines.push(w.slice(0, best) + '…');
        // we’re done: we spent the whole line on an oversized word
        break;
      }
      cur = words.shift();
    } else {
      const candidate = cur + ' ' + words[0];
      if (widthOf(candidate) <= maxWidthPx) {
        cur = candidate; words.shift();
      } else {
        lines.push(cur);
        cur = '';
      }
    }
  }
  if (cur && lines.length < maxLines) lines.push(cur);

  // If words remain, ellipsize last line to indicate more content
  if (words.length) {
    let last = lines.pop() || '';
    while (last && widthOf(last + '…') > maxWidthPx) last = last.slice(0, -1);
    lines.push((last || '…') + '…');
  }

  // Clear and create tspans, vertically centered via absolute y
  while (textEl.firstChild) textEl.removeChild(textEl.firstChild);

  const x = textEl.getAttribute('x') || '0';
  const yCenter = Number(textEl.getAttribute('y') || 0);
  const totalH = (lines.length - 1) * lineHeight;
  const yStart = yCenter - totalH / 2; // top line

  for (let i = 0; i < lines.length; i++) {
    const tspan = document.createElementNS(SVG_NS, 'tspan');
    tspan.setAttribute('x', x);
    tspan.setAttribute('y', String(yStart + i * lineHeight));
    tspan.textContent = lines[i];
    textEl.appendChild(tspan);
  }
}

function renderDag() {
  if (!dagSvg) return;

  // set up / reuse defs + viewport + root
  const layers = ensureDagLayers(dagSvg);
  if (!layers) return;
  const { viewport, root, panzoom } = layers;

  // --- Clear ONLY the content root; keep defs and viewport in place
  while (root.firstChild) root.removeChild(root.firstChild);

  // collect nodes & edges from existing maps
  const nodeW = 150, nodeH = 42;
  const nodes = Array.from(runsIndex.keys()).map(id => ({ id }));
  const edges = [];
  parentsOf.forEach((ps, child) => ps.forEach(p => edges.push({ source: p, target: child })));

  // min viewBox sizes from CSS (fallbacks OK)
  const cs = getComputedStyle(document.documentElement);
  const MIN_W = parseFloat(cs.getPropertyValue('--dag-minW')) || 960;
  const MIN_H = parseFloat(cs.getPropertyValue('--dag-minH')) || 560;

  if (!nodes.length) {
    dagSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    dagSvg.setAttribute('viewBox', `0 0 ${MIN_W} ${MIN_H}`);
    dagSvg.setAttribute('width', '100%');
    dagSvg.setAttribute('height', '100%');

    const t = document.createElementNS(SVG_NS, 'text');
    t.setAttribute('x', '50%'); t.setAttribute('y', '50%');
    t.setAttribute('text-anchor', 'middle');
    t.setAttribute('fill', '#fff');
    t.textContent = 'No runs yet';
    root.appendChild(t);
    return;
  }

  // --- compute layout
  const { pos, routed, size } = window.layoutDAG(nodes, edges, {
    nodeW, nodeH,
    margin: 48,
    rankSep: 200,
    nodeSep: 28,
    iterations: 6, // harmless extra param
  });

  // --- clamp the viewBox so content never over-zooms
  const W = size.W, H = size.H;
  const vbW = Math.max(W, MIN_W);
  const vbH = Math.max(H, MIN_H);

  dagSvg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  dagSvg.setAttribute('viewBox', `0 0 ${vbW} ${vbH}`);
  dagSvg.setAttribute('width', '100%');
  dagSvg.setAttribute('height', '100%');

  // --- center the content within the larger clamped viewBox
  const offsetX = (vbW - W) / 2;
  const offsetY = (vbH - H) / 2;
  root.setAttribute('transform', `translate(${offsetX}, ${offsetY})`);

  // --- draw edges (under nodes) — append to root
  routed.forEach(e => {
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('class', 'dagEdge');
    path.setAttribute('d', e.d);
    path.setAttribute('marker-end', 'url(#arrowHead)');
    root.appendChild(path);
  });

  // --- draw nodes — append to root
  pos.forEach(({ x, y }, id) => {
    const isSelected = selectedForMerge.has(id);

    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('class', `dagNode${isSelected ? ' selected' : ''}`);
    g.setAttribute('transform', `translate(${x},${y})`);
    g.style.cursor = 'pointer';

    const label = (runsIndex.get(id)?.name || id);
    g.setAttribute('role', 'button');
    g.setAttribute('tabindex', '0');
    g.setAttribute('aria-label', label);
    g.setAttribute('aria-pressed', String(isSelected));

    const rect = document.createElementNS(SVG_NS, 'rect');
    rect.setAttribute('width', String(nodeW));
    rect.setAttribute('height', String(nodeH));
    rect.setAttribute('rx', '10'); rect.setAttribute('ry', '10');
    g.appendChild(rect);

    // (unchanged foreignObject label)
    const fo = document.createElementNS(SVG_NS, 'foreignObject');
    fo.setAttribute('x', '0'); fo.setAttribute('y', '0');
    fo.setAttribute('width', String(nodeW)); fo.setAttribute('height', String(nodeH));
    fo.style.pointerEvents = 'none';

    const outer = document.createElement('div');
    outer.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
    Object.assign(outer.style, {
      display: 'flex', width: '100%', height: '100%',
      alignItems: 'center', justifyContent: 'center',
      padding: '4px 8px'
    });

    const inner = document.createElement('div');
    Object.assign(inner.style, {
      display: '-webkit-box',
      WebkitBoxOrient: 'vertical',
      WebkitLineClamp: '2',
      overflow: 'hidden',
      textAlign: 'center',
      lineHeight: '1.2',
      fontWeight: '600',
      fontSize: '12px',
      overflowWrap: 'anywhere',
      wordBreak: 'break-word',
    });
    inner.textContent = label;

    outer.appendChild(inner);
    fo.appendChild(outer);
    g.appendChild(fo);

    const title = document.createElementNS(SVG_NS, 'title');
    title.textContent = label;
    g.appendChild(title);

    // IMPORTANT: prevent the pan layer from hijacking node clicks.
    // This stops pointerdown from bubbling to the SVG's pan handler,
    // so Shift-click multi-select works reliably.
    g.addEventListener('pointerdown', (evt) => {
      evt.stopPropagation();
    });

    function handlePrimarySelect() {
      followTemporarilyOff();
      gotoPageByRunId(id);
      closeDag();
    }

    function toggleMergeSelection() {
      if (selectedForMerge.has(id)) selectedForMerge.delete(id);
      else selectedForMerge.add(id);
      // Update this node in place (no full re-render)
      const selected = selectedForMerge.has(id);
      g.classList.toggle('selected', selected);
      g.setAttribute('aria-pressed', String(selected));
      updateMergeUi?.();
    }

    g.addEventListener('click', (evt) => {
      if (evt.shiftKey) toggleMergeSelection();
      else handlePrimarySelect();
    });
    g.addEventListener('keydown', (evt) => {
      if (evt.key === 'Enter' || evt.key === ' ') {
        evt.preventDefault();
        if (evt.shiftKey) toggleMergeSelection();
        else handlePrimarySelect();
      }
    });

    root.appendChild(g);
  });

  // If later we want to reset when the graph shape changes a lot, we can:
  // panzoom?.reset();
}

function addAutoForkMarker(runKey, step, kind = 'suggested') {
  const s = Number(step);
  if (!Number.isFinite(s) || !runKey) return;
  const arr = AF_MARKERS.get(runKey) || [];
  // de-dupe by step+kind
  if (!arr.some(m => m.step === s && m.kind === kind)) {
    arr.push({ step: s, kind });
    // keep memory bounded
    if (arr.length > 128) arr.splice(0, arr.length - 128);
    AF_MARKERS.set(runKey, arr);
    scheduleChartsRedraw(); // redraw charts to show the line
  }
}

function clearMarkersFor(runKey) {
  if (!runKey) return;
  AF_MARKERS.delete(runKey);
}
