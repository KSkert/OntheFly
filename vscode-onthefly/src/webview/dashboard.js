/**
 * =====================================================================
 * dashboard.js (refactor-ready edition)
 * =====================================================================
 * TABLE OF CONTENTS
 * -----------------
 *  1) Core Handles & Helpers
 *  2) DOM References & Basic Wiring
 *  3) Run State & Navigation
 *  4) Export Hook
 *  5) Chart & Report Plumbing
 *  6) DAG & Merge UI
 *  7) IPC & Controls Bridge
 *  8) Window & Message Bus


/* ====================================================================
 * 1) Core Handles & Helpers
 * --------------------------------------------------------------------
 * VS Code bridge, RunState accessors, and shared helper utilities.
 * ==================================================================== */

/* global acquireVsCodeApi, Chart */
const vscode = acquireVsCodeApi();
// Expose vscode handle so earlier scripts (chart utils/export helpers) can reach it.
if (typeof window !== 'undefined') {
  window.vscode = vscode;
}
const RunState = window.RunState || (() => { throw new Error('run_state.js missing'); })();
const {
  parentsOf,
  childrenOf,
  runsIndex,
  AF_MARKERS,
  RUN_STATE,
  LAST_PAUSED_STEP,
  keyOf,
  runIdOf,
  edges,
  addRun: storeAddRun,
  rebuildNavListFromRows: storeRebuildNavList,
  gotoPageByIndex: storeGotoPageByIndex,
  gotoPageByRunId: storeGotoPageByRunId,
  updateModelNav: computeModelNavSnapshot,
  setLiveRun: storeSetLiveRun,
  getLiveRun: storeGetLiveRun,
  streamTargetRunId,
  followTemporarilyOff: storeFollowTemporarilyOff,
  isFollowActive,
  setFollowActive: storeSetFollowActive,
  getNavList,
  getPageIndex,
  setLastRowsRunKey,
  getLastRowsRunKey,
} = RunState;

const byId = (id) => document.getElementById(id);
const on = (el, evt, fn) => { if (el) el.addEventListener(evt, fn); };
const nf = v => (Number.isFinite(v) ? v : NaN);

// ===== DEBUG SHIM =====
window.DEBUG_NAV = false;
const groupNav = (label, fn) => {
  if (!window.DEBUG_NAV) return fn();
  console.groupCollapsed(`%c[NAV] ${label}`, 'color:#0ea5e9');
  try { fn(); } finally { console.groupEnd(); }
};

function notify(text, level = 'info') {
  vscode.postMessage({ command: 'notify', level, text });
}


/* ====================================================================
 * 2) DOM References & Basic Wiring
 * --------------------------------------------------------------------
 * Cache commonly used DOM elements once to keep lookups cheap.
 * ==================================================================== */

/* -------- DOM refs -------- */
//setup env
const btnChoose = byId('btnChoose');
const btnSetPy = byId('btnSetPy');
const pyPath = byId('pyPath');
const scriptName = byId('scriptName');

//common basics
const btnPause = byId('btnPause');
const btnResume = byId('btnResume');
const btnTestNow = byId('btnTestNow');

const btnAutoSave = byId('btnAutoSave');
const btnLoad = byId('btnLoad');

// report
const btnReport = byId('btnGenerateReport') || byId('btnReport');
const reportNote = byId('reportNote');
const reportMeta = byId('reportMeta');

// health
const btnDistHealth = byId('btnDistHealth')
const btnActivationsHealth = byId('btnActivationsHealth')
const btnNumericsHealth = byId('btnNumericsHealth')
const btnDeterminismHealth = byId('btnDeterminismHealth')
const btnThroughputHealth = byId('btnThroughputHealth')

//nav
const btnPrevModel = byId('btnPrevModel');
const btnNextModel = byId('btnNextModel');
const runSel = byId('runSel');
const btnRefreshRuns = byId('btnRefreshRuns');

//dag
const btnOpenDag = byId('btnOpenDag');
const btnExportLoss = byId('exportLossBtn');
const btnExportLossDist = byId('exportLossDistBtn');
const btnExportValLoss = byId('exportValLossBtn');

const dagOverlay = byId('dagOverlay');
const dagSvg     = byId('dagSvg');
const dagClose   = byId('dagClose');
const dagMergeBtn= byId('dagMergeBtn');
const dagStrategy= byId('dagStrategy');

//exports
const btnExportSubset = byId('btnExportSubset');
const exportSubsetFmt = byId('exportSubsetFmt');
const axisModeStepBtn = byId('axisModeStep');
const axisModeEpochBtn = byId('axisModeEpoch');

// axis helpers
const RUN_EPOCH = new Map();

/* ====================================================================
 * 3) Run State & Navigation
 * --------------------------------------------------------------------
 * Tracks paused/running flags, follow state, and model navigation UI.
 * ==================================================================== */

let IS_RUNNING = false;
let IS_PAUSED  = false;
let TEST_PENDING = false;
gateHealthButtons();
function curRunKey() { return keyOf(currentPageRunId()); }


function reportMatchesPause(runKey = curRunKey()) {
  const cache = ChartReport.getReport(runKey);
  const pausedAt = LAST_PAUSED_STEP.get(runKey);
  const atStep = cache?.at_step;
  return Number.isFinite(atStep) && Number.isFinite(pausedAt) && atStep === pausedAt;
}

function updateSelectRegionGate() {
  window.ChartReportSelection?.updateSelectRegionGate?.();
}

function updateSelectedExportGate() {
  window.ChartReportSelection?.updateSelectedExportGate?.();
}

function gateHealthButtons() {
  const needPauseMsg = 'Pause training to run health checks.';
  const enabled = !!IS_PAUSED;   // only when explicitly paused

  [
    btnDistHealth,
    btnThroughputHealth,
    btnActivationsHealth,
    btnNumericsHealth,
    btnDeterminismHealth,
  ].forEach(b => {
    if (!b) return;
    b.disabled = !enabled;
    b.title = enabled ? '' : needPauseMsg;
    if (!enabled) {
      b.setAttribute('aria-disabled', 'true');
      b.style.opacity = '0.5';
      b.style.pointerEvents = 'none';
    } else {
      b.removeAttribute('aria-disabled');
      b.style.opacity = '';
      b.style.pointerEvents = '';
    }
  });
}

function updateTestNowGate() {
  if (!btnTestNow) return;
  const shouldDisable = IS_RUNNING || TEST_PENDING;
  btnTestNow.disabled = shouldDisable;
}

function notifyModelNavSelected(runId) {
  vscode.postMessage({ command: 'modelNav.select', runId: String(runId || '') });
}

function currentPageRunId() {
  return RunState.currentPageRunId();
}

function applyRunSelection(runId) {
  if (!runId) {
    updateModelNavUI();
    return;
  }

  if (runSel) runSel.value = runId;
  notifyModelNavSelected(runId);
  MetricHistory.setRun(runId);

  clearCharts();
  setLastRowsRunKey(runId);
  send('requestRows',   { runId });
  send('requestReport', { runId });
  ChartReport.showReportFor(runId);

  updateSelectRegionGate();
  updateSelectedExportGate();
  updateModelNavUI();
}

function selectRunByIndex(i) {
  const runId = storeGotoPageByIndex(i);
  applyRunSelection(runId);
}

function selectRunByRunId(runId) {
  const next = storeGotoPageByRunId(runId);
  if (next) applyRunSelection(next);
}

function updateModelNavUI() {
  const { prev, next } = navEls();
  if (!prev || !next) return;

  const navState = computeModelNavSnapshot();
  setArrow(prev, navState.hasPrev, navState.prevVal);
  setArrow(next, navState.hasNext, navState.nextVal);
}

function setArrow(btn, enabled, targetId) {
  if (!btn) { return; }
  btn.dataset.target = enabled ? keyOf(targetId) : '';

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

function navEls() {
  const els = {
    prev: document.getElementById('btnPrevModel'),
    next: document.getElementById('btnNextModel'),
    sel:  document.getElementById('runSel'),
  };
  return els;
}



function wireModelNavClicks() {
  const { prev, next } = navEls();

  if (prev && !prev._wired) {
    prev.addEventListener('click', () => {
      storeFollowTemporarilyOff();
      const idx = getPageIndex();
      if (idx > 0) selectRunByIndex(idx - 1);
    });
    prev._wired = true;
  }
  if (next && !next._wired) {
    next.addEventListener('click', () => {
      storeFollowTemporarilyOff();
      const idx = getPageIndex();
      const navList = getNavList();
      if (idx < navList.length - 1) selectRunByIndex(idx + 1);
    });
    next._wired = true;
  }
}

if (runSel) {
  runSel.onchange = () => {
    storeFollowTemporarilyOff();
    const id  = String(runSel.value);
    const navList = getNavList();
    const idx = navList.indexOf(id);
    if (idx >= 0) selectRunByIndex(idx);
  };
}

function fillRunSel(rows) {
  if (!runSel) return;

  storeRebuildNavList(rows);
  const navList = getNavList();

  const prevViewed = currentPageRunId();
  const prevSelect = String(runSel.value || '');

  runSel.innerHTML = '';
  for (const id of navList) {
    const row = runsIndex.get(id) || {};
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = row?.name || id;
    runSel.appendChild(opt);
  }

  if (!navList.length) {
    clearCharts();
    storeSetLiveRun(null);
    updateModelNavUI();
    return;
  }

  // Prefer follow-live, then the page the user was on, else newest
  const exists = (id) => !!id && navList.includes(String(id));
  let targetIdx = 0;

  const liveRun = storeGetLiveRun();
  if (isFollowActive() && exists(liveRun)) {
    targetIdx = navList.indexOf(String(liveRun));
  } else if (exists(prevViewed)) {
    targetIdx = navList.indexOf(String(prevViewed));
  } else if (exists(prevSelect)) {
    targetIdx = navList.indexOf(String(prevSelect));
  }

  selectRunByIndex(targetIdx);
}

function setRunning(running) {
  IS_RUNNING = !!running;

  if (btnPause)  btnPause.disabled  = !running;
  if (btnResume) btnResume.disabled = running;
  if (btnChoose) btnChoose.disabled = running;
  if (btnSetPy)  btnSetPy.disabled  = running;
  if (btnAutoSave) btnAutoSave.disabled = running;
  if (btnLoad) btnLoad.disabled = running;
  if (btnRefreshRuns) btnRefreshRuns.disabled = running;
  if (btnExportSubset) btnExportSubset.disabled = running;

  if (btnReport) btnReport.disabled = running;
  updateTestNowGate();
  updateSelectRegionGate();
  updateSelectedExportGate();
  gateHealthButtons();
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
    const effectiveRunning = !paused && !!st.running;
    setRunning(effectiveRunning);
    updateSelectRegionGate();
    updateSelectedExportGate();
    gateHealthButtons();
  }
}


function formatWhen(step, epoch) {
  const stepTxt = (Number.isFinite(step) ? step : '—');
  const epTxt   = (Number.isFinite(epoch) ? epoch : '—');
  return `Analyzed at step ${stepTxt} (epoch ${epTxt})`;
}

/* ====================================================================
 * 4) subset export button wiring.
 * ==================================================================== */



// --- subset export UI ---

on(btnExportSubset, 'click', () => {
  const runId = currentPageRunId();
  if (!runId) { notify('No run selected.', 'warn'); return; }
  const format = (exportSubsetFmt && exportSubsetFmt.value) || 'parquet';
  vscode.postMessage({ command: 'exportSubset', runId, format });
});


/* ====================================================================
 * 5) Chart & Report Plumbing
 * --------------------------------------------------------------------
 * ChartStream wiring plus report helpers (chart_* scripts do the heavy lifting).
 * ==================================================================== */

const STREAM_METRICS = [
  'accuracy',
  'lr',
  'grad_norm',
  'weight_norm',
  'activation_zero_frac',
  'throughput',
  'mem_vram',
  'gpu_util',
];

window.ChartStream?.trackMetrics?.(['loss', 'val_loss', ...STREAM_METRICS]);

const AXIS_MODE_KEY = 'fs.axisMode.v1';
const axisModeButtons = [axisModeStepBtn, axisModeEpochBtn].filter(Boolean);

function setAxisMode(mode, persist = true) {
  const normalized = mode === 'epoch' ? 'epoch' : 'step';
  window.ChartStream?.setAxisMode?.(normalized);
  axisModeButtons.forEach(btn => {
    if (!btn) return;
    const isActive = (btn.dataset.axisMode === normalized);
    btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
    btn.classList.toggle('active', isActive);
  });
  if (persist) {
    try { localStorage.setItem(AXIS_MODE_KEY, normalized); } catch {}
  }
}

const storedAxisMode = (() => {
  try { return localStorage.getItem(AXIS_MODE_KEY); }
  catch { return null; }
})();
setAxisMode(storedAxisMode === 'epoch' ? 'epoch' : 'step', false);
axisModeButtons.forEach(btn => {
  on(btn, 'click', () => {
    const target = btn?.dataset.axisMode === 'epoch' ? 'epoch' : 'step';
    setAxisMode(target);
  });
});

const MetricHistory = (() => {
  const MAX_POINTS = 4096;
  let runKey = '';
  const store = new Map();
  let backfillTimer = null;
  let awaitingBackfill = false;
  const BACKFILL_DEBOUNCE_MS = 500;

  function setRun(nextKey) {
    const normalized = keyOf(nextKey);
    if (!normalized || normalized === runKey) return;
    runKey = normalized;
    store.clear();
    window.ChartStream?.reset?.();
  }

  function pushPoint(metric, point, skipChart) {
    if (!Number.isFinite(point?.x) || !Number.isFinite(point?.y)) return;
    const arr = store.get(metric) || [];
    arr.push(point);
    if (arr.length > MAX_POINTS) arr.splice(0, arr.length - MAX_POINTS);
    store.set(metric, arr);
  }

  function rebuild(key, rows) {
    const normalized = keyOf(key);
    if (!normalized) return;
    runKey = normalized;
    store.clear();
    if (!Array.isArray(rows)) return;
    for (const row of rows) {
      const step = Number(row.step);
      if (!Number.isFinite(step)) continue;
      for (const metric of STREAM_METRICS) {
        const val = nf(row?.[metric]);
        if (!Number.isFinite(val)) continue;
        pushPoint(metric, { x: step, y: val }, true);
      }
    }
    hydrateOpenCharts();
    markHydrated();
  }

  function pushLive(key, row) {
    const normalized = keyOf(key);
    if (!normalized) return;
    if (!runKey || runKey !== normalized) {
      setRun(normalized);
    }
    const step = Number(row?.step);
    if (!Number.isFinite(step)) return;
    for (const metric of STREAM_METRICS) {
      const val = nf(row?.[metric]);
      if (!Number.isFinite(val)) continue;
      pushPoint(metric, { x: step, y: val }, false);
    }
  }

  function hydrate(metric) {
    window.ChartStream?.applyMetric?.(metric);
  }

  function hydrateOpenCharts() {
    window.ChartStream?.applyStateToCharts?.();
  }

  function hasData(metric) {
    const arr = store.get(metric);
    return Array.isArray(arr) && arr.length > 0;
  }

  function requestBackfill() {
    const runId = keyOf(currentPageRunId());
    if (!runId) return;
    if (awaitingBackfill) return;
    awaitingBackfill = true;
    if (backfillTimer) clearTimeout(backfillTimer);
    vscode.postMessage({ command: 'requestRows', runId, reason: 'metrics-backfill' });
    backfillTimer = setTimeout(() => { awaitingBackfill = false; }, BACKFILL_DEBOUNCE_MS);
  }

  function markHydrated() {
    awaitingBackfill = false;
    if (backfillTimer) {
      clearTimeout(backfillTimer);
      backfillTimer = null;
    }
  }

  return {
    STREAM_METRICS,
    setRun,
    replaceAll: rebuild,
    pushLive,
    hydrate,
    hydrateOpenCharts,
    hasData,
    requestBackfill,
    markHydrated,
    currentRun: () => runKey,
  };
})();

window.MetricHistory = MetricHistory;
window.MetricHydrator = {
  hydrateMetric(metric) {
    MetricHistory.hydrate(metric);
  }
};

function clearCharts() {
  // reset streaming buffers + visuals without tearing down chart instances
  ChartStream.reset();

  const ids = (window.OnTheFlyExports && OnTheFlyExports.ids) || {};
  ChartCreation.ensurePluginsRegistered();
  ChartCreation.initDefaultCanvases();

  const lossChart = ChartCreation.get('loss') || ChartCreation.createLineChart({
    name: 'loss',
    canvasId: ids.lossChart,
    label: 'Loss'
  });

  const valChart = ChartCreation.get('val_loss') || ChartCreation.createLineChart({
    name: 'val_loss',
    canvasId: ids.valLossChart,
    label: 'VAL'
  });

  const accuracyChart = ChartCreation.get('accuracy') || ChartCreation.createLineChart({
    name: 'accuracy',
    canvasId: ids.accuracyChart,
    label: 'Accuracy'
  });

  const hist = ChartCreation.get('loss_dist') || ChartCreation.createHistogramChart({
    name: 'loss_dist',
    canvasId: ids.lossDistChart
  });

  // ensure histogram visuals are cleared between runs
  if (hist?.data?.datasets) {
    hist.data.datasets.forEach(ds => { if (ds?.data) ds.data = []; });
    hist.update('none');
  }

  ChartStream.attachCharts({ lossChart, valLossChart: valChart, accuracyChart });

  // Back-compat globals for any older code still reading these
  window.lossChart = lossChart;
  window.val_lossChart = valChart;
  window.accuracyChart = accuracyChart;
  window.lossDistChart = hist;
}

function clearReportChart() {
  ChartReport.clearReportChart();
}

function addAutoForkMarker(runKey, step, kind = 'suggested') {
  const s = Number(step);
  if (!Number.isFinite(s) || !runKey) return;
  const arr = AF_MARKERS.get(runKey) || [];
  if (!arr.some(m => m.step === s && m.kind === kind)) {
    arr.push({ step: s, kind });
    if (arr.length > 128) arr.splice(0, arr.length - 128);
    AF_MARKERS.set(runKey, arr);
    scheduleChartsRedraw();
  }
}

function clearMarkersFor(runKey) {
  if (!runKey) return;
  AF_MARKERS.delete(runKey);
}

/* ====================================================================
 * 6) DAG & Merge UI
 * --------------------------------------------------------------------
 * Overlay UX for DAG selection, merge gating banner, and render helpers.
 * ==================================================================== */

const selectedForMerge = new Set();

function pickPrimaryParent(childId) {
  const ps = Array.from(parentsOf.get(childId) || []);
  if (!ps.length) return null;
  ps.sort((a, b) => (runsIndex.get(b)?.created_at || 0) - (runsIndex.get(a)?.created_at || 0));
  return ps[0];
}

function openDag(){
  if (!dagOverlay) return;
  dagOverlay.classList.add('show');
  renderDag();
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

function requestDagMerge() {
  if (selectedForMerge.size !== 2) return;
  const parents = Array.from(selectedForMerge);
  const strategy = (dagStrategy && dagStrategy.value) || 'swa';
  vscode.postMessage({ command: 'merge', payload: { parents, strategy } });
  notify(`Requested merge: ${parents.join(' + ')} (${strategy})`);
}

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
  el.textContent = '';
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

const LAST_MERGE_REASON = new Map();

function shouldShowMergeGating(m) {
  const rk = String(m.run_id || curRunKey());
  if (rk !== String(curRunKey())) return false;
  const prev = LAST_MERGE_REASON.get(rk);
  if (prev === m.reason) return false;
  LAST_MERGE_REASON.set(rk, m.reason);
  return true;
}

function hideMergeBanner() {
  const el = document.getElementById('mergeBanner');
  if (el) el.style.display = 'none';
}

function renderDag() {
  if (!dagSvg) return;
  const renderer = window.DagRender?.render;
  if (typeof renderer !== 'function') return;

  renderer({
    svg: dagSvg,
    runsIndex,
    edges,
    selectedForMerge,
    onPrimarySelect: (runId) => {
      storeFollowTemporarilyOff();
      selectRunByRunId(runId);
      closeDag();
    },
    updateMergeUi,
  });
}

/* ====================================================================
 * 7) IPC & Controls Bridge
 * --------------------------------------------------------------------
 * Log buffer plumbing plus the shared send() helper + ControlsBridge.
 * ==================================================================== */

const fallbackLogBuffer = {
  log: (msg) => console.log(String(msg)),
  clearLogs: () => {}
};
const { log, clearLogs } = window.LogBuffer || fallbackLogBuffer;

function send(command, extra = {}) { vscode.postMessage({ command, ...extra }); }

const NullReportRequests = {
  has: () => false,
  runFor: () => null,
  latestReqForRun: () => null,
};

const ControlsBridge = window.IPCControls?.init({
  send,
  currentPageRunId,
  keyOf,
  setRunningFor,
  setPausedFor,
  notify,
  getPyPathValue: () => (pyPath && pyPath.value) || 'python',
  onOpenDag: openDag,
  onCloseDag: closeDag,
  onRequestDagMerge: requestDagMerge,
});

const reportRequests = ControlsBridge?.reportRequests || NullReportRequests;

/* ====================================================================
 * 8) Window & Message Bus
 * --------------------------------------------------------------------
 * Global listeners (resize + message) that keep the webview responsive.
 * ==================================================================== */

window.addEventListener('resize', scheduleChartsRedraw);

window.addEventListener('message', (e) => {
  const m = e.data;
  if (m?.type === 'subsetExported') {
    const rows = Number(m.rows || 0);
    const fmt  = (m.format || '').toUpperCase();
  }
});

window.addEventListener('message', (e) => {
  const m = e.data;

  switch (m.type) {
    case 'resetOk': {
      // Now it’s safe to clear the webview state because the user confirmed.
      try {
        clearLogs();
        RunState.resetLineage();
        selectedForMerge.clear();
        ChartReport.clearAllReports();

        clearCharts();
        clearReportChart();
        updateSelectedExportGate();

        if (runSel) runSel.innerHTML = '';
        IS_RUNNING = false;
        IS_PAUSED = false;
        setRunning(false);
        TEST_PENDING = false;
        if (btnTestNow) {
          btnTestNow.removeAttribute('aria-busy');
          updateTestNowGate();
        }
        updateModelNavUI();

        window.ChartReportSelection?.cancelSelection?.();

        storeSetLiveRun(null);
        storeSetFollowActive(true);

        log('Session reset.');
      } catch (e) {
        console.error('Reset UI error:', e);
      }
      break;
    }
    case 'scriptChosen':
      if (scriptName) scriptName.textContent = `Chosen Python Script: ${m.file}`;
      break;
    case 'logs': {
      const rows = Array.isArray(m.rows) ? m.rows : [];
      //  only clear if the run changed
      clearLogs();
      for (const r of rows) {
        const ts  = r.ts ? new Date(r.ts).toLocaleTimeString() : '';
        const stp = Number.isFinite(r.step) ? ` [s:${r.step}]` : '';
        const msg = r.text || '';
        log(`${ts}${stp} ${msg}`.trim());
      }
      break;
    }
    case 'session_started':
      if (m.run_id) storeSetLiveRun(m.run_id);
      send('requestRuns');
      break;
    case 'testNow':
      if (btnTestNow && m?.status) {
        if (m.status === 'pending') {
          TEST_PENDING = true;
          updateTestNowGate();
          btnTestNow.setAttribute('aria-busy', 'true');
          notify(`Running test on ${m.run_id || 'current run'}…`);
        } else {
          TEST_PENDING = false;
          btnTestNow.removeAttribute('aria-busy');
          updateTestNowGate();
          if (m.status === 'completed') {
            const avg = Number(m?.data?.avg_loss);
            const lossText = Number.isFinite(avg) ? ` (avg loss ${avg.toFixed(4)})` : '';
            notify(`Test complete${lossText}`);
          } else if (m.status === 'error') {
            notify(`Test failed: ${m.error || 'unknown error'}`, 'error');
          }
        }
      }
      break;
    case 'newRun': {
      const child = keyOf(m.run_id || '');
      if (child) storeSetLiveRun(child);     // newest run becomes live

      if (child) {
        // --- provenance logging (single place) ---
        try {
          const kind = m?.meta?.kind;
        } catch (e) {
          console.warn('[newRun provenance log failed]', e);
        }

        const ps = Array.isArray(m.parents) ? m.parents.map(keyOf).filter(Boolean) : [];
        storeAddRun({
          id: child,
          parents: ps,
          name: m.name,
          created_at: m.created_at ?? Date.now(),
          ...m, // keep extra fields if you need them elsewhere
        });

        send('requestRuns');            // always refresh the list
      } else {
        send('requestRuns');
        wireModelNavClicks();
        updateModelNavUI();
      }
      if ((m?.meta?.kind || '').toLowerCase() === 'fork') {
        storeSetFollowActive(true);       // keep following live
        selectRunByRunId(child);
      }
      break;
    }

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
    case 'status': {
      const rk = keyOf(m.run_id || curRunKey());
      setRunningFor(rk, !!m.running);
      if (m.running) {
        storeSetLiveRun(rk);                    // always switch to the run that is actually running
        if (isFollowActive()) selectRunByRunId(rk); // snap view immediately if following
      } else if (!m.running && storeGetLiveRun() === rk) {
        storeSetLiveRun(null);                  // clear stale live when it stops
      }
      log(`Status: ${m.running ? 'running' : 'idle'}${m.run_id ? ` (run ${rk})` : ''}`);
      break;
    }
    case 'runs': {
      const rows = Array.isArray(m.rows) ? m.rows : [];

      // 1) Upsert every row into runsIndex + parents/children + edges (deduped)
      for (const r of rows) {
        const id = runIdOf(r);
        if (!id) continue;

        // normalize parents (supports array + legacy single-field)
        const parents =
          Array.isArray(r.parents)
            ? r.parents
            : [r.parent ?? r.parent_run ?? r.parent_run_id ?? r.parentId ?? null];

        storeAddRun({
          id,
          parents: parents.map(keyOf).filter(Boolean),
          name: r.name,
          created_at: r.created_at ?? r.createdAt ?? r.created ?? r.timestamp ?? r.ts,
          ...r, // keep other fields you rely on elsewhere
        });
      }

      // 2) Rebuild selector/nav order (this does NOT touch lineage)
      fillRunSel(rows);

      wireModelNavClicks();
      updateModelNavUI();

      if (dagOverlay && dagOverlay.classList.contains('show')) renderDag();
      updateSelectRegionGate();
      updateSelectedExportGate();
      break;
    }
    case 'rows': {
      const rows = m.rows || [];
      if (getLastRowsRunKey() !== keyOf(currentPageRunId())) break; // ignore stale/off-page rows
      const targetRun = keyOf(currentPageRunId());
      MetricHistory.replaceAll(targetRun, rows);
      const tracked = ['loss', 'val_loss', ...STREAM_METRICS];
      const sanitized = [];
      for (const r of rows) {
        const step = Number(r.step);
        if (!Number.isFinite(step)) continue;
        const epoch = Number(r?.epoch);
        const rowObj = { step };
        if (Number.isFinite(epoch)) rowObj.epoch = epoch;
        for (const metric of tracked) {
          const val = nf(r?.[metric]);
          rowObj[metric] = Number.isFinite(val) ? val : NaN;
        }
        sanitized.push(rowObj);
      }
      window.ChartStream?.ingestRows?.(sanitized);
      break;
    }
    case 'trainStep': {
      // Primary: does this message belong to the run we're tracking?
      const msgRunId = keyOf(m.run_id);
      const liveRun = keyOf(storeGetLiveRun());
      
      // If we have a live run set, only accept messages from it
      if (liveRun && msgRunId !== liveRun) break;
      
      // If we don't have a live run but DO have a current page, only show that
      const pageRun = keyOf(currentPageRunId());
      if (!liveRun && pageRun && msgRunId !== pageRun) break;
      
      // If we have neither (fresh start), accept any message and set it as live
      if (!liveRun && !pageRun) {
        storeSetLiveRun(msgRunId);
      }

      // ---- Epoch handling: only act when a real epoch comes in ----
      let msgEpoch = null;

      // Only treat it as an epoch update if the field exists and is finite
      if (Object.prototype.hasOwnProperty.call(m, 'epoch') &&
          m.epoch !== null && m.epoch !== undefined) {
        const n = Number(m.epoch);
        if (Number.isFinite(n)) {
          msgEpoch = n;

          // This is still useful for other UI pieces (e.g. “Epoch: X” somewhere)
          if (msgRunId) {
            RUN_EPOCH.set(msgRunId, n);
          }
        }
      }

      MetricHistory.pushLive(msgRunId, m);

      const tracked = ['loss', 'val_loss', ...STREAM_METRICS];
      const metricPayload = {};
      for (const metric of tracked) {
        const val = nf(m?.[metric]);
        metricPayload[metric] = Number.isFinite(val) ? val : NaN;
      }

      // 🔴 IMPORTANT: for the chart, NO fallback.
      // We only ever tag the exact steps that carried an epoch.
      const effectiveEpoch = msgEpoch;
      console.log('effective_epoch=====', effectiveEpoch);

      window.ChartStream?.pendPush?.(m.step, metricPayload, effectiveEpoch);
      window.ChartStream?.scheduleChartsRedraw?.();
      break;
    }


    case 'paused': {
      const rk = keyOf(m.run_id || currentPageRunId());
      LAST_PAUSED_STEP.set(rk, Number(m.step) || 0);
      setRunningFor(rk, false);   // <- add this
      setPausedFor(rk, true);
      if (rk === curRunKey() && btnReport) btnReport.disabled = false;
      break;
    }
    case 'resumed': {
      log('Training ...');
      const rk = keyOf(m.run_id || currentPageRunId());
      setRunningFor(rk, true);
      setPausedFor(rk, false);
      storeSetLiveRun(rk);
      if (rk === curRunKey() && btnReport) btnReport.disabled = true;
      break;
    }
    case 'trainingFinished': {
      log('Training finished.');
      const rk = keyOf(m.run_id || currentPageRunId());
      setRunningFor(rk, false);
      setPausedFor(rk, false);
      break;
    }
    case 'sessionLoaded':
      log('Session loaded. Refreshing runs...');
      send('requestRuns');
      const cr = currentPageRunId();
      if (cr) notifyModelNavSelected(cr);
      break;
    case 'log': {
      const epochVal = Number(m?.epoch);
      const rk = keyOf(m.run_id || currentPageRunId());
      if (rk && Number.isFinite(epochVal)) RUN_EPOCH.set(rk, epochVal);
      log(m.text);
      break;
    }
    case 'error':
      log(`[stderr] ${m.text}`);
      break;
    case 'reportData': {
      notify('Done!');
      const { losses, sample_indices, meta, reqId, owner_run_id } = m;

      // Route only the intended reply
      if (reqId == null || !reportRequests.has(reqId)) { log('(ignored) Report without a known reqId'); return; }
      const intendedRunKey = reportRequests.runFor(reqId);
      if (reportRequests.latestReqForRun(intendedRunKey) !== reqId) { log(`(ignored) Stale report for ${intendedRunKey} (reqId=${reqId})`); return; }
      if (keyOf(owner_run_id) !== intendedRunKey) { log(`(ignored) Report attributed to ${owner_run_id}, expected ${intendedRunKey} (reqId=${reqId})`); return; }

      // Persist + render via ChartReport
      ChartReport.updateReportForRun(intendedRunKey, losses || [], meta || {}, sample_indices || []);
      if (keyOf(currentPageRunId()) === intendedRunKey) ChartReport.showReportFor(intendedRunKey);

      updateSelectRegionGate();
      updateSelectedExportGate();
      break;
    }
    case 'reportFromDb': {
      const { losses, meta, owner_run_id } = m;
      const intendedRunKey = keyOf(owner_run_id || currentPageRunId());

      // Persist + render via ChartReport (no indices from DB)
      ChartReport.updateReportForRun(intendedRunKey, losses || [], meta || {}, []);
      if (keyOf(currentPageRunId()) === intendedRunKey) ChartReport.showReportFor(intendedRunKey);

      updateSelectRegionGate();
      updateSelectedExportGate();
      break;
    }
  }
});
