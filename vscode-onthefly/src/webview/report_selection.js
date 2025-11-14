/* report_selection.js
 * Region selection + manual fork overlay shared across the dashboard.
 */
(function(){
  if (window.ChartReportSelection) return;

  const ids = (window.OnTheFlyExports && OnTheFlyExports.ids) || {};
  const byId = window.ChartUtils?.byId || ((id) => document.getElementById(id));
  const listen = (el, evt, fn) => { if (el) el.addEventListener(evt, fn); };

  let regionSelectMode = false;
  const regionSel = { active: false, aVal: null, bVal: null, dragging: null, activeHandle: 'a' };
  let regionOverlay = null;
  let regionCountPill = null;
  let btnSelectRegion = null;
  let btnExportSelectedSubset = null;
  let exportSubsetFmt = null;
  let overlayFmt = null;

  function notify(text, level = 'info') {
    if (typeof window.notify === 'function') window.notify(text, level);
    else console.log(`[${level}] ${text}`);
  }

  function send(command, payload = {}) {
    if (typeof window.send === 'function') window.send(command, payload);
    else if (window.vscode) window.vscode.postMessage({ command, ...payload });
  }

  function currentRunId() {
    return typeof window.currentPageRunId === 'function' ? window.currentPageRunId() : null;
  }

  function getLossChart() {
    const factory = window.ChartCreation;
    return (factory && factory.get('loss_dist')) || window.lossDistChart || null;
  }

  function getCanvas() {
    return byId(ids.lossDistChart || 'lossDistChart');
  }

  function selectionActive() {
    return regionSelectMode && regionSel.active;
  }

  function cloneSel() {
    return {
      active: !!regionSel.active,
      aVal: regionSel.aVal,
      bVal: regionSel.bVal,
      dragging: regionSel.dragging,
      activeHandle: regionSel.activeHandle
    };
  }

  function updateSelectRegionGate() {
    if (!btnSelectRegion) btnSelectRegion = byId(ids.btnSelectRegion || 'btnSelectRegion');
    if (!btnSelectRegion || !window.ChartReport?.hasReport) return;
    const runId = currentRunId();
    const runKey = typeof window.keyOf === 'function' ? window.keyOf(runId) : (runId == null ? '' : String(runId));
    const hasReport = window.ChartReport.hasReport(runKey);
    btnSelectRegion.disabled = !hasReport;
    btnSelectRegion.style.display = hasReport ? '' : 'none';
    btnSelectRegion.title = hasReport ? '' : 'Generate a report to select a region.';
  }

  function updateSelectedExportGate() {
    if (!btnExportSelectedSubset) btnExportSelectedSubset = byId('btnExportSelectedSubset');
    if (!btnExportSelectedSubset) return;
    const available = selectionActive();
    btnExportSelectedSubset.disabled = !available;
    btnExportSelectedSubset.style.display = available ? '' : 'none';
    btnExportSelectedSubset.title = available ? '' : 'Click “Select Region” first.';
  }

  function computeSelectedSampleCount(minLoss, maxLoss) {
    const chart = getLossChart();
    if (!chart) return 0;
    const bars = chart.data?.datasets?.[0]?.data || [];
    if (!Array.isArray(bars) || !bars.length) return 0;
    const lo = Math.min(minLoss, maxLoss);
    const hi = Math.max(minLoss, maxLoss);
    let total = 0;
    for (const p of bars) {
      if (p && Number.isFinite(p.x) && p.x >= lo && p.x <= hi) total += Number.isFinite(p.y) ? p.y : 0;
    }
    return total;
  }

  function updateSelectedCountPill() {
    if (!regionCountPill || !selectionActive()) return;
    if (regionSel.aVal == null || regionSel.bVal == null) {
      regionCountPill.textContent = 'Total Samples Selected: 0';
      return;
    }
    const n = computeSelectedSampleCount(regionSel.aVal, regionSel.bVal);
    regionCountPill.textContent = `Total Samples Selected: ${n}`;
  }

  function clampValue(v) {
    const chart = getLossChart();
    const s = chart?.scales?.x;
    if (!s) return v;
    return Math.max(s.min, Math.min(s.max, v));
  }

  function setActiveHandleFromEvent(evt) {
    const chart = getLossChart();
    const canvas = getCanvas();
    if (!chart || !canvas || regionSel.aVal == null || regionSel.bVal == null) return;
    const rect = canvas.getBoundingClientRect();
    const xPix = evt.clientX - rect.left;
    const scaleX = chart.scales.x;
    const aPix = scaleX.getPixelForValue(regionSel.aVal);
    const bPix = scaleX.getPixelForValue(regionSel.bVal);
    regionSel.activeHandle = Math.abs(xPix - aPix) <= Math.abs(xPix - bPix) ? 'a' : 'b';
  }

  function bumpHandle(which, delta) {
    const chart = getLossChart();
    if (!chart || !selectionActive()) return;
    const scaleX = chart.scales.x;
    const step = (scaleX.max - scaleX.min) / 100;
    if (which === 'a') regionSel.aVal = clampValue((regionSel.aVal ?? scaleX.min) + delta * step);
    else regionSel.bVal = clampValue((regionSel.bVal ?? scaleX.max) + delta * step);
    chart.update('none');
  }

  function ensureRegionOverlay() {
    if (regionOverlay) return regionOverlay;
    regionOverlay = document.createElement('div');
    Object.assign(regionOverlay.style, {
      position: 'static',
      background: '#334155',
      border: '1px solid #ddd',
      borderRadius: '10px',
      padding: '10px',
      boxShadow: '0 6px 20px rgba(0,0,0,0.15)',
      marginTop: '12px',
      display: 'none'
    });

    const chartWrap = getCanvas()?.closest('.chartWrap');
    if (chartWrap && chartWrap.parentNode) {
      chartWrap.parentNode.insertBefore(regionOverlay, chartWrap.nextSibling);
    } else {
      document.body.appendChild(regionOverlay);
    }

    regionOverlay.innerHTML = `
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <span id="forkCountPill"
          style="
            padding:4px 10px;border-radius:9999px;background:var(--btn-bg);color:var(--btn-fg);
            font-weight:600;font-size:12px;box-shadow:var(--btn-shadow);user-select:none;
          "
        >Total Samples Selected: 0</span>

        <label>LR <input id="forkLR" type="number" step="0.0001" value="0.001" style="width:90px"></label>
        <label>Batch <input id="forkBS" type="number" step="1" value="32" style="width:80px"></label>
        <label>Patience <input id="forkPat" type="number" step="1" value="5" style="width:80px"></label>

        <button id="forkOk">Okay, fork</button>
        <button id="forkALeft"  class="ghostBtn" style="padding:6px 8px;">A◀</button>
        <button id="forkARight" class="ghostBtn" style="padding:6px 8px;">A▶</button>
        <button id="forkBLeft"  class="ghostBtn" style="padding:6px 8px;">B◀</button>
        <button id="forkBRight" class="ghostBtn" style="padding:6px 8px;">B▶</button>
        <label for="exportSelectedSubsetFmt">Format</label>
        <select id="exportSelectedSubsetFmt" aria-label="Export format (selected subset)">
          <option value="parquet">.parquet</option>
          <option value="csv">.csv</option>
          <option value="feather">.feather</option>
        </select>
        <button id="btnExportSelectedSubset" title="Export only the currently selected samples">
          Export subset
        </button>
        <button id="forkCancel">Cancel</button>
      </div>`;

    regionCountPill = byId('forkCountPill');
    btnExportSelectedSubset = byId('btnExportSelectedSubset');
    overlayFmt = byId('exportSelectedSubsetFmt');

    if (!exportSubsetFmt) exportSubsetFmt = byId('exportSubsetFmt');
    if (overlayFmt) {
      overlayFmt.value = (exportSubsetFmt && exportSubsetFmt.value) || 'parquet';
      listen(overlayFmt, 'change', () => { if (exportSubsetFmt) exportSubsetFmt.value = overlayFmt.value; });
      if (exportSubsetFmt) listen(exportSubsetFmt, 'change', () => { overlayFmt.value = exportSubsetFmt.value; });
    }

    listen(btnExportSelectedSubset, 'click', exportSelectedSubset);
    listen(byId('forkOk'), 'click', confirmFork);
    listen(byId('forkCancel'), 'click', cancelSelection);

    listen(byId('forkALeft'),  'click', () => { regionSel.dragging = 'a'; bumpHandle('a', -1); });
    listen(byId('forkARight'), 'click', () => { regionSel.dragging = 'a'; bumpHandle('a', +1); });
    listen(byId('forkBLeft'),  'click', () => { regionSel.dragging = 'b'; bumpHandle('b', -1); });
    listen(byId('forkBRight'), 'click', () => { regionSel.dragging = 'b'; bumpHandle('b', +1); });

    updateSelectedCountPill();
    updateSelectedExportGate();

    return regionOverlay;
  }

  function hideOverlay() {
    if (regionOverlay) regionOverlay.style.display = 'none';
    if (btnSelectRegion) btnSelectRegion.style.display = '';
  }

  function resetSelection() {
    regionSel.aVal = null;
    regionSel.bVal = null;
    regionSel.activeHandle = 'a';
    regionSel.dragging = null;
    regionSel.active = false;
    regionSelectMode = false;
    hideOverlay();
    updateSelectedExportGate();
  }

  function exportSelectedSubset() {
    const runId = currentRunId();
    if (!runId) { notify('No run selected.', 'warn'); return; }
    if (!selectionActive()) {
      notify('Click “Select Region” first, then pick a range.', 'warn');
      return;
    }
    if (regionSel.aVal == null || regionSel.bVal == null) {
      notify('Selection handles are not set yet. Click/drag on the chart.', 'warn');
      return;
    }
    const format = (exportSubsetFmt && exportSubsetFmt.value) || 'parquet';
    const minLoss = Math.min(regionSel.aVal, regionSel.bVal);
    const maxLoss = Math.max(regionSel.aVal, regionSel.bVal);

    const subset_indices = window.ChartReport?.selectedIndicesForRun?.(runId, minLoss, maxLoss) || [];
    if (!subset_indices.length) {
      notify('No samples in the selected range.', 'warn');
      return;
    }

    send('exportSubset', { runId, format, subset_indices });
    notify(`Exporting selected region (${subset_indices.length} samples, ${format.toUpperCase()})…`);
  }

  function confirmFork() {
    const lr = parseFloat(byId('forkLR')?.value || '0.001');
    const bs = parseInt(byId('forkBS')?.value || '32', 10);
    const pat = parseInt(byId('forkPat')?.value || '5', 10);

    if (regionSel.aVal == null || regionSel.bVal == null) {
      notify('Selection handles not set. Click/drag on the chart first.', 'warn');
      return;
    }
    const minLoss = Math.min(regionSel.aVal, regionSel.bVal);
    const maxLoss = Math.max(regionSel.aVal, regionSel.bVal);

    send('fork', {
      runId: currentRunId(),
      region: { minLoss, maxLoss },
      hparams: { lr, batch_size: bs, patience: pat }
    });

    resetSelection();
    const chart = getLossChart();
    chart?.update('none');
  }

  function cancelSelection() {
    resetSelection();
    const chart = getLossChart();
    chart?.update('none');
  }

  function beginSelection() {
    const chart = getLossChart();
    if (!chart) { notify('Report chart not ready yet.', 'warn'); return; }
    const points = chart.data?.datasets?.[0]?.data?.length || 0;
    if (!points) { notify('Generate the report first.', 'warn'); return; }

    regionSelectMode = true;
    regionSel.active = true;
    regionSel.dragging = null;

    const s = chart.scales?.x;
    if (s && Number.isFinite(s.min) && Number.isFinite(s.max)) {
      const span = (s.max - s.min) || 1;
      if (regionSel.aVal == null) regionSel.aVal = s.min + 0.25 * span;
      if (regionSel.bVal == null) regionSel.bVal = s.min + 0.75 * span;
    }

    ensureRegionOverlay();
    if (regionOverlay) regionOverlay.style.display = 'block';
    if (btnSelectRegion) btnSelectRegion.style.display = 'none';
    chart.update('none');
    updateSelectedExportGate();
    updateSelectedCountPill();
  }

  function valFromEvent(evt) {
    const chart = getLossChart();
    const canvas = getCanvas();
    if (!chart || !canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const xPix = evt.clientX - rect.left;
    const scaleX = chart.scales.x;
    const v = scaleX.getValueForPixel(xPix);
    const min = scaleX.min, max = scaleX.max;
    return Math.max(min, Math.min(max, v));
  }

  function nearHandle(evt) {
    const chart = getLossChart();
    const canvas = getCanvas();
    if (!chart || !canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const xPix = evt.clientX - rect.left;
    const scaleX = chart.scales.x;
    if (regionSel.aVal == null || regionSel.bVal == null) return null;
    const aPix = scaleX.getPixelForValue(regionSel.aVal);
    const bPix = scaleX.getPixelForValue(regionSel.bVal);
    const hitRadius = 8;
    if (Math.abs(xPix - aPix) <= hitRadius) return 'a';
    if (Math.abs(xPix - bPix) <= hitRadius) return 'b';
    return null;
  }

  function enableDragOnReportChart() {
    const chart = getLossChart();
    const canvas = getCanvas();
    if (!canvas || !chart || canvas._forkDragWired) return;
    canvas._forkDragWired = true;

    listen(canvas, 'mousedown', (e) => {
      if (!selectionActive()) return;
      const h = nearHandle(e);
      if (h) {
        regionSel.dragging = h;
        regionSel.activeHandle = h;
      }
    });

    listen(canvas, 'click', (e) => {
      if (!selectionActive()) return;
      setActiveHandleFromEvent(e);
      const v = valFromEvent(e);
      if (v == null) return;
      if (regionSel.activeHandle === 'a') regionSel.aVal = v;
      else regionSel.bVal = v;
      chart.update('none');
    });

    listen(window, 'mousemove', (e) => {
      if (!selectionActive() || !regionSel.dragging) return;
      const v = valFromEvent(e);
      if (v == null) return;
      if (regionSel.dragging === 'a') regionSel.aVal = v;
      else regionSel.bVal = v;
      chart.update('none');
    });

    listen(window, 'mouseup', () => { regionSel.dragging = null; });
  }

  function onReportRendered(ownerKey) {
    hideOverlay();
    const chart = getLossChart();
    if (chart) chart.$ownerKey = ownerKey;
    updateSelectRegionGate();
    updateSelectedExportGate();
  }

  function init() {
    btnSelectRegion = byId(ids.btnSelectRegion || 'btnSelectRegion');
    exportSubsetFmt = byId('exportSubsetFmt');
    if (btnSelectRegion && !btnSelectRegion._wiredSelectRegion) {
      btnSelectRegion.addEventListener('click', beginSelection);
      btnSelectRegion._wiredSelectRegion = true;
    }

    updateSelectRegionGate();
    updateSelectedExportGate();

    listen(window, 'keydown', (e) => {
      if (!selectionActive()) return;
      const which = regionSel.dragging || regionSel.activeHandle || 'a';
      if (e.key === 'ArrowLeft')  { bumpHandle(which, -1); e.preventDefault(); }
      if (e.key === 'ArrowRight') { bumpHandle(which, +1); e.preventDefault(); }
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

  window.ChartReportSelection = {
    beginSelection,
    cancelSelection,
    updateSelectRegionGate,
    updateSelectedExportGate,
    onReportRendered,
    getState: () => ({ enabled: selectionActive(), sel: cloneSel() })
  };

  window.enableDragOnReportChart = enableDragOnReportChart;
  window.updateSelectedCountPill = updateSelectedCountPill;

  window.ChartPlugins?.configure?.({ getForkSelectionState: () => ({ enabled: selectionActive(), sel: cloneSel() }) });
})();
