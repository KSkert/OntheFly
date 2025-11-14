/* chart-report.js
 * Report cache + histogram chart hookup. Independent of streaming.
 */
(function(){
  const { computeLossHistogram, byId } = window.ChartUtils;
  const CH = window.ChartCreation;

  // one report per runId
  const REPORT_CACHE = (window.__REPORT_CACHE__ ||= new Map());
  const keyOf = id => (id == null ? '' : String(id));
  const cacheKeyFor = (runId) => `${keyOf(runId)}`;

  function hasReport(runId) {
  return REPORT_CACHE.has(cacheKeyFor(runId));
  }
  function getReport(runId) {
    return REPORT_CACHE.get(cacheKeyFor(runId));
  }
  function clearAllReports() {
    REPORT_CACHE.clear();
  }

  function selectedIndicesForRun(runId, lo, hi) {
    const c = getReport(runId);
    if (!c || !c.lossesRaw || !c.indexMap) return [];
    const a = Math.min(lo, hi), b = Math.max(lo, hi);
    const out = [];
    const L = c.lossesRaw.length;
    for (let i = 0; i < L; i++) {
      const v = c.lossesRaw[i];
      if (Number.isFinite(v) && v >= a && v <= b) out.push(c.indexMap[i]);
    }
    return out;
  }

  function updateReportForRun(runId, values, meta = {}, sample_indices = []) {
    const arr = Array.isArray(values) ? values.map(Number) : [];
    const h   = computeLossHistogram(arr, 30);
    const n   = arr.length;
    let idx   = Array.isArray(sample_indices) ? sample_indices.slice(0, n) : [];
    if (idx.length !== n) idx = Array.from({ length: n }, (_, i) => i);

    REPORT_CACHE.set(cacheKeyFor(runId), {
      ...h,
      note: meta.note || '',
      at_step: meta.at_step ?? null,
      at_epoch: meta.at_epoch ?? null,
      lossesRaw: new Float32Array(arr),
      indexMap:  new Uint32Array(idx.map(v => (Number(v) | 0))),
    });
  }

  function clearReportChart() {
    const ch = CH.get('loss_dist');
    if (!ch) return;
    const bars = ch.data?.datasets?.[0];
    const line = ch.data?.datasets?.[1];
    if (bars) bars.data = [];
    if (line) line.data = [];
    ch.update('none');
    const noteEl = byId((OnTheFlyExports.ids||{}).reportNote);
    const metaEl = byId((OnTheFlyExports.ids||{}).reportMeta);
    if (noteEl) noteEl.textContent = '';
    if (metaEl) metaEl.textContent = '—';
    window.ChartReportSelection?.cancelSelection?.();
  }

  function showReportFor(runId) {
    const ch = CH.get('loss_dist');
    if (!ch) return;
    const cache = REPORT_CACHE.get(cacheKeyFor(runId));
    if (!cache) { clearReportChart(); return; }

    const bars = ch.data?.datasets?.[0];
    const line = ch.data?.datasets?.[1];
    if (bars) bars.data = cache.bars.slice();
    if (line) line.data = cache.line.slice();

    if (ch.options?.scales?.x) {
      ch.options.scales.x.min = cache.xmin;
      ch.options.scales.x.max = cache.xmax;
    }

    const noteEl = byId((OnTheFlyExports.ids||{}).reportNote);
    const metaEl = byId((OnTheFlyExports.ids||{}).reportMeta);
    if (noteEl) noteEl.textContent = cache.note || '';

    const fmt = (window.formatWhen || ((s,e)=>`Analyzed at step ${Number.isFinite(s)?s:'—'} (epoch ${Number.isFinite(e)?e:'—'})`));
    if (metaEl) metaEl.textContent = fmt(cache.at_step, cache.at_epoch);

    ch.update('none');
    window.ChartReportSelection?.onReportRendered?.(cacheKeyFor(runId));
  }

  // (optional) external API
  window.ChartReport = {
    REPORT_CACHE,
    updateReportForRun,
    showReportFor,
    clearReportChart,
    hasReport,
    getReport,
    clearAllReports,
    selectedIndicesForRun
  };
})();

(function(){
  if (window.__REPORT_HELPERS_WIRED__) return;
  window.__REPORT_HELPERS_WIRED__ = true;

  const ids = (window.OnTheFlyExports && OnTheFlyExports.ids) || {};
  const HELP_TIP = 'Generate this report to manually fork.';

  function wireReportHelpBadge(){
    var btn = document.getElementById(ids.btnGenerateReport || 'btnGenerateReport');
    if (!btn || btn.dataset.fsHelpWrapped) return;

    var wrap = document.createElement('span');
    wrap.className = 'fs-helpWrap';
    btn.parentElement.insertBefore(wrap, btn);
    wrap.appendChild(btn);

    var mark = document.createElement('span');
    mark.className = 'fs-helpMark';
    mark.textContent = '?';
    mark.setAttribute('data-tip', HELP_TIP);
    wrap.appendChild(mark);

    btn.dataset.fsHelpWrapped = '1';
  }

  function wireMoveAndToggle(){
    var btn   = document.getElementById(ids.btnSelectRegion || 'btnSelectRegion');
    var aside = document.querySelector('.reportSide');
    if (!btn || !aside) return;
    aside.appendChild(btn);
    btn.style.display = 'none';

    var meta = document.getElementById(ids.reportMeta || 'reportMeta');
    var note = document.getElementById(ids.reportNote || 'reportNote');

    function hasReport() {
      var metaTxt = (meta && meta.textContent || '').trim();
      var noteTxt = (note && note.textContent || '').trim();
      return (metaTxt && metaTxt !== '—') || (noteTxt.length > 0);
    }
    function updateVisibility(){ btn.style.display = hasReport() ? '' : 'none'; }

    var gen = document.getElementById(ids.btnGenerateReport || 'btnGenerateReport');
    if (gen) gen.addEventListener('click', function(){ setTimeout(updateVisibility, 60); });

    var mo = new MutationObserver(updateVisibility);
    if (meta) mo.observe(meta, { childList: true, characterData: true, subtree: true });
    if (note) mo.observe(note, { childList: true, characterData: true, subtree: true });

    updateVisibility();
  }

  function init(){
    wireReportHelpBadge();
    wireMoveAndToggle();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();

  (window.ChartReportUI ||= { wireReportHelpBadge, wireMoveAndToggle });
})();


(function(){
  if (window.__REPORT_HELP_BADGE__) return;
  window.__REPORT_HELP_BADGE__ = true;

  function init(){
    window.ChartReportUI && typeof window.ChartReportUI.wireReportHelpBadge === 'function' &&
      window.ChartReportUI.wireReportHelpBadge();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
