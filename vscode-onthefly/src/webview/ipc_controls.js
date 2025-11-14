/* ipc_controls.js
 * Centralizes dashboard button wiring + report request tracking.
 */
(function () {
  const ids = (window.OnTheFlyExports && OnTheFlyExports.ids) || {};
  const byId = (id) => (id ? document.getElementById(id) : null);
  const on = (el, evt, fn) => { if (el && typeof fn === 'function') el.addEventListener(evt, fn); };
  const warn = (...args) => console.warn('[ipc-controls]', ...args);

  function defaultSend(command, extra = {}) {
    if (!window.vscode) {
      warn('vscode API missing; cannot send', command);
      return;
    }
    window.vscode.postMessage({ command, ...extra });
  }

  function init(options = {}) {
    const {
      send = defaultSend,
      currentPageRunId = () => null,
      keyOf = (id) => (id == null ? '' : String(id)),
      setRunningFor = () => {},
      setPausedFor = () => {},
      notify = () => {},
      getPyPathValue = () => {
        const el = byId(ids.pyPath);
        return (el && el.value) || 'python';
      },
      onOpenDag = () => {},
      onCloseDag = () => {},
      onRequestDagMerge = () => {},
    } = options;

    const els = {
      btnChoose: byId(ids.btnChoose),
      btnSetPy: byId(ids.btnSetPy),
      btnPause: byId(ids.btnPause),
      btnResume: byId(ids.btnResume),
      btnTestNow: byId(ids.btnTestNow),
      btnAutoSave: byId(ids.btnAutoSave),
      btnLoad: byId(ids.btnLoad),
      btnReport: byId(ids.btnGenerateReport) || byId(ids.btnReport),
      btnDistHealth: byId(ids.btnDistHealth),
      btnActivationsHealth: byId(ids.btnActivationsHealth),
      btnNumericsHealth: byId(ids.btnNumericsHealth),
      btnDeterminismHealth: byId(ids.btnDeterminismHealth),
      btnThroughputHealth: byId(ids.btnThroughputHealth),
      btnRefreshRuns: byId(ids.btnRefreshRuns),
      btnOpenDag: byId(ids.btnOpenDag),
      dagClose: byId(ids.dagClose),
      dagMergeBtn: byId(ids.dagMergeBtn),
    };

    const reportState = createReportTracker();

    // Environment + interpreter controls
    on(els.btnChoose, 'click', () => send('chooseScript'));
    on(els.btnSetPy, 'click', () => send('setPython', { path: getPyPathValue() }));

    // Training lifecycle controls
    on(els.btnPause, 'click', () => send('pause', { runId: currentPageRunId() }));
    on(els.btnTestNow, 'click', () => send('testNow', { runId: currentPageRunId() }));
    on(els.btnResume, 'click', () => {
      const rk = keyOf(currentPageRunId());
      setRunningFor(rk, true);
      setPausedFor(rk, false);
      send('resume', { runId: rk });
    });

    // Session persistence controls
    on(els.btnAutoSave, 'click', () => send('exportSession'));
    on(els.btnLoad, 'click', () => send('loadSession'));

    // Report requests (tracks reqId -> run mapping)
    on(els.btnReport, 'click', () => {
      const runId = currentPageRunId();
      if (!runId) { notify('No run selected.', 'warn'); return; }
      const reqId = reportState.nextReqId();
      const runKey = keyOf(runId);
      reportState.track(reqId, runKey);
      send('generateReport', { runId, reqId });
      notify('Generating fresh report…');
    });

    // Health monitors
    [
      [els.btnDistHealth, 'dist_health'],
      [els.btnThroughputHealth, 'throughput_health'],
      [els.btnActivationsHealth, 'activations_health'],
      [els.btnNumericsHealth, 'numerics_health'],
      [els.btnDeterminismHealth, 'determinism_health'],
    ].forEach(([btn, command]) => on(btn, 'click', () => send(command)));

    // Misc
    on(els.btnRefreshRuns, 'click', () => send('resetAll'));

    // DAG overlay controls
    on(els.btnOpenDag, 'click', onOpenDag);
    on(els.dagClose, 'click', onCloseDag);
    on(els.dagMergeBtn, 'click', onRequestDagMerge);

    return {
      reportRequests: {
        has: (reqId) => reportState.has(reqId),
        runFor: (reqId) => reportState.runFor(reqId),
        latestReqForRun: (runKey) => reportState.latestForRun(runKey),
      },
    };
  }

  function createReportTracker() {
    let seq = 0;
    const reqToRun = new Map();
    const latestForRun = new Map();

    return {
      nextReqId() {
        seq += 1;
        return seq;
      },
      track(reqId, runKey) {
        reqToRun.set(reqId, runKey);
        latestForRun.set(runKey, reqId);
      },
      has(reqId) {
        return reqToRun.has(reqId);
      },
      runFor(reqId) {
        return reqToRun.get(reqId);
      },
      latestForRun(runKey) {
        return latestForRun.get(runKey);
      },
    };
  }

  window.IPCControls = { init };
})();
