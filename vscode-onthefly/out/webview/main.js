"use strict";
/**
 * main.ts
 * ------------------------------------------------------------
 * Bootstraps the webview app. No UI logic here—just wiring.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const host_js_1 = require("./host.js");
const index_js_1 = require("./store/index.js");
const dom_js_1 = require("./ui/utils/dom.js");
const logPanel_js_1 = require("./ui/logPanel.js");
const ChartManager_js_1 = require("./ui/charts/ChartManager.js");
const runSelector_js_1 = require("./ui/runSelector.js");
const dagView_js_1 = require("./ui/dagView.js");
const autoforkPanel_js_1 = require("./ui/autoforkPanel.js");
const compareView_js_1 = require("./ui/compareView.js");
const mergeBanner_js_1 = require("./ui/mergeBanner.js");
/* ---------------- small helpers ---------------- */
function computeLossHistogram(values, numBins = 30) {
    if (!Array.isArray(values) || !values.length) {
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
        if (idx >= numBins)
            idx = numBins - 1;
        if (idx < 0)
            idx = 0;
        counts[idx]++;
    }
    const centers = counts.map((_, i) => edges[i] + step * 0.5);
    const n = values.length;
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const variance = values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, (n - 1));
    const std = Math.sqrt(Math.max(variance, 0));
    let h = 1.06 * (std || (step / 1.06)) * Math.pow(n, -1 / 5);
    if (!Number.isFinite(h) || h <= 1e-12)
        h = step;
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
    };
}
function chartToPngDataURL(canvas) {
    if (!canvas)
        return null;
    const src = canvas;
    const w = src.width, h = src.height;
    const bg = getComputedStyle(src.closest(".chartWrap") || document.body).backgroundColor || "#ffffff";
    const out = document.createElement("canvas");
    out.width = w;
    out.height = h;
    const ctx = out.getContext("2d");
    ctx.save();
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(src, 0, 0);
    ctx.restore();
    return out.toDataURL("image/png");
}
/* ---------------- boot ---------------- */
function boot() {
    // 1) IPC host
    const host = (0, host_js_1.createHost)();
    // post commands to the extension (shape: { command, ...payload })
    const send = (command, payload = {}) => host.send?.({ command, ...payload });
    const waiters = {
        fsSummaryGet: new Set(),
        fsSummaryList: new Set(),
        fsSessionRuns: new Set(),
        fsSessionSummaryGet: new Set(),
    };
    // 2) App store (aggregates all slices)
    const store = (0, index_js_1.createAppStore)();
    // 3) Debug handles
    window.__SEAMLESS_HOST__ = host;
    window.__SEAMLESS_STORE__ = store;
    // 4) UI modules
    const logPanel = new logPanel_js_1.LogPanel("log");
    const charts = new ChartManager_js_1.ChartManager();
    charts.initCharts();
    const runSel = (0, runSelector_js_1.createRunSelector)({
        onChange: (runId) => {
            send("requestRows", { runId });
            send("requestReport", { runId });
        },
    });
    runSel.wireArrows();
    // DAG view (derive data from incoming `runs` rows)
    let _lastRunsRows = [];
    const dag = (0, dagView_js_1.createDagView)({
        getData: () => {
            const nodes = _lastRunsRows.map(r => ({ id: (0, runSelector_js_1.runIdOf)(r), label: r?.name || (0, runSelector_js_1.runIdOf)(r) }));
            const edges = [];
            for (const r of _lastRunsRows) {
                const child = (0, runSelector_js_1.runIdOf)(r);
                const rowParents = Array.isArray(r.parents) ? r.parents : [r.parent ?? r.parent_run ?? r.parent_run_id ?? r.parentId ?? null];
                for (const p of rowParents) {
                    if (p && p !== child)
                        edges.push({ source: String(p), target: child });
                }
            }
            return { nodes, edges };
        },
        onPickRun: (id) => {
            const sel = (0, dom_js_1.byId)("runSel");
            if (!sel)
                return;
            const idx = Array.from(sel.options).findIndex(o => o.value === id);
            if (idx >= 0) {
                sel.selectedIndex = idx;
                sel.dispatchEvent(new Event("change", { bubbles: true }));
                dag.close();
            }
        },
        onMergeClick: (parents, strategy) => {
            send("merge", { payload: { parents, strategy } });
            // toast comes from merge_gating events
        },
    });
    // Compare view: ask extension; resolve on next matching result event
    const compare = (0, compareView_js_1.createCompareView)({
        getRunsForSession: (sessionId) => new Promise((resolve) => {
            waiters.fsSessionRuns.add({
                match: (m) => String(m?.sessionId || "") === String(sessionId),
                resolve: (m) => resolve((m?.runs || []).map((r) => ({ id: String(r.id), label: r.label ?? r.id }))),
            });
            send("fs.session.runs", { sessionId });
        }),
        getSummary: (args) => new Promise((resolve) => {
            if (args.type === "run") {
                waiters.fsSummaryGet.add({
                    match: (m) => String(m?.runId || "") === String(args.runId) && String(m?.view || "") === String(args.view),
                    resolve: (m) => resolve(String(m?.text || "")),
                });
                send("fs.summary.get", { runId: args.runId, view: args.view });
            }
            else {
                waiters.fsSessionSummaryGet.add({
                    match: (m) => String(m?.sessionId || "") === String(args.sessionId) &&
                        String(m?.runId || "") === String(args.runId) &&
                        String(m?.view || "") === String(args.view),
                    resolve: (m) => resolve(String(m?.text || "")),
                });
                send("fs.session.summary.get", { sessionId: args.sessionId, runId: args.runId, view: args.view });
            }
        }),
        requestRunList: () => send("fs.summary.list"),
    });
    // Autofork panel: prefill + tabs
    (0, autoforkPanel_js_1.prefillAutoforkUi)();
    (0, autoforkPanel_js_1.wireAfTabs)();
    /* -------- buttons / UI commands -------- */
    (0, dom_js_1.on)((0, dom_js_1.byId)("btnReport"), "click", () => {
        const runId = runSel.current();
        if (!runId)
            return;
        send("generateReport", { runId, reqId: Date.now() });
    });
    (0, dom_js_1.on)((0, dom_js_1.byId)("btnAutoForkOn"), "click", () => {
        (0, autoforkPanel_js_1.prefillAutoforkUi)();
        const cfg = (0, autoforkPanel_js_1.readAutoForkConfig)();
        cfg.runtime = { ...cfg.runtime, auto_execute: true };
        send("applyAutoForkRules", { config: cfg });
        (0, autoforkPanel_js_1.setAutoModeUI)(true);
    });
    (0, dom_js_1.on)((0, dom_js_1.byId)("btnAutoForkApply"), "click", () => {
        const cfg = (0, autoforkPanel_js_1.readAutoForkConfig)();
        send("applyAutoForkRules", { config: cfg });
    });
    (0, dom_js_1.on)((0, dom_js_1.byId)("btnAutoForkExec"), "click", () => {
        const plan = window._lastAutoForkPlan || null;
        const runId = runSel.current();
        const idx = Number((0, dom_js_1.byId)("afVariantIndex")?.value || 0);
        if (!plan || !runId)
            return;
        send("executeAutoForkPlan", { plan, variantIndex: Number.isFinite(idx) ? idx : 0, runId });
    });
    (0, dom_js_1.on)((0, dom_js_1.byId)("btnOpenDag"), "click", () => dag.open());
    (0, dom_js_1.on)((0, dom_js_1.byId)("dagClose"), "click", () => dag.close());
    // Export chart PNGs via extension
    const exportVia = (canvasId, suggested) => {
        const url = chartToPngDataURL((0, dom_js_1.byId)(canvasId));
        if (!url)
            return;
        send("exportChart", { filename: suggested, dataUrl: url });
    };
    (0, dom_js_1.on)((0, dom_js_1.byId)("exportLossBtn"), "click", () => exportVia("lossChart", `loss_chart_${Date.now()}.png`));
    (0, dom_js_1.on)((0, dom_js_1.byId)("exportLossDistBtn"), "click", () => exportVia("lossDistChart", `loss_distribution_${Date.now()}.png`));
    (0, dom_js_1.on)((0, dom_js_1.byId)("exportValLossBtn"), "click", () => exportVia("valLossChart", `val_loss_${Date.now()}.png`));
    // Simple “subset export” button (optional)
    (0, dom_js_1.on)((0, dom_js_1.byId)("btnExportSubset"), "click", () => {
        const runId = runSel.current();
        if (!runId)
            return;
        const format = ((0, dom_js_1.byId)("exportSubsetFmt")?.value || "parquet");
        send("exportSubset", { runId, format });
    });
    // pause / resume
    (0, dom_js_1.on)((0, dom_js_1.byId)("btnPause"), "click", () => { const id = runSel.current(); if (id)
        send("pause", { runId: id }); });
    (0, dom_js_1.on)((0, dom_js_1.byId)("btnResume"), "click", () => { const id = runSel.current(); if (id)
        send("resume", { runId: id }); });
    // “refresh runs” (and clear columns)
    (0, dom_js_1.on)((0, dom_js_1.byId)("btnRefreshRuns"), "click", () => {
        send("resetAll");
        (0, dom_js_1.byId)("columnsGrid")?.querySelectorAll(".modelCol")?.forEach(n => n.remove());
    });
    /* -------- inbound routing -------- */
    // cache for markers per run (for the charts plugin)
    const MARKERS = new Map();
    const curRunKey = () => runSel.current() || "";
    host.route({
        "*": (m) => {
            // debug:
            // console.debug("[host->webview]", m);
        },
        log: (m) => logPanel.log(m.text),
        error: (m) => logPanel.log(`[stderr] ${m.text}`),
        runs: (m) => {
            _lastRunsRows = m.rows || [];
            const { activeChanged, activeId } = runSel.fillRunSel(_lastRunsRows);
            runSel.updateArrows();
            if (activeChanged && activeId) {
                send("requestRows", { runId: activeId });
                send("requestReport", { runId: activeId });
            }
            dag.render();
        },
        rows: (m) => {
            const rows = m.rows || [];
            charts.setAll(rows.map((r) => r.step), rows.map((r) => Number.isFinite(r.loss) ? r.loss : NaN), rows.map((r) => Number.isFinite(r.val_loss) ? r.val_loss : NaN));
        },
        trainStep: (m) => {
            charts.push(m.step, m.loss, m.val_loss);
        },
        status: (m) => {
            store.setState({ isRunning: !!m.running }, "status");
        },
        paused: () => {
            store.setState({ isRunning: false }, "paused");
            const btn = (0, dom_js_1.byId)("btnReport");
            if (btn)
                btn.disabled = false;
        },
        resumed: () => {
            store.setState({ isRunning: true }, "resumed");
            const btn = (0, dom_js_1.byId)("btnReport");
            if (btn)
                btn.disabled = true;
        },
        trainingFinished: () => {
            store.setState({ isRunning: false }, "trainingFinished");
        },
        // Report payload computed here → feed to ChartManager
        reportData: (m) => {
            const { losses, meta, owner_run_id } = m;
            const intended = (0, runSelector_js_1.keyOf)(owner_run_id || curRunKey());
            let bars = [], line = [], xmin, xmax;
            if (Array.isArray(losses) && losses.length) {
                ({ bars, line, xmin, xmax } = computeLossHistogram(losses, 30));
            }
            charts.setReport(bars, line, xmin, xmax);
            const metaEl = (0, dom_js_1.byId)("reportMeta");
            if (metaEl)
                metaEl.textContent = `Analyzed at step ${meta?.at_step ?? "—"} (epoch ${meta?.at_epoch ?? "—"})`;
            const noteEl = (0, dom_js_1.byId)("reportNote");
            if (noteEl)
                noteEl.textContent = Array.isArray(losses) ? `Samples: ${losses.length} (run: ${intended})` : (meta?.note || "No per-sample losses available.");
        },
        reportFromDb: (m) => {
            const { losses, meta, owner_run_id } = m;
            const intended = (0, runSelector_js_1.keyOf)(owner_run_id || curRunKey());
            let bars = [], line = [], xmin, xmax;
            if (Array.isArray(losses) && losses.length) {
                ({ bars, line, xmin, xmax } = computeLossHistogram(losses, 30));
            }
            charts.setReport(bars, line, xmin, xmax);
            const metaEl = (0, dom_js_1.byId)("reportMeta");
            if (metaEl)
                metaEl.textContent = `Analyzed at step ${meta?.at_step ?? "—"} (epoch ${meta?.at_epoch ?? "—"})`;
            const noteEl = (0, dom_js_1.byId)("reportNote");
            if (noteEl)
                noteEl.textContent = Array.isArray(losses) ? `Samples: ${losses.length} (run: ${intended})` : (meta?.note || "No per-sample losses available.");
        },
        // Compare results → resolve our waiters
        "fs.summary.get.result": (m) => {
            for (const w of Array.from(waiters.fsSummaryGet)) {
                if (w.match(m)) {
                    waiters.fsSummaryGet.delete(w);
                    w.resolve(m);
                    break;
                }
            }
        },
        "fs.summary.list.result": (m) => {
            for (const w of Array.from(waiters.fsSummaryList)) {
                if (w.match(m)) {
                    waiters.fsSummaryList.delete(w);
                    w.resolve(m);
                    break;
                }
            }
        },
        "fs.session.runs.result": (m) => {
            for (const w of Array.from(waiters.fsSessionRuns)) {
                if (w.match(m)) {
                    waiters.fsSessionRuns.delete(w);
                    w.resolve(m);
                    break;
                }
            }
        },
        "fs.session.summary.get.result": (m) => {
            for (const w of Array.from(waiters.fsSessionSummaryGet)) {
                if (w.match(m)) {
                    waiters.fsSessionSummaryGet.delete(w);
                    w.resolve(m);
                    break;
                }
            }
        },
        // Autofork UX (camelCase, matching your extension.ts)
        autoForkSuggested: (m) => {
            window._lastAutoForkPlan = m.plan || null;
            (0, autoforkPanel_js_1.renderAutoForkPlan)(m.plan || null);
            const runKey = (0, runSelector_js_1.keyOf)(m.run_id || curRunKey());
            const step = Number(m.plan?.at_step ?? m.step);
            if (Number.isFinite(step)) {
                const arr = MARKERS.get(runKey) || [];
                if (!arr.some(a => a.step === step && a.kind === "suggested"))
                    arr.push({ step, kind: "suggested" });
                MARKERS.set(runKey, arr);
                if (runKey === curRunKey())
                    charts.setMarkers(arr);
            }
        },
        autoForkExecuted: (m) => {
            const runKey = (0, runSelector_js_1.keyOf)(m.run_id || curRunKey());
            const step = Number(m.plan?.at_step ?? m.step);
            if (Number.isFinite(step)) {
                const arr = MARKERS.get(runKey) || [];
                if (!arr.some(a => a.step === step && a.kind === "executed"))
                    arr.push({ step, kind: "executed" });
                MARKERS.set(runKey, arr);
                if (runKey === curRunKey())
                    charts.setMarkers(arr);
            }
            send("requestRuns");
        },
        autoforkRulesSet: (m) => {
            if (m.config?.runtime && typeof m.config.runtime.auto_execute === "boolean") {
                (0, autoforkPanel_js_1.setAutoModeUI)(!!m.config.runtime.auto_execute);
            }
        },
        // Merge gating toasts
        merge_gating: (m) => {
            if (m.reason === "cleared") {
                (0, mergeBanner_js_1.hideMergeBanner)();
                return;
            }
            const parents = Array.isArray(m.parents) ? m.parents : []; // guard for TS/narrowing
            const map = {
                engine_error: "Merge engine error. Check logs for details.",
                awaiting_signal: "Merge pending: waiting for a suggestion.",
                auto_merge_disabled: "Auto-merge disabled. Enable it to proceed automatically.",
                awaiting_checkpoint: `Merge pending: checkpoints (${m.have_parent_ckpt ? "parent ✓" : "parent ✗"}, ${m.have_child_ckpt ? "child ✓" : "child ✗"}).`,
                saving_child_checkpoint: `Saving checkpoint for child ${m.child_id || ""}…`,
                merging: `Merging ${parents.map(String).join(" + ")}…`,
            };
            const msg = map[String(m.reason)] ?? "Paused";
            const kind = m.reason === "engine_error" ? "error" : (m.reason === "merging" ? "busy" : "info");
            (0, mergeBanner_js_1.showMergeBanner)(msg, kind, { sticky: m.reason === "merging" });
            logPanel.log(`[merge] ${msg}`);
        },
        autoMergeExecuted: (m) => {
            (0, mergeBanner_js_1.hideMergeBanner)();
            if (m.new_run)
                send("requestRuns");
        },
    });
    // first population if the extension hasn’t sent anything yet
    send("requestRuns");
    return { host, store };
}
/* boot once the DOM is ready */
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
}
else {
    boot();
}
//# sourceMappingURL=main.js.map