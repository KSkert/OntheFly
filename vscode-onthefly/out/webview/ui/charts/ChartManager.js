"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ChartManager = void 0;
// ui/charts/ChartManager.ts
const plugins_js_1 = require("./plugins.js");
class ChartManager {
    lossChart = null;
    valChart = null;
    distChart = null;
    SERIES = { labels: [], loss: [], val_loss: [] };
    PEND = { labels: [], loss: [], val_loss: [] };
    rafPending = false;
    lastDrawTs = 0;
    MIN_DRAW_MS = 120;
    forkSel = { active: false, aVal: null, bVal: null };
    forkEnabled = false;
    markers = [];
    constructor() { }
    /** Call once when Chart.js is available and canvases exist. */
    initCharts() {
        const Chart = window.Chart;
        if (!Chart)
            return;
        // register plugins with state getters
        if (!Chart.registry.plugins.get("forkSelection"))
            Chart.register((0, plugins_js_1.forkSelectionPlugin)(() => ({ enabled: this.forkEnabled, sel: this.forkSel })));
        if (!Chart.registry.plugins.get("autoForkMarkers"))
            Chart.register((0, plugins_js_1.autoForkMarkerPlugin)(() => this.markers));
        const commonOpts = {
            animation: false,
            animations: { colors: false, x: false, y: false },
            parsing: true,
            normalized: true,
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false },
            devicePixelRatio: 1,
            plugins: { legend: { display: false }, decimation: { enabled: true, algorithm: "lttb", samples: 500 } },
            scales: { x: { type: "category", ticks: { maxTicksLimit: 8 } } },
            elements: { point: { radius: 0, hitRadius: 0 } },
        };
        const lossCtx = document.getElementById("lossChart")?.getContext("2d");
        if (lossCtx) {
            this.lossChart = new Chart(lossCtx, {
                type: "line",
                data: { labels: [], datasets: [{ label: "Loss", data: [], borderColor: "blue", fill: false, yAxisID: "y" }] },
                options: { ...commonOpts, scales: { ...commonOpts.scales, y: { type: "linear", position: "left", title: { display: true, text: "Train Loss" } } } },
            });
        }
        const valCtx = document.getElementById("valLossChart")?.getContext("2d");
        if (valCtx) {
            this.valChart = new Chart(valCtx, {
                type: "line",
                data: { labels: [], datasets: [{ label: "VAL", data: [], borderColor: "orange", fill: false }] },
                options: { ...commonOpts, scales: { ...commonOpts.scales, y: { type: "linear", position: "left", title: { display: true, text: "Validation Loss" } } } },
            });
        }
        const distCtx = document.getElementById("lossDistChart")?.getContext("2d");
        if (distCtx) {
            this.distChart = new Chart(distCtx, {
                type: "bar",
                data: {
                    datasets: [
                        { type: "bar", label: "Loss frequency", data: [], parsing: false, borderWidth: 1, backgroundColor: "rgba(128, 0, 128, 0.15)" },
                        { type: "line", label: "Loss density (smooth)", data: [], parsing: false, pointRadius: 0, tension: 0.3, borderWidth: 2, borderColor: "purple" },
                    ],
                },
                options: {
                    ...commonOpts,
                    plugins: { ...commonOpts.plugins, decimation: { enabled: false } },
                    scales: { x: { type: "linear", ticks: { maxTicksLimit: 8 }, offset: false }, y: { type: "linear", beginAtZero: true } },
                },
            });
        }
    }
    destroy() {
        this.lossChart?.destroy();
        this.lossChart = null;
        this.valChart?.destroy();
        this.valChart = null;
        this.distChart?.destroy();
        this.distChart = null;
        this.SERIES.labels.length = this.SERIES.loss.length = this.SERIES.val_loss.length = 0;
        this.PEND.labels.length = this.PEND.loss.length = this.PEND.val_loss.length = 0;
        this.rafPending = false;
        this.lastDrawTs = 0;
    }
    /** push one step (frame-paced apply) */
    push(step, loss, valLoss) {
        if (Number.isFinite(step) && Number.isFinite(loss)) {
            this.PEND.labels.push(step);
            this.PEND.loss.push(loss);
            this.PEND.val_loss.push(Number.isFinite(valLoss) ? valLoss : NaN);
        }
        this.scheduleRedraw();
    }
    /** replace whole series (e.g., on rows load) */
    setAll(steps, loss, valLoss) {
        this.SERIES.labels = steps.slice();
        this.SERIES.loss = loss.slice();
        this.SERIES.val_loss = valLoss.slice();
        this.PEND.labels.length = this.PEND.loss.length = this.PEND.val_loss.length = 0;
        this.applyNow();
    }
    setMarkers(m) {
        this.markers = m.slice();
        this.scheduleRedraw();
    }
    enableForkOverlay(enabled) { this.forkEnabled = enabled; this.scheduleRedraw(); }
    setForkSelection(sel) { Object.assign(this.forkSel, sel); this.scheduleRedraw(); }
    setReport(bars, line, xmin, xmax) {
        if (!this.distChart)
            return;
        const ds0 = this.distChart.data?.datasets?.[0];
        const ds1 = this.distChart.data?.datasets?.[1];
        if (ds0)
            ds0.data = (bars || []).slice();
        if (ds1)
            ds1.data = (line || []).slice();
        if (this.distChart.options?.scales?.x) {
            this.distChart.options.scales.x.min = xmin;
            this.distChart.options.scales.x.max = xmax;
        }
        this.distChart.update("none");
    }
    scheduleRedraw() {
        const now = performance.now();
        if (now - this.lastDrawTs < this.MIN_DRAW_MS)
            return;
        if (this.rafPending)
            return;
        this.rafPending = true;
        requestAnimationFrame(() => { this.rafPending = false; this.flushPend(); this.applyNow(); this.lastDrawTs = performance.now(); });
    }
    flushPend() {
        if (!this.PEND.labels.length)
            return;
        this.SERIES.labels.push(...this.PEND.labels);
        this.SERIES.loss.push(...this.PEND.loss);
        this.SERIES.val_loss.push(...this.PEND.val_loss);
        this.PEND.labels.length = this.PEND.loss.length = this.PEND.val_loss.length = 0;
    }
    applyNow() {
        const L = this.SERIES.labels.length;
        if (this.lossChart) {
            this.lossChart.data.labels = this.SERIES.labels.map(Number);
            this.lossChart.data.datasets[0].data = this.SERIES.loss.slice(0, L);
            this.syncY(this.lossChart, this.SERIES.loss);
            this.lossChart.update("none");
        }
        if (this.valChart) {
            this.valChart.data.labels = this.SERIES.labels.map(Number);
            this.valChart.data.datasets[0].data = this.SERIES.val_loss.slice(0, L);
            this.valChart.update("none");
        }
    }
    syncY(chart, loss) {
        if (!chart?.options?.scales)
            return;
        const ls = (loss || []).filter(Number.isFinite);
        if (!ls.length)
            return;
        const minL = Math.min(...ls), maxL = Math.max(...ls);
        const pad = (maxL - minL) * 0.05 || 1e-6;
        chart.options.scales.y.min = minL - pad;
        chart.options.scales.y.max = maxL + pad;
    }
}
exports.ChartManager = ChartManager;
//# sourceMappingURL=ChartManager.js.map