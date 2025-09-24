"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.autoForkMarkerPlugin = exports.forkSelectionPlugin = void 0;
const forkSelectionPlugin = (getState) => ({
    id: "forkSelection",
    afterDraw(chart) {
        if (!chart || chart.canvas.id !== "lossDistChart")
            return;
        const { enabled, sel } = getState();
        if (!enabled || !sel.active || sel.aVal == null || sel.bVal == null)
            return;
        const scaleX = chart.scales?.x;
        const area = chart.chartArea;
        if (!scaleX || !area)
            return;
        const clamp = (x) => Math.max(area.left, Math.min(area.right, x));
        const aX = clamp(scaleX.getPixelForValue(sel.aVal));
        const bX = clamp(scaleX.getPixelForValue(sel.bVal));
        const left = Math.min(aX, bX);
        const right = Math.max(aX, bX);
        const ctx = chart.ctx;
        ctx.save();
        ctx.fillStyle = "rgba(100, 149, 237, 0.18)";
        ctx.fillRect(left, area.top, right - left, area.bottom - area.top);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#efe444ff";
        ctx.beginPath();
        ctx.moveTo(aX, area.top);
        ctx.lineTo(aX, area.bottom);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(bX, area.top);
        ctx.lineTo(bX, area.bottom);
        ctx.stroke();
        const cap = 6;
        ctx.fillStyle = "#efe444ff";
        ctx.fillRect(aX - 3, area.top - cap, 6, cap);
        ctx.fillRect(bX - 3, area.top - cap, 6, cap);
        ctx.restore();
    },
});
exports.forkSelectionPlugin = forkSelectionPlugin;
const autoForkMarkerPlugin = (getMarkers) => ({
    id: "autoForkMarkers",
    afterDatasetsDraw(chart) {
        const id = chart?.canvas?.id;
        if (!id || (id !== "lossChart" && id !== "valLossChart"))
            return;
        const marks = getMarkers?.();
        if (!marks?.length)
            return;
        const scaleX = chart.scales?.x;
        const area = chart.chartArea;
        if (!scaleX || !area)
            return;
        const labels = chart.data?.labels || [];
        const pixelForStep = (step) => {
            let idx = labels.findIndex((v) => Number(v) === Number(step));
            if (idx >= 0)
                return scaleX.getPixelForTick(idx);
            let best = -1, dBest = Infinity;
            for (let i = 0; i < labels.length; i++) {
                const d = Math.abs(Number(labels[i]) - Number(step));
                if (d < dBest) {
                    dBest = d;
                    best = i;
                }
            }
            return (best >= 0) ? scaleX.getPixelForTick(best) : NaN;
        };
        const ctx = chart.ctx;
        ctx.save();
        ctx.setLineDash([6, 5]);
        ctx.lineWidth = 1;
        for (const m of marks) {
            const x = pixelForStep(m.step);
            if (!Number.isFinite(x))
                continue;
            ctx.strokeStyle = (m.kind === "executed") ? "#22c55e" : "#94a3b8";
            ctx.beginPath();
            ctx.moveTo(x + 0.5, area.top);
            ctx.lineTo(x + 0.5, area.bottom);
            ctx.stroke();
        }
        ctx.restore();
    },
});
exports.autoForkMarkerPlugin = autoForkMarkerPlugin;
//# sourceMappingURL=plugins.js.map