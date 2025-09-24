"use strict";
/**
 * store/seriesSlice.ts
 * Streaming step series: pending buffers + committed series.
 * Scheduling is UI’s concern; this slice just provides push/flush helpers.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.initialSeriesSlice = void 0;
exports.pendPush = pendPush;
exports.flushPending = flushPending;
exports.resetSeries = resetSeries;
exports.initialSeriesSlice = {
    labels: [],
    loss: [],
    val_loss: [],
    pendLabels: [],
    pendLoss: [],
    pendValLoss: [],
};
function pendPush(state, step, loss, valLoss) {
    const s = Number(step);
    const l = Number(loss);
    if (!Number.isFinite(s) || !Number.isFinite(l))
        return;
    state.pendLabels.push(s);
    state.pendLoss.push(l);
    const v = Number(valLoss);
    state.pendValLoss.push(Number.isFinite(v) ? v : Number.NaN);
}
function flushPending(state) {
    const n = state.pendLabels.length;
    if (!n)
        return;
    // Commit
    state.labels.push(...state.pendLabels);
    state.loss.push(...state.pendLoss);
    state.val_loss.push(...state.pendValLoss);
    // Clear buffers
    state.pendLabels.length = 0;
    state.pendLoss.length = 0;
    state.pendValLoss.length = 0;
}
function resetSeries(state) {
    state.labels.length = 0;
    state.loss.length = 0;
    state.val_loss.length = 0;
    state.pendLabels.length = 0;
    state.pendLoss.length = 0;
    state.pendValLoss.length = 0;
}
/**
 * NOTE: you said you’re keeping computeLossHistogram in dashboard.js for now.
 * If/when you want it here, move it into this slice (or utils/math) and export.
 */
//# sourceMappingURL=seriesSlice.js.map