"use strict";
/**
 * store/compareSlice.ts
 * Comparison/summary state used by the “columns” view.
 * No DOM; just identifiers and caches.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.initialCompareSlice = void 0;
exports.sumKey = sumKey;
/** helper to form stable keys for compare summaries */
function sumKey(sessionId, runId, view) {
    return `${String(sessionId)}|${String(runId)}:${String(view)}`;
}
exports.initialCompareSlice = {
    compareView: "train",
    colState: new Map(),
    summaryCache: new Map(),
    aggregateKey: "__aggregate__",
};
//# sourceMappingURL=compareSlice.js.map