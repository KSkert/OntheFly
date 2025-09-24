"use strict";
/**
 * store/reportSlice.ts
 * Report cache + request/response sequencing guards (anti-bleed across runs).
 * Pure data (no Chart, no DOM).
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.initialReportSlice = void 0;
exports.beginReportRequest = beginReportRequest;
exports.isReportResponseCurrent = isReportResponseCurrent;
exports.cacheReport = cacheReport;
exports.getCachedReport = getCachedReport;
exports.initialReportSlice = {
    reportSeq: 0,
    reportReqToRun: new Map(),
    latestReqForRun: new Map(),
    reportCache: new Map(),
};
function beginReportRequest(state, runKey) {
    const reqId = ++state.reportSeq;
    state.reportReqToRun.set(reqId, runKey);
    state.latestReqForRun.set(runKey, reqId);
    return reqId;
}
function isReportResponseCurrent(state, reqId, attributedRunKey) {
    // Known request?
    if (!state.reportReqToRun.has(reqId))
        return false;
    // Mapped to the same run?
    const intended = state.reportReqToRun.get(reqId);
    if (intended !== attributedRunKey)
        return false;
    // Latest for that run?
    return state.latestReqForRun.get(attributedRunKey) === reqId;
}
function cacheReport(state, runKey, entry) {
    state.reportCache.set(runKey, {
        bars: entry.bars?.map(p => ({ x: +p.x, y: +p.y })) || [],
        line: entry.line?.map(p => ({ x: +p.x, y: +p.y })) || [],
        xmin: entry.xmin,
        xmax: entry.xmax,
        note: entry.note || "",
        at_step: entry.at_step ?? null,
        at_epoch: entry.at_epoch ?? null,
    });
}
function getCachedReport(state, runKey) {
    return state.reportCache.get(runKey);
}
//# sourceMappingURL=reportSlice.js.map