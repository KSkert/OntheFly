"use strict";
/**
 * store/runSlice.ts
 * Run-focused state: active run, per-run running/paused flags, follow-live.
 * No DOM; pure data + helpers.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.initialRunSlice = void 0;
exports.keyOf = keyOf;
exports.setActiveRun = setActiveRun;
exports.setFollowActive = setFollowActive;
exports.setRunningFor = setRunningFor;
exports.setPausedFor = setPausedFor;
exports.markPausedStep = markPausedStep;
exports.isEffectivelyRunning = isEffectivelyRunning;
exports.initialRunSlice = {
    activeRunId: null,
    followActive: true,
    lastPausedStep: new Map(),
    runState: new Map(),
};
/** Utility: normalize “run id like” to a stable key (string). */
function keyOf(id) {
    return id == null ? "" : String(id);
}
function setActiveRun(state, runId) {
    state.activeRunId = keyOf(runId) || null;
}
function setFollowActive(state, on) {
    state.followActive = !!on;
}
function setRunningFor(state, runId, running) {
    const key = keyOf(runId);
    const st = state.runState.get(key) || {};
    st.running = !!running;
    state.runState.set(key, st);
}
function setPausedFor(state, runId, paused) {
    const key = keyOf(runId);
    const st = state.runState.get(key) || {};
    st.paused = !!paused;
    state.runState.set(key, st);
}
function markPausedStep(state, runId, step) {
    const key = keyOf(runId);
    const s = Number(step);
    state.lastPausedStep.set(key, Number.isFinite(s) ? s : 0);
}
function isEffectivelyRunning(state, runId) {
    const key = keyOf(runId || state.activeRunId || "");
    const st = state.runState.get(key) || {};
    return !!st.running && !st.paused;
}
//# sourceMappingURL=runSlice.js.map