"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.extensionState = exports.LAST_EXPORT_DIR_KEY = void 0;
exports.setRunActivityState = setRunActivityState;
exports.post = post;
exports.postErr = postErr;
exports.postStatus = postStatus;
exports.postCurrentSession = postCurrentSession;
exports.stripUiOnlyFields = stripUiOnlyFields;
exports.LAST_EXPORT_DIR_KEY = 'onthefly.lastExportDir';
exports.extensionState = {
    panel: null,
    currentRunId: null,
    seenRuns: new Set(),
    currentSessionId: null,
    nativeRunsThisSession: new Set(),
    modelNavSelectedRunId: null,
    runActivityState: null,
    pauseInFlight: false,
    resumeInFlight: false,
    needDiskCleanOnNextTrainer: true,
    lastExtensionContext: null,
    hasTrainerConnectedOnce: false,
    pendingResumeAwaitingFirstRun: false,
};
function setRunActivityState(activity) {
    exports.extensionState.runActivityState = activity;
}
function post(msg) {
    try {
        exports.extensionState.panel?.webview.postMessage(msg);
    }
    catch (e) {
        console.log('[EXT->WEB] post threw:', e);
    }
}
function postErr(e) {
    post({ type: 'error', text: String(e?.message || e) });
}
function postStatus(connected) {
    const activity = exports.extensionState.runActivityState;
    const running = connected && activity === 'running';
    const paused = connected && activity === 'paused';
    post({
        type: 'status',
        connected,
        running,
        paused,
        run_id: exports.extensionState.currentRunId || null,
    });
}
function postCurrentSession() {
    if (exports.extensionState.currentSessionId) {
        post({ type: 'fs.session.current', id: exports.extensionState.currentSessionId });
    }
}
function isUiLogLike(v) {
    return !!v && typeof v === 'object';
}
function stripUiOnlyFields(rows) {
    const cleanOne = (r) => {
        if (!isUiLogLike(r))
            return r;
        const { ts, ...rest } = r;
        if (typeof rest.text === 'string') {
            const mentionsStep = /\bstep\b\s*(?:[:#]\s*)?\d+(?:\s*[:#])?/i.test(rest.text);
            if (mentionsStep)
                delete rest.step;
        }
        return rest;
    };
    return Array.isArray(rows) ? rows.map(cleanOne) : cleanOne(rows);
}
//# sourceMappingURL=state.js.map