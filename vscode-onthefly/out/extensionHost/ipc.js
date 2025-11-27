"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.registerHardResetHandler = registerHardResetHandler;
exports.trainerActive = trainerActive;
exports.sendReq = sendReq;
exports.ensureTrainerServer = ensureTrainerServer;
exports.shutdownTrainerServer = shutdownTrainerServer;
exports.disconnectTrainer = disconnectTrainer;
exports.setRunActivity = setRunActivity;
exports.sendCtl = sendCtl;
exports.requestBackendPause = requestBackendPause;
exports.resumeTrainerOn = resumeTrainerOn;
exports.resumeAfterReset = resumeAfterReset;
exports.ensureTrainerOnRun = ensureTrainerOnRun;
exports.getSelectedRegionIndices = getSelectedRegionIndices;
exports.runHealth = runHealth;
const vscode = __importStar(require("vscode"));
const net = __importStar(require("net"));
const crypto = __importStar(require("crypto"));
const storage_1 = require("../storage");
const state_1 = require("./state");
const pending = new Map();
const DASHBOARD_PORT = Number(process.env.ONTHEFLY_DASHBOARD_PORT || '47621');
let trainerSocket = null;
let trainerBuffer = '';
let trainerServer = null;
let hardResetHandler = null;
function registerHardResetHandler(handler) {
    hardResetHandler = handler;
}
function trainerActive() {
    return Boolean(trainerSocket && !trainerSocket.destroyed);
}
function sendReq(cmd, payload = {}, timeoutMs = 15000) {
    if (!trainerSocket || trainerSocket.destroyed) {
        return Promise.reject(new Error('No Trainer connected. Run your script with OnTheFlyTrainer first.'));
    }
    const id = crypto.randomUUID();
    const line = JSON.stringify({ id, cmd, payload }) + '\n';
    trainerSocket.write(line, 'utf8');
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            pending.delete(id);
            reject(new Error(`timeout waiting for ${cmd}`));
        }, timeoutMs);
        pending.set(id, { resolve, reject, timer });
    });
}
function ensureTrainerServer() {
    if (trainerServer)
        return;
    trainerServer = net.createServer((socket) => {
        if (trainerSocket && !trainerSocket.destroyed) {
            socket.destroy();
            vscode.window.showWarningMessage('Another Trainer tried to connect while one is active. Close the existing run first.');
            return;
        }
        (async () => {
            if (state_1.extensionState.hasTrainerConnectedOnce && state_1.extensionState.lastExtensionContext && hardResetHandler) {
                try {
                    await hardResetHandler(state_1.extensionState.lastExtensionContext, { fromUser: false });
                }
                catch (e) {
                    console.warn('[onthefly] automatic session reset on new trainer failed:', e);
                }
            }
            state_1.extensionState.hasTrainerConnectedOnce = true;
            trainerSocket = socket;
            trainerBuffer = '';
            socket.setEncoding('utf8');
            socket.on('data', (chunk) => handleTrainerData(chunk));
            socket.on('error', (err) => {
                (0, state_1.post)({ type: 'error', text: `[trainer] ${err?.message || err}` });
            });
            socket.on('close', () => {
                (0, state_1.post)({ type: 'log', text: 'Trainer disconnected.' });
                disconnectTrainer(false);
            });
            (0, state_1.postStatus)(true);
            (0, state_1.post)({ type: 'log', text: 'Trainer connected. Streaming events live.' });
            const target = (state_1.extensionState.modelNavSelectedRunId && state_1.extensionState.modelNavSelectedRunId.trim()) ||
                (state_1.extensionState.currentRunId && state_1.extensionState.currentRunId.trim()) ||
                null;
            if (target) {
                seedTrainerForRun(target).catch((e) => {
                    console.warn('[onthefly] attach_context failed on connect:', e);
                });
            }
        })().catch((err) => {
            console.warn('[onthefly] trainer connection handler failed:', err);
        });
    });
    trainerServer.on('error', (err) => {
        const msg = `OnTheFly dashboard could not listen on port ${DASHBOARD_PORT}: ${err?.message || err}`;
        vscode.window.showErrorMessage(msg);
        (0, state_1.post)({ type: 'error', text: msg });
    });
    trainerServer.listen(DASHBOARD_PORT, '127.0.0.1', () => {
        (0, state_1.post)({
            type: 'log',
            text: `Waiting for Trainer connections on localhost:${DASHBOARD_PORT}. Run your script to attach.`,
        });
    });
}
function shutdownTrainerServer() {
    if (trainerServer) {
        try {
            trainerServer.close();
        }
        catch { }
        trainerServer = null;
    }
    disconnectTrainer(false);
}
function disconnectTrainer(notify = true) {
    if (trainerSocket) {
        try {
            trainerSocket.destroy();
        }
        catch { }
        trainerSocket = null;
    }
    trainerBuffer = '';
    setRunActivity(null);
    state_1.extensionState.pauseInFlight = false;
    state_1.extensionState.resumeInFlight = false;
    for (const [, p] of pending) {
        clearTimeout(p.timer);
        p.reject(new Error('Trainer disconnected'));
    }
    pending.clear();
    if (notify) {
        (0, state_1.post)({ type: 'log', text: 'Trainer connection closed.' });
    }
    (0, state_1.postStatus)(false);
}
function handleTrainerData(chunk) {
    trainerBuffer += chunk;
    let idx;
    while ((idx = trainerBuffer.indexOf('\n')) >= 0) {
        const line = trainerBuffer.slice(0, idx);
        trainerBuffer = trainerBuffer.slice(idx + 1);
        if (line.trim()) {
            handleLine(line);
        }
    }
}
function setRunActivity(state) {
    (0, state_1.setRunActivityState)(state);
    (0, state_1.postStatus)(trainerActive());
}
function requireTrainerConnection() {
    if (!trainerActive()) {
        vscode.window.showErrorMessage('No Trainer connection. Run your training script with an OnTheFlyTrainer to stream data.');
        (0, state_1.postStatus)(false);
        return false;
    }
    return true;
}
async function waitForTrainerConnection(timeoutMs = 0) {
    const start = Date.now();
    while (true) {
        if (trainerSocket && !trainerSocket.destroyed)
            return;
        if (timeoutMs && Date.now() - start > timeoutMs) {
            throw new Error('Timed out waiting for Trainer connection.');
        }
        await new Promise((res) => setTimeout(res, 250));
    }
}
async function startRun() {
    ensureTrainerServer();
    if (trainerSocket && !trainerSocket.destroyed) {
        (0, state_1.postStatus)(true);
        return;
    }
    await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Waiting for OnTheFly Trainer connection…' }, async () => {
        try {
            await waitForTrainerConnection(60000);
            (0, state_1.postStatus)(true);
        }
        catch (e) {
            throw new Error('No Trainer connected. Run your training script in a terminal and instantiate OnTheFlyTrainer.');
        }
    });
}
const optimisticEcho = {
    pause: 'paused',
    resume: 'resumed',
    save_ckpt: 'checkpointSaved',
    merge: 'merged',
};
function sendCtl(cmd) {
    if (!trainerSocket || trainerSocket.destroyed) {
        (0, state_1.post)({ type: 'error', text: 'No Trainer connected. Run your script first.' });
        return;
    }
    try {
        trainerSocket.write(JSON.stringify(cmd) + '\n');
        const t = cmd.cmd;
        if (t === 'resume')
            setRunActivity('running');
        if (t && optimisticEcho[t])
            (0, state_1.post)({ type: optimisticEcho[t], payload: cmd });
    }
    catch (e) {
        (0, state_1.postErr)(e);
    }
}
async function requestBackendPause(timeoutMs = 30_000) {
    const data = await sendReq('pause', {}, timeoutMs);
    setRunActivity('paused');
    return data;
}
async function resumeTrainerOn(target) {
    if (state_1.extensionState.resumeInFlight) {
        return;
    }
    state_1.extensionState.resumeInFlight = true;
    try {
        await ensureTrainerOnRun(target);
        sendCtl({ cmd: 'resume' });
        setRunActivity('running');
    }
    catch (e) {
        (0, state_1.postErr)(e);
        (0, state_1.postStatus)(false);
    }
    finally {
        state_1.extensionState.resumeInFlight = false;
    }
}
async function resumeAfterReset() {
    if (state_1.extensionState.resumeInFlight) {
        return;
    }
    state_1.extensionState.resumeInFlight = true;
    state_1.extensionState.pendingResumeAwaitingFirstRun = true;
    try {
        await startRun();
        state_1.extensionState.pendingResumeAwaitingFirstRun = false;
        if (!trainerSocket || trainerSocket.destroyed) {
            throw new Error('Trainer disconnected before resume.');
        }
        sendCtl({ cmd: 'resume' });
        setRunActivity('running');
    }
    catch (e) {
        (0, state_1.postErr)(e);
        throw e;
    }
    finally {
        state_1.extensionState.pendingResumeAwaitingFirstRun = false;
        state_1.extensionState.resumeInFlight = false;
    }
}
function computeForkMergeCounters() {
    let forkMax = 0;
    let mergeMax = 0;
    try {
        const runs = (0, storage_1.listRuns)();
        const reFork = /^fork(\d+)$/i;
        const reMerge = /^merge(\d+)$/i;
        for (const r of runs) {
            for (const v of [r.name, r.run_id]) {
                if (!v)
                    continue;
                const s = String(v).trim();
                let m = reFork.exec(s);
                if (m) {
                    const n = parseInt(m[1], 10);
                    if (Number.isFinite(n) && n > forkMax)
                        forkMax = n;
                }
                m = reMerge.exec(s);
                if (m) {
                    const n = parseInt(m[1], 10);
                    if (Number.isFinite(n) && n > mergeMax)
                        mergeMax = n;
                }
            }
        }
    }
    catch (e) {
        console.warn('[onthefly] computeForkMergeCounters failed:', e);
    }
    return { forkMax, mergeMax };
}
function latestResumeHints(runId) {
    if (!runId)
        return null;
    const ck = (0, storage_1.latestCheckpointForRun)(runId);
    return ck ? { init_ckpt: ck.path, init_step: Number(ck.step) || 0 } : null;
}
async function seedTrainerForRun(targetRunId) {
    const { forkMax, mergeMax } = computeForkMergeCounters();
    const hints = latestResumeHints(targetRunId) || {};
    if (!state_1.extensionState.currentRunId || state_1.extensionState.currentRunId !== targetRunId) {
        await sendReq('attach_context', {
            run_id: targetRunId,
            ...hints,
            fork_counter_init: forkMax || 0,
            merge_counter_init: mergeMax || 0,
        }, 30_000);
    }
    else {
        await sendReq('set_counter_seeds', {
            fork_counter_init: forkMax || 0,
            merge_counter_init: mergeMax || 0,
        }, 30_000);
    }
}
async function ensureTrainerOnRun(targetRunId) {
    if (!targetRunId)
        throw new Error('No target run.');
    await startRun();
    if (!requireTrainerConnection())
        return;
    const alreadyOn = state_1.extensionState.currentRunId && state_1.extensionState.currentRunId === targetRunId;
    if (alreadyOn) {
        return;
    }
    const hints = latestResumeHints(targetRunId) || {};
    const { forkMax, mergeMax } = computeForkMergeCounters();
    await sendReq('switch_run', {
        run_id: targetRunId,
        ...hints,
        fork_counter_init: forkMax || 0,
        merge_counter_init: mergeMax || 0,
    }, 30_000);
    state_1.extensionState.currentRunId = targetRunId;
    state_1.extensionState.modelNavSelectedRunId = targetRunId;
    (0, state_1.post)({ type: 'modelNav.select', runId: targetRunId });
}
async function getSelectedRegionIndices(runId, minLoss, maxLoss) {
    const data = await sendReq('get_selected_region_indices', {
        run_id: String(runId),
        min_loss: Number(minLoss),
        max_loss: Number(maxLoss),
    }, 60_000);
    const arr = Array.isArray(data?.indices) ? data.indices : [];
    return arr
        .map((n) => Number(n) | 0)
        .filter((n) => Number.isFinite(n) && n >= 0);
}
async function runHealth(cmd, payload, eventType, timeoutMs = 60_000, forRunId) {
    const run_id = forRunId || state_1.extensionState.modelNavSelectedRunId || state_1.extensionState.currentRunId || null;
    try {
        (0, state_1.post)({ type: eventType, run_id, pending: true });
        try {
            await requestBackendPause(30_000);
        }
        catch { }
        const data = await sendReq(cmd, payload, timeoutMs);
        (0, state_1.post)({ type: eventType, cmd, payload, data, run_id });
    }
    catch (e) {
        (0, state_1.post)({ type: eventType, run_id, error: String(e?.message || e) });
        (0, state_1.postErr)(e);
    }
}
function normalizeParents(from) {
    const raw = (from && (from.parents ?? from.parent)) ?? [];
    if (Array.isArray(raw))
        return raw.filter(Boolean).map((s) => String(s));
    if (raw == null || raw === '')
        return [];
    return [String(raw)];
}
function handleLine(line) {
    let obj = null;
    try {
        obj = JSON.parse(line);
    }
    catch {
        (0, state_1.post)({ type: 'log', text: line });
        return;
    }
    if (obj && obj.event && !obj.type) {
        obj.type = obj.event;
        delete obj.event;
    }
    if (obj && obj.id && (obj.ok === true || obj.ok === false)) {
        const p = pending.get(obj.id);
        if (p) {
            clearTimeout(p.timer);
            pending.delete(obj.id);
            obj.ok ? p.resolve(obj.data) : p.reject(new Error(obj.error || 'python error'));
        }
        return;
    }
    if (obj && obj.type === 'testStep') {
        const run_id = obj.run_id || state_1.extensionState.currentRunId || 'live';
        const step = Number(obj.step) || 0;
        const loss = Number(obj.loss);
        (0, state_1.post)({ type: 'testStep', run_id, step, loss: Number.isFinite(loss) ? loss : null, ts: Date.now() });
        try {
            (0, storage_1.insertTestMetric)(run_id, step, Number.isFinite(loss) ? loss : null);
        }
        catch { }
        return;
    }
    if (obj?.type === 'paused') {
        setRunActivity('paused');
        state_1.extensionState.pauseInFlight = false;
    }
    else if (obj?.type === 'resumed') {
        setRunActivity('running');
        state_1.extensionState.resumeInFlight = false;
    }
    else if (obj?.type === 'trainingFinished') {
        setRunActivity(null);
        state_1.extensionState.pauseInFlight = false;
        state_1.extensionState.resumeInFlight = false;
    }
    if (obj?.type === 'log') {
        const run_id = obj.run_id || state_1.extensionState.currentRunId || 'live';
        const session_id = obj.session_id || obj.sessionId || state_1.extensionState.currentSessionId || null;
        if (run_id && session_id && state_1.extensionState.currentSessionId && session_id === state_1.extensionState.currentSessionId) {
            state_1.extensionState.nativeRunsThisSession.add(String(run_id));
        }
        const level = obj.level || 'info';
        const text = String(obj.text || '');
        const pRaw = (obj.phase && String(obj.phase).toLowerCase()) || null;
        let phase = (pRaw === 'train' || pRaw === 'test' || pRaw === 'info') ? pRaw : 'info';
        let step = null;
        let epoch = null;
        const mEpoch = text.match(/^\s*epoch\s+(\d+)\s*$/i);
        if (mEpoch && phase === 'info') {
            epoch = Number(mEpoch[1]);
            phase = 'train';
        }
        const mTest = text.match(/step\s+(\d+)\s*:\s*test[_\s]*loss\s*=\s*([0-9.eE+\-]+)/i);
        if (mTest && phase === 'info') {
            step = Number(mTest[1]);
            phase = 'test';
        }
        if (/\btesting\b/i.test(text) && phase === 'info')
            phase = 'test';
        if (/\b(?:eval|evaluation)\b/i.test(text) && /\b(loss|acc|metric)\b/i.test(text) && phase === 'info')
            phase = 'test';
        const mTrain = text.match(/step\s+(\d+)\s*:\s*train[_\s]*loss\s*=\s*([0-9.eE+\-]+).*?val[_\s]*loss\s*=\s*([0-9.eE+\-]+|None)/i);
        if (mTrain && phase === 'info') {
            step = Number(mTrain[1]);
            phase = 'train';
        }
        if ((/\btrain[_\s]*loss\b/i.test(text) || /\bval[_\s]*loss\b/i.test(text)) && phase === 'info') {
            phase = 'train';
        }
        const tsMs = (Number(obj.ts) > 0 ? Math.round(Number(obj.ts) * 1000) : Date.now());
        const hasStepInText = /\bstep\b\s*(?:[:#]\s*)?\d+(?:\s*[:#])?/i.test(text);
        const stepForUI = hasStepInText ? null : step;
        try {
            (0, storage_1.insertLog)({ run_id, session_id, level, text, phase, step, epoch, ts: tsMs });
        }
        catch { }
        (0, state_1.post)({ type: 'log', run_id, session_id, level, text, phase, step: stepForUI, epoch });
        return;
    }
    if (obj?.type === 'reportData') {
        return;
    }
    if (obj?.type === 'merge_gating') {
        (0, state_1.post)({
            type: 'merge_gating',
            reason: obj.reason || 'unknown',
            parents: Array.isArray(obj.parents) ? obj.parents.map(String) : [],
            step: Number(obj.step) || null,
            run_id: obj.run_id || state_1.extensionState.currentRunId || null,
            ...obj,
        });
        return;
    }
    if (obj && obj.type === 'trainStep') {
        const run_id = obj.run_id || state_1.extensionState.currentRunId || 'live';
        const num = (value) => (Number.isFinite(value) ? Number(value) : null);
        state_1.extensionState.currentRunId = run_id;
        if (!state_1.extensionState.modelNavSelectedRunId) {
            state_1.extensionState.modelNavSelectedRunId = run_id;
        }
        const row = {
            run_id,
            step: Number(obj.step) || 0,
            epoch: num(obj.epoch),
            loss: num(obj.loss),
            val_loss: num(obj.val_loss),
            accuracy: num(obj.accuracy),
            lr: num(obj.lr),
            grad_norm: num(obj.grad_norm),
            weight_norm: num(obj.weight_norm),
            activation_zero_frac: num(obj.activation_zero_frac),
            throughput: num(obj.throughput),
            mem_vram: num(obj.mem_vram),
            gpu_util: num(obj.gpu_util),
            ts: Number(obj.ts) || Date.now(),
        };
        const lastVal = (obj.val_loss == null ? 'None' : String(obj.val_loss));
        (0, state_1.post)({ type: 'trainStep', ...row });
        const logLine = `step ${row.step}: train_loss = ${row.loss ?? 'None'}, val_loss = ${lastVal}`;
        if (state_1.extensionState.runActivityState !== 'running') {
            setRunActivity('running');
        }
        (0, state_1.post)({ type: 'log', level: 'info', phase: 'train', text: logLine });
        try {
            (0, storage_1.insertMetric)(run_id, row);
        }
        catch (e) {
            console.warn('[onthefly] metric persist error:', e);
        }
        return;
    }
    if (obj?.type === 'log' && obj.text && /model\s+session_id/i.test(obj.text)) {
        const m = String(obj.text).match(/session_id\s*=\s*([^\s]+)/i);
        if (m && m[1]) {
            state_1.extensionState.currentSessionId = m[1];
            (0, state_1.postCurrentSession)();
        }
    }
    if (obj && obj.type === 'newRun') {
        const id = String(obj.run_id || '').trim();
        if (id)
            state_1.extensionState.nativeRunsThisSession.add(id);
        if (!id)
            return;
        if (obj.session_id || obj.sessionId) {
            state_1.extensionState.currentSessionId = String(obj.session_id || obj.sessionId);
            (0, state_1.postCurrentSession)();
        }
        const parents = normalizeParents(obj);
        const project = obj.project || 'default';
        const name = obj.run_name || id;
        if (!state_1.extensionState.seenRuns.has(id)) {
            state_1.extensionState.seenRuns.add(id);
            state_1.extensionState.currentRunId = id;
            const primaryParent = (parents && parents[0]) ?? null;
            const existingRuns = (0, storage_1.listRuns)();
            const existsInDb = existingRuns.some(r => r.run_id === id);
            if (!existsInDb) {
                (0, storage_1.insertRun)(id, project, name, primaryParent);
            }
        }
        else {
            state_1.extensionState.currentRunId = id;
        }
        (0, state_1.post)({
            type: 'newRun',
            run_id: id,
            parents,
            meta: obj.meta,
            session_id: state_1.extensionState.currentSessionId || null,
        });
        (0, state_1.post)({ type: 'runs', rows: (0, storage_1.listRuns)() });
        return;
    }
    if (obj?.type === 'auto_test_complete') {
        const runId = String(obj.run_id || state_1.extensionState.currentRunId || '').trim();
        const ckptPath = obj.ckpt_path ? String(obj.ckpt_path) : '';
        const step = Number(obj.step) || 0;
        if (runId && ckptPath) {
            try {
                const ckptId = `${runId}:${step}:${Date.now()}`;
                (0, storage_1.insertCheckpoint)(ckptId, runId, step, ckptPath);
                (0, state_1.post)({
                    type: 'log',
                    level: 'info',
                    text: `[auto-test] checkpoint recorded for run ${runId} at step ${step}`,
                });
            }
            catch (e) {
                console.warn('[onthefly] failed to persist auto-test checkpoint', e);
                (0, state_1.post)({
                    type: 'log',
                    level: 'warn',
                    text: `[auto-test] failed to persist checkpoint for run ${runId}: ${String(e?.message || e)}`,
                });
            }
        }
        else {
            (0, state_1.post)({
                type: 'log',
                level: 'warn',
                text: '[auto-test] auto_test_complete event missing run_id or ckpt_path; not recorded.',
            });
        }
        return;
    }
    (0, state_1.post)(obj);
    try {
        switch (obj?.type) {
            case 'session_started': {
                state_1.extensionState.currentRunId = obj.run_id || state_1.extensionState.currentRunId;
                if (obj.session_id || obj.sessionId) {
                    state_1.extensionState.currentSessionId = String(obj.session_id || obj.sessionId);
                    (0, state_1.postCurrentSession)();
                }
                if (state_1.extensionState.needDiskCleanOnNextTrainer && trainerActive()) {
                    state_1.extensionState.needDiskCleanOnNextTrainer = false;
                    (async () => {
                        try {
                            const res = await sendReq('clean_disk', { scope: 'all' }, 60_000);
                            (0, state_1.post)({
                                type: 'log',
                                level: res?.ok ? 'info' : 'warn',
                                text: res?.ok
                                    ? '[startup] clean_disk: removed old runs from save_dir'
                                    : `[startup] clean_disk reported an issue: ${res?.error ?? 'unknown error'}`,
                            });
                        }
                        catch (e) {
                            (0, state_1.post)({
                                type: 'log',
                                level: 'warn',
                                text: `[startup] clean_disk failed (will continue anyway): ${e?.message || String(e)}`,
                            });
                        }
                    })();
                }
                break;
            }
            case 'checkpoint_saved': {
                const ckptId = `${obj.run_id}:${obj.step}:${Date.now()}`;
                (0, storage_1.insertCheckpoint)(ckptId, obj.run_id, Number(obj.step) || 0, obj.path || '');
                break;
            }
            case 'epoch_end': {
                if (Number.isFinite(obj.val_loss)) {
                    (0, storage_1.upsertSummary)(obj.run_id, null, obj.val_loss);
                }
                break;
            }
            default:
                break;
        }
    }
    catch (e) {
        console.warn('[onthefly] sqlite persist error:', e);
    }
}
//# sourceMappingURL=ipc.js.map