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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const child_process_1 = require("child_process");
const crypto = __importStar(require("crypto"));
const storage_1 = require("./storage");
const os = __importStar(require("os"));
// Stash latest config (webview can set it before or after starting the run)
let autoForkConfig = {
    rules: {
        enabled: true,
        loss_plateau_patience: 200,
        loss_plateau_delta: 1e-4,
        per_sample_window: 5000,
        kmeans_k: 5,
        dead_cluster_z: 1.0,
        high_loss_quantile: 0.85,
        spike_sigma: 3.0,
        ema_decay: 0.98,
        max_parallel_children: 2,
        fork_cooldown_steps: 1000,
        gate_epochs: 30,
        warmup_steps_before_first_fork: 200, // not implemented yet
        base_cooldown_steps: 200, // not implemented yet
    },
    sampling: {
        psl_every: 200,
        psl_budget: 4000,
        mirror_train: true,
        amp_for_psl: true,
        compute_margins: true,
        compute_embeddings: false,
        embed_max_dim: 256,
    },
    runtime: {
        auto_execute: false,
        variant_policy: 'first',
        variant_index: 0,
        name_template: '{parent}-auto@{step}',
        min_train_steps_between_autoforks: 200
    }
};
const pending = new Map();
function sendReq(cmd, payload = {}, timeoutMs = 15000) {
    if (!proc || !proc.stdin.writable) {
        return Promise.reject(new Error('No python process running. Press play first.'));
    }
    const id = crypto.randomUUID();
    const line = JSON.stringify({ id, cmd, payload }) + '\n';
    proc.stdin.write(line, 'utf8');
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
            pending.delete(id);
            reject(new Error(`timeout waiting for ${cmd}`));
        }, timeoutMs);
        pending.set(id, { resolve, reject, timer });
    });
}
function isRunImportedForThisSession(runId) {
    if (nativeRunsThisSession.has(runId))
        return false;
    if (!currentSessionId)
        return true; // no session yet → treat as imported
    const seen = ((0, storage_1.runsForSession)(currentSessionId) || []).some(id => String(id) === runId);
    return !seen;
}
const optimisticEcho = {
    pause: 'paused',
    resume: 'resumed',
    rewind_steps: 'rewound', // updated
    save_ckpt: 'checkpointSaved',
    merge: 'merged',
    set_mode: 'mode_set',
};
/* ============================ Globals ============================ */
let panel = null;
let proc = null;
const CHILD_KILL_TIMEOUT_MS = 1500;
let lastMode = 'single'; ///////////////
let pythonConfirmed = false;
let scriptConfirmed = false;
let currentRunId = null;
const seenRuns = new Set();
let currentSessionId = null; // sticky session id for log stamping
const nativeRunsThisSession = new Set(); // runs touched by THIS python session
let modelNavSelectedRunId = null;
/* ============================ Activate / Deactivate ============================ */
function activate(context) {
    context.subscriptions.push(vscode.commands.registerCommand('onthefly.showDashboard', () => openPanel(context)));
    (0, storage_1.initStorage)(context);
}
function deactivate() {
    try {
        killProc();
    }
    catch { }
    try {
        (0, storage_1.closeStorage)();
    }
    catch { }
}
/* ============================ Model Comparison Compact Text-Only Column ============================ */
function fmt(n, p = 6) {
    const x = Number(n);
    return Number.isFinite(x) ? x.toFixed(p).replace(/\.?0+$/, '') : '—';
}
function buildSummaryText(key, view) {
    // Heuristic: if this key matches any run_id we know, treat as run; otherwise treat as session_id.
    const allRuns = (0, storage_1.listRuns)();
    const isRun = allRuns.some(r => String(r.run_id) === key);
    const phase = (view === 'test' ? 'test' : (view === 'info' ? 'info' : 'train'));
    if (isRun) {
        // --- existing per-run behavior (unchanged) ---
        const rows = (0, storage_1.getRunRows)(key);
        const tests = (0, storage_1.getTestRows)(key);
        const logs = (0, storage_1.getLogs)(key, phase);
        const steps = rows.map(r => r.step);
        const train = rows.map(r => Number(r.loss)).filter(Number.isFinite);
        const vals = rows.map(r => Number(r.val_loss)).filter(Number.isFinite);
        const tsts = tests.map(r => Number(r.loss)).filter(Number.isFinite);
        const lastStep = steps.length ? steps[steps.length - 1] : null;
        const bestTrain = train.length ? Math.min(...train) : null;
        const bestVal = vals.length ? Math.min(...vals) : null;
        const bestTest = tsts.length ? Math.min(...tsts) : null;
        const recent = logs.slice(-30).map(l => {
            const tag = l.level ? l.level.toUpperCase() : 'LOG';
            const step = Number.isFinite(l.step) ? ` s=${l.step}` : '';
            const ep = Number.isFinite(l.epoch) ? ` e=${l.epoch}` : '';
            return `[${tag}]${step}${ep}  ${String(l.text || '').trim()}`;
        });
        const header = `Run: ${key}
      Last step: ${lastStep ?? '—'}
      Best train: ${fmt(bestTrain)}
      Best val:   ${fmt(bestVal)}
      Best test:  ${fmt(bestTest)}
      View: ${phase.toUpperCase()}
      ────────────────────────────────`;
        return [header, ...recent].join('\n');
    }
    // ---  session aggregation ---
    const runIds = (0, storage_1.runsForSession)(key);
    // aggregate series
    const rowsAll = runIds.flatMap(id => (0, storage_1.getRunRows)(id));
    const testsAll = runIds.flatMap(id => (0, storage_1.getTestRows)(id));
    const logsAll = (0, storage_1.getLogsBySession)(key, phase);
    const steps = rowsAll.map(r => r.step).sort((a, b) => a - b);
    const train = rowsAll.map(r => Number(r.loss)).filter(Number.isFinite);
    const vals = rowsAll.map(r => Number(r.val_loss)).filter(Number.isFinite);
    const tsts = testsAll.map(r => Number(r.loss)).filter(Number.isFinite);
    const lastStep = steps.length ? steps[steps.length - 1] : null;
    const bestTrain = train.length ? Math.min(...train) : null;
    const bestVal = vals.length ? Math.min(...vals) : null;
    const bestTest = tsts.length ? Math.min(...tsts) : null;
    const recent = logsAll.slice(-30).map(l => {
        const tag = l.level ? l.level.toUpperCase() : 'LOG';
        const step = Number.isFinite(l.step) ? ` s=${l.step}` : '';
        const ep = Number.isFinite(l.epoch) ? ` e=${l.epoch}` : '';
        return `[${tag}]${step}${ep}  ${String(l.text || '').trim()}`;
    });
    const header = `Session: ${key}
    Runs in session: ${runIds.length}
    Last step: ${lastStep ?? '—'}
    Best train: ${fmt(bestTrain)}
    Best val:   ${fmt(bestVal)}
    Best test:  ${fmt(bestTest)}
    View: ${phase.toUpperCase()}
    ────────────────────────────────`;
    return [header, ...recent].join('\n');
}
/* ============================ Panel & HTML ============================ */
function openPanel(context) {
    if (panel) {
        panel.reveal(vscode.ViewColumn.Active);
        return;
    }
    const nonce = getNonce();
    const roots = [
        vscode.Uri.file(context.extensionPath),
        vscode.Uri.file(path.join(context.extensionPath, 'src', 'webview')),
        vscode.Uri.file(path.join(context.extensionPath, 'media')),
        vscode.Uri.file(path.join(context.extensionPath, 'node_modules')),
    ];
    let webviewVisible = false;
    panel = vscode.window.createWebviewPanel('ontheflyDashboard', 'On the Fly Dash', vscode.ViewColumn.Active, {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: roots,
    });
    webviewVisible = true;
    panel.onDidChangeViewState(({ webviewPanel }) => {
        webviewVisible = webviewPanel.visible;
        if (webviewVisible) {
            // when user returns, cheaply re-sync UI (no heavy streams)
            try {
                post({ type: 'runs', rows: (0, storage_1.listRuns)() });
            }
            catch { }
            postStatus(!!proc);
        }
    });
    panel.onDidDispose(async () => {
        // Close = explicit end of session → kill everything & reset
        try {
            killProc();
        }
        catch { }
        try {
            for (const [, p] of pending) {
                clearTimeout(p.timer);
                p.reject(new Error('Panel disposed'));
            }
            pending.clear();
        }
        catch { }
        // fresh ephemeral storage
        try {
            (0, storage_1.closeStorage)();
        }
        catch { }
        (0, storage_1.initStorage)(context);
        pythonConfirmed = false;
        scriptConfirmed = false;
        panel = null;
    });
    panel.webview.onDidReceiveMessage((m) => { onMessage(context, m); });
    panel.webview.html = getHtml(context, panel.webview, nonce);
    setTimeout(() => panel?.webview.postMessage({ type: 'log', text: 'Webview ready.' }), 50);
}
function getHtml(context, webview, nonce) {
    const htmlPath = [
        path.join(context.extensionPath, 'dashboard.html'),
        path.join(context.extensionPath, 'src', 'webview', 'dashboard.html'),
    ].find(fs.existsSync);
    if (!htmlPath)
        return `<!doctype html><html><body><h2>dashboard.html not found</h2></body></html>`;
    let html = fs.readFileSync(htmlPath, 'utf8');
    const jsPath = [
        path.join(context.extensionPath, 'dashboard.js'),
        path.join(context.extensionPath, 'src', 'webview', 'dashboard.js'),
    ].find(fs.existsSync);
    const dagLayoutPath = [
        path.join(context.extensionPath, 'dag_layout.js'),
        path.join(context.extensionPath, 'src', 'webview', 'dag_layout.js'),
    ].find(fs.existsSync);
    const chartPathCandidates = [
        path.join(context.extensionPath, 'media', 'chart.min.js'), // if I ever want a copy
        path.join(context.extensionPath, 'node_modules', 'chart.js', 'dist', 'chart.min.js'),
        path.join(context.extensionPath, 'node_modules', 'chart.js', 'dist', 'chart.js'),
    ];
    const chartPath = chartPathCandidates.find(fs.existsSync);
    const jsUri = jsPath ? webview.asWebviewUri(vscode.Uri.file(jsPath)).toString() : '';
    const dagLayoutUri = dagLayoutPath ? webview.asWebviewUri(vscode.Uri.file(dagLayoutPath)).toString() : '';
    const chartUri = chartPath ? webview.asWebviewUri(vscode.Uri.file(chartPath)).toString() : '';
    const flyPath = path.join(context.extensionPath, 'src', 'webview', 'images', 'fly.png');
    const flyUri = webview.asWebviewUri(vscode.Uri.file(flyPath)).toString();
    html = html
        .replace(/__NONCE__/g, nonce)
        .replace(/__DASHBOARD_JS__/g, jsUri || '')
        .replace(/__CHART_JS__/g, chartUri || '')
        .replace(/__DAG_LAYOUT_JS__/g, dagLayoutUri || '')
        .replace(/__FLY__/g, flyUri || '');
    ;
    if (!/__DASHBOARD_JS__/.test(html) && jsUri) {
        html = html.replace(/<script\s+[^>]*src=["']\.\/dashboard\.(ts|js)["'][^>]*><\/script>/, `<script nonce="${nonce}" src="${jsUri}"></script>`);
    }
    const cspMeta = `
    <meta http-equiv="Content-Security-Policy"
      content="
        default-src 'none';
        img-src ${webview.cspSource} https: data:;
        font-src ${webview.cspSource} https:;
        style-src ${webview.cspSource} 'unsafe-inline';
        script-src 'nonce-${nonce}' ${webview.cspSource};
        connect-src ${webview.cspSource} https:;
      ">
  `.trim();
    if (/<meta[^>]+Content-Security-Policy/i.test(html)) {
        html = html.replace(/<meta[^>]+Content-Security-Policy[^>]*>/i, cspMeta);
    }
    else {
        html = html.replace(/<head>/i, `<head>\n${cspMeta}\n`);
    }
    if (!chartUri)
        console.warn('[onthefly] Chart.js not found at media/chart.umd.js or node_modules/chart.js/dist/chart.umd.js');
    if (!jsUri)
        console.warn('[onthefly] dashboard.js not found. Buttons will not work.');
    if (!dagLayoutUri)
        console.warn('[onthefly] dag_layout.js not found. DAG will fall back or render linearly.');
    return html;
}
function getNonce() {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++)
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    return text;
}
/* ============================ Utilities ============================ */
function post(msg) {
    try {
        panel?.webview.postMessage(msg);
    }
    catch (e) {
        console.log('[EXT->WEB] post threw:', e);
    }
}
function postErr(e) { post({ type: 'error', text: String(e?.message || e) }); }
function postStatus(running) { post({ type: 'status', running, mode: lastMode }); }
function postCurrentSession() {
    if (currentSessionId)
        post({ type: 'fs.session.current', id: currentSessionId });
}
const LAST_EXPORT_DIR_KEY = 'onthefly.lastExportDir';
function ensurePng(name) {
    return name.toLowerCase().endsWith('.png') ? name : `${name}.png`;
}
function systemDownloadsDir() {
    const home = os.homedir();
    // Linux: respect XDG user dirs if present
    try {
        const cfg = path.join(home, '.config', 'user-dirs.dirs');
        if (fs.existsSync(cfg)) {
            const txt = fs.readFileSync(cfg, 'utf8');
            const m = txt.match(/XDG_DOWNLOAD_DIR="?(.+?)"?$/m);
            if (m && m[1]) {
                let p = m[1].replace('$HOME', home).replace(/^"|"$/g, '');
                if (fs.existsSync(p))
                    return p;
            }
        }
    }
    catch { }
    // Common default on macOS / Linux / Windows
    const candidates = [
        path.join(home, 'Downloads'),
        path.join(home, 'downloads'),
    ];
    for (const p of candidates)
        if (fs.existsSync(p))
            return p;
    // Fallback: user home
    return home;
}
function getInitialExportDir(context) {
    const remembered = context.globalState.get(LAST_EXPORT_DIR_KEY);
    if (remembered && fs.existsSync(remembered))
        return remembered;
    return systemDownloadsDir();
}
function timestampSlug() {
    return new Date().toISOString().replace(/[:.]/g, '-');
}
/* ============================ Message handling ============================ */
async function onMessage(context, m) {
    const ws = context.workspaceState;
    switch (m.command) {
        case 'chooseScript': {
            const picked = await vscode.window.showOpenDialog({
                canSelectMany: false,
                filters: { Python: ['py'] },
                title: 'Choose a Python training script',
            });
            if (!picked || !picked[0])
                return;
            const p = picked[0].fsPath;
            await ws.update("onthefly.scriptPath" /* Keys.ScriptPath */, p);
            scriptConfirmed = true;
            vscode.window.showInformationMessage(`Training script selected: ${path.basename(p)}`);
            post({ type: 'scriptChosen', file: p });
            break;
        }
        case 'setPython': {
            const chosen = m.path || 'python';
            await ws.update("onthefly.pythonPath" /* Keys.PythonPath */, chosen);
            pythonConfirmed = true;
            post({ type: 'log', text: `Python set to: ${chosen}` });
            vscode.window.showInformationMessage(`Python interpreter set to: ${chosen}`);
            break;
        }
        case 'setMode': {
            lastMode = m.mode;
            sendCtl({ cmd: 'set_mode', mode: m.mode });
            break;
        }
        case 'exportChart': {
            try {
                const defName = ensurePng(m.filename || 'chart.png');
                const initialDir = getInitialExportDir(context);
                const defaultUri = vscode.Uri.file(path.join(initialDir, defName));
                const picked = await vscode.window.showSaveDialog({
                    title: 'Export chart as PNG',
                    filters: { 'PNG Image': ['png'] },
                    defaultUri,
                });
                if (!picked)
                    break;
                const base64 = (m.dataUrl || '').split(',')[1];
                if (!base64)
                    throw new Error('Invalid PNG data.');
                const buf = Buffer.from(base64, 'base64');
                fs.writeFileSync(picked.fsPath, buf);
                await context.globalState.update(LAST_EXPORT_DIR_KEY, path.dirname(picked.fsPath));
                vscode.window.showInformationMessage(`Chart exported to: ${picked.fsPath}`);
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'exportSubset': {
            try {
                const runId = m.runId || currentRunId;
                if (!runId) {
                    vscode.window.showWarningMessage('No run selected.');
                    break;
                }
                const trace = m._trace || 'no-trace';
                const fromWebview = m.subset_indices;
                const fmtRaw = String(m.format || 'parquet').toLowerCase();
                const fmt = fmtRaw === 'csv' ? 'csv' : (fmtRaw === 'feather' ? 'feather' : 'parquet');
                const initialDir = getInitialExportDir(context);
                const defName = `subset_${runId}.${fmt === 'feather' ? 'feather' : fmt}`;
                const defaultUri = vscode.Uri.file(path.join(initialDir, defName));
                const label = fmt === 'csv' ? 'CSV' : fmt === 'feather' ? 'Feather' : 'Parquet';
                const ext = fmt === 'csv' ? 'csv' : fmt === 'feather' ? 'feather' : 'parquet';
                const filters = { [label]: [ext] };
                const picked = await vscode.window.showSaveDialog({ title: 'Export subset', defaultUri, filters });
                if (!picked)
                    break;
                await context.globalState.update(LAST_EXPORT_DIR_KEY, path.dirname(picked.fsPath));
                const payload = {
                    run_id: String(runId),
                    format: fmt,
                    out_path: picked.fsPath,
                    _trace: trace,
                };
                // --- Primary path: trust webview indices ---
                if (Array.isArray(fromWebview) && fromWebview.length > 0) {
                    const norm = Array.from(new Set(fromWebview.map(n => Number(n) | 0).filter(n => Number.isFinite(n) && n >= 0))).sort((a, b) => a - b);
                    if (norm.length > 0)
                        payload.subset_indices = norm;
                }
                // --- Fallbacks (optional) ---
                if (!payload.subset_indices) {
                    const region = m.region;
                    const stored = (0, storage_1.getRunSubset)(String(runId)) || [];
                    const haveStored = Array.isArray(stored) && stored.length > 0;
                    if (region && Number.isFinite(region.minLoss) && Number.isFinite(region.maxLoss)) {
                        try {
                            const indices = await getSelectedRegionIndices(String(runId), region.minLoss, region.maxLoss);
                            if (indices.length)
                                payload.subset_indices = indices;
                            else if (haveStored)
                                payload.subset_indices = stored.map((n) => Number(n) | 0);
                        }
                        catch {
                            if (haveStored)
                                payload.subset_indices = stored.map((n) => Number(n) | 0);
                        }
                    }
                    else if (haveStored) {
                        payload.subset_indices = stored.map((n) => Number(n) | 0);
                    }
                }
                const data = await sendReq('export_subset', payload, 10 * 60 * 1000);
                post({ type: 'subsetExported', run_id: String(runId), ...data });
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'modelNav.select': {
            const id = String(m.runId || '').trim();
            if (id) {
                modelNavSelectedRunId = id;
                post({ type: 'log', text: `[modelNav] selected: ${id}` });
            }
            break;
        }
        case 'requestLogs': {
            try {
                const rows = (0, storage_1.getLogs)(String(m.runId), m.phase);
                post({ type: 'logs', run_id: String(m.runId), rows });
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'requestTestRows': {
            try {
                const rows = (0, storage_1.getTestRows)(String(m.runId));
                post({ type: 'testRows', run_id: String(m.runId), rows });
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'pause':
            sendCtl({ cmd: 'pause' });
            break;
        case 'resume': {
            try {
                const rk = String(m.runId || '').trim();
                if (rk)
                    modelNavSelectedRunId = rk;
                if (proc) {
                    sendCtl({ cmd: 'resume' });
                }
                else {
                    post({ type: 'log', text: 'Starting training (via Resume)...' });
                    await startRun(context, lastMode);
                }
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'fork':
            sendCtl({ cmd: 'fork', payload: m.payload });
            break;
        case 'merge': {
            try {
                await sendReq('merge', m.payload, 120000);
                // Python will emit `newRun` → handled in handleLine().
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'generateReport': {
            try {
                const runId = m.runId ?? currentRunId;
                const reqId = m.reqId;
                if (!runId) {
                    post({ type: 'error', text: 'No active run selected for report.' });
                    break;
                }
                const subset = (0, storage_1.getRunSubset)(String(runId));
                const subset_on = 'train';
                const data = await sendReq('generate_report', {
                    owner_run_id: runId,
                    run_id: runId,
                    subset_indices: subset.length ? subset : undefined,
                    subset_on,
                    reqId
                });
                const losses = Array.isArray(data?.losses)
                    ? data.losses.map(Number).filter(Number.isFinite)
                    : [];
                let sample_indices = Array.isArray(data?.sample_indices)
                    ? data.sample_indices.map(v => Math.trunc(Number(v)))
                        .filter(v => Number.isFinite(v) && v >= 0)
                    : [];
                if (sample_indices.length !== losses.length) {
                    sample_indices = Array.from({ length: losses.length }, (_, i) => i);
                }
                const meta = data?.meta || {};
                const stepNum = Number(meta?.at_step);
                const epochNum = Number(meta?.at_epoch);
                const at_step = Number.isFinite(stepNum) ? stepNum : null;
                const at_epoch = Number.isFinite(epochNum) ? epochNum : null;
                const note = (typeof meta?.note === 'string' ? meta.note : '') || '';
                (0, storage_1.upsertReportLossDist)(String(runId), subset_on, {
                    losses,
                    sample_indices,
                    note,
                    at_step,
                    at_epoch,
                    samples: losses.length
                });
                post({
                    type: 'reportData',
                    run_id: String(runId),
                    owner_run_id: String(runId),
                    reqId,
                    losses,
                    sample_indices,
                    meta: { ...meta, at_step, at_epoch, subset_on, samples: losses.length, note }
                });
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'requestReport': {
            try {
                const runId = m.runId;
                if (!runId)
                    break;
                const row = (0, storage_1.getReportLossDist)(String(runId), 'train');
                if (row) {
                    post({
                        type: 'reportFromDb',
                        run_id: String(runId),
                        owner_run_id: String(runId),
                        losses: row.losses,
                        meta: { note: row.note, at_step: row.at_step, at_epoch: row.at_epoch, samples: row.samples, subset_on: row.subset_on }
                    });
                }
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'applyAutoForkRules': {
            try {
                autoForkConfig = m.config;
                if (proc) {
                    await sendReq('set_autofork_rules', autoForkConfig, 30000);
                    post({ type: 'log', text: '[AutoFork] rules applied to running session.' });
                }
                else {
                    post({ type: 'log', text: '[AutoFork] rules staged; will apply at next start.' });
                }
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'executeAutoForkPlan': {
            try {
                const plan = m.plan || {};
                const parent = String(m.runId || currentRunId || '');
                const vIdx = Math.max(0, Number(m.variantIndex ?? 0));
                const variants = Array.isArray(plan?.training_recipe?.variants)
                    ? plan.training_recipe.variants
                    : [];
                const chosen = variants[vIdx] || {};
                // provenance from the background worker
                const initFrom = typeof plan?.init_from === 'string' ? plan.init_from :
                    typeof plan?.initFrom === 'string' ? plan.initFrom : undefined;
                const payload = {
                    parent_run_id: parent,
                    owner_run_id: parent,
                    selection: plan.selection || undefined,
                    hparams: chosen || {},
                    run_name: plan.proposed_run_name || undefined,
                    //  execute the fork FROM the exact checkpoint the plan used
                    parent_ckpt_path: initFrom,
                };
                const data = await sendReq('fork', payload, 120000);
                if (data?.new_run && Array.isArray(data?.subset_indices)) {
                    try {
                        (0, storage_1.setRunSubset)(String(data.new_run), data.subset_indices.map((n) => Number(n) | 0));
                    }
                    catch (e) {
                        console.warn('[onthefly] persist subset failed:', e);
                    }
                }
                post({
                    type: 'log',
                    text: `[AutoFork] executed fork for plan "${plan.reason || 'plan'}"${initFrom ? ` (from ckpt: ${path.basename(initFrom)})` : ''}.`,
                });
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'manualFork': {
            try {
                await sendReq('pause', {}, 30000);
            }
            catch { /* ignore */ }
            try {
                const parent = String(m.runId || currentRunId || '');
                const data = await sendReq('fork', {
                    parent_run_id: parent,
                    owner_run_id: parent,
                    region: m.region,
                    hparams: m.hparams
                });
                if (data?.new_run && Array.isArray(data?.subset_indices)) {
                    try {
                        (0, storage_1.setRunSubset)(String(data.new_run), data.subset_indices.map((n) => Number(n) | 0));
                    }
                    catch (e) {
                        console.warn('[onthefly] failed to persist subset:', e);
                    }
                }
                sendCtl({ cmd: 'resume' });
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'notify': {
            const lvl = m.level || 'info';
            const type = lvl === 'error' ? 'error' : 'log';
            post({ type, text: m.text });
            break;
        }
        case 'exportSession': {
            try {
                // ask user where to put the bundle (folder is created, we copy into it)
                const initialDir = getInitialExportDir(context);
                const defaultName = `onthefly_bundle_${timestampSlug()}`;
                const picked = await vscode.window.showSaveDialog({
                    title: 'Export: choose a folder name & location',
                    defaultUri: vscode.Uri.file(path.join(initialDir, defaultName)),
                });
                if (!picked)
                    break;
                const bundleDir = picked.fsPath;
                await context.globalState.update(LAST_EXPORT_DIR_KEY, path.dirname(bundleDir));
                fs.mkdirSync(bundleDir, { recursive: true });
                // NOTE: "modelNav" → we don't keep a separate nav map here; the DB is source of truth.
                // Grab *every* run_id we know and treat that as the nav list.
                const allRuns = (0, storage_1.listRuns)().map(r => String(r.run_id));
                const owners = Array.from(new Set(allRuns)).filter(Boolean);
                let spillRoot = null;
                if (proc && proc.stdin.writable) {
                    spillRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'onthefly-spill-'));
                    // 1) try the new multi-owner prepare_export API (forces snapshots/ckpts for *all* runs)
                    let prep = null;
                    try {
                        // latest_only keeps the bundle trim
                        prep = await sendReq('prepare_export', { dir: spillRoot, latest_only: true, owners }, 10 * 60 * 1000);
                    }
                    catch (e) {
                        // old servers won't know prepare_export(owners=...), fall back to a generic spill
                        post({ type: 'log', text: '[export] prepare_export not available or failed; falling back to spill_all().' });
                        try {
                            const recs = await sendReq('spill_all', { dir: spillRoot, latest_only: true }, 10 * 60 * 1000);
                            prep = { ckpt: null, snapshots: recs };
                        }
                        catch (e2) {
                            console.warn('[export] spill_all RPC also failed; proceeding with whatever is already in DB:', e2);
                            prep = { ckpt: null, snapshots: [] };
                        }
                    }
                    // 2) write snapshot rows into DB so the bundler knows what to copy
                    try {
                        const snaps = Array.isArray(prep?.snapshots) ? prep.snapshots : [];
                        for (const r of snaps) {
                            if (!r || !r.ckpt_id || !r.owner || !r.path)
                                continue;
                            (0, storage_1.insertCheckpoint)(String(r.ckpt_id), String(r.owner), Number(r.step) | 0, String(r.path));
                        }
                    }
                    catch (e) {
                        console.warn('[export] failed to insert spilled snapshots into DB:', e);
                    }
                    // 3) if python returned a ring ckpt for the current run, stash it too (nice-to-have)
                    try {
                        const ck = prep?.ckpt;
                        if (ck?.path && ck?.run) {
                            const ckId = `${ck.run}:${ck.step}:ring`;
                            (0, storage_1.insertCheckpoint)(ckId, String(ck.run), Number(ck.step) | 0, String(ck.path));
                        }
                    }
                    catch (e) {
                        console.warn('[export] failed to insert ring checkpoint into DB:', e);
                    }
                    // 4) paranoia: ensure every owner has at least *one* checkpoint row pre-bundle.
                    //    if any run is missing, scan the save_dir heuristically and backfill.
                    try {
                        // best guess at the training script's dir → usually where save_dir lives or near it
                        const scriptPath = context.workspaceState.get("onthefly.scriptPath" /* Keys.ScriptPath */) || '';
                        const saveDirGuess = scriptPath ? path.dirname(scriptPath) : process.cwd();
                        const files = fs.existsSync(saveDirGuess) ? fs.readdirSync(saveDirGuess) : [];
                        for (const runId of owners) {
                            const have = (0, storage_1.latestCheckpointForRun)(runId);
                            if (have)
                                continue;
                            const patt = new RegExp(`__${runId}__step(\\d+)\\.pt$`);
                            const candidates = files
                                .filter(f => patt.test(f))
                                .map(f => ({ f, step: Number((f.match(patt) || [])[1] || 0) }))
                                .sort((a, b) => a.step - b.step);
                            if (candidates.length) {
                                const last = candidates[candidates.length - 1];
                                const abs = path.join(saveDirGuess, last.f);
                                (0, storage_1.insertCheckpoint)(`${runId}:${last.step}:scan`, runId, last.step, abs);
                            }
                        }
                    }
                    catch (e) {
                        console.warn('[export] backfill scan failed (non-fatal):', e);
                    }
                }
                else {
                    // not running → just bundle whatever DB already references
                    post({ type: 'log', text: '[export] Python not running; bundling existing DB checkpoints only.' });
                }
                // 5) build the portable bundle (copies DB + referenced ckpts into bundle)
                (0, storage_1.exportBundle)(bundleDir);
                // 6) cleanup tmp spill dir
                if (spillRoot) {
                    try {
                        fs.rmSync(spillRoot, { recursive: true, force: true });
                    }
                    catch { }
                }
                const choice = await vscode.window.showInformationMessage(`Export complete: ${bundleDir}`, 'Reveal in Finder/Explorer');
                if (choice)
                    vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(bundleDir));
                post({ type: 'sessionSaved', path: bundleDir });
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'loadSession': {
            try {
                // Let the user pick either a bundle FOLDER or bundle.json FILE.
                const initialDir = getInitialExportDir(context);
                const picked = await vscode.window.showOpenDialog({
                    title: 'Load bundle',
                    canSelectMany: false,
                    canSelectFiles: true, // allow picking bundle.json directly
                    canSelectFolders: true, // or the bundle directory
                    defaultUri: vscode.Uri.file(initialDir),
                    // No sqlite/db filters — bundles only
                    openLabel: 'Load',
                });
                if (!picked || !picked[0])
                    break;
                const chosen = picked[0].fsPath;
                const isDir = fs.existsSync(chosen) && fs.statSync(chosen).isDirectory();
                await context.globalState.update(LAST_EXPORT_DIR_KEY, isDir ? chosen : path.dirname(chosen));
                let bundleDir = null;
                if (isDir) {
                    // Expect bundle.json inside the directory
                    const manifestPath = path.join(chosen, 'bundle.json');
                    if (!fs.existsSync(manifestPath)) {
                        vscode.window.showErrorMessage(`No bundle.json found in folder:\n${chosen}`);
                        break;
                    }
                    bundleDir = chosen;
                }
                else {
                    // If a file was picked, it must be bundle.json
                    const base = path.basename(chosen).toLowerCase();
                    if (base !== 'bundle.json') {
                        vscode.window.showErrorMessage('Unsupported selection. Pick a bundle folder (with bundle.json) or bundle.json itself.');
                        break;
                    }
                    bundleDir = path.dirname(chosen);
                }
                // Load bundle: copies sqlite into live db folder and rewrites ckpt paths
                (0, storage_1.loadBundle)(bundleDir, context);
                post({ type: 'sessionLoaded' });
                post({ type: 'runs', rows: (0, storage_1.listRuns)() });
                const choice = await vscode.window.showInformationMessage(`Bundle loaded from: ${bundleDir}`, 'Reveal in Finder/Explorer');
                if (choice)
                    vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(bundleDir));
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'exportSubset': {
            try {
                const runId = m.runId || currentRunId;
                if (!runId) {
                    vscode.window.showWarningMessage('No run selected.');
                    break;
                }
                const fmtRaw = String(m.format || 'parquet').toLowerCase();
                const fmt = fmtRaw === 'csv' ? 'csv' : (fmtRaw === 'feather' ? 'feather' : 'parquet');
                const initialDir = getInitialExportDir(context);
                const defName = `subset_${runId}.${fmt === 'feather' ? 'feather' : fmt}`;
                const defaultUri = vscode.Uri.file(path.join(initialDir, defName));
                const label = fmt === 'csv' ? 'CSV' : fmt === 'feather' ? 'Feather' : 'Parquet';
                const ext = fmt === 'csv' ? 'csv' : fmt === 'feather' ? 'feather' : 'parquet';
                const filters = { [label]: [ext] };
                const picked = await vscode.window.showSaveDialog({ title: 'Export subset', defaultUri, filters });
                if (!picked)
                    break;
                await context.globalState.update(LAST_EXPORT_DIR_KEY, path.dirname(picked.fsPath));
                const incoming = m.subset_indices;
                let subset_indices = Array.isArray(incoming)
                    ? incoming.map(v => Math.trunc(Number(v))).filter(v => Number.isFinite(v) && v >= 0)
                    : undefined;
                if (!subset_indices || subset_indices.length === 0) {
                    const stored = (0, storage_1.getRunSubset)(String(runId)) || [];
                    if (Array.isArray(stored) && stored.length > 0) {
                        subset_indices = stored.map((n) => Number(n) | 0).filter(n => Number.isFinite(n) && n >= 0);
                    }
                }
                const payload = {
                    run_id: String(runId),
                    format: fmt,
                    out_path: picked.fsPath,
                };
                if (subset_indices && subset_indices.length)
                    payload.subset_indices = subset_indices;
                const data = await sendReq('export_subset', payload, 10 * 60 * 1000);
                const outPath = String(data?.out_path || picked.fsPath);
                const rows = Number(data?.rows || 0);
                const effFmt = String(data?.format || fmt).toUpperCase();
                const choice = await vscode.window.showInformationMessage(`Subset exported (${rows} rows, ${effFmt}): ${outPath}`, 'Reveal in Finder/Explorer');
                if (choice)
                    vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(outPath));
                post({ type: 'subsetExported', run_id: String(runId), ...data });
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'resetAll': {
            // (We already show a confirm dialog in the webview, but this native modal is a safe second guard.)
            const pick = await vscode.window.showWarningMessage('This will erase all models from memory, are you sure you want to refresh?', { modal: true, detail: 'Any running training will be stopped. Nothing will be saved if you have not exported session.' }, 'Erase & Refresh', 'Cancel');
            if (pick !== 'Erase & Refresh') {
                post({ type: 'log', text: 'Refresh cancelled.' });
                break;
            }
            try {
                // 1) Stop Python if running
                try {
                    killProc();
                }
                catch { }
                // 2) Reject any outstanding RPCs cleanly
                try {
                    for (const [, p] of pending) {
                        clearTimeout(p.timer);
                        // make the rejection explicit so callers don’t hang
                        p.reject(new Error('Reset requested'));
                    }
                    pending.clear();
                }
                catch { }
                // 3) Clear in-memory extension state
                currentRunId = null;
                seenRuns.clear();
                // 4) Reset storage: close and re-init a fresh DB
                try {
                    (0, storage_1.closeStorage)();
                }
                catch { }
                (0, storage_1.initStorage)(context);
                // 5) Tell the webview we’re clean
                post({ type: 'resetOk' }); // “done” signal for UI log
                post({ type: 'runs', rows: [] }); // empty list = nothing to select
                postStatus(false); // not running
                vscode.window.setStatusBarMessage('ForkSmith: session reset', 2000);
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'requestRuns': {
            try {
                const rows = (0, storage_1.listRuns)();
                post({ type: 'runs', rows });
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'requestRows': {
            try {
                post({ type: 'rows', rows: (0, storage_1.getRunRows)(m.runId) });
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
    }
}
/* ============================ Python process (training) ============================ */
async function startRun(context, mode) {
    if (proc) {
        vscode.window.showWarningMessage('A run is already active.');
        return;
    }
    // Require explicit confirmation in *this* panel session
    if (!pythonConfirmed) {
        vscode.window.showErrorMessage('Set a Python interpreter first (gear icon).');
        return;
    }
    if (!scriptConfirmed) {
        vscode.window.showErrorMessage('Choose a Python training script first.');
        return;
    }
    const ws = context.workspaceState;
    const python = ws.get("onthefly.pythonPath" /* Keys.PythonPath */)?.trim();
    const script = ws.get("onthefly.scriptPath" /* Keys.ScriptPath */)?.trim();
    if (!python) {
        vscode.window.showErrorMessage('Set a Python interpreter first (gear icon).');
        return;
    }
    if (!script) {
        vscode.window.showErrorMessage('Choose a Python training script first.');
        return;
    }
    if (!fs.existsSync(script)) {
        vscode.window.showErrorMessage(`Training script not found: ${script}`);
        post({ type: 'error', text: `Script not found: ${script}` });
        return;
    }
    const args = ['-u', script, '--seamless'];
    post({ type: 'log', text: `Spawning: ${python} ${args.join(' ')}` });
    const resumeRunId = (modelNavSelectedRunId && modelNavSelectedRunId.trim()) ||
        (currentRunId && currentRunId.trim()) ||
        null;
    const imported = resumeRunId ? isRunImportedForThisSession(resumeRunId) : false;
    const resume = imported && resumeRunId ? (0, storage_1.latestCheckpointForRun)(resumeRunId) : null;
    console.log('[onthefly] starting run', { mode, resumeRunId, imported });
    console.log('[onthefly] resume lookup', { resumeRunId, resume });
    const envBlock = {
        ...process.env,
        ONTHEFLY_MODE: mode,
        ...(imported && resumeRunId ? { ONTHEFLY_RESUME_RUN_ID: resumeRunId } : {}),
        ...(imported && resume && resume.path ? {
            ONTHEFLY_INIT_CKPT: resume.path,
            ONTHEFLY_INIT_STEP: String(resume.step ?? 0),
        } : {}),
    };
    console.log('[onthefly] env resume block', {
        ONTHEFLY_RESUME_RUN_ID: envBlock.ONTHEFLY_RESUME_RUN_ID,
        ONTHEFLY_INIT_CKPT: envBlock.ONTHEFLY_INIT_CKPT,
        ONTHEFLY_INIT_STEP: envBlock.ONTHEFLY_INIT_STEP,
    });
    try {
        proc = (0, child_process_1.spawn)(python, args, {
            cwd: path.dirname(script),
            env: envBlock,
            stdio: ['pipe', 'pipe', 'pipe'],
            //  make it a group leader so we can kill the entire tree on POSIX
            detached: process.platform !== 'win32',
        });
    }
    catch (e) {
        const msg = `Error starting training: ${e?.message || e}`;
        vscode.window.showErrorMessage(msg);
        post({ type: 'error', text: msg });
        return;
    }
    proc.on('error', (err) => {
        const msg = `Error starting training: ${String(err?.message || err)}`;
        vscode.window.showErrorMessage(msg);
        post({ type: 'error', text: msg });
        postStatus(false);
        proc = null;
    });
    proc.stdout.setEncoding('utf8');
    proc.stderr.setEncoding('utf8');
    let sawStdout = false;
    const markRunningOnce = () => {
        if (!sawStdout) {
            sawStdout = true;
            postStatus(true);
            post({ type: 'log', text: 'Training process started.' });
        }
    };
    let buffer = '';
    proc.stdout.on('data', (chunk) => {
        markRunningOnce();
        buffer += chunk;
        let idx;
        while ((idx = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 1);
            if (line.trim()) {
                handleLine(line);
            }
        }
    });
    proc.stderr.on('data', (chunk) => {
        const t = String(chunk);
        post({ type: 'error', text: t });
        markRunningOnce();
    });
    proc.on('close', (code) => {
        if (code === 0)
            post({ type: 'log', text: `Process exited cleanly.` });
        else
            post({ type: 'error', text: `Process exited with code ${code}` });
        post({ type: 'trainingFinished' });
        postStatus(false);
        proc = null;
        for (const [, p] of pending) {
            clearTimeout(p.timer);
            p.reject(new Error('python exited'));
        }
        pending.clear();
    });
}
function sendCtl(cmd) {
    if (!proc || !proc.stdin.writable) {
        post({ type: 'error', text: 'No active run. Press Play first.' });
        return;
    }
    try {
        proc.stdin.write(JSON.stringify(cmd) + '\n');
        const t = cmd.cmd;
        if (t && optimisticEcho[t])
            post({ type: optimisticEcho[t], payload: cmd });
    }
    catch (e) {
        postErr(e);
    }
}
function killProc() {
    const p = proc; // snapshot + narrow (TS knows p !== null after guard)
    if (!p)
        return;
    try {
        // Detach listeners and destroy streams to break retention chains
        try {
            p.stdout?.removeAllListeners?.();
        }
        catch { }
        try {
            p.stderr?.removeAllListeners?.();
        }
        catch { }
        try {
            p.removeAllListeners?.();
        }
        catch { }
        try {
            p.stdin?.destroy?.();
        }
        catch { }
        try {
            p.stdout?.destroy?.();
        }
        catch { }
        try {
            p.stderr?.destroy?.();
        }
        catch { }
        const pid = typeof p.pid === 'number' ? p.pid : undefined;
        if (process.platform === 'win32') {
            // Kill entire tree on Windows
            try {
                (0, child_process_1.spawn)('taskkill', ['/pid', String(p.pid), '/f', '/t']);
            }
            catch { }
        }
        else {
            // POSIX: if we spawned with detached:true, child is a process-group leader.
            // Kill the whole group first, then the process as a fallback.
            if (pid && pid > 0) {
                try {
                    process.kill(-pid, 'SIGTERM');
                }
                catch { } // group kill
                try {
                    process.kill(pid, 'SIGTERM');
                }
                catch { } // parent as fallback
                setTimeout(() => {
                    try {
                        process.kill(-pid, 'SIGKILL');
                    }
                    catch { }
                    try {
                        process.kill(pid, 'SIGKILL');
                    }
                    catch { }
                }, CHILD_KILL_TIMEOUT_MS);
            }
            else {
                // No pid? Fall back to per-process signals
                try {
                    p.kill('SIGTERM');
                }
                catch { }
                setTimeout(() => { try {
                    p.kill('SIGKILL');
                }
                catch { } ; }, CHILD_KILL_TIMEOUT_MS);
            }
        }
    }
    finally {
        proc = null; // release reference for GC
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
/* ============================ Line handler ============================ */
function handleLine(line) {
    let obj = null;
    try {
        obj = JSON.parse(line);
    }
    catch {
        post({ type: 'log', text: line });
        return;
    }
    if (obj && obj.event && !obj.type) {
        obj.type = obj.event;
        delete obj.event;
    }
    // ==== RPC responses (leave as-is) ====
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
        const run_id = obj.run_id || currentRunId || 'live';
        const step = Number(obj.step) || 0;
        const loss = Number(obj.loss);
        post({ type: 'testStep', run_id, step, loss: Number.isFinite(loss) ? loss : null, ts: Date.now() });
        try {
            (0, storage_1.insertTestMetric)(run_id, step, Number.isFinite(loss) ? loss : null);
        }
        catch { }
        return;
    }
    if (obj?.type === 'log') {
        const run_id = obj.run_id || currentRunId || 'live';
        // Prefer explicit, then fallback to sticky session
        const session_id = obj.session_id || obj.sessionId || currentSessionId || null;
        if (run_id && session_id && currentSessionId && session_id === currentSessionId) {
            nativeRunsThisSession.add(String(run_id));
        }
        const level = obj.level || 'info';
        const text = String(obj.text || '');
        // 1) honor explicit phase if provided by backend
        const pRaw = (obj.phase && String(obj.phase).toLowerCase()) || null;
        let phase = (pRaw === 'train' || pRaw === 'test' || pRaw === 'info') ? pRaw : 'info';
        // parse a few common patterns to enrich rows (prefer TEST first)
        let step = null;
        let epoch = null;
        const mEpoch = text.match(/^\s*epoch\s+(\d+)\s*$/i);
        if (mEpoch && phase === 'info') {
            epoch = Number(mEpoch[1]);
            phase = 'train';
        }
        // ---- TEST first (looser match) ----
        const mTest = text.match(/step\s+(\d+)\s*:\s*test[_\s]*loss\s*=\s*([0-9.eE+\-]+)/i);
        if (mTest && phase === 'info') {
            step = Number(mTest[1]);
            phase = 'test';
        }
        if (/\btesting\b/i.test(text) && phase === 'info')
            phase = 'test';
        if (/\b(?:eval|evaluation)\b/i.test(text) && /\b(loss|acc|metric)\b/i.test(text) && phase === 'info')
            phase = 'test';
        // ---- TRAIN (slightly looser too) ----
        const mTrain = text.match(/step\s+(\d+)\s*:\s*train[_\s]*loss\s*=\s*([0-9.eE+\-]+).*?val[_\s]*loss\s*=\s*([0-9.eE+\-]+|None)/i);
        if (mTrain && phase === 'info') {
            step = Number(mTrain[1]);
            phase = 'train';
        }
        if ((/\btrain[_\s]*loss\b/i.test(text) || /\bval[_\s]*loss\b/i.test(text)) && phase === 'info') {
            phase = 'train';
        }
        const tsMs = (Number(obj.ts) > 0 ? Math.round(Number(obj.ts) * 1000) : Date.now());
        // If the raw text already includes "step N", don't also render the UI "s: N" badge.
        const hasStepInText = /\bstep\s*[:#]?\s*\d+/i.test(text);
        const stepForUI = hasStepInText ? null : step;
        try {
            // keep full fidelity in storage
            (0, storage_1.insertLog)({ run_id, session_id, level, text, phase, step, epoch, ts: tsMs });
        }
        catch { }
        // suppress redundant "s:" in the webview
        post({ type: 'log', run_id, session_id, level, text, phase, step: stepForUI, epoch, ts: tsMs });
        return;
    }
    // swallow unsolicited reportData (RPC handles reply)
    if (obj?.type === 'reportData') {
        console.log('[onthefly] swallowed unsolicited reportData from python');
        return;
    }
    if (obj?.type === 'merge_gating') {
        post({
            type: 'merge_gating',
            reason: obj.reason || 'unknown',
            parents: Array.isArray(obj.parents) ? obj.parents.map(String) : [],
            step: Number(obj.step) || null,
            run_id: obj.run_id || currentRunId || null,
            // pass through any other fields (e.g., have_parent_ckpt, have_child_ckpt, child_id, etc.)
            ...obj
        });
        return;
    }
    // ==== streaming steps ====
    if (obj && obj.type === 'trainStep') {
        const run_id = obj.run_id || currentRunId || 'live';
        const row = {
            run_id,
            step: Number(obj.step) || 0,
            loss: Number.isFinite(obj.loss) ? obj.loss : null,
            val_loss: Number.isFinite(obj.val_loss) ? obj.val_loss : null,
        };
        post({ type: 'trainStep', ...row, ts: Date.now() });
        try {
            (0, storage_1.insertMetric)(run_id, row);
        }
        catch (e) {
            console.warn('[onthefly] sqlite persist error for step:', e);
        }
        return;
    }
    // ==== canonical run creation ====
    if (obj && obj.type === 'newRun') {
        const id = String(obj.run_id || '').trim();
        if (id)
            nativeRunsThisSession.add(id);
        if (!id)
            return;
        // If backend includes a session id on newRun, remember it
        if (obj.session_id || obj.sessionId) {
            currentSessionId = String(obj.session_id || obj.sessionId);
            postCurrentSession();
        }
        const parents = normalizeParents(obj);
        const project = obj.project || 'default';
        const name = obj.run_name || id;
        if (!seenRuns.has(id)) {
            seenRuns.add(id);
            currentRunId = id;
            // If storage only supports single parent, store the first (primary)
            const primaryParent = (parents && parents[0]) ?? null;
            (0, storage_1.insertRun)(id, project, name, lastMode || 'single', primaryParent);
        }
        else {
            currentRunId = id;
        }
        // include meta so webview can gate auto/manual UI
        post({
            type: 'newRun',
            run_id: id,
            parents,
            meta: obj.meta,
            session_id: currentSessionId || null,
        });
        post({ type: 'runs', rows: (0, storage_1.listRuns)() });
        return;
    }
    // ==== map event names to what dashboard.js expects ====
    if (obj?.type === 'auto_fork_suggested') {
        const plan = obj.plan || obj || {};
        post({
            type: 'autoForkSuggested',
            step: Number(obj.step) || null,
            run_id: obj.run_id || currentRunId || null,
            plan: {
                ...plan,
                at_step: plan.at_step ?? obj.at_step ?? null,
                init_from: plan.init_from ?? obj.init_from ?? null,
            },
        });
        return;
    }
    if (obj?.type === 'auto_fork_executed') {
        const plan = obj.plan || {};
        post({
            type: 'autoForkExecuted',
            step: Number(obj.step) || null,
            run_id: obj.run_id || currentRunId || null,
            plan: {
                ...plan,
                at_step: plan.at_step ?? obj.at_step ?? null,
                init_from: plan.init_from ?? obj.init_from ?? null,
            },
            child_run: obj.child_run || null,
            variant_index: Number.isFinite(obj.variant_index) ? Number(obj.variant_index) : null,
        });
        return;
    }
    if (obj?.type === 'auto_merge_suggested') {
        const plan = obj.plan || {};
        post({
            type: 'autoMergeSuggested',
            step: Number(obj.step) || null,
            run_id: obj.run_id || currentRunId || null,
            // pass through diagnostics so UI can render context/tooltips
            plan,
        });
        return;
    }
    if (obj?.type === 'auto_merge_executed') {
        post({
            type: 'autoMergeExecuted',
            step: Number(obj.step) || null,
            run_id: obj.run_id || currentRunId || null,
            new_run: obj.new_run || null,
            merged: Array.isArray(obj.merged) ? obj.merged.map(String) : [],
        });
        return;
    }
    if (obj?.type === 'autofork_rules_set') {
        autoForkConfig = obj.config;
        post({ type: 'autoforkRulesSet', config: obj.config });
        return;
    }
    if (obj?.type === 'training_finished') {
        post({ type: 'trainingFinished', code: obj.code });
        postStatus(false);
        return;
    }
    // Default: forward to webview, then extra persistence for other events
    post(obj);
    try {
        switch (obj?.type) {
            case 'session_started': {
                currentRunId = obj.run_id || currentRunId;
                // Remember active session for later log writes
                if (obj.session_id || obj.sessionId) {
                    currentSessionId = String(obj.session_id || obj.sessionId);
                    postCurrentSession();
                }
                // Apply staged AutoFork rules as soon as the session is live
                if (autoForkConfig) {
                    sendReq('set_autofork_rules', normalizeRulesForServer(autoForkConfig), 30000)
                        .catch(e => post({ type: 'error', text: `[AutoFork] failed to apply rules: ${String(e?.message || e)}` }));
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
/* ============================ Data explorer helper ============================ */
async function runDataExplorer(context, subArgs) {
    const ws = context.workspaceState;
    const python = ws.get("onthefly.pythonPath" /* Keys.PythonPath */) || 'python';
    return new Promise((resolve, reject) => {
        const p = (0, child_process_1.spawn)(python, ['-m', 'onthefly_backend.data_explorer', ...subArgs], { env: process.env });
        let out = '';
        let err = '';
        p.stdout.on('data', (d) => (out += d.toString()));
        p.stderr.on('data', (d) => (err += d.toString()));
        p.on('close', (code) => {
            if (code === 0) {
                try {
                    resolve(JSON.parse(out || 'null'));
                }
                catch (e) {
                    reject(new Error(`Failed to parse JSON from data explorer.\n${String(e)}\nSTDOUT:\n${out}\nSTDERR:\n${err}`));
                }
            }
            else {
                reject(new Error(`data_explorer exited with code ${code}\nSTDERR:\n${err}`));
            }
        });
    });
}
async function getSelectedRegionIndices(runId, minLoss, maxLoss) {
    const data = await sendReq('get_selected_region_indices', {
        run_id: String(runId),
        min_loss: Number(minLoss),
        max_loss: Number(maxLoss),
    }, 60_000);
    const arr = Array.isArray(data?.indices) ? data.indices : [];
    // normalize to ints
    return arr
        .map((n) => Number(n) | 0)
        .filter((n) => Number.isFinite(n) && n >= 0);
}
// auto fork helper
function normalizeRulesForServer(cfg) {
    const rules = { ...(cfg.rules || {}) };
    // cluster.k from legacy kmeans_k
    const cluster = { ...rules.cluster };
    if (typeof rules.kmeans_k === 'number') {
        cluster.k = rules.kmeans_k;
        delete rules.kmeans_k;
    }
    if (Object.keys(cluster).length)
        rules.cluster = cluster;
    // base_cooldown_steps from legacy fork_cooldown_steps
    if (typeof rules.fork_cooldown_steps === 'number') {
        rules.base_cooldown_steps = rules.fork_cooldown_steps;
        delete rules.fork_cooldown_steps;
    }
    return {
        rules,
        sampling: { ...(cfg.sampling || {}) },
        runtime: { ...(cfg.runtime || {}) },
    };
}
//# sourceMappingURL=extension.js.map