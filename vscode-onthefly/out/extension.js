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
const storage_1 = require("./storage");
const os = __importStar(require("os"));
const ipc_1 = require("./extensionHost/ipc");
const state_1 = require("./extensionHost/state");
// Stash latest config (webview can set it before or after starting the run)
function isRunImportedForThisSession(runId) {
    if (!(0, ipc_1.trainerActive)())
        return true;
    return !state_1.extensionState.nativeRunsThisSession.has(runId);
}
async function activate(context) {
    state_1.extensionState.lastExtensionContext = context;
    context.subscriptions.push(vscode.commands.registerCommand('onthefly.showDashboard', () => openPanel(context)));
    context.subscriptions.push(vscode.window.registerWebviewPanelSerializer('ontheflyDashboard', {
        async deserializeWebviewPanel(webviewPanel) {
            await revivePanel(context, webviewPanel);
        },
    }));
    await (0, storage_1.initStorage)(context);
}
function deactivate() {
    (0, ipc_1.shutdownTrainerServer)();
    try {
        (0, storage_1.closeStorage)({ retainFile: true });
    }
    catch { }
}
async function hardResetSession(context, opts) {
    let trainerSeed = null;
    const hadTrainer = (0, ipc_1.trainerActive)();
    if (hadTrainer && opts?.fromUser) {
        try {
            const res = await (0, ipc_1.sendReq)('reset_session', { reason: 'manual' }, 60000);
            trainerSeed = res || null;
            if (trainerSeed?.run_id) {
                (0, state_1.post)({
                    type: 'log',
                    text: `[reset] Trainer cleared. Next run "${trainerSeed.run_id}" waits for resume.`,
                });
            }
        }
        catch (e) {
            (0, state_1.post)({
                type: 'log',
                level: 'warn',
                text: `[reset] Backend reset skipped or failed: ${e?.message || String(e)}`,
            });
        }
    }
    // 0) If a backend is alive, ask it to clean its save_dir now (best effort).
    if ((0, ipc_1.trainerActive)()) {
        try {
            const res = await (0, ipc_1.sendReq)('clean_disk', { scope: 'all' }, 60000);
            (0, state_1.post)({
                type: 'log',
                level: res?.ok ? 'info' : 'warn',
                text: res?.ok
                    ? `[reset] Disk cleanup ok: removed ${res.removed?.length ?? 0} run(s)`
                    : `[reset] Disk cleanup reported an issue: ${res?.error ?? 'unknown error'}`,
            });
        }
        catch (e) {
            (0, state_1.post)({
                type: 'log',
                level: 'warn',
                text: `[reset] Disk cleanup skipped or failed: ${e?.message || String(e)}`,
            });
        }
    }
    // 1) Disconnect trainer (does not stop user's script; it will reconnect if still running)
    (0, ipc_1.disconnectTrainer)(true);
    // 3) Clear in-memory extension state
    state_1.extensionState.currentRunId = null;
    state_1.extensionState.seenRuns.clear();
    state_1.extensionState.currentSessionId = null;
    state_1.extensionState.nativeRunsThisSession.clear();
    state_1.extensionState.modelNavSelectedRunId = null;
    state_1.extensionState.runActivityState = null;
    state_1.extensionState.pauseInFlight = false;
    state_1.extensionState.resumeInFlight = false;
    state_1.extensionState.pendingResumeAwaitingFirstRun = false;
    // 4) Reset storage
    try {
        (0, storage_1.closeStorage)({ retainFile: false });
    }
    catch { }
    await (0, storage_1.initStorage)(context);
    state_1.extensionState.needDiskCleanOnNextTrainer = true;
    // 5) Tell webview, if it exists
    (0, state_1.post)({ type: 'resetOk' });
    if (trainerSeed?.run_id) {
        const runId = String(trainerSeed.run_id);
        const friendly = (trainerSeed.display_name && String(trainerSeed.display_name)) || runId;
        const projectName = (trainerSeed.project && String(trainerSeed.project)) || 'default';
        (0, storage_1.insertRun)(runId, projectName, friendly, null);
        state_1.extensionState.seenRuns.add(runId);
        state_1.extensionState.currentRunId = runId;
        state_1.extensionState.modelNavSelectedRunId = runId;
        state_1.extensionState.nativeRunsThisSession.add(runId);
        if (trainerSeed.session_id) {
            state_1.extensionState.currentSessionId = String(trainerSeed.session_id);
            (0, state_1.postCurrentSession)();
        }
        (0, state_1.post)({
            type: 'newRun',
            run_id: runId,
            parents: [],
            meta: { display_name: friendly, kind: 'reset' },
            session_id: state_1.extensionState.currentSessionId || null,
        });
        (0, state_1.post)({ type: 'modelNav.select', runId });
    }
    (0, state_1.post)({ type: 'runs', rows: (0, storage_1.listRuns)() });
    (0, state_1.postStatus)(false);
    if (opts?.fromUser) {
        vscode.window.setStatusBarMessage('Onthefly: session reset', 2000);
    }
}
(0, ipc_1.registerHardResetHandler)(hardResetSession);
/* ============================ Panel & HTML ============================ */
function getLocalResourceRoots(context) {
    return [
        vscode.Uri.file(context.extensionPath),
        vscode.Uri.file(path.join(context.extensionPath, 'src', 'webview')),
        vscode.Uri.file(path.join(context.extensionPath, 'media')),
        vscode.Uri.file(path.join(context.extensionPath, 'node_modules')),
    ];
}
async function openPanel(context) {
    if (state_1.extensionState.panel) {
        state_1.extensionState.panel.reveal(vscode.ViewColumn.Active);
        return;
    }
    const newPanel = vscode.window.createWebviewPanel('ontheflyDashboard', 'OnTheFly', vscode.ViewColumn.Active, {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: getLocalResourceRoots(context),
    });
    configurePanel(context, newPanel);
}
async function revivePanel(context, webviewPanel) {
    configurePanel(context, webviewPanel);
}
function configurePanel(context, webviewPanel) {
    (0, ipc_1.ensureTrainerServer)();
    state_1.extensionState.panel = webviewPanel;
    state_1.extensionState.panel.webview.options = {
        enableScripts: true,
        localResourceRoots: getLocalResourceRoots(context),
    };
    const nonce = getNonce();
    let webviewVisible = state_1.extensionState.panel.visible;
    state_1.extensionState.panel.onDidChangeViewState(({ webviewPanel }) => {
        webviewVisible = webviewPanel.visible;
        if (webviewVisible) {
            // when user returns, cheaply re-sync UI (no heavy streams)
            try {
                (0, state_1.post)({ type: 'runs', rows: (0, storage_1.listRuns)() });
            }
            catch { }
            (0, state_1.postStatus)((0, ipc_1.trainerActive)());
        }
    });
    state_1.extensionState.panel.onDidDispose(() => {
        // Just drop the reference; let deactivate() handle shutdown.
        if (state_1.extensionState.panel === webviewPanel) {
            state_1.extensionState.panel = null;
        }
    });
    state_1.extensionState.panel.webview.onDidReceiveMessage((m) => { onMessage(context, m); });
    state_1.extensionState.panel.webview.html = getHtml(context, state_1.extensionState.panel.webview, nonce);
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
    const autoforkPanelPath = [
        path.join(context.extensionPath, 'autofork_panel.js'),
        path.join(context.extensionPath, 'src', 'webview', 'autofork_panel.js'),
    ].find(fs.existsSync);
    const dagLayoutPath = [
        path.join(context.extensionPath, 'dag_layout.js'),
        path.join(context.extensionPath, 'src', 'webview', 'dag_layout.js'),
    ].find(fs.existsSync);
    const dagRenderPath = [
        path.join(context.extensionPath, 'dag_render.js'),
        path.join(context.extensionPath, 'src', 'webview', 'dag_render.js'),
    ].find(fs.existsSync);
    const healthPath = [
        path.join(context.extensionPath, 'health_monitor.js'),
        path.join(context.extensionPath, 'src', 'webview', 'health_monitor.js'),
    ].find(fs.existsSync);
    const chartCreationPath = [
        path.join(context.extensionPath, 'chart_creation.js'),
        path.join(context.extensionPath, 'src', 'webview', 'chart_creation.js'),
    ].find(fs.existsSync);
    const chartPluginsPath = [
        path.join(context.extensionPath, 'chart_plugins.js'),
        path.join(context.extensionPath, 'src', 'webview', 'chart_plugins.js'),
    ].find(fs.existsSync);
    const chartBootstrapPath = [
        path.join(context.extensionPath, 'chart_bootstrap.js'),
        path.join(context.extensionPath, 'src', 'webview', 'chart_bootstrap.js'),
    ].find(fs.existsSync);
    const chartReportPath = [
        path.join(context.extensionPath, 'chart_report.js'),
        path.join(context.extensionPath, 'src', 'webview', 'chart_report.js'),
    ].find(fs.existsSync);
    const reportSelectionPath = [
        path.join(context.extensionPath, 'report_selection.js'),
        path.join(context.extensionPath, 'src', 'webview', 'report_selection.js'),
    ].find(fs.existsSync);
    const chartStreamPath = [
        path.join(context.extensionPath, 'chart_stream.js'),
        path.join(context.extensionPath, 'src', 'webview', 'chart_stream.js'),
    ].find(fs.existsSync);
    const logBufferPath = [
        path.join(context.extensionPath, 'log_buffer.js'),
        path.join(context.extensionPath, 'src', 'webview', 'log_buffer.js'),
    ].find(fs.existsSync);
    const ipcControlsPath = [
        path.join(context.extensionPath, 'ipc_controls.js'),
        path.join(context.extensionPath, 'src', 'webview', 'ipc_controls.js'),
    ].find(fs.existsSync);
    const runStatePath = [
        path.join(context.extensionPath, 'run_state.js'),
        path.join(context.extensionPath, 'src', 'webview', 'run_state.js'),
    ].find(fs.existsSync);
    const chartUtilsPath = [
        path.join(context.extensionPath, 'chart_utils.js'),
        path.join(context.extensionPath, 'src', 'webview', 'chart_utils.js'),
    ].find(fs.existsSync);
    const chartPathCandidates = [
        path.join(context.extensionPath, 'media', 'chart.min.js'), // if I ever want a copy
        path.join(context.extensionPath, 'node_modules', 'chart.js', 'dist', 'chart.min.js'),
        path.join(context.extensionPath, 'node_modules', 'chart.js', 'dist', 'chart.js'),
    ];
    const chartPath = chartPathCandidates.find(fs.existsSync);
    const jsUri = jsPath ? webview.asWebviewUri(vscode.Uri.file(jsPath)).toString() : '';
    const dagLayoutUri = dagLayoutPath ? webview.asWebviewUri(vscode.Uri.file(dagLayoutPath)).toString() : '';
    const dagRenderUri = dagRenderPath ? webview.asWebviewUri(vscode.Uri.file(dagRenderPath)).toString() : '';
    const chartCreationUri = chartCreationPath ? webview.asWebviewUri(vscode.Uri.file(chartCreationPath)).toString() : '';
    const chartPluginsUri = chartPluginsPath ? webview.asWebviewUri(vscode.Uri.file(chartPluginsPath)).toString() : '';
    const chartBootstrapUri = chartBootstrapPath ? webview.asWebviewUri(vscode.Uri.file(chartBootstrapPath)).toString() : '';
    const chartReportUri = chartReportPath ? webview.asWebviewUri(vscode.Uri.file(chartReportPath)).toString() : '';
    const reportSelectionUri = reportSelectionPath ? webview.asWebviewUri(vscode.Uri.file(reportSelectionPath)).toString() : '';
    const chartStreamUri = chartStreamPath ? webview.asWebviewUri(vscode.Uri.file(chartStreamPath)).toString() : '';
    const logBufferUri = logBufferPath ? webview.asWebviewUri(vscode.Uri.file(logBufferPath)).toString() : '';
    const ipcControlsUri = ipcControlsPath ? webview.asWebviewUri(vscode.Uri.file(ipcControlsPath)).toString() : '';
    const runStateUri = runStatePath ? webview.asWebviewUri(vscode.Uri.file(runStatePath)).toString() : '';
    const chartUtilsUri = chartUtilsPath ? webview.asWebviewUri(vscode.Uri.file(chartUtilsPath)).toString() : '';
    const hmUri = healthPath ? webview.asWebviewUri(vscode.Uri.file(healthPath)).toString() : '';
    // const autoforkPanelUri = autoforkPanelPath ? webview.asWebviewUri(vscode.Uri.file(autoforkPanelPath)).toString() : '';
    const chartUri = chartPath ? webview.asWebviewUri(vscode.Uri.file(chartPath)).toString() : '';
    const flyPath = path.join(context.extensionPath, 'src', 'webview', 'images', 'fly.png');
    const flyUri = webview.asWebviewUri(vscode.Uri.file(flyPath)).toString();
    html = html
        .replace(/__NONCE__/g, nonce)
        .replace(/__DASHBOARD_JS__/g, jsUri || '')
        .replace(/__CHART_JS__/g, chartUri || '')
        .replace(/__DAG_LAYOUT_JS__/g, dagLayoutUri || '')
        .replace(/__DAG_RENDER_JS__/g, dagRenderUri || '')
        .replace(/__HEALTH_MONITOR_JS__/g, hmUri || '')
        .replace(/__CHART_CREATION_JS__/g, chartCreationUri || '')
        .replace(/__CHART_PLUGINS_JS__/g, chartPluginsUri || '')
        .replace(/__CHART_BOOTSTRAP_JS__/g, chartBootstrapUri || '')
        .replace(/__CHART_REPORT_JS__/g, chartReportUri || '')
        .replace(/__REPORT_SELECTION_JS__/g, reportSelectionUri || '')
        .replace(/__CHART_STREAM_JS__/g, chartStreamUri || '')
        .replace(/__LOG_BUFFER_JS__/g, logBufferUri || '')
        .replace(/__IPC_CONTROLS_JS__/g, ipcControlsUri || '')
        .replace(/__CHART_UTILS__/g, chartUtilsUri || '')
        // .replace(/__AUTOFORK_PANEL_JS__/g, autoforkPanelUri || '')
        .replace(/__RUN_STATE_JS__/g, runStateUri || '')
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
    if (!runStateUri)
        console.warn('[onthefly] run_state.js not found. Lineage store will not load.');
    if (!ipcControlsUri)
        console.warn('[onthefly] ipc_controls.js not found. Button wiring will not initialize.');
    // if (!autoforkPanelUri) console.warn('[onthefly] autofork_panel.js not found. AutoFork UI will be disabled.');
    if (!dagLayoutUri)
        console.warn('[onthefly] dag_layout.js not found. DAG will fall back or render linearly.');
    if (!dagRenderUri)
        console.warn('[onthefly] dag_render.js not found. DAG overlay will not render.');
    if (!logBufferUri)
        console.warn('[onthefly] log_buffer.js not found. Log textarea will not update.');
    if (!reportSelectionUri)
        console.warn('[onthefly] report_selection.js not found. Report selection overlay will not load.');
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
const TEST_NOW_TIMEOUT_MS = 10 * 60 * 1000;
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
    const remembered = context.globalState.get(state_1.LAST_EXPORT_DIR_KEY);
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
                await context.globalState.update(state_1.LAST_EXPORT_DIR_KEY, path.dirname(picked.fsPath));
                vscode.window.showInformationMessage(`Chart exported to: ${picked.fsPath}`);
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'exportSubset': {
            try {
                const runId = m.runId || state_1.extensionState.currentRunId;
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
                await context.globalState.update(state_1.LAST_EXPORT_DIR_KEY, path.dirname(picked.fsPath));
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
                // --- Fallbacks ---
                if (!payload.subset_indices) {
                    const region = m.region;
                    const stored = (0, storage_1.getRunSubset)(String(runId)) || [];
                    const haveStored = Array.isArray(stored) && stored.length > 0;
                    if (region && Number.isFinite(region.minLoss) && Number.isFinite(region.maxLoss)) {
                        try {
                            const indices = await (0, ipc_1.getSelectedRegionIndices)(String(runId), region.minLoss, region.maxLoss);
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
                const data = await (0, ipc_1.sendReq)('export_subset', payload, 10 * 60 * 1000);
                (0, state_1.post)({ type: 'subsetExported', run_id: String(runId), ...data });
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'modelNav.select': {
            const id = String(m.runId || '').trim();
            if (id) {
                state_1.extensionState.modelNavSelectedRunId = id;
                (0, state_1.post)({ type: 'log', text: `[modelNav] selected: ${id}` });
                (0, state_1.post)({ type: 'modelNav.select', runId: id }); // echo to webview so it knows which run is selected
                try {
                    const rows = (0, storage_1.getLogs)(id); // all phases
                    (0, state_1.post)({ type: 'logs', run_id: id, rows: (0, state_1.stripUiOnlyFields)(rows) });
                }
                catch (e) {
                    console.warn('[onthefly] failed to load logs for selected run', id, e);
                }
            }
            break;
        }
        case 'requestLogs': {
            try {
                const phase = m.phase;
                const requested = String(m.runId || '').trim();
                const selected = state_1.extensionState.modelNavSelectedRunId && state_1.extensionState.modelNavSelectedRunId.trim();
                const active = state_1.extensionState.currentRunId && state_1.extensionState.currentRunId.trim();
                const rid = requested || selected || active || '';
                let rows = rid ? (0, storage_1.getLogs)(rid, phase) : [];
                if ((!rows || rows.length === 0) && state_1.extensionState.currentSessionId) {
                    rows = (0, storage_1.getLogsBySession)(state_1.extensionState.currentSessionId, phase);
                }
                (0, state_1.post)({ type: 'logs', run_id: rid || null, rows: (0, state_1.stripUiOnlyFields)(rows) });
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'requestTestRows': {
            try {
                const rows = (0, storage_1.getTestRows)(String(m.runId));
                (0, state_1.post)({ type: 'testRows', run_id: String(m.runId), rows });
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'pause': {
            if (state_1.extensionState.pauseInFlight) {
                (0, state_1.post)({ type: 'log', text: '[pause] Request already in progress.' });
                break;
            }
            if (state_1.extensionState.runActivityState === 'paused') {
                break;
            }
            state_1.extensionState.pauseInFlight = true;
            try {
                // 1) Ask the backend to pause (finish current step, flush state, etc.)
                const pauseInfo = await (0, ipc_1.requestBackendPause)(30_000); // backend will do its own checkpointing if it wants
                // 2) Figure out which run this pause applies to
                const runId = (state_1.extensionState.modelNavSelectedRunId && state_1.extensionState.modelNavSelectedRunId.trim()) ||
                    (state_1.extensionState.currentRunId && state_1.extensionState.currentRunId.trim()) ||
                    null;
                if (!runId) {
                    (0, state_1.post)({ type: 'warn', text: '[pause] No runId available to attach checkpoint to.' });
                    break;
                }
                const pausedStep = Number(pauseInfo?.step);
                const payload = { cmd: 'pause', run_id: runId };
                if (Number.isFinite(pausedStep))
                    payload.step = pausedStep;
                (0, state_1.post)({
                    type: 'paused',
                    payload,
                });
                // 3) Explicitly save a checkpoint and persist it in storage
                try {
                    const ck = await (0, ipc_1.sendReq)('save_ckpt', {}, 120_000);
                    if (ck?.path) {
                        const stepNum = Number(ck.step) || 0;
                        const ckptId = `${runId}:${stepNum}:${Date.now()}`; // same style as elsewhere
                        (0, storage_1.insertCheckpoint)(ckptId, runId, stepNum, String(ck.path));
                        (0, state_1.post)({
                            type: 'log',
                            level: 'info',
                            text: `[pause] checkpoint saved for run ${runId} at step ${stepNum}`,
                        });
                    }
                    else {
                        (0, state_1.post)({
                            type: 'warn',
                            text: '[pause] save_ckpt returned no path; checkpoint not recorded in DB.',
                        });
                    }
                }
                catch (ckErr) {
                    console.warn('[onthefly] pause: save_ckpt / insertCheckpoint failed:', ckErr);
                    (0, state_1.post)({
                        type: 'warn',
                        text: `[pause] Failed to save/persist checkpoint: ${ckErr?.message || String(ckErr)}`,
                    });
                }
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            finally {
                state_1.extensionState.pauseInFlight = false;
            }
            break;
        }
        case 'resume': {
            if (state_1.extensionState.resumeInFlight) {
                (0, state_1.post)({ type: 'log', text: '[resume] Request already in progress.' });
                break;
            }
            const requested = String(m.runId || '').trim();
            if (requested)
                state_1.extensionState.modelNavSelectedRunId = requested;
            const target = requested || state_1.extensionState.modelNavSelectedRunId || state_1.extensionState.currentRunId || null;
            if (!target) {
                try {
                    const rows = (0, storage_1.listRuns)();
                    if (!rows.length) {
                        state_1.extensionState.pendingResumeAwaitingFirstRun = true;
                        (0, state_1.post)({ type: 'log', text: '[resume] No runs exist yet. Waiting for a fresh trainer session…' });
                        await (0, ipc_1.resumeAfterReset)();
                    }
                    else {
                        (0, state_1.post)({ type: 'error', text: '[resume] No run selected.' });
                    }
                }
                catch (e) {
                    state_1.extensionState.pendingResumeAwaitingFirstRun = false;
                    (0, state_1.postErr)(e);
                }
                break;
            }
            await (0, ipc_1.resumeTrainerOn)(target);
            break;
        }
        case 'testNow': {
            const requested = m.runId;
            const candidate = (requested && requested.trim())
                || (state_1.extensionState.modelNavSelectedRunId && state_1.extensionState.modelNavSelectedRunId.trim())
                || (state_1.extensionState.currentRunId && state_1.extensionState.currentRunId.trim())
                || null;
            try {
                if (!candidate) {
                    (0, state_1.post)({ type: 'log', text: 'Select a model in Model Nav before testing.' });
                    break;
                }
                const runs = (0, storage_1.listRuns)();
                const friendly = runs.find(r => r.run_id === candidate)?.name || candidate;
                const choice = await vscode.window.showWarningMessage(`This will override the final tested model for this session. Test model "${friendly}" now?`, {
                    modal: true,
                    detail: 'Confirming will test immediately and replace the stored final checkpoint for this session. You can continue training this and other models after.'
                }, 'Confirm Test');
                if (choice !== 'Confirm Test') {
                    (0, state_1.post)({ type: 'log', text: 'Test cancelled.' });
                    break;
                }
                const target = candidate;
                await (0, ipc_1.ensureTrainerOnRun)(target);
                try {
                    await (0, ipc_1.requestBackendPause)(30_000);
                }
                catch (pauseErr) {
                    console.warn('[onthefly] pause before test failed', pauseErr);
                }
                (0, state_1.post)({ type: 'testNow', status: 'pending', run_id: target });
                const data = await (0, ipc_1.sendReq)('test_now', { label: 'final' }, TEST_NOW_TIMEOUT_MS);
                const normalizedLoss = Number.isFinite(Number(data?.avg_loss)) ? Number(data.avg_loss) : null;
                const sessionIdRaw = (data?.session_id && String(data.session_id)) || state_1.extensionState.currentSessionId || '';
                const sessionId = (sessionIdRaw && sessionIdRaw.trim()) || `session-${target}`;
                const ckptPath = data?.ckpt_path ? String(data.ckpt_path) : '';
                if (sessionId && ckptPath) {
                    try {
                        (0, storage_1.upsertFinalModel)({
                            session_id: sessionId,
                            run_id: target,
                            ckpt_path: ckptPath,
                            step: Number(data?.step) || 0,
                            avg_test_loss: normalizedLoss,
                        });
                    }
                    catch (persistErr) {
                        console.warn('[onthefly] failed to persist final model', persistErr);
                    }
                }
                (0, state_1.post)({
                    type: 'testNow',
                    status: 'completed',
                    run_id: target,
                    data: { ...data, avg_loss: normalizedLoss, ckpt_path: ckptPath },
                });
            }
            catch (e) {
                const fallback = candidate || state_1.extensionState.modelNavSelectedRunId || state_1.extensionState.currentRunId || null;
                (0, state_1.post)({ type: 'testNow', status: 'error', run_id: fallback, error: String(e?.message || e) });
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'fork': {
            try {
                // Accept both message shapes
                const payloadIn = (() => {
                    const top = m;
                    const inner = (top && top.payload) || {};
                    const merged = { ...inner, ...top };
                    delete merged.command;
                    delete merged.type;
                    delete merged.payload;
                    return merged;
                })();
                // Also consider runId as a parent hint
                const hinted = String(payloadIn.owner_run_id || payloadIn.parent_run_id || payloadIn.runId || '').trim();
                const target = hinted ||
                    state_1.extensionState.modelNavSelectedRunId ||
                    state_1.extensionState.currentRunId ||
                    pickAnchorRunForAction(payloadIn) ||
                    '';
                if (!target) {
                    (0, state_1.postErr)('No run selected to fork from.');
                    break;
                }
                await (0, ipc_1.ensureTrainerOnRun)(target);
                try {
                    await (0, ipc_1.requestBackendPause)(30_000);
                }
                catch { }
                let ckptPath = (0, storage_1.latestCheckpointForRun)(target)?.path;
                if (!ckptPath) {
                    try {
                        const ck = await (0, ipc_1.sendReq)('save_ckpt', {}, 120_000);
                        ckptPath = ck?.path;
                    }
                    catch { }
                }
                // Forward region/hparams through to backend
                const data = await (0, ipc_1.sendReq)('fork', {
                    parent_run_id: target,
                    owner_run_id: target,
                    ...payloadIn,
                    ...(ckptPath ? { parent_ckpt_path: ckptPath } : {}),
                }, 120_000);
                const child = String(data?.new_run || data?.child_id || data?.run_id || '').trim();
                if (child) {
                    state_1.extensionState.currentRunId = child;
                    state_1.extensionState.modelNavSelectedRunId = child;
                    state_1.extensionState.nativeRunsThisSession.add(child);
                    (0, state_1.post)({ type: 'modelNav.select', runId: child });
                    if (Array.isArray(data?.subset_indices)) {
                        try {
                            (0, storage_1.setRunSubset)(child, data.subset_indices.map((n) => Number(n) | 0));
                        }
                        catch { }
                    }
                    try {
                        (0, ipc_1.sendCtl)({ cmd: 'resume' });
                    }
                    catch { }
                }
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'merge': {
            try {
                const payload = m.payload;
                const anchor = pickAnchorRunForAction(payload);
                if (!anchor) {
                    (0, state_1.postErr)('No runs available to anchor the merge. Select a run or include parents[].');
                    break;
                }
                await (0, ipc_1.ensureTrainerOnRun)(anchor);
                try {
                    await (0, ipc_1.requestBackendPause)(30_000);
                }
                catch { }
                // Preflight: all parents must have checkpoints.
                const parents = Array.isArray(payload.parents) ? payload.parents : [];
                const missing = parents.filter(p => !(0, storage_1.latestCheckpointForRun)(p));
                if (missing.length) {
                    (0, state_1.post)({
                        type: 'error',
                        text: `[merge] Missing checkpoints for parent runs: ${missing.join(', ')}. Pause runs in your script to create checkpoints first.`,
                    });
                    break;
                }
                // Perform the merge.
                const data = await (0, ipc_1.sendReq)('merge', payload, 120000);
                // --- immediately navigate to the merged child & start running (mirror fork behavior) ---
                const child = String((data && (data.new_run ?? data.child_id ?? data.run_id)) || '').trim();
                if (child) {
                    state_1.extensionState.currentRunId = child;
                    state_1.extensionState.modelNavSelectedRunId = child;
                    state_1.extensionState.nativeRunsThisSession.add(child);
                    (0, state_1.post)({ type: 'modelNav.select', runId: child });
                    // If backend provided a subset, persist it like we do on fork.
                    if (Array.isArray(data?.subset_indices)) {
                        try {
                            (0, storage_1.setRunSubset)(child, data.subset_indices.map((n) => Number(n) | 0));
                        }
                        catch { }
                    }
                    // Nudge the backend to resume the new merged run.
                    try {
                        (0, ipc_1.sendCtl)({ cmd: 'resume' });
                    }
                    catch { }
                }
                else {
                    // If the backend didn't return an explicit child id, the usual 'newRun' event
                    // will still keep things in sync. No-op here.
                }
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'dist_health': {
            try {
                const rid = m.runId;
                const target = await requireActiveRun(rid);
                if (!target)
                    break;
                await (0, ipc_1.ensureTrainerOnRun)(target);
                state_1.extensionState.modelNavSelectedRunId = target;
                (0, state_1.post)({ type: 'modelNav.select', runId: target });
                await (0, ipc_1.runHealth)('dist_health', { require_all_ranks: false, sample_weights: 0 }, 'distHealth', 30_000, target);
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'throughput_health': {
            try {
                const rid = m.runId;
                const target = await requireActiveRun(rid);
                if (!target)
                    break;
                await (0, ipc_1.ensureTrainerOnRun)(target);
                state_1.extensionState.modelNavSelectedRunId = target;
                (0, state_1.post)({ type: 'modelNav.select', runId: target });
                await (0, ipc_1.runHealth)('throughput_health', { budget_steps: 2, micro_backward: false }, 'throughputHealth', 45_000, target);
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'activations_health': {
            try {
                const rid = m.runId;
                const target = await requireActiveRun(rid);
                if (!target)
                    break;
                await (0, ipc_1.ensureTrainerOnRun)(target);
                state_1.extensionState.modelNavSelectedRunId = target;
                (0, state_1.post)({ type: 'modelNav.select', runId: target });
                await (0, ipc_1.runHealth)('activations_health', { budget_steps: 2, topk: 12, eps: 1e-3 }, 'activationsHealth', 60_000, target);
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'numerics_health': {
            try {
                const rid = m.runId;
                const target = await requireActiveRun(rid);
                if (!target)
                    break;
                await (0, ipc_1.ensureTrainerOnRun)(target);
                state_1.extensionState.modelNavSelectedRunId = target;
                (0, state_1.post)({ type: 'modelNav.select', runId: target });
                await (0, ipc_1.runHealth)('numerics_health', { sample_layers: 25 }, 'numericsHealth', 30_000, target);
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'determinism_health': {
            try {
                const rid = m.runId;
                const target = await requireActiveRun(rid);
                if (!target)
                    break;
                await (0, ipc_1.ensureTrainerOnRun)(target);
                state_1.extensionState.modelNavSelectedRunId = target;
                (0, state_1.post)({ type: 'modelNav.select', runId: target });
                await (0, ipc_1.runHealth)('determinism_health', {}, 'determinismHealth', 20_000, target);
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'generateReport': {
            try {
                let runId = m.runId ?? undefined;
                if (!runId || !String(runId).trim()) {
                    const anchor = pickAnchorRunForAction(); // choose something sensible if not provided
                    if (!anchor) {
                        (0, state_1.post)({ type: 'error', text: 'No active run selected for report.' });
                        break;
                    }
                    runId = anchor;
                }
                runId = String(runId).trim();
                await (0, ipc_1.ensureTrainerOnRun)(runId);
                const target = await requireActiveRun(runId);
                if (!target)
                    break;
                runId = target;
                state_1.extensionState.modelNavSelectedRunId = runId;
                (0, state_1.post)({ type: 'modelNav.select', runId });
                await (0, ipc_1.requestBackendPause)(30_000);
                const subset = (0, storage_1.getRunSubset)(String(target));
                const subset_on = 'train';
                const data = await (0, ipc_1.sendReq)('generate_report', {
                    owner_run_id: target,
                    subset_indices: subset.length ? subset : undefined,
                    subset_on,
                    reqId: m.reqId
                }, 10 * 60 * 1000);
                const losses = Array.isArray(data?.losses)
                    ? data.losses.map(Number).filter(Number.isFinite)
                    : [];
                let sample_indices = Array.isArray(data?.sample_indices)
                    ? data.sample_indices.map(v => Math.trunc(Number(v))).filter(v => Number.isFinite(v) && v >= 0)
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
                (0, state_1.post)({
                    type: 'reportData',
                    run_id: String(runId),
                    owner_run_id: String(runId),
                    reqId: m.reqId,
                    losses,
                    sample_indices,
                    meta: { ...meta, at_step, at_epoch, subset_on, samples: losses.length, note }
                });
            }
            catch (e) {
                (0, state_1.postErr)(e);
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
                    (0, state_1.post)({
                        type: 'reportFromDb',
                        run_id: String(runId),
                        owner_run_id: String(runId),
                        losses: row.losses,
                        meta: { note: row.note, at_step: row.at_step, at_epoch: row.at_epoch, samples: row.samples, subset_on: row.subset_on }
                    });
                }
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'notify': {
            const lvl = m.level || 'info';
            const type = lvl === 'error' ? 'error' : 'log';
            (0, state_1.post)({ type, text: m.text });
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
                await context.globalState.update(state_1.LAST_EXPORT_DIR_KEY, path.dirname(bundleDir));
                fs.mkdirSync(bundleDir, { recursive: true });
                // NOTE: "modelNav" → we don't keep a separate nav map here; the DB is source of truth.
                // Grab every run_id we know and treat that as the nav list.
                const allRuns = (0, storage_1.listRuns)().map(r => String(r.run_id));
                const owners = Array.from(new Set(allRuns)).filter(Boolean);
                let spillRoot = null;
                if ((0, ipc_1.trainerActive)()) {
                    spillRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'onthefly-spill-'));
                    // 1) try the multi-owner prepare_export API (forces snapshots/ckpts for all runs)
                    let prep = null;
                    try {
                        // latest_only keeps the bundle trim
                        prep = await (0, ipc_1.sendReq)('prepare_export', { dir: spillRoot, latest_only: true, owners }, 10 * 60 * 1000);
                    }
                    catch (e) {
                        // old servers won't know prepare_export(owners=...), fall back to a generic spill
                        (0, state_1.post)({ type: 'log', text: '[export] prepare_export not available or failed; falling back to spill_all().' });
                        try {
                            const recs = await (0, ipc_1.sendReq)('spill_all', { dir: spillRoot, latest_only: true }, 10 * 60 * 1000);
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
                    // 3) if python returned a ring ckpt for the current run, stash it too
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
                }
                else {
                    // not running → just bundle whatever DB already references
                    (0, state_1.post)({ type: 'log', text: '[export] Python not running; bundling existing DB checkpoints only.' });
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
                (0, state_1.post)({ type: 'sessionSaved', path: bundleDir });
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'loadSession': {
            try {
                const initialDir = getInitialExportDir(context);
                const picked = await vscode.window.showOpenDialog({
                    title: 'Load bundle',
                    canSelectMany: false,
                    canSelectFiles: true,
                    canSelectFolders: true,
                    defaultUri: vscode.Uri.file(initialDir),
                    openLabel: 'Load',
                });
                if (!picked || !picked[0])
                    break;
                const chosen = picked[0].fsPath;
                const isDir = fs.existsSync(chosen) && fs.statSync(chosen).isDirectory();
                await context.globalState.update(state_1.LAST_EXPORT_DIR_KEY, isDir ? chosen : path.dirname(chosen));
                let bundleDir = null;
                if (isDir) {
                    const manifestPath = path.join(chosen, 'bundle.json');
                    if (!fs.existsSync(manifestPath)) {
                        vscode.window.showErrorMessage(`No bundle.json found in folder:\n${chosen}`);
                        break;
                    }
                    bundleDir = chosen;
                }
                else {
                    const base = path.basename(chosen).toLowerCase();
                    if (base !== 'bundle.json') {
                        vscode.window.showErrorMessage('Unsupported selection. Pick a bundle folder (with bundle.json) or bundle.json itself.');
                        break;
                    }
                    bundleDir = path.dirname(chosen);
                }
                // Load the bundle into the live DB
                (0, storage_1.loadBundle)(bundleDir, context);
                for (const s of (0, storage_1.listSessions)()) {
                    (0, state_1.post)({ type: 'log', text: `[print] session ${s.session_id}: ${(0, storage_1.getLogsBySession)(String(s.session_id)).length} logs` });
                }
                // Notify UI & refresh runs
                (0, state_1.post)({ type: 'sessionLoaded' });
                const runs = (0, storage_1.listRuns)();
                (0, state_1.post)({ type: 'runs', rows: runs });
                try {
                    const sessions = (0, storage_1.listSessions)();
                    (0, state_1.post)({ type: 'fs.session.list', items: sessions });
                }
                catch { }
                // Hydrate logs for each run so the webview has something to render immediately
                for (const r of runs) {
                    try {
                        const rows = (0, storage_1.getLogs)(String(r.run_id)); // all phases
                        (0, state_1.post)({ type: 'logs', run_id: String(r.run_id), rows: (0, state_1.stripUiOnlyFields)(rows) });
                    }
                    catch (e) {
                        console.warn('[onthefly] failed to load logs for run', r.run_id, e);
                    }
                }
                const choice = await vscode.window.showInformationMessage(`Bundle loaded from: ${bundleDir}`, 'Reveal in Finder/Explorer');
                if (choice)
                    vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(bundleDir));
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'resetAll': {
            const pick = await vscode.window.showWarningMessage('This will erase all models from memory, are you sure you want to refresh?', {
                modal: true,
                detail: 'Any running training will be stopped. Nothing will be saved if you have not exported session.',
            }, 'Erase & Refresh');
            if (pick !== 'Erase & Refresh') {
                (0, state_1.post)({ type: 'log', text: 'Refresh cancelled.' });
                break;
            }
            await hardResetSession(context, { fromUser: true });
            break;
        }
        case 'requestRuns': {
            try {
                const rows = (0, storage_1.listRuns)();
                (0, state_1.post)({ type: 'runs', rows });
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
        case 'requestRows': {
            try {
                (0, state_1.post)({ type: 'rows', rows: (0, storage_1.getRunRows)(m.runId) });
            }
            catch (e) {
                (0, state_1.postErr)(e);
            }
            break;
        }
    }
}
/* ============================ Python process (training) ============================ */
function pickAnchorRunForAction(hint) {
    const fromRunId = typeof hint?.runId === 'string' && hint.runId.trim() ? String(hint.runId).trim() : null;
    const fromParents = Array.isArray(hint?.parents) && hint.parents.length ? String(hint.parents[0]) : null;
    if (fromRunId)
        return fromRunId;
    if (fromParents)
        return fromParents;
    if (state_1.extensionState.modelNavSelectedRunId?.trim())
        return state_1.extensionState.modelNavSelectedRunId.trim();
    if (state_1.extensionState.currentRunId?.trim())
        return state_1.extensionState.currentRunId.trim();
    try {
        const first = (0, storage_1.listRuns)()[0]?.run_id;
        if (first)
            return String(first);
    }
    catch { }
    return null;
}
async function requireActiveRun(requestedRunId) {
    const target = String(requestedRunId || state_1.extensionState.modelNavSelectedRunId || state_1.extensionState.currentRunId || '').trim();
    if (!target)
        return null;
    await (0, ipc_1.ensureTrainerOnRun)(target);
    return target;
}
//# sourceMappingURL=extension.js.map