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
exports.stripUiOnlyFields = stripUiOnlyFields;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const child_process_1 = require("child_process");
const crypto = __importStar(require("crypto"));
const semver = __importStar(require("semver"));
const storage_1 = require("./storage");
const os = __importStar(require("os"));
// ───────────────────────── Ensure onthefly-ai in a private venv ─────────────────────────
async function ensureOntheflyInPrivateVenv(context, opts) {
    const ask = opts?.askBeforeInstall ?? true;
    const storageDir = context.globalStorageUri.fsPath;
    fs.mkdirSync(storageDir, { recursive: true });
    const venvDir = path.join(storageDir, 'pyenv');
    // Pick a launcher to create the venv if it doesn't exist yet. Prefers newer versions.
    function pickLauncher() {
        const candidates = ['python3.12', 'python3.11', 'python3.10', 'python3.9', 'python3', 'python'];
        const cp = require('child_process');
        for (const c of candidates) {
            try {
                cp.execFileSync(c, ['--version'], { stdio: 'ignore' });
                return c;
            }
            catch { }
        }
        return 'python';
    }
    const pyBin = process.platform === 'win32'
        ? path.join(venvDir, 'Scripts', 'python.exe')
        : path.join(venvDir, 'bin', 'python');
    if (!fs.existsSync(pyBin)) {
        // Create venv
        require('child_process').execFileSync(pickLauncher(), ['-m', 'venv', venvDir], { stdio: 'inherit' });
    }
    try {
        const v = require('child_process')
            .execFileSync(pyBin, ['-c', 'import sys; print(".".join(map(str, sys.version_info[:3])))'], { encoding: 'utf8' })
            .trim();
        const [maj, min] = v
            .split('.')
            .map((s) => parseInt(s, 10));
        if (maj < 3 || (maj === 3 && min < 9)) {
            throw new Error(`Python ${v} found in extension venv; need >= 3.9. Please install Python 3.9+ and try again.`);
        }
    }
    catch (e) {
        vscode.window.showErrorMessage(e?.message || String(e));
        throw e; // abort ensureOntheflyInPrivateVenv
    }
    // Try to upgrade pip quietly
    try {
        require('child_process').execFileSync(pyBin, ['-m', 'pip', 'install', '--upgrade', 'pip'], { stdio: 'ignore' });
    }
    catch { }
    // Read configured minimum version
    const cfg = vscode.workspace.getConfiguration('onthefly');
    const minVersion = String(cfg.get('minPythonPackage') || '0.0.3.post1');
    // Discover installed version (if any)
    let installed = null;
    try {
        installed = require('child_process')
            .execFileSync(pyBin, ['-c',
            'import sys\n' +
                'try:\n' +
                '  import importlib.metadata as m\n' +
                'except Exception:\n' +
                '  import importlib_metadata as m\n' +
                'print(m.version("onthefly-ai"))'
        ], { encoding: 'utf8' }).trim();
    }
    catch {
        // not installed
    }
    const needInstall = !installed;
    // for future upgrades, 0.0.3.post2 will not be possible because 
    // semver.coerce('0.0.3.post2') === '0.0.3', so post1 vs post 2 won't be distinguished
    const needUpgrade = installed ? semver.lt(semver.coerce(installed), semver.coerce(minVersion)) : false;
    if (needInstall) {
        if (!ask || (await vscode.window.showInformationMessage(`Install onthefly-ai ${minVersion}+ for this extension?`, 'Install', 'Cancel')) === 'Install') {
            await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Installing onthefly-ai' }, async () => {
                require('child_process').execFileSync(pyBin, ['-m', 'pip', 'install', `onthefly-ai>=${minVersion}`], { stdio: 'inherit' });
            });
        }
        else {
            throw new Error('onthefly-ai is required but was not installed.');
        }
    }
    else if (needUpgrade) {
        if (!ask || (await vscode.window.showInformationMessage(`Upgrade onthefly-ai to at least ${minVersion}? (found ${installed})`, 'Upgrade', 'Skip')) === 'Upgrade') {
            await vscode.window.withProgress({ location: vscode.ProgressLocation.Notification, title: 'Upgrading onthefly-ai' }, async () => {
                require('child_process').execFileSync(pyBin, ['-m', 'pip', 'install', `onthefly-ai>=${minVersion}`, '-U'], { stdio: 'inherit' });
            });
        }
    }
    // Return interpreter + site-packages (if I ever need to import/inspect)
    let site = '';
    try {
        site = require('child_process').execFileSync(pyBin, ['-c', 'import sysconfig, json; print(sysconfig.get_paths()["purelib"])'], { encoding: 'utf8' }).trim();
    }
    catch { }
    return { python: pyBin, sitePackages: site };
}
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
    // Imported == NOT created/touched by the current live Python session.
    // If there is no live session yet, treat EVERYTHING as imported.
    if (!proc || !proc.stdin || !proc.stdin.writable)
        return true;
    return !nativeRunsThisSession.has(runId);
}
const optimisticEcho = {
    pause: 'paused',
    resume: 'resumed',
    save_ckpt: 'checkpointSaved',
    merge: 'merged'
};
/* ============================ Globals ============================ */
let panel = null;
let proc = null;
const CHILD_KILL_TIMEOUT_MS = 1500;
let currentRunId = null;
const seenRuns = new Set();
let currentSessionId = null; // sticky session id for log stamping
const nativeRunsThisSession = new Set(); // runs touched by THIS python session
let modelNavSelectedRunId = null;
let pythonConfigConfirmedThisSession = false;
let scriptConfigConfirmedThisSession = false;
let needDiskCleanOnNextBackend = true;
/* ============================ Activate / Deactivate ============================ */
async function activate(context) {
    context.subscriptions.push(vscode.commands.registerCommand('onthefly.showDashboard', () => openPanel(context)));
    await (0, storage_1.initStorage)(context);
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
async function hardResetSession(context, opts) {
    // 0) If a backend is alive, ask it to clean its save_dir now (best effort).
    if (proc && proc.stdin?.writable) {
        try {
            const res = await sendReq('clean_disk', { scope: 'all' }, 60000);
            post({
                type: 'log',
                level: res?.ok ? 'info' : 'warn',
                text: res?.ok
                    ? `[reset] Disk cleanup ok: removed ${res.removed?.length ?? 0} run(s)`
                    : `[reset] Disk cleanup reported an issue: ${res?.error ?? 'unknown error'}`,
            });
        }
        catch (e) {
            post({
                type: 'log',
                level: 'warn',
                text: `[reset] Disk cleanup skipped or failed: ${e?.message || String(e)}`,
            });
        }
    }
    // 1) Kill Python
    try {
        killProc();
    }
    catch { }
    // 2) Reject outstanding RPCs
    try {
        for (const [, p] of pending) {
            clearTimeout(p.timer);
            p.reject(new Error('Reset requested'));
        }
        pending.clear();
    }
    catch { }
    // 3) Clear in-memory extension state
    currentRunId = null;
    seenRuns.clear();
    currentSessionId = null;
    nativeRunsThisSession.clear();
    modelNavSelectedRunId = null;
    pythonConfigConfirmedThisSession = false;
    scriptConfigConfirmedThisSession = false;
    // 4) Reset storage
    try {
        (0, storage_1.closeStorage)();
    }
    catch { }
    await (0, storage_1.initStorage)(context);
    needDiskCleanOnNextBackend = true;
    // 5) Tell webview, if it exists
    post({ type: 'resetOk' });
    post({ type: 'runs', rows: [] });
    postStatus(false);
    if (opts?.fromUser) {
        vscode.window.setStatusBarMessage('Onthefly: session reset', 2000);
    }
}
/* ============================ Panel & HTML ============================ */
async function openPanel(context) {
    try {
        await ensureOntheflyInPrivateVenv(context, { askBeforeInstall: true });
    }
    catch (e) {
        console.warn('[onthefly] backend not ready yet:', e);
    }
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
        await (0, storage_1.initStorage)(context);
        needDiskCleanOnNextBackend = true;
        panel = null;
    });
    panel.webview.onDidReceiveMessage((m) => { onMessage(context, m); });
    panel.webview.html = getHtml(context, panel.webview, nonce);
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
function post(msg) {
    try {
        panel?.webview.postMessage(msg);
    }
    catch (e) {
        console.log('[EXT->WEB] post threw:', e);
    }
}
function postErr(e) { post({ type: 'error', text: String(e?.message || e) }); }
function postStatus(running) { post({ type: 'status', running }); }
function postCurrentSession() {
    if (currentSessionId)
        post({ type: 'fs.session.current', id: currentSessionId });
}
function havePython(context) {
    const p = context.workspaceState.get("onthefly.pythonPath" /* Keys.PythonPath */)?.trim();
    if (!p)
        return false;
    try {
        require('child_process').execFileSync(p, ['--version'], { stdio: 'ignore' });
        return true;
    }
    catch {
        return false;
    }
}
function haveScript(context) {
    const s = context.workspaceState.get("onthefly.scriptPath" /* Keys.ScriptPath */)?.trim();
    return !!s && fs.existsSync(s);
}
async function ensureTrainingConfigConfirmed(context) {
    const ws = context.workspaceState;
    const python = ws.get("onthefly.pythonPath" /* Keys.PythonPath */)?.trim();
    const script = ws.get("onthefly.scriptPath" /* Keys.ScriptPath */)?.trim();
    if (!python) {
        vscode.window.showErrorMessage('Set a Python interpreter first.');
        return false;
    }
    if (!script) {
        vscode.window.showErrorMessage('Choose a Python training script first.');
        return false;
    }
    // If both were set explicitly in this session, we’re done.
    if (pythonConfigConfirmedThisSession && scriptConfigConfirmedThisSession) {
        return true;
    }
    // Otherwise, they came from a previous session → ask once.
    const choice = await vscode.window.showWarningMessage(`Use saved Python interpreter:\n${python}\nand training script:\n${script}?`, {
        modal: true,
        detail: 'These were remembered from a previous session. If they look wrong, cancel and set them again from the dashboard.',
    }, 'Use');
    if (choice === 'Use') {
        pythonConfigConfirmedThisSession = true;
        scriptConfigConfirmedThisSession = true;
        return true;
    }
    return false;
}
const TEST_NOW_TIMEOUT_MS = 10 * 60 * 1000;
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
function isUiLogLike(v) {
    return !!v && typeof v === 'object';
}
// clean up `ts` logs -- python needs more explicit logs but user doesn't
function stripUiOnlyFields(rows) {
    const cleanOne = (r) => {
        if (!isUiLogLike(r))
            return r;
        const { ts, ...rest } = r; // drop ts
        if (typeof rest.text === 'string') {
            const mentionsStep = /\bstep\b\s*(?:[:#]\s*)?\d+(?:\s*[:#])?/i.test(rest.text);
            if (mentionsStep)
                delete rest.step; // drop step to suppress [s:N]
        }
        return rest;
    };
    return Array.isArray(rows) ? rows.map(cleanOne) : cleanOne(rows);
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
async function switchToRun(context, runId) {
    try {
        await sendReq('pause', {}, 30_000);
    }
    catch { }
    try {
        await sendReq('save_ckpt', {}, 120_000);
    }
    catch { }
    try {
        await new Promise(res => setTimeout(res, 200));
    }
    catch { }
    try {
        killProc();
    }
    catch { }
    modelNavSelectedRunId = runId;
    currentRunId = runId;
    await startRun(context);
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
            scriptConfigConfirmedThisSession = true; // <-- NEW
            vscode.window.showInformationMessage(`Training script selected: ${path.basename(p)}`);
            post({ type: 'scriptChosen', file: p });
            break;
        }
        case 'setPython': {
            const chosen = m.path || 'python';
            await ws.update("onthefly.pythonPath" /* Keys.PythonPath */, chosen);
            pythonConfigConfirmedThisSession = true; // <-- NEW
            post({ type: 'log', text: `Python set to: ${chosen}` });
            vscode.window.showInformationMessage(`Python interpreter set to: ${chosen}`);
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
                // --- Fallbacks ---
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
                post({ type: 'modelNav.select', runId: id }); // echo to webview so it knows which run is selected
                try {
                    const rows = (0, storage_1.getLogs)(id); // all phases
                    post({ type: 'logs', run_id: id, rows: stripUiOnlyFields(rows) });
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
                const selected = modelNavSelectedRunId && modelNavSelectedRunId.trim();
                const active = currentRunId && currentRunId.trim();
                const rid = requested || selected || active || '';
                let rows = rid ? (0, storage_1.getLogs)(rid, phase) : [];
                if ((!rows || rows.length === 0) && currentSessionId) {
                    rows = (0, storage_1.getLogsBySession)(currentSessionId, phase);
                }
                post({ type: 'logs', run_id: rid || null, rows: stripUiOnlyFields(rows) });
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
        case 'pause': {
            try {
                // 1) Ask the backend to pause (finish current step, flush state, etc.)
                await sendReq('pause', {}, 30_000); // backend will do its own checkpointing if it wants
                // 2) Figure out which run this pause applies to
                const runId = (modelNavSelectedRunId && modelNavSelectedRunId.trim()) ||
                    (currentRunId && currentRunId.trim()) ||
                    null;
                if (!runId) {
                    post({ type: 'warn', text: '[pause] No runId available to attach checkpoint to.' });
                    break;
                }
                // 3) Explicitly save a checkpoint and persist it in storage
                try {
                    const ck = await sendReq('save_ckpt', {}, 120_000);
                    if (ck?.path) {
                        const stepNum = Number(ck.step) || 0;
                        const ckptId = `${runId}:${stepNum}:${Date.now()}`; // same style as elsewhere
                        (0, storage_1.insertCheckpoint)(ckptId, runId, stepNum, String(ck.path));
                        post({
                            type: 'log',
                            level: 'info',
                            text: `[pause] checkpoint saved for run ${runId} at step ${stepNum}`,
                        });
                    }
                    else {
                        post({
                            type: 'warn',
                            text: '[pause] save_ckpt returned no path; checkpoint not recorded in DB.',
                        });
                    }
                }
                catch (ckErr) {
                    console.warn('[onthefly] pause: save_ckpt / insertCheckpoint failed:', ckErr);
                    post({
                        type: 'warn',
                        text: `[pause] Failed to save/persist checkpoint: ${ckErr?.message || String(ckErr)}`,
                    });
                }
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'resume': {
            try {
                const requested = String(m.runId || '').trim();
                if (requested)
                    modelNavSelectedRunId = requested;
                const target = requested ||
                    currentRunId ||
                    modelNavSelectedRunId ||
                    null;
                // If a process exists AND user wants a different run → clean switch
                if (proc && target && currentRunId && target !== currentRunId) {
                    post({ type: 'log', text: `Switching active run: ${currentRunId} → ${target}` });
                    await switchToRun(context, target);
                    post({ type: 'resumed', payload: { run_id: target } });
                    break;
                }
                // If the process exists and we are already on the correct run → send resume
                if (proc) {
                    sendCtl({ cmd: 'resume' });
                    break;
                }
                // No process → start a run (startRun handles interpreter/script validation + user confirmation)
                post({ type: 'log', text: 'Starting training (via Resume)...' });
                await startRun(context);
            }
            catch (e) {
                postErr(e);
                postStatus(false);
            }
            break;
        }
        case 'testNow': {
            const requested = m.runId;
            const candidate = (requested && requested.trim())
                || (modelNavSelectedRunId && modelNavSelectedRunId.trim())
                || (currentRunId && currentRunId.trim())
                || null;
            try {
                if (!candidate) {
                    post({ type: 'log', text: 'Select a model in Model Nav before testing.' });
                    break;
                }
                const runs = (0, storage_1.listRuns)();
                const friendly = runs.find(r => r.run_id === candidate)?.name || candidate;
                const choice = await vscode.window.showWarningMessage(`This will override the final tested model for this session. Test model "${friendly}" now?`, {
                    modal: true,
                    detail: 'Confirming will test immediately and replace the stored final checkpoint for this session. You can continue training this and other models after.'
                }, 'Confirm Test');
                if (choice !== 'Confirm Test') {
                    post({ type: 'log', text: 'Test cancelled.' });
                    break;
                }
                const target = await ensureProcOnRun(context, candidate);
                try {
                    await sendReq('pause', {}, 30_000);
                }
                catch (pauseErr) {
                    console.warn('[onthefly] pause before test failed', pauseErr);
                }
                post({ type: 'testNow', status: 'pending', run_id: target });
                const data = await sendReq('test_now', { label: 'final' }, TEST_NOW_TIMEOUT_MS);
                const normalizedLoss = Number.isFinite(Number(data?.avg_loss)) ? Number(data.avg_loss) : null;
                const sessionIdRaw = (data?.session_id && String(data.session_id)) || currentSessionId || '';
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
                post({
                    type: 'testNow',
                    status: 'completed',
                    run_id: target,
                    data: { ...data, avg_loss: normalizedLoss, ckpt_path: ckptPath },
                });
            }
            catch (e) {
                const fallback = candidate || modelNavSelectedRunId || currentRunId || null;
                post({ type: 'testNow', status: 'error', run_id: fallback, error: String(e?.message || e) });
                postErr(e);
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
                    modelNavSelectedRunId ||
                    currentRunId ||
                    pickAnchorRunForAction(payloadIn) ||
                    '';
                if (!target) {
                    postErr('No run selected to fork from.');
                    break;
                }
                // Ensure backend on target…
                if (!proc || !proc.stdin?.writable) {
                    await ensureMaintenanceBackend(context, target);
                    modelNavSelectedRunId = target;
                    post({ type: 'modelNav.select', runId: target });
                }
                else if (currentRunId && currentRunId !== target) {
                    await switchToRun(context, target);
                }
                try {
                    await sendReq('pause', {}, 30_000);
                }
                catch { }
                let ckptPath = (0, storage_1.latestCheckpointForRun)(target)?.path;
                if (!ckptPath) {
                    try {
                        const ck = await sendReq('save_ckpt', {}, 120_000);
                        ckptPath = ck?.path;
                    }
                    catch { }
                }
                // Forward region/hparams through to backend
                const data = await sendReq('fork', {
                    parent_run_id: target,
                    owner_run_id: target,
                    ...payloadIn,
                    ...(ckptPath ? { parent_ckpt_path: ckptPath } : {}),
                }, 120_000);
                const child = String(data?.new_run || data?.child_id || data?.run_id || '').trim();
                if (child) {
                    currentRunId = child;
                    modelNavSelectedRunId = child;
                    nativeRunsThisSession.add(child);
                    post({ type: 'modelNav.select', runId: child });
                    if (Array.isArray(data?.subset_indices)) {
                        try {
                            (0, storage_1.setRunSubset)(child, data.subset_indices.map((n) => Number(n) | 0));
                        }
                        catch { }
                    }
                    try {
                        sendCtl({ cmd: 'resume' });
                    }
                    catch { }
                }
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'merge': {
            try {
                const payload = m.payload;
                // Ensure we have a backend anchored to something sensible for merge ops.
                if (!proc || !proc.stdin?.writable) {
                    const anchor = pickAnchorRunForAction(payload);
                    if (!anchor) {
                        postErr('No runs available to anchor the merge. Select a run or include parents[].');
                        break;
                    }
                    await ensureMaintenanceBackend(context, anchor);
                    modelNavSelectedRunId = anchor;
                    post({ type: 'modelNav.select', runId: anchor });
                }
                else {
                    try {
                        await sendReq('pause', {}, 30_000);
                    }
                    catch { }
                }
                // Preflight: all parents must have checkpoints.
                const parents = Array.isArray(payload.parents) ? payload.parents : [];
                for (const p of parents) {
                    if (!(0, storage_1.latestCheckpointForRun)(p)) {
                        await ensureProcOnRun(context, p);
                        try {
                            await sendReq('pause', {}, 30_000);
                        }
                        catch { }
                        try {
                            const ck = await sendReq('save_ckpt', {}, 120_000);
                            // belt-and-suspenders: persist immediately in case the backend doesn't emit the JSON event
                            if (ck?.path)
                                (0, storage_1.insertCheckpoint)(`${p}:${ck.step}:${Date.now()}`, p, Number(ck.step) || 0, String(ck.path));
                        }
                        catch { }
                    }
                }
                // now run the existing preflight
                const missing = parents.filter(p => !(0, storage_1.latestCheckpointForRun)(p));
                // Perform the merge.
                const data = await sendReq('merge', payload, 120000);
                // --- immediately navigate to the merged child & start running (mirror fork behavior) ---
                const child = String((data && (data.new_run ?? data.child_id ?? data.run_id)) || '').trim();
                if (child) {
                    currentRunId = child;
                    modelNavSelectedRunId = child;
                    nativeRunsThisSession.add(child);
                    post({ type: 'modelNav.select', runId: child });
                    // If backend provided a subset (optional), persist it like we do on fork.
                    if (Array.isArray(data?.subset_indices)) {
                        try {
                            (0, storage_1.setRunSubset)(child, data.subset_indices.map((n) => Number(n) | 0));
                        }
                        catch { }
                    }
                    // Nudge the backend to resume the new merged run.
                    try {
                        sendCtl({ cmd: 'resume' });
                    }
                    catch { }
                }
                else {
                    // If the backend didn't return an explicit child id, the usual 'newRun' event
                    // will still keep things in sync. No-op here.
                }
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'dist_health': {
            try {
                const rid = m.runId;
                const target = await ensureProcOnRun(context, rid);
                modelNavSelectedRunId = target;
                post({ type: 'modelNav.select', runId: target });
                await runHealth('dist_health', { require_all_ranks: false, sample_weights: 0 }, 'distHealth', 30_000, target);
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'throughput_health': {
            try {
                const rid = m.runId;
                const target = await ensureProcOnRun(context, rid);
                modelNavSelectedRunId = target;
                post({ type: 'modelNav.select', runId: target });
                await runHealth('throughput_health', { budget_steps: 2, micro_backward: false }, 'throughputHealth', 45_000, target);
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'activations_health': {
            try {
                const rid = m.runId;
                const target = await ensureProcOnRun(context, rid);
                modelNavSelectedRunId = target;
                post({ type: 'modelNav.select', runId: target });
                await runHealth('activations_health', { budget_steps: 2, topk: 12, eps: 1e-3 }, 'activationsHealth', 60_000, target);
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'numerics_health': {
            try {
                const rid = m.runId;
                const target = await ensureProcOnRun(context, rid);
                modelNavSelectedRunId = target;
                post({ type: 'modelNav.select', runId: target });
                await runHealth('numerics_health', { sample_layers: 25 }, 'numericsHealth', 30_000, target);
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'determinism_health': {
            try {
                const rid = m.runId;
                const target = await ensureProcOnRun(context, rid);
                modelNavSelectedRunId = target;
                post({ type: 'modelNav.select', runId: target });
                await runHealth('determinism_health', {}, 'determinismHealth', 20_000, target);
            }
            catch (e) {
                postErr(e);
            }
            break;
        }
        case 'generateReport': {
            try {
                let runId = m.runId ?? undefined;
                if (!runId || !String(runId).trim()) {
                    const anchor = pickAnchorRunForAction(); // choose something sensible if not provided
                    if (!anchor) {
                        post({ type: 'error', text: 'No active run selected for report.' });
                        break;
                    }
                    runId = anchor;
                }
                runId = String(runId).trim();
                // If nothing is running yet, spin up a maintenance backend on the target.
                if (!proc || !proc.stdin?.writable) {
                    await ensureMaintenanceBackend(context, runId);
                    modelNavSelectedRunId = runId;
                    post({ type: 'modelNav.select', runId });
                }
                else if (currentRunId && currentRunId !== runId) {
                    // Running but on a different model → switch cleanly.
                    await switchToRun(context, runId);
                }
                await sendReq('pause', {}, 30_000);
                const subset = (0, storage_1.getRunSubset)(String(runId));
                const subset_on = 'train';
                const data = await sendReq('generate_report', {
                    owner_run_id: runId,
                    subset_indices: subset.length ? subset : undefined,
                    subset_on,
                    reqId: m.reqId
                });
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
                post({
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
                // Grab every run_id we know and treat that as the nav list.
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
                    // 4) paranoia: ensure every owner has at least *one* checkpoint row pre-bundle.
                    //    if any run is missing, scan the save_dir heuristically and backfill.
                    try {
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
                await context.globalState.update(LAST_EXPORT_DIR_KEY, isDir ? chosen : path.dirname(chosen));
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
                    post({ type: 'log', text: `[print] session ${s.session_id}: ${(0, storage_1.getLogsBySession)(String(s.session_id)).length} logs` });
                }
                // Notify UI & refresh runs
                post({ type: 'sessionLoaded' });
                const runs = (0, storage_1.listRuns)();
                post({ type: 'runs', rows: runs });
                try {
                    const sessions = (0, storage_1.listSessions)();
                    post({ type: 'fs.session.list', items: sessions });
                }
                catch { }
                // Hydrate logs for each run so the webview has something to render immediately
                for (const r of runs) {
                    try {
                        const rows = (0, storage_1.getLogs)(String(r.run_id)); // all phases
                        post({ type: 'logs', run_id: String(r.run_id), rows: stripUiOnlyFields(rows) });
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
                postErr(e);
            }
            break;
        }
        case 'resetAll': {
            const pick = await vscode.window.showWarningMessage('This will erase all models from memory, are you sure you want to refresh?', {
                modal: true,
                detail: 'Any running training will be stopped. Nothing will be saved if you have not exported session.',
            }, 'Erase & Refresh');
            if (pick !== 'Erase & Refresh') {
                post({ type: 'log', text: 'Refresh cancelled.' });
                break;
            }
            await hardResetSession(context, { fromUser: true });
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
async function startRun(context) {
    if (proc) {
        vscode.window.showWarningMessage('A run is already active.');
        return;
    }
    // Ensure onthefly-ai is available in our private venv (required)
    try {
        await ensureOntheflyInPrivateVenv(context, { askBeforeInstall: true });
    }
    catch (e) {
        vscode.window.showErrorMessage(`Backend not available: ${e?.message || e}`);
        postStatus(false);
        return;
    }
    // confirm we’re allowed to use the stored interpreter + script
    if (!(await ensureTrainingConfigConfirmed(context))) {
        postStatus(false);
        return;
    }
    // Still keep the runtime sanity checks (path exists, python runs).
    if (!havePython(context)) {
        vscode.window.showErrorMessage('Set a Python interpreter first.');
        postStatus(false);
        return;
    }
    if (!haveScript(context)) {
        vscode.window.showErrorMessage('Choose a Python training script first.');
        postStatus(false);
        return;
    }
    const ws = context.workspaceState;
    const python = ws.get("onthefly.pythonPath" /* Keys.PythonPath */)?.trim();
    const script = ws.get("onthefly.scriptPath" /* Keys.ScriptPath */)?.trim();
    if (!python) {
        vscode.window.showErrorMessage('Set a Python interpreter first.');
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
    const { forkMax, mergeMax } = computeForkMergeCounters();
    const args = ['-u', script];
    post({ type: 'log', text: `Spawning: ${python} ${args.join(' ')}` });
    const resumeRunId = (modelNavSelectedRunId && modelNavSelectedRunId.trim()) ||
        (currentRunId && currentRunId.trim()) ||
        null;
    const imported = resumeRunId ? isRunImportedForThisSession(resumeRunId) : false;
    const resume = resumeRunId ? (0, storage_1.latestCheckpointForRun)(resumeRunId) : null;
    if (resumeRunId && !resume) {
        post({ type: 'warn', text: `[resume] No checkpoint found for run ${resumeRunId}; starting fresh.` });
    }
    const envBlock = {
        ...process.env,
        ...(resumeRunId ? { ONTHEFLY_RESUME_RUN_ID: resumeRunId } : {}),
        ...(resume && resume.path ? {
            ONTHEFLY_INIT_CKPT: resume.path,
            ONTHEFLY_INIT_STEP: String(resume.step ?? 0),
        } : {}),
        ONTHEFLY_FORK_COUNTER_INIT: String(forkMax || 0),
        ONTHEFLY_MERGE_COUNTER_INIT: String(mergeMax || 0),
    };
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
async function runHealth(cmd, payload, eventType, timeoutMs = 60_000, forRunId) {
    const run_id = forRunId || modelNavSelectedRunId || currentRunId || null;
    try {
        // tell the webview to show a loading placeholder
        post({ type: eventType, run_id, pending: true });
        try {
            await sendReq('pause', {}, 30_000);
        }
        catch { }
        const data = await sendReq(cmd, payload, timeoutMs);
        // normal success event (what your webview already handles)
        post({ type: eventType, cmd, payload, data, run_id });
    }
    catch (e) {
        // let the UI show an error if needed
        post({ type: eventType, run_id, error: String(e?.message || e) });
        postErr(e);
    }
}
async function ensureProcOnRun(context, requestedRunId) {
    const target = String(requestedRunId || modelNavSelectedRunId || currentRunId || '').trim();
    if (!target)
        throw new Error('No run selected. Open Model Nav and pick a run first.');
    // Already on target?
    if (proc && currentRunId === target)
        return target;
    // Switch if a different run is active
    if (proc && currentRunId && currentRunId !== target) {
        await switchToRun(context, target);
        return target;
    }
    // No proc → start on the target (same behavior as Resume/Report)
    modelNavSelectedRunId = target;
    await startRun(context);
    return target;
}
function pickAnchorRunForAction(hint) {
    const fromRunId = typeof hint?.runId === 'string' && hint.runId.trim() ? String(hint.runId).trim() : null;
    const fromParents = Array.isArray(hint?.parents) && hint.parents.length ? String(hint.parents[0]) : null;
    if (fromRunId)
        return fromRunId;
    if (fromParents)
        return fromParents;
    if (modelNavSelectedRunId?.trim())
        return modelNavSelectedRunId.trim();
    if (currentRunId?.trim())
        return currentRunId.trim();
    try {
        const first = (0, storage_1.listRuns)()[0]?.run_id;
        if (first)
            return String(first);
    }
    catch { }
    return null;
}
async function ensureMaintenanceBackend(context, anchor) {
    // Start/attach to backend on the chosen run, then pause to avoid races.
    await ensureProcOnRun(context, anchor);
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
        const hasStepInText = /\bstep\b\s*(?:[:#]\s*)?\d+(?:\s*[:#])?/i.test(text);
        const stepForUI = hasStepInText ? null : step;
        try {
            // keep full fidelity in storage
            (0, storage_1.insertLog)({ run_id, session_id, level, text, phase, step, epoch, ts: tsMs });
        }
        catch { }
        // suppress redundant "s:" in the webview
        post({ type: 'log', run_id, session_id, level, text, phase, step: stepForUI, epoch });
        return;
    }
    // swallow unsolicited reportData (RPC handles reply)
    if (obj?.type === 'reportData') {
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
        const num = (value) => (Number.isFinite(value) ? Number(value) : null);
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
            // modelNavSelectedRunId = id;   // keep selection in sync with what the UI shows
            // If storage only supports single parent, store the first (primary)
            const primaryParent = (parents && parents[0]) ?? null;
            const existingRuns = (0, storage_1.listRuns)();
            const existsInDb = existingRuns.some(r => r.run_id === id);
            if (!existsInDb) {
                (0, storage_1.insertRun)(id, project, name, primaryParent);
            }
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
    if (obj?.type === 'auto_test_complete') {
        const runId = String(obj.run_id || currentRunId || '').trim();
        const ckptPath = obj.ckpt_path ? String(obj.ckpt_path) : '';
        const step = Number(obj.step) || 0;
        if (runId && ckptPath) {
            try {
                const ckptId = `${runId}:${step}:${Date.now()}`;
                (0, storage_1.insertCheckpoint)(ckptId, runId, step, ckptPath);
                post({
                    type: 'log',
                    level: 'info',
                    text: `[auto-test] checkpoint recorded for run ${runId} at step ${step}`,
                });
            }
            catch (e) {
                console.warn('[onthefly] failed to persist auto-test checkpoint', e);
                post({
                    type: 'log',
                    level: 'warn',
                    text: `[auto-test] failed to persist checkpoint for run ${runId}: ${String(e?.message || e)}`,
                });
            }
        }
        else {
            post({
                type: 'log',
                level: 'warn',
                text: '[auto-test] auto_test_complete event missing run_id or ckpt_path; not recorded.',
            });
        }
        // We already emitted a generic test_complete earlier, and that fell
        // through to the default `post(obj)` below. This extra event is
        // extension-only plumbing; no need to forward it again.
        return;
    }
    // Default: forward to webview, then extra persistence for other events
    post(obj);
    try {
        switch (obj?.type) {
            case 'session_started': {
                currentRunId = obj.run_id || currentRunId;
                if (obj.session_id || obj.sessionId) {
                    currentSessionId = String(obj.session_id || obj.sessionId);
                    postCurrentSession();
                }
                // Backend has finished booting and announced the session.
                // Safe to call clean_disk once for this logical backend.
                if (needDiskCleanOnNextBackend && proc && proc.stdin?.writable) {
                    needDiskCleanOnNextBackend = false;
                    (async () => {
                        try {
                            const res = await sendReq('clean_disk', { scope: 'all' }, 60000);
                            post({
                                type: 'log',
                                level: res?.ok ? 'info' : 'warn',
                                text: res?.ok
                                    ? '[startup] clean_disk: removed old runs from save_dir'
                                    : `[startup] clean_disk reported an issue: ${res?.error ?? 'unknown error'}`,
                            });
                        }
                        catch (e) {
                            post({
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
//helpers
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
//# sourceMappingURL=extension.js.map