import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as net from 'net';
import * as crypto from 'crypto';
import * as semver from 'semver';

import {
  initStorage,
  insertRun,
  insertMetric,
  insertCheckpoint,
  upsertSummary,
  upsertFinalModel,
  listRuns,
  getRunRows,
  setRunSubset,
  getRunSubset,
  upsertReportLossDist,
  getReportLossDist,
  closeStorage,
  insertLog,
  insertTestMetric,
  getTestRows,
  getLogs,
  runsForSession,
  getLogsBySession,
  listSessions,
  exportBundle as storageExportBundle,
  loadBundle as storageLoadBundle,
  latestCheckpointForRun
} from './storage';
import * as os from 'os';


// Stash latest config (webview can set it before or after starting the run)
/* ============================ RPC state ============================ */

type Pending = {
  resolve: (v: any) => void;
  reject: (e: any) => void;
  timer: NodeJS.Timeout;
};
const pending = new Map<string, Pending>();

const DASHBOARD_PORT = Number(process.env.ONTHEFLY_DASHBOARD_PORT || '47621');
let trainerSocket: net.Socket | null = null;
let trainerBuffer = '';
let trainerServer: net.Server | null = null;

function sendReq(cmd: string, payload: any = {}, timeoutMs = 15000): Promise<any> {
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

function isRunImportedForThisSession(runId: string): boolean {
  // Imported == NOT created/touched by the current live Python session.
  // If there is no live session yet, treat EVERYTHING as imported.
  if (!trainerSocket || trainerSocket.destroyed) return true;
  return !nativeRunsThisSession.has(runId);
}

/* ============================ Trainer socket ============================ */

function ensureTrainerServer() {
  if (trainerServer) return;

  trainerServer = net.createServer((socket) => {
    if (trainerSocket && !trainerSocket.destroyed) {
      socket.destroy();
      vscode.window.showWarningMessage(
        'Another Trainer tried to connect while one is active. Close the existing run first.'
      );
      return;
    }

    // Handle new trainer connection in an async block so we can await the reset
    (async () => {
      // If we've ever had a trainer before, a new one means "new session" → hard reset.
      if (hasTrainerConnectedOnce && lastExtensionContext) {
        try {
          await hardResetSession(lastExtensionContext, { fromUser: false });
        } catch (e) {
          console.warn('[onthefly] automatic session reset on new trainer failed:', e);
        }
      }

      hasTrainerConnectedOnce = true;

      // Now wire up the newly connected Trainer
      trainerSocket = socket;
      trainerBuffer = '';
      socket.setEncoding('utf8');
      socket.on('data', (chunk: string) => handleTrainerData(chunk));
      socket.on('error', (err: any) => {
        post({ type: 'error', text: `[trainer] ${err?.message || err}` });
      });
      socket.on('close', () => {
        post({ type: 'log', text: 'Trainer disconnected.' });
        // NOTE: this does NOT reset DB/session/dashboard anymore; it just tears down the socket.
        disconnectTrainer(false);
      });

      postStatus(true);
      post({ type: 'log', text: 'Trainer connected. Streaming events live.' });

      // if user had a run selected, seed + attach immediately
      const target =
        (modelNavSelectedRunId && modelNavSelectedRunId.trim()) ||
        (currentRunId && currentRunId.trim()) ||
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

  trainerServer.on('error', (err: any) => {
    const msg = `OnTheFly dashboard could not listen on port ${DASHBOARD_PORT}: ${err?.message || err}`;
    vscode.window.showErrorMessage(msg);
    post({ type: 'error', text: msg });
  });

  trainerServer.listen(DASHBOARD_PORT, '127.0.0.1', () => {
    post({
      type: 'log',
      text: `Waiting for Trainer connections on localhost:${DASHBOARD_PORT}. Run your script to attach.`,
    });
  });
}

function trainerActive(): boolean {
  return Boolean(trainerSocket && !trainerSocket.destroyed);
}

function requireTrainerConnection(): boolean {
  if (!trainerActive()) {
    vscode.window.showErrorMessage('No Trainer connection. Run your training script with an OnTheFlyTrainer to stream data.');
    postStatus(false);
    return false;
  }
  return true;
}

function shutdownTrainerServer() {
  if (trainerServer) {
    try { trainerServer.close(); } catch {}
    trainerServer = null;
  }
  disconnectTrainer(false);
}

function disconnectTrainer(notify = true) {
  if (trainerSocket) {
    try { trainerSocket.destroy(); } catch {}
    trainerSocket = null;
  }
  trainerBuffer = '';
  setRunActivity(null);
  pauseInFlight = false;
  resumeInFlight = false;

  for (const [, p] of pending) {
    clearTimeout(p.timer);
    p.reject(new Error('Trainer disconnected'));
  }
  pending.clear();

  if (notify) {
    post({ type: 'log', text: 'Trainer connection closed.' });
  }
  postStatus(false);
}

function handleTrainerData(chunk: string) {
  trainerBuffer += chunk;
  let idx: number;
  while ((idx = trainerBuffer.indexOf('\n')) >= 0) {
    const line = trainerBuffer.slice(0, idx);
    trainerBuffer = trainerBuffer.slice(idx + 1);
    if (line.trim()) {
      handleLine(line);
    }
  }
}


/* ============================ Persisted keys ============================ */

/* ============================ Webview -> Ext messages ============================ */

type CompareView = 'train' | 'test' | 'info'; //not using but hypothetically, ifwe wanted to display test/train logs separate, or do cross-session compare

type WebMsg =
  | { command: 'pause' }
  | { command: 'resume' }
  | { command: 'testNow'; runId?: string }
  | { command: 'fork'; payload?: any }
  | { command: 'merge'; payload: { paths?: string[]; parents?: string[]; strategy?: string; new_name?: string } }
  | { command: 'exportSession' }
  | { command: 'loadSession' }
  | { command: 'requestRuns' }
  | { command: 'requestRows'; runId: string }
  | { command: 'requestReport'; runId: string }
  | { command: 'generateReport'; runId?: string; reqId?: number }
  | { command: 'dist_health' }
  | { command: 'throughput_health' }
  | { command: 'numerics_health' }
  | { command: 'activations_health' }
  | { command: 'determinism_health' }
  | { command: 'throughput_health' }
  | { command: 'notify'; level?: 'info' | 'warn' | 'error'; text: string }
  | { command: 'modelNav.select'; runId: string }
  | { command: 'exportChart'; filename?: string; dataUrl: string }
  | { command: 'resetAll' }
  | { command: 'requestLogs'; runId: string; phase?: CompareView }
  | { command: 'requestTestRows'; runId: string }
  | {
      command: 'exportSubset';
      runId?: string;
      format?: 'parquet' | 'csv' | 'feather';
      region?: { minLoss: number; maxLoss: number };
      subset_indices?: number[];
    };

/* ============================ Control messages to Python ============================ */

type Ctl =
  | { cmd: 'pause' }
  | { cmd: 'resume' }
  | { cmd: 'save_ckpt' }
  | { cmd: 'fork'; payload?: any }
  | { cmd: 'merge'; payload: { paths?: string[]; parents?: string[]; strategy?: string; new_name?: string } };

type RunActivityState = 'running' | 'paused' | null;

const optimisticEcho: Record<string, string> = {
  pause: 'paused',
  resume: 'resumed',
  save_ckpt: 'checkpointSaved',
  merge: 'merged'
};

/* ============================ Globals ============================ */

let panel: vscode.WebviewPanel | null = null;
let currentRunId: string | null = null;
const seenRuns = new Set<string>();
let currentSessionId: string | null = null;   // sticky session id for log stamping
const nativeRunsThisSession = new Set<string>(); // runs touched by THIS python session
let modelNavSelectedRunId: string | null = null;
let runActivityState: RunActivityState = null;
let pauseInFlight = false;
let resumeInFlight = false;
let needDiskCleanOnNextTrainer = true;
let lastExtensionContext: vscode.ExtensionContext | null = null;
let hasTrainerConnectedOnce = false;

/* ============================ Types ============================ */

type StepRow = {
  run_id: string;
  step: number;
  epoch?: number | null;
  loss: number | null;
  val_loss: number | null;
  accuracy?: number | null;
  lr?: number | null;
  grad_norm?: number | null;
  weight_norm?: number | null;
  activation_zero_frac?: number | null;
  throughput?: number | null;
  mem_vram?: number | null;
  gpu_util?: number | null;
  ts?: number | null;
};

/* ============================ Activate / Deactivate ============================ */

export async function activate(context: vscode.ExtensionContext) {
  lastExtensionContext = context;
  context.subscriptions.push(
    vscode.commands.registerCommand('onthefly.showDashboard', () => openPanel(context)),
  );
  context.subscriptions.push(
    vscode.window.registerWebviewPanelSerializer('ontheflyDashboard', {
      async deserializeWebviewPanel(webviewPanel: vscode.WebviewPanel) {
        await revivePanel(context, webviewPanel);
      },
    }),
  );
  await initStorage(context);
}

export function deactivate() {
  shutdownTrainerServer();
  try { closeStorage({ retainFile: true }); } catch {}
}

async function hardResetSession(context: vscode.ExtensionContext, opts?: { fromUser?: boolean }) {

  // 0) If a backend is alive, ask it to clean its save_dir now (best effort).
  if (trainerSocket && !trainerSocket.destroyed) {
    try {
      const res = await sendReq('clean_disk', { scope: 'all' }, 60000);
      post({
        type: 'log',
        level: res?.ok ? 'info' : 'warn',
        text: res?.ok
          ? `[reset] Disk cleanup ok: removed ${res.removed?.length ?? 0} run(s)`
          : `[reset] Disk cleanup reported an issue: ${res?.error ?? 'unknown error'}`,
      });
    } catch (e: any) {
      post({
        type: 'log',
        level: 'warn',
        text: `[reset] Disk cleanup skipped or failed: ${e?.message || String(e)}`,
      });
    }
  }
  // 1) Disconnect trainer (does not stop user's script; it will reconnect if still running)
  disconnectTrainer(true);

  // 3) Clear in-memory extension state
  currentRunId = null;
  seenRuns.clear();
  currentSessionId = null;
  nativeRunsThisSession.clear();
  modelNavSelectedRunId = null;
  runActivityState = null;
  pauseInFlight = false;
  resumeInFlight = false;

  // 4) Reset storage
  try { closeStorage({ retainFile: false }); } catch {}
  await initStorage(context);
  
  needDiskCleanOnNextTrainer = true;

  // 5) Tell webview, if it exists
  post({ type: 'resetOk' });
  post({ type: 'runs', rows: [] });
  postStatus(false);

  if (opts?.fromUser) {
    vscode.window.setStatusBarMessage('Onthefly: session reset', 2000);
  }
}


/* ============================ Panel & HTML ============================ */

function getLocalResourceRoots(context: vscode.ExtensionContext): vscode.Uri[] {
  return [
    vscode.Uri.file(context.extensionPath),
    vscode.Uri.file(path.join(context.extensionPath, 'src', 'webview')),
    vscode.Uri.file(path.join(context.extensionPath, 'media')),
    vscode.Uri.file(path.join(context.extensionPath, 'node_modules')),
  ];
}

async function openPanel(context: vscode.ExtensionContext) {
  if (panel) { panel.reveal(vscode.ViewColumn.Active); return; }

  const newPanel = vscode.window.createWebviewPanel(
    'ontheflyDashboard',
    'OnTheFly',
    vscode.ViewColumn.Active,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
      localResourceRoots: getLocalResourceRoots(context),
    }
  );
  configurePanel(context, newPanel);
}

async function revivePanel(context: vscode.ExtensionContext, webviewPanel: vscode.WebviewPanel) {
  configurePanel(context, webviewPanel);
}

function configurePanel(context: vscode.ExtensionContext, webviewPanel: vscode.WebviewPanel) {
  ensureTrainerServer();
  panel = webviewPanel;
  panel.webview.options = {
    enableScripts: true,
    localResourceRoots: getLocalResourceRoots(context),
  };

  const nonce = getNonce();
  let webviewVisible = panel.visible;

  panel.onDidChangeViewState(({ webviewPanel }) => {
    webviewVisible = webviewPanel.visible;
    if (webviewVisible) {
      // when user returns, cheaply re-sync UI (no heavy streams)
      try { post({ type: 'runs', rows: listRuns() }); } catch {}
      postStatus(trainerActive());
    }
  });

  panel.onDidDispose(() => {
    // Just drop the reference; let deactivate() handle shutdown.
    if (panel === webviewPanel) {
      panel = null;
    }
  });

  panel.webview.onDidReceiveMessage((m: any) => { onMessage(context, m); });

  panel.webview.html = getHtml(context, panel.webview, nonce);
}

function getHtml(context: vscode.ExtensionContext, webview: vscode.Webview, nonce: string) {
  const htmlPath = [
    path.join(context.extensionPath, 'dashboard.html'),
    path.join(context.extensionPath, 'src', 'webview', 'dashboard.html'),
  ].find(fs.existsSync);

  if (!htmlPath) return `<!doctype html><html><body><h2>dashboard.html not found</h2></body></html>`;

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
  const flyUri  = webview.asWebviewUri(vscode.Uri.file(flyPath)).toString();


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
    .replace(/__FLY__/g, flyUri || '');;

  if (!/__DASHBOARD_JS__/.test(html) && jsUri) {
    html = html.replace(
      /<script\s+[^>]*src=["']\.\/dashboard\.(ts|js)["'][^>]*><\/script>/,
      `<script nonce="${nonce}" src="${jsUri}"></script>`
    );
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
  } else {
    html = html.replace(/<head>/i, `<head>\n${cspMeta}\n`);
  }

  if (!chartUri) console.warn('[onthefly] Chart.js not found at media/chart.umd.js or node_modules/chart.js/dist/chart.umd.js');
  if (!jsUri) console.warn('[onthefly] dashboard.js not found. Buttons will not work.');
  if (!runStateUri) console.warn('[onthefly] run_state.js not found. Lineage store will not load.');
  if (!ipcControlsUri) console.warn('[onthefly] ipc_controls.js not found. Button wiring will not initialize.');
  // if (!autoforkPanelUri) console.warn('[onthefly] autofork_panel.js not found. AutoFork UI will be disabled.');
  if (!dagLayoutUri) console.warn('[onthefly] dag_layout.js not found. DAG will fall back or render linearly.');
  if (!dagRenderUri) console.warn('[onthefly] dag_render.js not found. DAG overlay will not render.');
  if (!logBufferUri) console.warn('[onthefly] log_buffer.js not found. Log textarea will not update.');
  if (!reportSelectionUri) console.warn('[onthefly] report_selection.js not found. Report selection overlay will not load.');

  return html;
}

function getNonce() {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) text += possible.charAt(Math.floor(Math.random() * possible.length));
  return text;
}

/* ============================ Utilities ============================ */

function post(msg: any) {
  
  try { panel?.webview.postMessage(msg); } catch (e) { console.log('[EXT->WEB] post threw:', e); }
}
function postErr(e: any) { post({ type: 'error', text: String(e?.message || e) }); }

function postStatus(connected: boolean) {
  const activity = runActivityState;
  const running = connected && activity === 'running';
  const paused  = connected && activity === 'paused';

  post({
    type: 'status',
    connected,
    running,   // "is actively training"
    paused,    // "trainer attached but paused"
    run_id: currentRunId || null,
  });
}

function setRunActivity(state: RunActivityState) {
  runActivityState = state;
  postStatus(trainerActive());
}

function postCurrentSession() {
  if (currentSessionId) post({ type: 'fs.session.current', id: currentSessionId });
}

const TEST_NOW_TIMEOUT_MS = 10 * 60 * 1000;
const LAST_EXPORT_DIR_KEY = 'onthefly.lastExportDir';

function ensurePng(name: string) {
  return name.toLowerCase().endsWith('.png') ? name : `${name}.png`;
}

function systemDownloadsDir(): string {
  const home = os.homedir();

  // Linux: respect XDG user dirs if present
  try {
    const cfg = path.join(home, '.config', 'user-dirs.dirs');
    if (fs.existsSync(cfg)) {
      const txt = fs.readFileSync(cfg, 'utf8');
      const m = txt.match(/XDG_DOWNLOAD_DIR="?(.+?)"?$/m);
      if (m && m[1]) {
        let p = m[1].replace('$HOME', home).replace(/^"|"$/g, '');
        if (fs.existsSync(p)) return p;
      }
    }
  } catch {}

  // Common default on macOS / Linux / Windows
  const candidates = [
    path.join(home, 'Downloads'),
    path.join(home, 'downloads'),
  ];
  for (const p of candidates) if (fs.existsSync(p)) return p;

  // Fallback: user home
  return home;
}

function getInitialExportDir(context: vscode.ExtensionContext): string {
  const remembered = context.globalState.get<string>(LAST_EXPORT_DIR_KEY);
  if (remembered && fs.existsSync(remembered)) return remembered;
  return systemDownloadsDir();
}

function timestampSlug() {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

type UiLogLike = { text?: string; step?: number | null; ts?: number | null };

function isUiLogLike(v: unknown): v is UiLogLike & Record<string, any> {
  return !!v && typeof v === 'object';
}
// clean up `ts` logs -- python needs more explicit logs but user doesn't
export function stripUiOnlyFields(rows: unknown): unknown {
  const cleanOne = (r: unknown): unknown => {
    if (!isUiLogLike(r)) return r;
    const { ts, ...rest } = r; // drop ts

    if (typeof rest.text === 'string') {
      const mentionsStep = /\bstep\b\s*(?:[:#]\s*)?\d+(?:\s*[:#])?/i.test(rest.text);
      if (mentionsStep) delete (rest as any).step; // drop step to suppress [s:N]
    }
    return rest;
  };

  return Array.isArray(rows) ? rows.map(cleanOne) : cleanOne(rows);
}

function computeForkMergeCounters(): { forkMax: number; mergeMax: number } {
  let forkMax = 0;
  let mergeMax = 0;

  try {
    const runs = listRuns() as Array<{ run_id: string; name?: string }>;
    const reFork = /^fork(\d+)$/i;
    const reMerge = /^merge(\d+)$/i;

    for (const r of runs) {
      for (const v of [r.name, r.run_id]) {
        if (!v) continue;
        const s = String(v).trim();
        let m = reFork.exec(s);
        if (m) {
          const n = parseInt(m[1], 10);
          if (Number.isFinite(n) && n > forkMax) forkMax = n;
        }
        m = reMerge.exec(s);
        if (m) {
          const n = parseInt(m[1], 10);
          if (Number.isFinite(n) && n > mergeMax) mergeMax = n;
        }
      }
    }
  } catch (e) {
    console.warn('[onthefly] computeForkMergeCounters failed:', e);
  }

  return { forkMax, mergeMax };
}


/* ============================ Message handling ============================ */

async function onMessage(context: vscode.ExtensionContext, m: WebMsg) {
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
        if (!picked) break;

        const base64 = (m.dataUrl || '').split(',')[1];
        if (!base64) throw new Error('Invalid PNG data.');

        const buf = Buffer.from(base64, 'base64');
        fs.writeFileSync(picked.fsPath, buf);

        await context.globalState.update(LAST_EXPORT_DIR_KEY, path.dirname(picked.fsPath));

        vscode.window.showInformationMessage(`Chart exported to: ${picked.fsPath}`);
      } catch (e: any) {
        postErr(e);
      }
      break;
    }

    case 'exportSubset': {
      try {
        const runId = (m as any).runId || currentRunId;
        if (!runId) { vscode.window.showWarningMessage('No run selected.'); break; }

        const trace = (m as any)._trace || 'no-trace';
        const fromWebview = (m as any).subset_indices as any[] | undefined;

        const fmtRaw = String((m as any).format || 'parquet').toLowerCase();
        const fmt: 'parquet'|'csv'|'feather' = fmtRaw === 'csv' ? 'csv' : (fmtRaw === 'feather' ? 'feather' : 'parquet');

        const initialDir = getInitialExportDir(context);
        const defName = `subset_${runId}.${fmt === 'feather' ? 'feather' : fmt}`;
        const defaultUri = vscode.Uri.file(path.join(initialDir, defName));
        const label = fmt === 'csv' ? 'CSV' : fmt === 'feather' ? 'Feather' : 'Parquet';
        const ext   = fmt === 'csv' ? 'csv' : fmt === 'feather' ? 'feather' : 'parquet';
        const filters: { [name: string]: string[] } = { [label]: [ext] };

        const picked = await vscode.window.showSaveDialog({ title: 'Export subset', defaultUri, filters });
        if (!picked) break;
        await context.globalState.update(LAST_EXPORT_DIR_KEY, path.dirname(picked.fsPath));

        const payload: any = {
          run_id: String(runId),
          format: fmt,
          out_path: picked.fsPath,
          _trace: trace,
        };

        // --- Primary path: trust webview indices ---
        if (Array.isArray(fromWebview) && fromWebview.length > 0) {
          const norm = Array.from(
            new Set(fromWebview.map(n => Number(n) | 0).filter(n => Number.isFinite(n) && n >= 0))
          ).sort((a, b) => a - b);
          if (norm.length > 0) payload.subset_indices = norm;
        }

        // --- Fallbacks ---
        if (!payload.subset_indices) {
          const region = (m as any).region as { minLoss: number; maxLoss: number } | undefined;
          const stored = getRunSubset(String(runId)) || [];
          const haveStored = Array.isArray(stored) && stored.length > 0;

          if (region && Number.isFinite(region.minLoss) && Number.isFinite(region.maxLoss)) {
            try {
              const indices = await getSelectedRegionIndices(String(runId), region.minLoss, region.maxLoss);
              if (indices.length) payload.subset_indices = indices;
              else if (haveStored) payload.subset_indices = stored.map((n: any) => Number(n) | 0);
            } catch {
              if (haveStored) payload.subset_indices = stored.map((n: any) => Number(n) | 0);
            }
          } else if (haveStored) {
            payload.subset_indices = stored.map((n: any) => Number(n) | 0);
          }
        }
        const data = await sendReq('export_subset', payload, 10 * 60 * 1000);

        post({ type: 'subsetExported', run_id: String(runId), ...data });
      } catch (e:any) {
        postErr(e);
      }
      break;
    }

    case 'modelNav.select': {
      const id = String((m as any).runId || '').trim();
      if (id) {
        modelNavSelectedRunId = id;

        post({ type: 'log', text: `[modelNav] selected: ${id}` });
        post({ type: 'modelNav.select', runId: id }); // echo to webview so it knows which run is selected
        try {
          const rows = getLogs(id); // all phases
          post({ type: 'logs', run_id: id, rows: stripUiOnlyFields(rows) });
        } catch (e:any) {
          console.warn('[onthefly] failed to load logs for selected run', id, e);
        }
      }
      break;
    }


    case 'requestLogs': {
      try {
        const phase = (m as any).phase;
        const requested = String((m as any).runId || '').trim();
        const selected = modelNavSelectedRunId && modelNavSelectedRunId.trim();
        const active   = currentRunId && currentRunId.trim();

        const rid = requested || selected || active || '';
        let rows = rid ? getLogs(rid, phase) : [];
        if ((!rows || rows.length === 0) && currentSessionId) {
          rows = getLogsBySession(currentSessionId, phase);
        }
        post({ type: 'logs', run_id: rid || null, rows: stripUiOnlyFields(rows) });
      } catch (e:any) { postErr(e); }
      break;
    }

    case 'requestTestRows': {
      try {
        const rows = getTestRows(String((m as any).runId));
        post({ type: 'testRows', run_id: String((m as any).runId), rows });
      } catch (e:any) { postErr(e); }
      break;
    }

    case 'pause': {
      if (pauseInFlight) {
        post({ type: 'log', text: '[pause] Request already in progress.' });
        break;
      }
      if (runActivityState === 'paused') {
        break;
      }
      pauseInFlight = true;
      try {
        // 1) Ask the backend to pause (finish current step, flush state, etc.)
        const pauseInfo = await requestBackendPause(30_000); // backend will do its own checkpointing if it wants

        // 2) Figure out which run this pause applies to
        const runId =
          (modelNavSelectedRunId && modelNavSelectedRunId.trim()) ||
          (currentRunId && currentRunId.trim()) ||
          null;

        if (!runId) {
          post({ type: 'warn', text: '[pause] No runId available to attach checkpoint to.' });
          break;
        }

        const pausedStep = Number(pauseInfo?.step);

        const payload: any = { cmd: 'pause', run_id: runId };
        if (Number.isFinite(pausedStep)) payload.step = pausedStep;

        post({
          type: 'paused',
          payload,
        });

        // 3) Explicitly save a checkpoint and persist it in storage
        try {
          const ck = await sendReq('save_ckpt', {}, 120_000);
          if (ck?.path) {
            const stepNum = Number(ck.step) || 0;
            const ckptId = `${runId}:${stepNum}:${Date.now()}`; // same style as elsewhere

            insertCheckpoint(
              ckptId,
              runId,
              stepNum,
              String(ck.path),
            );

            post({
              type: 'log',
              level: 'info',
              text: `[pause] checkpoint saved for run ${runId} at step ${stepNum}`,
            });
          } else {
            post({
              type: 'warn',
              text: '[pause] save_ckpt returned no path; checkpoint not recorded in DB.',
            });
          }
        } catch (ckErr: any) {
          console.warn('[onthefly] pause: save_ckpt / insertCheckpoint failed:', ckErr);
          post({
            type: 'warn',
            text: `[pause] Failed to save/persist checkpoint: ${ckErr?.message || String(ckErr)}`,
          });
        }
      } catch (e: any) {
        postErr(e);
      } finally {
        pauseInFlight = false;
      }
      break;
    }
    case 'resume': {
      if (resumeInFlight) { post({ type: 'log', text: '[resume] Request already in progress.' }); break; }

      const requested = String((m as any).runId || '').trim();
      if (requested) modelNavSelectedRunId = requested;
      const target = requested || modelNavSelectedRunId || currentRunId || null;

      if (!target) { post({ type: 'error', text: '[resume] No run selected.' }); break; }

      resumeInFlight = true;
      try {
        await ensureTrainerOnRun(target);   // <<< switch-and-seed
        sendCtl({ cmd: 'resume' });
        setRunActivity('running');
      } catch (e:any) {
        postErr(e);
        postStatus(false);
      } finally {
        resumeInFlight = false;
      }
      break;
    }


    case 'testNow': {
      const requested = (m as any).runId as string | undefined;
      const candidate = (requested && requested.trim())
        || (modelNavSelectedRunId && modelNavSelectedRunId.trim())
        || (currentRunId && currentRunId.trim())
        || null;
      try {
        if (!candidate) {
          post({ type: 'log', text: 'Select a model in Model Nav before testing.' });
          break;
        }

        const runs = listRuns() as Array<{ run_id: string; name?: string }>;
        const friendly = runs.find(r => r.run_id === candidate)?.name || candidate;
        const choice = await vscode.window.showWarningMessage(
          `This will override the final tested model for this session. Test model "${friendly}" now?`,
          {
            modal: true,
            detail: 'Confirming will test immediately and replace the stored final checkpoint for this session. You can continue training this and other models after.'
          },
          'Confirm Test'
        );
        if (choice !== 'Confirm Test') {
          post({ type: 'log', text: 'Test cancelled.' });
          break;
        }

        const target = candidate;
        await ensureTrainerOnRun(target);
        try { await requestBackendPause(30_000); }
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
            upsertFinalModel({
              session_id: sessionId,
              run_id: target,
              ckpt_path: ckptPath,
              step: Number(data?.step) || 0,
              avg_test_loss: normalizedLoss,
            });
          } catch (persistErr) {
            console.warn('[onthefly] failed to persist final model', persistErr);
          }
        }
        post({
          type: 'testNow',
          status: 'completed',
          run_id: target,
          data: { ...data, avg_loss: normalizedLoss, ckpt_path: ckptPath },
        });
      } catch (e: any) {
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
          const top = m as any;
          const inner = (top && top.payload) || {};
          const merged = { ...inner, ...top };
          delete (merged as any).command;
          delete (merged as any).type;
          delete (merged as any).payload;
          return merged;
        })();

        // Also consider runId as a parent hint
        const hinted = String(
          payloadIn.owner_run_id || payloadIn.parent_run_id || payloadIn.runId || ''
        ).trim();

        const target =
          hinted ||
          modelNavSelectedRunId ||
          currentRunId ||
          pickAnchorRunForAction(payloadIn) ||
          '';

        if (!target) { postErr('No run selected to fork from.'); break; }

        await ensureTrainerOnRun(target);

        try { await requestBackendPause(30_000); } catch {}
        let ckptPath: string | undefined = latestCheckpointForRun(target)?.path;
        if (!ckptPath) {
          try { const ck = await sendReq('save_ckpt', {}, 120_000); ckptPath = ck?.path; } catch {}
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
            try { setRunSubset(child, data.subset_indices.map((n: any) => Number(n) | 0)); } catch {}
          }
          try { sendCtl({ cmd: 'resume' }); } catch {}
        }
      } catch (e:any) {
        postErr(e);
      }
      break;
    }

    case 'merge': {
      try {
        const payload = m.payload;

        const anchor = pickAnchorRunForAction(payload);
        if (!anchor) {
          postErr('No runs available to anchor the merge. Select a run or include parents[].');
          break;
        }
        await ensureTrainerOnRun(anchor);
        try { await requestBackendPause(30_000); } catch {}

        // Preflight: all parents must have checkpoints.
        const parents: string[] = Array.isArray(payload.parents) ? payload.parents : [];
        const missing = parents.filter(p => !latestCheckpointForRun(p));
        if (missing.length) {
          post({
            type: 'error',
            text: `[merge] Missing checkpoints for parent runs: ${missing.join(', ')}. Pause runs in your script to create checkpoints first.`,
          });
          break;
        }

        // Perform the merge.
        const data = await sendReq('merge', payload, 120000);

        // --- immediately navigate to the merged child & start running (mirror fork behavior) ---
        const child =
          String(
            (data && (data.new_run ?? data.child_id ?? data.run_id)) || ''
          ).trim();

        if (child) {
          currentRunId = child;
          modelNavSelectedRunId = child;
          nativeRunsThisSession.add(child);
          post({ type: 'modelNav.select', runId: child });

          // If backend provided a subset, persist it like we do on fork.
          if (Array.isArray(data?.subset_indices)) {
            try {
              setRunSubset(child, data.subset_indices.map((n: any) => Number(n) | 0));
            } catch {}
          }

          // Nudge the backend to resume the new merged run.
          try { sendCtl({ cmd: 'resume' }); } catch {}
        } else {
          // If the backend didn't return an explicit child id, the usual 'newRun' event
          // will still keep things in sync. No-op here.
        }
      } catch (e: any) {
        postErr(e);
      }
      break;
    }

    case 'dist_health': {
      try {
        const rid = (m as any).runId as string | undefined;
        const target = await requireActiveRun(rid);
        if (!target) break;
        await ensureTrainerOnRun(target);
        modelNavSelectedRunId = target;
        post({ type: 'modelNav.select', runId: target });
        await runHealth('dist_health', { require_all_ranks: false, sample_weights: 0 }, 'distHealth', 30_000, target);
      } catch (e:any) { postErr(e); }
      break;
    }

    case 'throughput_health': {
      try {
        const rid = (m as any).runId as string | undefined;
        const target = await requireActiveRun(rid);
        if (!target) break;
        await ensureTrainerOnRun(target);
        modelNavSelectedRunId = target;
        post({ type: 'modelNav.select', runId: target });
        await runHealth('throughput_health', { budget_steps: 2, micro_backward: false }, 'throughputHealth', 45_000, target);
      } catch (e:any) { postErr(e); }
      break;
    }

    case 'activations_health': {
      try {
        const rid = (m as any).runId as string | undefined;
        const target = await requireActiveRun(rid);
        if (!target) break;
        await ensureTrainerOnRun(target);
        modelNavSelectedRunId = target;
        post({ type: 'modelNav.select', runId: target });
        await runHealth('activations_health', { budget_steps: 2, topk: 12, eps: 1e-3 }, 'activationsHealth', 60_000, target);
      } catch (e:any) { postErr(e); }
      break;
    }

    case 'numerics_health': {
      try {
        const rid = (m as any).runId as string | undefined;
        const target = await requireActiveRun(rid);
        if (!target) break;
        await ensureTrainerOnRun(target);
        modelNavSelectedRunId = target;
        post({ type: 'modelNav.select', runId: target });
        await runHealth('numerics_health', { sample_layers: 25 }, 'numericsHealth', 30_000, target);
      } catch (e:any) { postErr(e); }
      break;
    }

    case 'determinism_health': {
      try {
        const rid = (m as any).runId as string | undefined;
        const target = await requireActiveRun(rid);
        if (!target) break;
        await ensureTrainerOnRun(target);
        modelNavSelectedRunId = target;
        post({ type: 'modelNav.select', runId: target });
        await runHealth('determinism_health', {}, 'determinismHealth', 20_000, target);
      } catch (e:any) { postErr(e); }
      break;
    }

    case 'generateReport': {
      try {
        let runId: string | undefined = (m as any).runId ?? undefined;
        if (!runId || !String(runId).trim()) {
          const anchor = pickAnchorRunForAction(); // choose something sensible if not provided
          if (!anchor) { post({ type: 'error', text: 'No active run selected for report.' }); break; }
          runId = anchor;
        }
        runId = String(runId).trim();
        await ensureTrainerOnRun(runId);
        const target = await requireActiveRun(runId);
        if (!target) break;
        runId = target;
        modelNavSelectedRunId = runId;
        post({ type: 'modelNav.select', runId });
        await requestBackendPause(30_000);

        const subset = getRunSubset(String(target));
        const subset_on: 'train' = 'train';

        const data = await sendReq('generate_report', {
          owner_run_id: target,
          subset_indices: subset.length ? subset : undefined,
          subset_on,
          reqId: (m as any).reqId
        });

        const losses: number[] = Array.isArray(data?.losses)
          ? data.losses.map(Number).filter(Number.isFinite)
          : [];

        let sample_indices: number[] = Array.isArray(data?.sample_indices)
          ? (data.sample_indices as any[]).map(v => Math.trunc(Number(v))).filter(v => Number.isFinite(v) && v >= 0)
          : [];

        if (sample_indices.length !== losses.length) {
          sample_indices = Array.from({ length: losses.length }, (_, i) => i);
        }

        const meta = data?.meta || {};
        const stepNum  = Number(meta?.at_step);
        const epochNum = Number(meta?.at_epoch);
        const at_step  = Number.isFinite(stepNum)  ? stepNum  : null;
        const at_epoch = Number.isFinite(epochNum) ? epochNum : null;
        const note = (typeof meta?.note === 'string' ? meta.note : '') || '';

        upsertReportLossDist(String(runId), subset_on, {
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
          reqId: (m as any).reqId,
          losses,
          sample_indices,
          meta: { ...meta, at_step, at_epoch, subset_on, samples: losses.length, note }
        });
      } catch (e: any) { postErr(e); }
      break;
    }

    case 'requestReport': {
      try {
        const runId = (m as any).runId;
        if (!runId) break;

        const row = getReportLossDist(String(runId), 'train');
        if (row) {
          post({
            type: 'reportFromDb',
            run_id: String(runId),
            owner_run_id: String(runId),
            losses: row.losses,
            meta: { note: row.note, at_step: row.at_step, at_epoch: row.at_epoch, samples: row.samples, subset_on: row.subset_on }
          });
        }
      } catch (e:any) { postErr(e); }
      break;
    }

    case 'notify': {
      const lvl = (m as any).level || 'info';
      const type = lvl === 'error' ? 'error' : 'log';
      post({ type, text: (m as any).text });
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
        if (!picked) break;

        const bundleDir = picked.fsPath;
        await context.globalState.update(LAST_EXPORT_DIR_KEY, path.dirname(bundleDir));
        fs.mkdirSync(bundleDir, { recursive: true });

        // NOTE: "modelNav" → we don't keep a separate nav map here; the DB is source of truth.
        // Grab every run_id we know and treat that as the nav list.
        const allRuns = (listRuns() as Array<{ run_id: string }>).map(r => String(r.run_id));
        const owners = Array.from(new Set(allRuns)).filter(Boolean);

        let spillRoot: string | null = null;

        if (trainerActive()) {
          spillRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'onthefly-spill-'));

          // 1) try the multi-owner prepare_export API (forces snapshots/ckpts for all runs)
          let prep: any = null;
          try {
            // latest_only keeps the bundle trim
            prep = await sendReq('prepare_export', { dir: spillRoot, latest_only: true, owners }, 10 * 60 * 1000);
          } catch (e) {
            // old servers won't know prepare_export(owners=...), fall back to a generic spill
            post({ type: 'log', text: '[export] prepare_export not available or failed; falling back to spill_all().' });
            try {
              const recs = await sendReq('spill_all', { dir: spillRoot, latest_only: true }, 10 * 60 * 1000);
              prep = { ckpt: null, snapshots: recs };
            } catch (e2) {
              console.warn('[export] spill_all RPC also failed; proceeding with whatever is already in DB:', e2);
              prep = { ckpt: null, snapshots: [] };
            }
          }

          // 2) write snapshot rows into DB so the bundler knows what to copy
          try {
            const snaps = Array.isArray(prep?.snapshots) ? prep.snapshots : [];
            for (const r of snaps) {
              if (!r || !r.ckpt_id || !r.owner || !r.path) continue;
              insertCheckpoint(String(r.ckpt_id), String(r.owner), Number(r.step) | 0, String(r.path));
            }
          } catch (e) {
            console.warn('[export] failed to insert spilled snapshots into DB:', e);
          }

          // 3) if python returned a ring ckpt for the current run, stash it too
          try {
            const ck = prep?.ckpt;
            if (ck?.path && ck?.run) {
              const ckId = `${ck.run}:${ck.step}:ring`;
              insertCheckpoint(ckId, String(ck.run), Number(ck.step) | 0, String(ck.path));
            }
          } catch (e) {
            console.warn('[export] failed to insert ring checkpoint into DB:', e);
          }

        } else {
          // not running → just bundle whatever DB already references
          post({ type: 'log', text: '[export] Python not running; bundling existing DB checkpoints only.' });
        }

        // 5) build the portable bundle (copies DB + referenced ckpts into bundle)
        storageExportBundle(bundleDir);

        // 6) cleanup tmp spill dir
        if (spillRoot) {
          try { fs.rmSync(spillRoot, { recursive: true, force: true }); } catch {}
        }

        const choice = await vscode.window.showInformationMessage(
          `Export complete: ${bundleDir}`,
          'Reveal in Finder/Explorer'
        );
        if (choice) vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(bundleDir));
        post({ type: 'sessionSaved', path: bundleDir });
      } catch (e: any) {
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
        if (!picked || !picked[0]) break;

        const chosen = picked[0].fsPath;
        const isDir = fs.existsSync(chosen) && fs.statSync(chosen).isDirectory();
        await context.globalState.update(LAST_EXPORT_DIR_KEY, isDir ? chosen : path.dirname(chosen));

        let bundleDir: string | null = null;
        if (isDir) {
          const manifestPath = path.join(chosen, 'bundle.json');
          if (!fs.existsSync(manifestPath)) {
            vscode.window.showErrorMessage(`No bundle.json found in folder:\n${chosen}`);
            break;
          }
          bundleDir = chosen;
        } else {
          const base = path.basename(chosen).toLowerCase();
          if (base !== 'bundle.json') {
            vscode.window.showErrorMessage(
              'Unsupported selection. Pick a bundle folder (with bundle.json) or bundle.json itself.'
            );
            break;
          }
          bundleDir = path.dirname(chosen);
        }

        // Load the bundle into the live DB
        storageLoadBundle(bundleDir, context);

        for (const s of listSessions()) {
          post({ type: 'log', text: `[print] session ${s.session_id}: ${getLogsBySession(String(s.session_id)).length} logs` });
        }

        // Notify UI & refresh runs
        post({ type: 'sessionLoaded' });
        const runs = listRuns() as Array<{ run_id: string }>;
        post({ type: 'runs', rows: runs });
        try {
          const sessions = listSessions();
          post({ type: 'fs.session.list', items: sessions });

        } catch {}

        // Hydrate logs for each run so the webview has something to render immediately
        for (const r of runs) {
          try {
            const rows = getLogs(String(r.run_id)); // all phases
            post({ type: 'logs', run_id: String(r.run_id), rows: stripUiOnlyFields(rows) });
          } catch (e:any) {
            console.warn('[onthefly] failed to load logs for run', r.run_id, e);
          }
        }

        const choice = await vscode.window.showInformationMessage(
          `Bundle loaded from: ${bundleDir}`,
          'Reveal in Finder/Explorer'
        );
        if (choice) vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(bundleDir));

      } catch (e: any) {
        postErr(e);
      }
      break;
    }

    case 'resetAll': {
      const pick = await vscode.window.showWarningMessage(
        'This will erase all models from memory, are you sure you want to refresh?',
        {
          modal: true,
          detail: 'Any running training will be stopped. Nothing will be saved if you have not exported session.',
        },
        'Erase & Refresh',
      );
      if (pick !== 'Erase & Refresh') {
        post({ type: 'log', text: 'Refresh cancelled.' });
        break;
      }

      await hardResetSession(context, { fromUser: true });
      break;
    }

    case 'requestRuns': {
      try {
        const rows = listRuns();
        post({ type: 'runs', rows });
      } catch (e: any) { postErr(e); }
      break;
    }

    case 'requestRows': {
      try {
        post({ type: 'rows', rows: getRunRows((m as any).runId) });
      } catch (e: any) { postErr(e); }
      break;
    }
  }
}

/* ============================ Python process (training) ============================ */

async function startRun() {
  ensureTrainerServer();
  if (trainerSocket && !trainerSocket.destroyed) {
    postStatus(true);
    return;
  }

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: 'Waiting for OnTheFly Trainer connection…' },
    async () => {
      try {
        await waitForTrainerConnection(60000);
        postStatus(true);
      } catch (e) {
        throw new Error('No Trainer connected. Run your training script in a terminal and instantiate OnTheFlyTrainer.');
      }
    }
  );
}

async function waitForTrainerConnection(timeoutMs = 0): Promise<void> {
  const start = Date.now();
  while (true) {
    if (trainerSocket && !trainerSocket.destroyed) return;
    if (timeoutMs && Date.now() - start > timeoutMs) {
      throw new Error('Timed out waiting for Trainer connection.');
    }
    await new Promise((res) => setTimeout(res, 250));
  }
}

function sendCtl(cmd: Ctl) {
  if (!trainerSocket || trainerSocket.destroyed) {
    post({ type: 'error', text: 'No Trainer connected. Run your script first.' });
    return;
  }
  try {
    trainerSocket.write(JSON.stringify(cmd) + '\n');
    const t = (cmd as any).cmd;
    if (t === 'resume') setRunActivity('running');
    if (t && optimisticEcho[t]) post({ type: optimisticEcho[t], payload: cmd });
  } catch (e: any) {
    postErr(e);
  }
}

async function requestBackendPause(timeoutMs = 30_000): Promise<any> {
  const data = await sendReq('pause', {}, timeoutMs);
  setRunActivity('paused');
  return data;
}

async function runHealth(
  cmd: string,
  payload: any,
  eventType: string,
  timeoutMs = 60_000,
  forRunId?: string
) {
  const run_id = forRunId || modelNavSelectedRunId || currentRunId || null;

  try {
    // tell the webview to show a loading placeholder
    post({ type: eventType, run_id, pending: true });

    try { await requestBackendPause(30_000); } catch {}

    const data = await sendReq(cmd, payload, timeoutMs);

    // normal success event (what your webview already handles)
    post({ type: eventType, cmd, payload, data, run_id });
  } catch (e:any) {
    // let the UI show an error if needed
    post({ type: eventType, run_id, error: String(e?.message || e) });
    postErr(e);
  }
}


function pickAnchorRunForAction(hint?: { parents?: string[]; runId?: string } | any): string | null {
  const fromRunId =
    typeof hint?.runId === 'string' && hint.runId.trim() ? String(hint.runId).trim() : null;

  const fromParents =
    Array.isArray(hint?.parents) && hint.parents.length ? String(hint.parents[0]) : null;

  if (fromRunId) return fromRunId;
  if (fromParents) return fromParents;
  if (modelNavSelectedRunId?.trim()) return modelNavSelectedRunId.trim();
  if (currentRunId?.trim()) return currentRunId.trim();

  try {
    const first = (listRuns() as Array<{ run_id: string }>)[0]?.run_id;
    if (first) return String(first);
  } catch {}

  return null;
}

async function requireActiveRun(requestedRunId?: string): Promise<string | null> {
  const target = String(requestedRunId || modelNavSelectedRunId || currentRunId || '').trim();
  if (!target) return null;
  await ensureTrainerOnRun(target);
  return target;
}

function normalizeParents(from: any): string[] {
  const raw = (from && (from.parents ?? from.parent)) ?? [];
  if (Array.isArray(raw)) return raw.filter(Boolean).map((s) => String(s));
  if (raw == null || raw === '') return [];
  return [String(raw)];
}


/* ============================ Line handler ============================ */

function handleLine(line: string) {
  let obj: any = null;

  try { obj = JSON.parse(line); }
  catch { post({ type: 'log', text: line }); return; }

  if (obj && obj.event && !obj.type) { obj.type = obj.event; delete obj.event; }

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
    try { insertTestMetric(run_id, step, Number.isFinite(loss) ? loss : null); } catch {}
    return;
  }

  if (obj?.type === 'paused') {
    setRunActivity('paused');
    pauseInFlight = false;
  } else if (obj?.type === 'resumed') {
    setRunActivity('running');
    resumeInFlight = false;
  } else if (obj?.type === 'trainingFinished') {
    setRunActivity(null);
    pauseInFlight = false;
    resumeInFlight = false;
  }

  if (obj?.type === 'log') {
    const run_id = obj.run_id || currentRunId || 'live';
    // Prefer explicit, then fallback to sticky session
    const session_id = obj.session_id || obj.sessionId || currentSessionId || null;
    if (run_id && session_id && currentSessionId && session_id === currentSessionId) {
      nativeRunsThisSession.add(String(run_id));
    }
    const level = obj.level || 'info';
    const text: string = String(obj.text || '');

    // 1) honor explicit phase if provided by backend
    const pRaw = (obj.phase && String(obj.phase).toLowerCase()) || null;
    let phase: 'train'|'test'|'info' =
      (pRaw === 'train' || pRaw === 'test' || pRaw === 'info') ? (pRaw as any) : 'info';

    // parse a few common patterns to enrich rows (prefer TEST first)
    let step: number | null = null;
    let epoch: number | null = null;

    const mEpoch = text.match(/^\s*epoch\s+(\d+)\s*$/i);
    if (mEpoch && phase === 'info') { epoch = Number(mEpoch[1]); phase = 'train'; }

    // ---- TEST first (looser match) ----
    const mTest = text.match(/step\s+(\d+)\s*:\s*test[_\s]*loss\s*=\s*([0-9.eE+\-]+)/i);
    if (mTest && phase === 'info') { step = Number(mTest[1]); phase = 'test'; }
    if (/\btesting\b/i.test(text) && phase === 'info') phase = 'test';
    if (/\b(?:eval|evaluation)\b/i.test(text) && /\b(loss|acc|metric)\b/i.test(text) && phase === 'info') phase = 'test';

    // ---- TRAIN (slightly looser too) ----
    const mTrain = text.match(/step\s+(\d+)\s*:\s*train[_\s]*loss\s*=\s*([0-9.eE+\-]+).*?val[_\s]*loss\s*=\s*([0-9.eE+\-]+|None)/i);
    if (mTrain && phase === 'info') { step = Number(mTrain[1]); phase = 'train'; }
    if ((/\btrain[_\s]*loss\b/i.test(text) || /\bval[_\s]*loss\b/i.test(text)) && phase === 'info') {
      phase = 'train';
    }

    const tsMs = (Number(obj.ts) > 0 ? Math.round(Number(obj.ts) * 1000) : Date.now());

    // If the raw text already includes "step N", don't also render the UI "s: N" badge.
    const hasStepInText = /\bstep\b\s*(?:[:#]\s*)?\d+(?:\s*[:#])?/i.test(text);
    const stepForUI = hasStepInText ? null : step;

    try {
      // keep full fidelity in storage
      insertLog({ run_id, session_id, level, text, phase, step, epoch, ts: tsMs });
    } catch {}

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
    const num = (value: any) => (Number.isFinite(value) ? Number(value) : null);

    //keep extension-side notion of "current run" in sync with the stream
    currentRunId = run_id;
    if (!modelNavSelectedRunId) {
      modelNavSelectedRunId = run_id;
    }

    const row: StepRow = {
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
    post({ type: 'trainStep', ...row });
    const logLine = `step ${row.step}: train_loss = ${row.loss ?? 'None'}, val_loss = ${lastVal}`;
    if (runActivityState !== 'running') {
      setRunActivity('running');   // this will now call postStatus(true) with run_id
    }
    post({ type: 'log', level: 'info', phase: 'train', text: logLine });
    try {
      insertMetric(run_id, row);
    } catch (e) {
      console.warn('[onthefly] metric persist error:', e);
    }
    return;
  }


  if (obj?.type === 'log' && obj.text && /model\s+session_id/i.test(obj.text)) {
    const m = String(obj.text).match(/session_id\s*=\s*([^\s]+)/i);
    if (m && m[1]) { currentSessionId = m[1]; postCurrentSession(); }
  }

  // ==== canonical run creation ====
  if (obj && obj.type === 'newRun') {
    const id = String(obj.run_id || '').trim();
    if (id) nativeRunsThisSession.add(id);
    if (!id) return;

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
      const existingRuns = listRuns() as Array<{ run_id: string }>;
      const existsInDb = existingRuns.some(r => r.run_id === id);
  
      if (!existsInDb) {
        insertRun(id, project, name, primaryParent);
      }
    } else {
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
    post({ type: 'runs', rows: listRuns() });
    return;
  }

  if (obj?.type === 'auto_test_complete') {
    const runId = String(obj.run_id || currentRunId || '').trim();
    const ckptPath = obj.ckpt_path ? String(obj.ckpt_path) : '';
    const step = Number(obj.step) || 0;

    if (runId && ckptPath) {
      try {
        const ckptId = `${runId}:${step}:${Date.now()}`;
        insertCheckpoint(ckptId, runId, step, ckptPath);
        post({
          type: 'log',
          level: 'info',
          text: `[auto-test] checkpoint recorded for run ${runId} at step ${step}`,
        });
      } catch (e) {
        console.warn('[onthefly] failed to persist auto-test checkpoint', e);
        post({
          type: 'log',
          level: 'warn',
          text: `[auto-test] failed to persist checkpoint for run ${runId}: ${String(
            (e as any)?.message || e,
          )}`,
        });
      }
    } else {
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

        if (needDiskCleanOnNextTrainer && trainerActive()) {
          needDiskCleanOnNextTrainer = false;
          (async () => {
            try {
              const res = await sendReq('clean_disk', { scope: 'all' }, 60_000);
              post({
                type: 'log',
                level: res?.ok ? 'info' : 'warn',
                text: res?.ok
                  ? '[startup] clean_disk: removed old runs from save_dir'
                  : `[startup] clean_disk reported an issue: ${res?.error ?? 'unknown error'}`,
              });
            } catch (e: any) {
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
        insertCheckpoint(ckptId, obj.run_id, Number(obj.step) || 0, obj.path || '');
        break;
      }
      case 'epoch_end': {
        if (Number.isFinite(obj.val_loss)) {
          upsertSummary(obj.run_id, null, obj.val_loss);
        }
        break;
      }
      default:
        break;
    }
  } catch (e) {
    console.warn('[onthefly] sqlite persist error:', e);
  }
}


//helpers

function latestResumeHints(runId: string | null) {
  if (!runId) return null;
  const ck = latestCheckpointForRun(runId);
  return ck ? { init_ckpt: ck.path, init_step: Number(ck.step) || 0 } : null;
}

async function seedTrainerForRun(targetRunId: string): Promise<void> {
  const { forkMax, mergeMax } = computeForkMergeCounters();
  const hints = latestResumeHints(targetRunId) || {};

  // Use attach_context only when we really need to bind/attach:
  if (!currentRunId || currentRunId !== targetRunId) {
    await sendReq('attach_context', {
      run_id: targetRunId,
      ...hints,
      fork_counter_init: forkMax || 0,
      merge_counter_init: mergeMax || 0,
    }, 30_000);
  } else {
    // Already on this run – just send counter seeds if you care:
    await sendReq('set_counter_seeds', {
      fork_counter_init: forkMax || 0,
      merge_counter_init: mergeMax || 0,
    }, 30_000);
  }
}


/** Try to attach the connected Trainer to a specific run, creating resume wiring. */
async function ensureTrainerOnRun(targetRunId: string): Promise<void> {
  if (!targetRunId) throw new Error('No target run.');
  await startRun();                    // wait for socket
  if (!requireTrainerConnection()) return;

  const alreadyOn = currentRunId && currentRunId === targetRunId;

  // If Trainer is already on this run, do nothing.
  // No attach_context, no _bind_to_run, no ckpt reload.
  if (alreadyOn) {
    return;
  }

  // We are switching runs -> use switch_run + init hints.
  const hints = latestResumeHints(targetRunId) || {};
  const { forkMax, mergeMax } = computeForkMergeCounters();

  await sendReq('switch_run', {
    run_id: targetRunId,
    ...hints,
    fork_counter_init: forkMax || 0,
    merge_counter_init: mergeMax || 0,
  }, 30_000);

  currentRunId = targetRunId;
  modelNavSelectedRunId = targetRunId;
  post({ type: 'modelNav.select', runId: targetRunId });
}


async function getSelectedRegionIndices(runId: string, minLoss: number, maxLoss: number): Promise<number[]> {
  const data = await sendReq('get_selected_region_indices', {
    run_id: String(runId),
    min_loss: Number(minLoss),
    max_loss: Number(maxLoss),
  }, 60_000);
  const arr = Array.isArray(data?.indices) ? data.indices : [];
  // normalize to ints
  return arr
    .map((n: number) => Number(n) | 0)
    .filter((n: number) => Number.isFinite(n) && n >= 0);

}
