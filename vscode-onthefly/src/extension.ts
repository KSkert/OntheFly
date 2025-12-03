import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import {
  initStorage,
  insertRun,
  insertCheckpoint,
  upsertFinalModel,
  listRuns,
  getRunRows,
  setRunSubset,
  getRunSubset,
  upsertReportLossDist,
  getReportLossDist,
  closeStorage,
  getTestRows,
  getLogs,
  getLogsBySession,
  listSessions,
  exportBundle as storageExportBundle,
  loadBundle as storageLoadBundle,
  latestCheckpointForRun
} from './storage';
import * as os from 'os';
import {
  disconnectTrainer,
  ensureTrainerOnRun,
  ensureTrainerServer,
  getSelectedRegionIndices,
  registerHardResetHandler,
  requestBackendPause,
  resumeAfterReset,
  resumeTrainerOn,
  runHealth,
  sendCtl,
  sendReq,
  shutdownTrainerServer,
  trainerActive,
} from './extensionHost/ipc';
import {
  extensionState,
  LAST_EXPORT_DIR_KEY,
  post,
  postCurrentSession,
  postErr,
  postStatus,
  stripUiOnlyFields,
} from './extensionHost/state';
import { TrainerResetSeed, WebMsg } from './extensionHost/types';

// Stash latest config (webview can set it before or after starting the run)
function isRunImportedForThisSession(runId: string): boolean {
  if (!trainerActive()) return true;
  return !extensionState.nativeRunsThisSession.has(runId);
}
export async function activate(context: vscode.ExtensionContext) {
  extensionState.lastExtensionContext = context;
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
  let trainerSeed: TrainerResetSeed | null = null;
  const hadTrainer = trainerActive();

  if (hadTrainer && opts?.fromUser) {
    try {
      const res = await sendReq('reset_session', { reason: 'manual' }, 60000);
      trainerSeed = res || null;
      if (trainerSeed?.run_id) {
        post({
          type: 'log',
          text: `[reset] Trainer cleared. Next run "${trainerSeed.run_id}" waits for resume.`,
        });
      }
    } catch (e: any) {
      post({
        type: 'log',
        level: 'warn',
        text: `[reset] Backend reset skipped or failed: ${e?.message || String(e)}`,
      });
    }
  }

  // 0) If a backend is alive, ask it to clean its save_dir now (best effort).
  if (trainerActive()) {
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
  extensionState.currentRunId = null;
  extensionState.seenRuns.clear();
  extensionState.currentSessionId = null;
  extensionState.nativeRunsThisSession.clear();
  extensionState.modelNavSelectedRunId = null;
  extensionState.runActivityState = null;
  extensionState.pauseInFlight = false;
  extensionState.resumeInFlight = false;
  extensionState.pendingResumeAwaitingFirstRun = false;

  // 4) Reset storage
  try { closeStorage({ retainFile: false }); } catch {}
  await initStorage(context);
  
  extensionState.needDiskCleanOnNextTrainer = true;

  // 5) Tell webview, if it exists
  post({ type: 'resetOk' });
  if (trainerSeed?.run_id) {
    const runId = String(trainerSeed.run_id);
    const friendly = (trainerSeed.display_name && String(trainerSeed.display_name)) || runId;
    const projectName = (trainerSeed.project && String(trainerSeed.project)) || 'default';
    insertRun(runId, projectName, friendly, null);
    extensionState.seenRuns.add(runId);
    extensionState.currentRunId = runId;
    extensionState.modelNavSelectedRunId = runId;
    extensionState.nativeRunsThisSession.add(runId);
    if (trainerSeed.session_id) {
      extensionState.currentSessionId = String(trainerSeed.session_id);
      postCurrentSession();
    }
    post({
      type: 'newRun',
      run_id: runId,
      parents: [],
      meta: { display_name: friendly, kind: 'reset' },
      session_id: extensionState.currentSessionId || null,
    });
    post({ type: 'modelNav.select', runId });
  }
  post({ type: 'runs', rows: listRuns() });
  postStatus(false);

  if (opts?.fromUser) {
    vscode.window.setStatusBarMessage('Onthefly: session reset', 2000);
  }
}
registerHardResetHandler(hardResetSession);


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
  if (extensionState.panel) { extensionState.panel.reveal(vscode.ViewColumn.Active); return; }

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
  extensionState.panel = webviewPanel;
  extensionState.panel.webview.options = {
    enableScripts: true,
    localResourceRoots: getLocalResourceRoots(context),
  };

  const nonce = getNonce();
  let webviewVisible = extensionState.panel.visible;

  extensionState.panel.onDidChangeViewState(({ webviewPanel }) => {
    webviewVisible = webviewPanel.visible;
    if (webviewVisible) {
      // when user returns, cheaply re-sync UI (no heavy streams)
      try { post({ type: 'runs', rows: listRuns() }); } catch {}
      postStatus(trainerActive());
    }
  });

  extensionState.panel.onDidDispose(() => {
    // Just drop the reference; let deactivate() handle shutdown.
    if (extensionState.panel === webviewPanel) {
      extensionState.panel = null;
    }
  });

  extensionState.panel.webview.onDidReceiveMessage((m: any) => { onMessage(context, m); });

  extensionState.panel.webview.html = getHtml(context, extensionState.panel.webview, nonce);

  if (extensionState.compatGateActive && extensionState.compatErrorMessage) {
    post({ type: 'compatError', message: extensionState.compatErrorMessage });
  }
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

const TEST_NOW_TIMEOUT_MS = 10 * 60 * 1000;

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
        const runId = (m as any).runId || extensionState.currentRunId;
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
        extensionState.modelNavSelectedRunId = id;

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
        const selected = extensionState.modelNavSelectedRunId && extensionState.modelNavSelectedRunId.trim();
        const active   = extensionState.currentRunId && extensionState.currentRunId.trim();

        const rid = requested || selected || active || '';
        let rows = rid ? getLogs(rid, phase) : [];
        if ((!rows || rows.length === 0) && extensionState.currentSessionId) {
          rows = getLogsBySession(extensionState.currentSessionId, phase);
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
      if (extensionState.pauseInFlight) {
        post({ type: 'log', text: '[pause] Request already in progress.' });
        break;
      }
      if (extensionState.runActivityState === 'paused') {
        break;
      }
      extensionState.pauseInFlight = true;
      try {
        // 1) Ask the backend to pause (finish current step, flush state, etc.)
        const pauseInfo = await requestBackendPause(30_000); // backend will do its own checkpointing if it wants

        // 2) Figure out which run this pause applies to
        const runId =
          (extensionState.modelNavSelectedRunId && extensionState.modelNavSelectedRunId.trim()) ||
          (extensionState.currentRunId && extensionState.currentRunId.trim()) ||
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
        extensionState.pauseInFlight = false;
      }
      break;
    }
    case 'resume': {
      if (extensionState.resumeInFlight) { post({ type: 'log', text: '[resume] Request already in progress.' }); break; }

      const requested = String((m as any).runId || '').trim();
      if (requested) extensionState.modelNavSelectedRunId = requested;
      const target = requested || extensionState.modelNavSelectedRunId || extensionState.currentRunId || null;

      if (!target) {
        try {
          const rows = listRuns();
          if (!rows.length) {
            extensionState.pendingResumeAwaitingFirstRun = true;
            post({ type: 'log', text: '[resume] No runs exist yet. Waiting for a fresh trainer session…' });
            await resumeAfterReset();
          } else {
            post({ type: 'error', text: '[resume] No run selected.' });
          }
        } catch (e: any) {
          extensionState.pendingResumeAwaitingFirstRun = false;
          postErr(e);
        }
        break;
      }

      await resumeTrainerOn(target);
      break;
    }


    case 'testNow': {
      const requested = (m as any).runId as string | undefined;
      const candidate = (requested && requested.trim())
        || (extensionState.modelNavSelectedRunId && extensionState.modelNavSelectedRunId.trim())
        || (extensionState.currentRunId && extensionState.currentRunId.trim())
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
        const sessionIdRaw = (data?.session_id && String(data.session_id)) || extensionState.currentSessionId || '';
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
        const fallback = candidate || extensionState.modelNavSelectedRunId || extensionState.currentRunId || null;
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
          extensionState.modelNavSelectedRunId ||
          extensionState.currentRunId ||
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
          extensionState.currentRunId = child;
          extensionState.modelNavSelectedRunId = child;
          extensionState.nativeRunsThisSession.add(child);
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
          extensionState.currentRunId = child;
          extensionState.modelNavSelectedRunId = child;
          extensionState.nativeRunsThisSession.add(child);
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
        extensionState.modelNavSelectedRunId = target;
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
        extensionState.modelNavSelectedRunId = target;
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
        extensionState.modelNavSelectedRunId = target;
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
        extensionState.modelNavSelectedRunId = target;
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
        extensionState.modelNavSelectedRunId = target;
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
        extensionState.modelNavSelectedRunId = runId;
        post({ type: 'modelNav.select', runId });
        await requestBackendPause(30_000);

        const subset = getRunSubset(String(target));
        const subset_on: 'train' = 'train';

        const data = await sendReq('generate_report', {
          owner_run_id: target,
          subset_indices: subset.length ? subset : undefined,
          subset_on,
          reqId: (m as any).reqId
        }, 10 * 60 * 1000);

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

function pickAnchorRunForAction(hint?: { parents?: string[]; runId?: string } | any): string | null {
  const fromRunId =
    typeof hint?.runId === 'string' && hint.runId.trim() ? String(hint.runId).trim() : null;

  const fromParents =
    Array.isArray(hint?.parents) && hint.parents.length ? String(hint.parents[0]) : null;

  if (fromRunId) return fromRunId;
  if (fromParents) return fromParents;
  if (extensionState.modelNavSelectedRunId?.trim()) return extensionState.modelNavSelectedRunId.trim();
  if (extensionState.currentRunId?.trim()) return extensionState.currentRunId.trim();

  try {
    const first = (listRuns() as Array<{ run_id: string }>)[0]?.run_id;
    if (first) return String(first);
  } catch {}

  return null;
}

async function requireActiveRun(requestedRunId?: string): Promise<string | null> {
  const target = String(requestedRunId || extensionState.modelNavSelectedRunId || extensionState.currentRunId || '').trim();
  if (!target) return null;
  await ensureTrainerOnRun(target);
  return target;
}
