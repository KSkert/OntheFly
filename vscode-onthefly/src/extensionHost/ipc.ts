import * as vscode from 'vscode';
import * as net from 'net';
import * as crypto from 'crypto';
import semver from 'semver';

import {
  insertRun,
  insertMetric,
  insertCheckpoint,
  upsertSummary,
  insertLog,
  insertTestMetric,
  listRuns,
  latestCheckpointForRun,
} from '../storage';
import {
  extensionState,
  post,
  postErr,
  postStatus,
  postCurrentSession,
  setRunActivityState,
} from './state';
import { Ctl, RunActivityState, StepRow } from './types';

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
const MIN_PYTHON_PACKAGE_FALLBACK = '0.1.1';
const VERSION_BLOCK_MESSAGE =
  'Current pip onthefly-ai version is outdated. Update it in your terminal with "pip install -U onthefly-ai" for the compatible version.';

type HardResetHandler = (context: vscode.ExtensionContext, opts?: { fromUser?: boolean }) => Promise<void>;
let hardResetHandler: HardResetHandler | null = null;

export function registerHardResetHandler(handler: HardResetHandler): void {
  hardResetHandler = handler;
}

export function trainerActive(): boolean {
  return Boolean(trainerSocket && !trainerSocket.destroyed);
}

export function sendReq(cmd: string, payload: any = {}, timeoutMs = 15000): Promise<any> {
  if (!trainerSocket || trainerSocket.destroyed) {
    return Promise.reject(
      new Error('No Trainer connected. Run your training script with an OnTheFlyTrainer to stream data.')
    );
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

function postCompatError(message: string): void {
  const msg = (message || '').trim() || VERSION_BLOCK_MESSAGE;
  const alreadySent = extensionState.compatGateActive && extensionState.compatErrorMessage === msg;
  extensionState.compatGateActive = true;
  extensionState.compatErrorMessage = msg;
  if (!alreadySent) {
    post({ type: 'compatError', message: msg });
  }
  if (!extensionState.compatNotified) {
    extensionState.compatNotified = true;
    vscode.window.showErrorMessage(msg);
  }
}

function clearCompatibilityBlock(): void {
  if (!extensionState.compatGateActive && !extensionState.compatErrorMessage) {
    extensionState.compatNotified = false;
    return;
  }
  extensionState.compatGateActive = false;
  extensionState.compatErrorMessage = null;
  extensionState.compatNotified = false;
  post({ type: 'compatClear' });
}

async function sendTrainerVersionAbort(message: string, required: string, reported: string | null): Promise<void> {
  if (!trainerSocket || trainerSocket.destroyed) return;
  try {
    await sendReq(
      'enforce_version',
      { message: (message || '').trim() || VERSION_BLOCK_MESSAGE, required, reported: reported ?? null },
      10_000
    );
  } catch (err) {
    console.warn('[onthefly] enforce_version command failed:', err);
  }
}

async function enforceMinPythonPackage(): Promise<void> {
  const cfg = vscode.workspace.getConfiguration('onthefly');
  const raw = String(cfg.get('minPythonPackage') || MIN_PYTHON_PACKAGE_FALLBACK).trim();
  const minVersion = semver.coerce(raw);
  if (!minVersion) {
    console.warn(`[onthefly] invalid onthefly.minPythonPackage value "${raw}" (unable to parse)`);
    return;
  }
  if (!trainerSocket || trainerSocket.destroyed) {
    return;
  }

  let info: any;
  try {
    info = await sendReq('server_info', {}, 10_000);
  } catch (err) {
    postCompatError(VERSION_BLOCK_MESSAGE);
    await sendTrainerVersionAbort(VERSION_BLOCK_MESSAGE, raw, null);
    throw err;
  }

  const reportedRaw = typeof info?.version === 'string' ? info.version.trim() : '';
  const reported = reportedRaw || (info?.version != null ? String(info.version) : '');
  const parsedReported = reported ? semver.coerce(reported) : null;
  if (!parsedReported) {
    postCompatError(VERSION_BLOCK_MESSAGE);
    await sendTrainerVersionAbort(VERSION_BLOCK_MESSAGE, raw, reported || null);
    throw new Error('Trainer reported an unrecognized onthefly-ai version.');
  }

  if (semver.lt(parsedReported, minVersion)) {
    postCompatError(VERSION_BLOCK_MESSAGE);
    await sendTrainerVersionAbort(VERSION_BLOCK_MESSAGE, raw, reported || parsedReported.version);
    throw new Error('Trainer onthefly-ai version is below the dashboard minimum.');
  }

  clearCompatibilityBlock();
}

export function ensureTrainerServer(): void {
  if (trainerServer) return;

  trainerServer = net.createServer((socket) => {
    if (trainerSocket && !trainerSocket.destroyed) {
      socket.destroy();
      vscode.window.showWarningMessage(
        'Another Trainer tried to connect while one is active. Close the existing run first.'
      );
      return;
    }

    (async () => {
      if (extensionState.lastExtensionContext && hardResetHandler) {
        try {
          let hasRuns = false;
          try {
            hasRuns = listRuns().length > 0;
          } catch {}

          const hasUiState = Boolean(
            extensionState.currentRunId ||
            extensionState.modelNavSelectedRunId ||
            extensionState.currentSessionId ||
            extensionState.seenRuns.size
          );

          if (hasRuns || hasUiState) {
            await hardResetHandler(extensionState.lastExtensionContext, { fromUser: false });
          }
        } catch (e) {
          console.warn('[onthefly] automatic session reset on new trainer failed:', e);
        }
      }

      extensionState.hasTrainerConnectedOnce = true;

      trainerSocket = socket;
      trainerBuffer = '';
      socket.setEncoding('utf8');
      socket.on('data', (chunk: string) => handleTrainerData(chunk));
      socket.on('error', (err: any) => {
        post({ type: 'error', text: `[trainer] ${err?.message || err}` });
      });
      socket.on('close', () => {
        post({ type: 'log', text: 'Trainer disconnected.' });
        disconnectTrainer(false);
      });

      postStatus(true);
      post({ type: 'log', text: 'Trainer connected. Streaming events live.' });

      try {
        await enforceMinPythonPackage();
      } catch (err: any) {
        if (!extensionState.compatGateActive) {
          const msg = err?.message || 'Trainer version check failed.';
          post({ type: 'error', text: msg });
          vscode.window.showErrorMessage(msg);
        }
        disconnectTrainer(false);
        return;
      }

      const target =
        (extensionState.modelNavSelectedRunId && extensionState.modelNavSelectedRunId.trim()) ||
        (extensionState.currentRunId && extensionState.currentRunId.trim()) ||
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

export function shutdownTrainerServer(): void {
  if (trainerServer) {
    try {
      trainerServer.close();
    } catch {}
    trainerServer = null;
  }
  disconnectTrainer(false);
}

export function disconnectTrainer(notify = true): void {
  if (trainerSocket) {
    try {
      trainerSocket.destroy();
    } catch {}
    trainerSocket = null;
  }
  trainerBuffer = '';
  setRunActivity(null);
  extensionState.pauseInFlight = false;
  extensionState.resumeInFlight = false;

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

function handleTrainerData(chunk: string): void {
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

export function setRunActivity(state: RunActivityState): void {
  setRunActivityState(state);
  postStatus(trainerActive());
}

function requireTrainerConnection(): boolean {
  if (!trainerActive()) {
    vscode.window.showErrorMessage('No Trainer connection. Run your training script with an OnTheFlyTrainer to stream data.');
    postStatus(false);
    return false;
  }
  return true;
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

async function startRun(): Promise<void> {
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

const optimisticEcho: Record<string, string> = {
  pause: 'paused',
  resume: 'resumed',
  save_ckpt: 'checkpointSaved',
  merge: 'merged',
};

export function sendCtl(cmd: Ctl): void {
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

export async function requestBackendPause(timeoutMs = 30_000): Promise<any> {
  const data = await sendReq('pause', {}, timeoutMs);
  setRunActivity('paused');
  return data;
}

export async function resumeTrainerOn(target: string): Promise<void> {
  if (extensionState.resumeInFlight) {
    return;
  }
  extensionState.resumeInFlight = true;
  try {
    await ensureTrainerOnRun(target);
    sendCtl({ cmd: 'resume' });
    setRunActivity('running');
  } catch (e: any) {
    postErr(e);
    postStatus(false);
  } finally {
    extensionState.resumeInFlight = false;
  }
}

export async function resumeAfterReset(): Promise<void> {
  if (extensionState.resumeInFlight) {
    return;
  }
  extensionState.resumeInFlight = true;
  extensionState.pendingResumeAwaitingFirstRun = true;
  try {
    await startRun();
    extensionState.pendingResumeAwaitingFirstRun = false;
    if (!trainerSocket || trainerSocket.destroyed) {
      throw new Error('Trainer disconnected before resume.');
    }
    sendCtl({ cmd: 'resume' });
    setRunActivity('running');
  } catch (e: any) {
    postErr(e);
    throw e;
  } finally {
    extensionState.pendingResumeAwaitingFirstRun = false;
    extensionState.resumeInFlight = false;
  }
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

function latestResumeHints(runId: string | null) {
  if (!runId) return null;
  const ck = latestCheckpointForRun(runId);
  return ck ? { init_ckpt: ck.path, init_step: Number(ck.step) || 0 } : null;
}

async function seedTrainerForRun(targetRunId: string): Promise<void> {
  const { forkMax, mergeMax } = computeForkMergeCounters();
  const hints = latestResumeHints(targetRunId) || {};

  if (!extensionState.currentRunId || extensionState.currentRunId !== targetRunId) {
    await sendReq('attach_context', {
      run_id: targetRunId,
      ...hints,
      fork_counter_init: forkMax || 0,
      merge_counter_init: mergeMax || 0,
    }, 30_000);
  } else {
    await sendReq('set_counter_seeds', {
      fork_counter_init: forkMax || 0,
      merge_counter_init: mergeMax || 0,
    }, 30_000);
  }
}

export async function ensureTrainerOnRun(targetRunId: string): Promise<void> {
  if (!targetRunId) throw new Error('No target run.');
  await startRun();
  if (!requireTrainerConnection()) return;

  const alreadyOn = extensionState.currentRunId && extensionState.currentRunId === targetRunId;

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

  extensionState.currentRunId = targetRunId;
  extensionState.modelNavSelectedRunId = targetRunId;
  post({ type: 'modelNav.select', runId: targetRunId });
}

export async function getSelectedRegionIndices(runId: string, minLoss: number, maxLoss: number): Promise<number[]> {
  const data = await sendReq('get_selected_region_indices', {
    run_id: String(runId),
    min_loss: Number(minLoss),
    max_loss: Number(maxLoss),
  }, 60_000);
  const arr = Array.isArray(data?.indices) ? data.indices : [];
  return arr
    .map((n: number) => Number(n) | 0)
    .filter((n: number) => Number.isFinite(n) && n >= 0);
}

export async function runHealth(
  cmd: string,
  payload: any,
  eventType: string,
  timeoutMs = 60_000,
  forRunId?: string
): Promise<void> {
  const run_id = forRunId || extensionState.modelNavSelectedRunId || extensionState.currentRunId || null;

  try {
    post({ type: eventType, run_id, pending: true });

    try { await requestBackendPause(30_000); } catch {}

    const data = await sendReq(cmd, payload, timeoutMs);

    post({ type: eventType, cmd, payload, data, run_id });
  } catch (e:any) {
    post({ type: eventType, run_id, error: String(e?.message || e) });
    postErr(e);
  }
}

function normalizeParents(from: any): string[] {
  const raw = (from && (from.parents ?? from.parent)) ?? [];
  if (Array.isArray(raw)) return raw.filter(Boolean).map((s) => String(s));
  if (raw == null || raw === '') return [];
  return [String(raw)];
}

function handleLine(line: string): void {
  let obj: any = null;

  try {
    obj = JSON.parse(line);
  } catch {
    post({ type: 'log', text: line });
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
    const run_id = obj.run_id || extensionState.currentRunId || 'live';
    const step = Number(obj.step) || 0;
    const loss = Number(obj.loss);
    post({ type: 'testStep', run_id, step, loss: Number.isFinite(loss) ? loss : null, ts: Date.now() });
    try { insertTestMetric(run_id, step, Number.isFinite(loss) ? loss : null); } catch {}
    return;
  }

  if (obj?.type === 'paused') {
    setRunActivity('paused');
    extensionState.pauseInFlight = false;
  } else if (obj?.type === 'resumed') {
    setRunActivity('running');
    extensionState.resumeInFlight = false;
  } else if (obj?.type === 'trainingFinished') {
    setRunActivity(null);
    extensionState.pauseInFlight = false;
    extensionState.resumeInFlight = false;
  }

  if (obj?.type === 'log') {
    const run_id = obj.run_id || extensionState.currentRunId || 'live';
    const session_id = obj.session_id || obj.sessionId || extensionState.currentSessionId || null;
    if (run_id && session_id && extensionState.currentSessionId && session_id === extensionState.currentSessionId) {
      extensionState.nativeRunsThisSession.add(String(run_id));
    }
    const level = obj.level || 'info';
    const text: string = String(obj.text || '');

    const pRaw = (obj.phase && String(obj.phase).toLowerCase()) || null;
    let phase: 'train' | 'test' | 'info' =
      (pRaw === 'train' || pRaw === 'test' || pRaw === 'info') ? (pRaw as any) : 'info';

    let step: number | null = null;
    let epoch: number | null = null;

    const mEpoch = text.match(/^\s*epoch\s+(\d+)\s*$/i);
    if (mEpoch && phase === 'info') { epoch = Number(mEpoch[1]); phase = 'train'; }

    const mTest = text.match(/step\s+(\d+)\s*:\s*test[_\s]*loss\s*=\s*([0-9.eE+\-]+)/i);
    if (mTest && phase === 'info') { step = Number(mTest[1]); phase = 'test'; }
    if (/\btesting\b/i.test(text) && phase === 'info') phase = 'test';
    if (/\b(?:eval|evaluation)\b/i.test(text) && /\b(loss|acc|metric)\b/i.test(text) && phase === 'info') phase = 'test';

    const mTrain = text.match(/step\s+(\d+)\s*:\s*train[_\s]*loss\s*=\s*([0-9.eE+\-]+).*?val[_\s]*loss\s*=\s*([0-9.eE+\-]+|None)/i);
    if (mTrain && phase === 'info') { step = Number(mTrain[1]); phase = 'train'; }
    if ((/\btrain[_\s]*loss\b/i.test(text) || /\bval[_\s]*loss\b/i.test(text)) && phase === 'info') {
      phase = 'train';
    }

    const tsMs = (Number(obj.ts) > 0 ? Math.round(Number(obj.ts) * 1000) : Date.now());

    const hasStepInText = /\bstep\b\s*(?:[:#]\s*)?\d+(?:\s*[:#])?/i.test(text);
    const stepForUI = hasStepInText ? null : step;

    try {
      insertLog({ run_id, session_id, level, text, phase, step, epoch, ts: tsMs });
    } catch {}

    post({ type: 'log', run_id, session_id, level, text, phase, step: stepForUI, epoch });
    return;
  }

  if (obj?.type === 'reportData') {
    return;
  }

  if (obj?.type === 'merge_gating') {
    post({
      type: 'merge_gating',
      reason: obj.reason || 'unknown',
      parents: Array.isArray(obj.parents) ? obj.parents.map(String) : [],
      step: Number(obj.step) || null,
      run_id: obj.run_id || extensionState.currentRunId || null,
      ...obj,
    });
    return;
  }

  if (obj && obj.type === 'trainStep') {
    const run_id = obj.run_id || extensionState.currentRunId || 'live';
    const num = (value: any) => (Number.isFinite(value) ? Number(value) : null);

    extensionState.currentRunId = run_id;
    if (!extensionState.modelNavSelectedRunId) {
      extensionState.modelNavSelectedRunId = run_id;
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
    if (extensionState.runActivityState !== 'running') {
      setRunActivity('running');
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
    if (m && m[1]) { extensionState.currentSessionId = m[1]; postCurrentSession(); }
  }

  if (obj && obj.type === 'newRun') {
    const id = String(obj.run_id || '').trim();
    if (id) extensionState.nativeRunsThisSession.add(id);
    if (!id) return;

    if (obj.session_id || obj.sessionId) {
      extensionState.currentSessionId = String(obj.session_id || obj.sessionId);
      postCurrentSession();
    }

    const parents = normalizeParents(obj);
    const project = obj.project || 'default';
    const name = obj.run_name || id;

    if (!extensionState.seenRuns.has(id)) {
      extensionState.seenRuns.add(id);
      extensionState.currentRunId = id;

      const primaryParent = (parents && parents[0]) ?? null;
      const existingRuns = listRuns() as Array<{ run_id: string }>;
      const existsInDb = existingRuns.some(r => r.run_id === id);

      if (!existsInDb) {
        insertRun(id, project, name, primaryParent);
      }
    } else {
      extensionState.currentRunId = id;
    }

    post({
      type: 'newRun',
      run_id: id,
      parents,
      meta: obj.meta,
      session_id: extensionState.currentSessionId || null,
    });
    post({ type: 'runs', rows: listRuns() });
    return;
  }

  if (obj?.type === 'auto_test_complete') {
    const runId = String(obj.run_id || extensionState.currentRunId || '').trim();
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
    return;
  }

  post(obj);

  try {
    switch (obj?.type) {
      case 'session_started': {
        extensionState.currentRunId = obj.run_id || extensionState.currentRunId;
        if (obj.session_id || obj.sessionId) {
          extensionState.currentSessionId = String(obj.session_id || obj.sessionId);
          postCurrentSession();
        }

        if (extensionState.needDiskCleanOnNextTrainer && trainerActive()) {
          extensionState.needDiskCleanOnNextTrainer = false;
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
