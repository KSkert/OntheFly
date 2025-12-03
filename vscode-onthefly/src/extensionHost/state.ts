import * as vscode from 'vscode';
import { RunActivityState, UiLogLike } from './types';

export const LAST_EXPORT_DIR_KEY = 'onthefly.lastExportDir';

export type ExtensionRuntimeState = {
  panel: vscode.WebviewPanel | null;
  currentRunId: string | null;
  seenRuns: Set<string>;
  currentSessionId: string | null;
  nativeRunsThisSession: Set<string>;
  modelNavSelectedRunId: string | null;
  runActivityState: RunActivityState;
  pauseInFlight: boolean;
  resumeInFlight: boolean;
  needDiskCleanOnNextTrainer: boolean;
  lastExtensionContext: vscode.ExtensionContext | null;
  hasTrainerConnectedOnce: boolean;
  pendingResumeAwaitingFirstRun: boolean;
  compatGateActive: boolean;
  compatErrorMessage: string | null;
  compatNotified: boolean;
};

export const extensionState: ExtensionRuntimeState = {
  panel: null,
  currentRunId: null,
  seenRuns: new Set<string>(),
  currentSessionId: null,
  nativeRunsThisSession: new Set<string>(),
  modelNavSelectedRunId: null,
  runActivityState: null,
  pauseInFlight: false,
  resumeInFlight: false,
  needDiskCleanOnNextTrainer: true,
  lastExtensionContext: null,
  hasTrainerConnectedOnce: false,
  pendingResumeAwaitingFirstRun: false,
  compatGateActive: false,
  compatErrorMessage: null,
  compatNotified: false,
};

export function setRunActivityState(activity: RunActivityState): void {
  extensionState.runActivityState = activity;
}

export function post(msg: any): void {
  try {
    extensionState.panel?.webview.postMessage(msg);
  } catch (e) {
    console.log('[EXT->WEB] post threw:', e);
  }
}

export function postErr(e: any): void {
  post({ type: 'error', text: String(e?.message || e) });
}

export function postStatus(connected: boolean): void {
  const activity = extensionState.runActivityState;
  const running = connected && activity === 'running';
  const paused = connected && activity === 'paused';

  post({
    type: 'status',
    connected,
    running,
    paused,
    run_id: extensionState.currentRunId || null,
  });
}

export function postCurrentSession(): void {
  if (extensionState.currentSessionId) {
    post({ type: 'fs.session.current', id: extensionState.currentSessionId });
  }
}

function isUiLogLike(v: unknown): v is UiLogLike & Record<string, any> {
  return !!v && typeof v === 'object';
}

export function stripUiOnlyFields(rows: unknown): unknown {
  const cleanOne = (r: unknown): unknown => {
    if (!isUiLogLike(r)) return r;
    const { ts, ...rest } = r;

    if (typeof rest.text === 'string') {
      const mentionsStep = /\bstep\b\s*(?:[:#]\s*)?\d+(?:\s*[:#])?/i.test(rest.text);
      if (mentionsStep) delete (rest as any).step;
    }
    return rest;
  };

  return Array.isArray(rows) ? rows.map(cleanOne) : cleanOne(rows);
}
