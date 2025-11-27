export type CompareView = 'train' | 'test' | 'info';

export type WebMsg =
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

export type Ctl =
  | { cmd: 'pause' }
  | { cmd: 'resume' }
  | { cmd: 'save_ckpt' }
  | { cmd: 'fork'; payload?: any }
  | { cmd: 'merge'; payload: { paths?: string[]; parents?: string[]; strategy?: string; new_name?: string } };

export type RunActivityState = 'running' | 'paused' | null;

export type StepRow = {
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

export type TrainerResetSeed = {
  run_id?: string;
  display_name?: string;
  project?: string;
  session_id?: string;
};

export type UiLogLike = {
  text?: string;
  step?: number | null;
  ts?: number | null;
};
