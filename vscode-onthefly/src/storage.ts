// storage.ts
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as crypto from 'crypto';
import type * as BetterSqlite3Types from 'better-sqlite3';
const BetterSqlite3 = require('better-sqlite3') as typeof import('better-sqlite3');

let db: BetterSqlite3Types.Database | null = null;
let dbFolder: string | undefined;
let dbPath: string | undefined;


// Single-flight async initializer
let initPromise: Promise<void> | null = null;

// Prepared statement cache
let _insStmt: BetterSqlite3Types.Statement | null = null;

export function isReady() { return !!db; }
export function whenReady(): Promise<void> { return initPromise ?? Promise.resolve(); }

function applyPragmas(d: BetterSqlite3Types.Database) {
  d.pragma('journal_mode = WAL');
  d.pragma('synchronous = NORMAL');
  d.pragma('temp_store = MEMORY');
  d.pragma('cache_size = -8000');
  d.pragma('busy_timeout = 3000');
  d.pragma('wal_autocheckpoint = 512');
}

function ensureSchema(d: BetterSqlite3Types.Database) {
  // Base tables (excluding deprecated metric fields)
  d.exec(
    ''
      + 'PRAGMA foreign_keys=ON;'
      + '\n\n'
      + 'CREATE TABLE IF NOT EXISTS runs ('
      + '  run_id TEXT PRIMARY KEY,'
      + '  parent_run_id TEXT,'
      + '  project TEXT,'
      + '  name TEXT,'
      + '  created_at INTEGER'
      + ');'
      + '\n\n'
      + 'CREATE TABLE IF NOT EXISTS run_subsets ('
      + '  run_id TEXT PRIMARY KEY,'
      + '  indices TEXT'
      + ');'
      + '\n\n'
      + 'CREATE TABLE IF NOT EXISTS checkpoints ('
      + '  ckpt_id TEXT PRIMARY KEY,'
      + '  run_id TEXT,'
      + '  step INTEGER,'
      + '  path TEXT,'
      + '  created_at INTEGER'
      + ');'
      + '\n\n'
      + "CREATE TABLE IF NOT EXISTS metrics ("
      + "  run_id TEXT,"
      + "  step INTEGER,"
      + "  loss REAL,"
      + "  val_loss REAL,"
      + "  accuracy REAL,"
      + "  lr REAL,"
      + "  grad_norm REAL,"
      + "  weight_norm REAL,"
      + "  activation_zero_frac REAL,"
      + "  throughput REAL,"
      + "  mem_vram REAL,"
      + "  gpu_util REAL,"
      + "  ts INTEGER"
      + ");"
      + '\n\n'
      + 'CREATE TABLE IF NOT EXISTS summaries ('
      + '  run_id TEXT PRIMARY KEY,'
      + '  final_loss REAL,'
      + '  val_loss REAL'
      + ');'
      + '\n\n'
      + 'CREATE TABLE IF NOT EXISTS kv ( k TEXT PRIMARY KEY, v TEXT );'
      + '\n\n'
      + 'CREATE TABLE IF NOT EXISTS run_edges ('
      + '  child_run_id  TEXT NOT NULL,'
      + '  parent_run_id TEXT NOT NULL,'
      + '  PRIMARY KEY (child_run_id, parent_run_id)'
      + ');'
      + '\n\n'
      + 'CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step);'
      + 'CREATE INDEX IF NOT EXISTS idx_metrics_run       ON metrics(run_id);'
      + 'CREATE INDEX IF NOT EXISTS idx_edges_child       ON run_edges(child_run_id);'
      + 'CREATE INDEX IF NOT EXISTS idx_edges_parent      ON run_edges(parent_run_id);'
  );

  // ── Free-form print/log lines
  d.exec(
    ''
      + 'CREATE TABLE IF NOT EXISTS logs ('
      + '  id         INTEGER PRIMARY KEY AUTOINCREMENT,'
      + '  run_id     TEXT,'
      + '  session_id TEXT,'
      + '  level      TEXT,'
      + '  text       TEXT,'
      + "  phase      TEXT,"   // 'train' | 'test' | 'info'
      + '  step       INTEGER,'
      + '  epoch      INTEGER,'
      + '  ts         INTEGER'
      + ');'
      + 'CREATE INDEX IF NOT EXISTS idx_logs_run   ON logs(run_id);'
      + 'CREATE INDEX IF NOT EXISTS idx_logs_phase ON logs(phase);'
  );

  // per-step test losses
  d.exec(
    ''
      + 'CREATE TABLE IF NOT EXISTS test_metrics ('
      + '  run_id TEXT,'
      + '  step   INTEGER,'
      + '  loss   REAL,'
      + '  ts     INTEGER'
      + ');'
      + 'CREATE INDEX IF NOT EXISTS idx_test_metrics_run_step ON test_metrics(run_id, step);'
  );

  d.exec(
    ''
      + 'CREATE TABLE IF NOT EXISTS session_final_models ('
      + '  session_id    TEXT PRIMARY KEY,'
      + '  run_id        TEXT,'
      + '  ckpt_path     TEXT,'
      + '  step          INTEGER,'
      + '  avg_test_loss REAL,'
      + '  updated_at    INTEGER'
      + ');'
  );

  // Backfill edges once from legacy column (idempotent)
  d.exec(
    ''
      + 'INSERT OR IGNORE INTO run_edges(child_run_id, parent_run_id) '
      + 'SELECT run_id, parent_run_id FROM runs WHERE parent_run_id IS NOT NULL;'
  );

  // ---- Metrics migration: remove unwanted columns if they exist ----
  {
    const infoStmt = d.prepare("PRAGMA table_info('metrics')");
    const cols = infoStmt.all() as Array<{ name: string }>;
    const hasTable = cols.length > 0;
    const colSet = new Set(cols.map(c => c.name));
    const forbidden = ['theta', 'c', 'genBound', 'gen_bound', 'constraintSatisfied'];
    const desired = [
      'run_id',
      'step',
      'loss',
      'val_loss',
      'accuracy',
      'lr',
      'grad_norm',
      'weight_norm',
      'activation_zero_frac',
      'throughput',
      'mem_vram',
      'gpu_util',
      'ts',
    ];
    const hasForbidden = cols.some(c => forbidden.includes(c.name));
    const matchesDesiredExactly =
      hasTable &&
      cols.length === desired.length &&
      desired.every(k => colSet.has(k));

    if (!hasTable || hasForbidden || !matchesDesiredExactly) {
      const hasValLoss = colSet.has('val_loss');
      const hasAccuracy = colSet.has('accuracy');
      const hasLr = colSet.has('lr');
      const hasGradNorm = colSet.has('grad_norm');
      const hasWeightNorm = colSet.has('weight_norm');
      const hasZeroFrac = colSet.has('activation_zero_frac');
      const hasThroughput = colSet.has('throughput');
      const hasMemVram = colSet.has('mem_vram');
      const hasGpuUtil = colSet.has('gpu_util');
      const hasTs  = colSet.has('ts');

      d.exec(
        ''
          + 'PRAGMA foreign_keys=OFF;'
          + 'BEGIN IMMEDIATE;'
          + 'DROP INDEX IF EXISTS idx_metrics_run_step;'
          + 'DROP INDEX IF EXISTS idx_metrics_run;'
          + 'CREATE TABLE IF NOT EXISTS metrics_new ('
          + '  run_id TEXT,'
          + '  step INTEGER,'
          + '  loss REAL,'
          + '  val_loss REAL,'
          + '  accuracy REAL,'
          + '  lr REAL,'
          + '  grad_norm REAL,'
          + '  weight_norm REAL,'
          + '  activation_zero_frac REAL,'
          + '  throughput REAL,'
          + '  mem_vram REAL,'
          + '  gpu_util REAL,'
          + '  ts INTEGER'
          + ');'
          + (hasTable
              ? ''
                + 'INSERT INTO metrics_new(run_id, step, loss, val_loss, accuracy, lr, grad_norm, weight_norm, activation_zero_frac, throughput, mem_vram, gpu_util, ts) '
                + 'SELECT '
                + '  run_id, '
                + '  step, '
                + '  loss, '
                + (hasValLoss ? '  val_loss, ' : '  NULL AS val_loss, ')
                + (hasAccuracy ? '  accuracy, ' : '  NULL AS accuracy, ')
                + (hasLr ? '  lr, ' : '  NULL AS lr, ')
                + (hasGradNorm ? '  grad_norm, ' : '  NULL AS grad_norm, ')
                + (hasWeightNorm ? '  weight_norm, ' : '  NULL AS weight_norm, ')
                + (hasZeroFrac ? '  activation_zero_frac, ' : '  NULL AS activation_zero_frac, ')
                + (hasThroughput ? '  throughput, ' : '  NULL AS throughput, ')
                + (hasMemVram ? '  mem_vram, ' : '  NULL AS mem_vram, ')
                + (hasGpuUtil ? '  gpu_util, ' : '  NULL AS gpu_util, ')
                + (hasTs ? '  ts ' : "  (CAST(strftime('%s','now') AS INTEGER) * 1000) AS ts ")
                + 'FROM metrics;'
                + 'DROP TABLE IF EXISTS metrics;'
              : ''
            )
          + 'ALTER TABLE metrics_new RENAME TO metrics;'
          + 'CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step);'
          + 'CREATE INDEX IF NOT EXISTS idx_metrics_run       ON metrics(run_id);'
          + 'COMMIT;'
          + 'PRAGMA foreign_keys=ON;'
      );
    }
  }

  // ---- Reports table migration ----
  {
    const infoStmt = d.prepare("PRAGMA table_info('report_loss_dist')");
    const cols = infoStmt.all() as Array<{ name: string }>;
    const hasTable  = cols.length > 0;
    const hasKind   = cols.some(c => c.name === 'kind');
    const hasLosses = cols.some(c => c.name === 'losses');
    const hasSample = cols.some(c => c.name === 'sample_indices');

    if (!hasTable || hasKind || !hasLosses || !hasSample) {
      d.exec(
        ''
          + 'DROP TABLE IF EXISTS report_loss_dist;'
          + 'DROP INDEX IF EXISTS idx_report_loss_dist_run;'
          + 'CREATE TABLE IF NOT EXISTS report_loss_dist ('
          + '  run_id          TEXT NOT NULL,'
          + '  subset_on       TEXT NOT NULL,'
          + '  losses          TEXT NOT NULL,'
          + "  sample_indices  TEXT NOT NULL DEFAULT '[]',"
          + '  note            TEXT,'
          + '  at_step         INTEGER,'
          + '  at_epoch        INTEGER,'
          + '  samples         INTEGER,'
          + "  created_at      INTEGER NOT NULL DEFAULT (strftime('%s','now')),"
          + '  PRIMARY KEY (run_id, subset_on)'
          + ');'
          + 'CREATE INDEX IF NOT EXISTS idx_report_loss_dist_run ON report_loss_dist(run_id);'
      );
    }
  }
}

// Make a byte-for-byte copy of the current DB to targetPath.
// Uses VACUUM INTO when available; falls back to a plain file copy.
// We also try to leave the copied file in "single file" mode (no WAL sidecars).
function _copyDbToSingleFile(targetPath: string) {
  if (!db || !dbPath) return; // guard if not ready
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });

  try { db.pragma('wal_checkpoint(FULL)'); } catch {}

  let ok = false;
  try {
    const esc = targetPath.replace(/'/g, "''");
    db.exec("VACUUM INTO '" + esc + "'");
    ok = true;
  } catch {
    if (fs.existsSync(dbPath)) {
      fs.copyFileSync(dbPath, targetPath);
      ok = true;
    }
  }
  if (!ok) return;

  try {
    const out = new BetterSqlite3(targetPath, { readonly: false });
    out.pragma('journal_mode = DELETE');
    out.exec('VACUUM');
    out.close();
  } catch {}
}

export function initStorage(_context: vscode.ExtensionContext): Promise<void> {
  if (db) return Promise.resolve();     // already open
  if (initPromise) return initPromise;  // single-flight

  initPromise = (async () => {
    // Always use a fresh temp folder per session
    if (!dbFolder) {
      dbFolder = path.join(os.tmpdir(), 'onthefly-session-' + crypto.randomUUID());
    }

    fs.mkdirSync(dbFolder, { recursive: true });
    dbPath = path.join(dbFolder, 'runs.sqlite');

    db = new BetterSqlite3(dbPath);
    applyPragmas(db);
    ensureSchema(db);

    _insStmt = null;
  })().finally(() => { initPromise = null; });

  return initPromise;
}


export function getDbPath() { return dbPath || ''; }

export function closeStorage() {
  try { db?.close?.(); } catch {}
  db = null;
  _insStmt = null;

  if (dbFolder) {
    try { fs.rmSync(dbFolder, { recursive: true, force: true }); } catch {}
  }

  dbFolder = undefined;
  dbPath = undefined;
}


function loadSessionFrom(sourcePath: string, _context: vscode.ExtensionContext) {
  // Ensure we have a temp folder / dbPath even if initStorage() wasn’t called
  if (!dbFolder) {
    dbFolder = path.join(os.tmpdir(), 'onthefly-session-' + crypto.randomUUID());
  }
  if (!dbPath) {
    dbPath = path.join(dbFolder, 'runs.sqlite');
  }

  try { db?.close?.(); } catch {}

  fs.mkdirSync(dbFolder, { recursive: true });
  fs.copyFileSync(sourcePath, dbPath);

  db = new BetterSqlite3(dbPath);
  applyPragmas(db);
  ensureSchema(db);
  _insStmt = null;
}


export function checkpointNow() {
  try { db?.pragma('wal_checkpoint(PASSIVE)'); } catch {}
}

/* ─────────────────────────── Bundle helpers (for importer-ready handoff) ─────────────────────────── */

// Copy sqlite + referenced checkpoint files into a directory with a tiny manifest.
// After export, the importer can zip the directory, share it, and load it with loadBundle().
function _safeSlug(base: string): string {
  const slug = (base || '').trim().toLowerCase().replace(/[^a-z0-9_-]+/gi, '-').replace(/^-+|-+$/g, '');
  return slug || 'session';
}

export function exportBundle(dir: string) {
  if (!db) { console.warn('[storage] exportBundle called before db initialized'); return; }

  fs.mkdirSync(dir, { recursive: true });

  // 1) Copy DB
  const sqlitePath = path.join(dir, 'runs.sqlite');
  _copyDbToSingleFile(sqlitePath);

  // 2) Copy checkpoints into bundle/checkpoints/<ckpt_id>.pt
  const ckptRows = db.prepare('SELECT ckpt_id, path FROM checkpoints').all() as Array<{ ckpt_id: string, path: string }>;
  const ckptDir = path.join(dir, 'checkpoints');
  fs.mkdirSync(ckptDir, { recursive: true });

  const map: Record<string, string> = {};
  for (const r of ckptRows) {
    if (!r.path || !fs.existsSync(r.path)) continue;
    const fname = r.ckpt_id + '.pt';
    fs.copyFileSync(r.path, path.join(ckptDir, fname));
    map[r.ckpt_id] = 'checkpoints/' + fname;
  }

  // 3) Copy final tested models into dedicated files
  const finalRows = db.prepare(
    'SELECT session_id, run_id, ckpt_path FROM session_final_models WHERE ckpt_path IS NOT NULL'
  ).all() as Array<{ session_id: string; run_id: string; ckpt_path: string }>;
  const finalDir = path.join(dir, 'final_models');
  const finalMap: Record<string, string> = {};
  if (finalRows.length) fs.mkdirSync(finalDir, { recursive: true });

  finalRows.forEach((row, idx) => {
    if (!row?.session_id || !row.ckpt_path || !fs.existsSync(row.ckpt_path)) return;
    const slugBase = _safeSlug(row.session_id || row.run_id || `session-${idx + 1}`);
    let fname = `${slugBase}-final.pt`;
    let attempt = 2;
    while (fs.existsSync(path.join(finalDir, fname))) {
      fname = `${slugBase}-final-${attempt}.pt`;
      attempt += 1;
    }
    const dest = path.join(finalDir, fname);
    try {
      fs.copyFileSync(row.ckpt_path, dest);
      finalMap[row.session_id] = path.relative(dir, dest);
    } catch (e) {
      console.warn('[storage] failed to copy final model for session', row.session_id, e);
    }
  });

  // 4) Manifest to help remap on import
  const manifest = {
    schema_version: '1.1',
    sqlite: 'runs.sqlite',
    checkpoints: map,
    final_models: finalMap,
  };
  fs.writeFileSync(path.join(dir, 'bundle.json'), JSON.stringify(manifest, null, 2));
}

// Load a bundle directory created by exportBundle(); also rewrites checkpoint paths in-place.
export function loadBundle(dir: string, _context: vscode.ExtensionContext) {
  const bundlePath = path.join(dir, 'bundle.json');
  const raw = fs.readFileSync(bundlePath, 'utf8');
  const manifest = JSON.parse(raw) as {
    checkpoints: Record<string, string>;
    final_models?: Record<string, string>;
  };

  // 1) Load sqlite from bundle
  const sqlitePath = path.join(dir, 'runs.sqlite');
  loadSessionFrom(sqlitePath, _context);

  // 2) Rewrite checkpoint paths to bundle-local absolute paths
  if (!db) return;
  const tx = db.transaction(() => {
    const upd = db!.prepare('UPDATE checkpoints SET path=? WHERE ckpt_id=?');
    const entries = manifest.checkpoints || {};
    for (const ckptId in entries) {
      if (!Object.prototype.hasOwnProperty.call(entries, ckptId)) continue;
      const rel = entries[ckptId];
      const abs = path.isAbsolute(rel) ? rel : path.join(dir, rel);
      upd.run(abs, ckptId);
    }

    const finals = manifest.final_models || {};
    if (Object.keys(finals).length) {
      const updFinal = db!.prepare('UPDATE session_final_models SET ckpt_path=? WHERE session_id=?');
      for (const sessionId in finals) {
        if (!Object.prototype.hasOwnProperty.call(finals, sessionId)) continue;
        const rel = finals[sessionId];
        if (!rel) continue;
        const abs = path.isAbsolute(rel) ? rel : path.join(dir, rel);
        updFinal.run(abs, sessionId);
      }
    }
  });
  tx();
}

/* ─────────────────────────── Write helpers ─────────────────────────── */

export function insertRun(
  run_id: string,
  project: string,
  name: string,
  parents?: string[] | string | null
) {
  if (!db) { console.error('[storage] insertRun called before db initialized'); return; }

  const toParents = (p: any): string[] => {
    if (Array.isArray(p)) return p.filter(Boolean).map(String);
    if (p == null || p === '') return [];
    return [String(p)];
  };
  const ps = toParents(parents);
  const primary = ps[0] ?? null;

  const now = Date.now();
  const tx = db.transaction(() => {
    db!.prepare(
      'INSERT OR IGNORE INTO runs(run_id, parent_run_id, project, name, created_at) VALUES (?,?,?,?,?)'
    ).run(run_id, primary, project, name, now);

    db!.prepare('DELETE FROM run_edges WHERE child_run_id=?').run(run_id);
    if (ps.length) {
      const ins = db!.prepare('INSERT OR IGNORE INTO run_edges(child_run_id, parent_run_id) VALUES (?,?)');
      for (const p of ps) ins.run(run_id, p);
    }
  });
  tx();
}

export function insertCheckpoint(ckpt_id: string, run_id: string, step: number, p: string) {
  if (!db) { console.error('[storage] insertCheckpoint called before db initialized'); return; }
  db.prepare(
    'INSERT OR REPLACE INTO checkpoints(ckpt_id,run_id,step,path,created_at) VALUES (?,?,?,?,?)'
  ).run(ckpt_id, run_id, step, p, Date.now());
}

type MetricInsertRow = {
  step: number;
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
};

function insStmt() {
  if (_insStmt) return _insStmt;
  if (!db) throw new Error('Database not initialized. Call initStorage() first.');
  _insStmt = db.prepare(
    'INSERT INTO metrics( run_id, step, loss, val_loss, accuracy, lr, grad_norm, weight_norm, activation_zero_frac, throughput, mem_vram, gpu_util, ts ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'
  );
  return _insStmt;
}

export function insertMetric(run_id: string, row: MetricInsertRow) {
  if (!db) {
    console.error('[storage] insertMetric called before db initialized');
    return;
  }
  insStmt().run(
    run_id,
    row.step,
    row.loss ?? null,
    row.val_loss ?? null,
    row.accuracy ?? null,
    row.lr ?? null,
    row.grad_norm ?? null,
    row.weight_norm ?? null,
    row.activation_zero_frac ?? null,
    row.throughput ?? null,
    row.mem_vram ?? null,
    row.gpu_util ?? null,
    Date.now()
  );
}

// batch insert
(insertMetric as any).batch = (rows: Array<{
  run_id: string;
  step: number;
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
}>) => {
  if (!rows || rows.length === 0) return;
  if (!db) {
    console.error('[storage] insertMetric.batch called before db initialized');
    return;
  }
  const runMany = db.transaction((xs: typeof rows) => {
    const st = insStmt();
    const now = Date.now();
    for (const r of xs) {
      st.run(
        r.run_id,
        r.step,
        r.loss ?? null,
        r.val_loss ?? null,
        r.accuracy ?? null,
        r.lr ?? null,
        r.grad_norm ?? null,
        r.weight_norm ?? null,
        r.activation_zero_frac ?? null,
        r.throughput ?? null,
        r.mem_vram ?? null,
        r.gpu_util ?? null,
        now
      );
    }
  });
  runMany(rows);
};

export function upsertSummary(
  run_id: string,
  final_loss?: number | null,
  val_loss?: number | null
) {
  if (!db) { console.error('[storage] upsertSummary called before db initialized'); return; }

  const exists = db.prepare('SELECT 1 FROM summaries WHERE run_id=?').get(run_id);
  if (exists) {
    db.prepare(
      'UPDATE summaries SET final_loss=COALESCE(?, final_loss), val_loss=COALESCE(?, val_loss) WHERE run_id=?'
    ).run(final_loss ?? null, val_loss ?? null, run_id);
  } else {
    db.prepare(
      'INSERT INTO summaries(run_id,final_loss,val_loss) VALUES (?,?,?)'
    ).run(run_id, final_loss ?? null, val_loss ?? null);
  }
}

export function upsertReportLossDist(
  run_id: string,
  subset_on: 'train' | 'val',
  payload: {
    losses: number[];
    sample_indices: number[];
    note?: string;
    at_step?: number | null;
    at_epoch?: number | null;
    samples?: number | null;
  }
) {
  if (!db) { console.error('[storage] upsertReportLossDist called before db initialized'); return; }

  const samples = payload.samples ?? payload.losses.length;
  const lossesJson = JSON.stringify(payload.losses || []);
  const idxJson = JSON.stringify((payload.sample_indices || []).map(v => Math.trunc(Number(v))));
  db.prepare(
    ''
      + 'INSERT INTO report_loss_dist(run_id, subset_on, losses, sample_indices, note, at_step, at_epoch, samples, created_at) '
      + 'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) '
      + 'ON CONFLICT(run_id, subset_on) DO UPDATE SET '
      + '  losses         = excluded.losses, '
      + '  sample_indices = excluded.sample_indices, '
      + '  note           = excluded.note, '
      + '  at_step        = excluded.at_step, '
      + '  at_epoch       = excluded.at_epoch, '
      + '  samples        = excluded.samples, '
      + '  created_at     = excluded.created_at'
  ).run(
    run_id,
    subset_on,
    lossesJson,
    idxJson,
    payload.note ?? null,
    payload.at_step ?? null,
    payload.at_epoch ?? null,
    samples,
    Date.now()
  );
}

export function getReportLossDist(run_id: string, subset_on: 'train' | 'val') {
  if (!db) return null;
  try {
    const row = db.prepare(
      'SELECT run_id, subset_on, losses, sample_indices, note, at_step, at_epoch, samples FROM report_loss_dist WHERE run_id=? AND subset_on=?'
    ).get(run_id, subset_on) as any;

    if (!row) return null;

    let losses: number[] = [];
    let sample_indices: number[] = [];

    try {
      const arr = JSON.parse(row.losses);
      if (Array.isArray(arr)) losses = arr.map(Number).filter(Number.isFinite);
    } catch {}

    try {
      const arr = JSON.parse(row.sample_indices);
      if (Array.isArray(arr)) {
        sample_indices = arr.map((v: any) => Math.trunc(Number(v)))
                            .filter((v: number) => Number.isFinite(v) && v >= 0);
      }
    } catch {}

    if (sample_indices.length !== losses.length) {
      sample_indices = Array.from({ length: losses.length }, (_, i) => i);
    }

    return {
      run_id,
      subset_on,
      losses,
      sample_indices,
      note: row.note ?? '',
      at_step: row.at_step ?? null,
      at_epoch: row.at_epoch ?? null,
      samples: Number.isFinite(row.samples) ? row.samples : losses.length,
    };
  } catch (e: any) {
    if (String(e && e.message || e).includes('no such table')) return null;
    throw e;
  }
}

/* ─────────────────────────── Subset helpers ─────────────────────────── */
export function setRunSubset(run_id: string, indices: number[]) {
  if (!db) { console.error('[storage] setRunSubset called before db initialized'); return; }
  const payload = JSON.stringify(Array.from(indices || []));
  db.prepare('INSERT OR REPLACE INTO run_subsets(run_id, indices) VALUES(?,?)').run(run_id, payload);
}

export function getRunSubset(run_id: string): number[] {
  if (!db) return [];
  const row = db.prepare('SELECT indices FROM run_subsets WHERE run_id=?').get(run_id) as { indices?: string } | undefined;
  if (!row || !row.indices) return [];
  try {
    const arr = JSON.parse(row.indices);
    if (Array.isArray(arr)) return arr.map((x: any) => Number(x)).filter((x: number) => Number.isInteger(x) && x >= 0);
  } catch {}
  return [];
}

/* ─────────────────────────── Read helpers ─────────────────────────── */

export function listRuns() {
  if (!db) {
    console.warn('[storage] listRuns called before db initialized');
    return [];
  }
  const rows = db.prepare('SELECT * FROM runs ORDER BY created_at DESC').all() as any[];

  const edges = db.prepare(
    'SELECT child_run_id AS child, parent_run_id AS parent FROM run_edges'
  ).all() as Array<{ child: string; parent: string }>;

  const parentsMap = new Map<string, string[]>();
  for (const e of edges) {
    if (!parentsMap.has(e.child)) parentsMap.set(e.child, []);
    parentsMap.get(e.child)!.push(e.parent);
  }

  return rows.map(r => ({
    ...r,
    parents: parentsMap.get(r.run_id) || [],
  }));
}

export function getRunRows(run_id: string) {
  if (!db) return [];
  return db.prepare(
    'SELECT step, loss, val_loss, accuracy, lr, grad_norm, weight_norm, activation_zero_frac, throughput, mem_vram, gpu_util FROM metrics WHERE run_id=? ORDER BY step ASC'
  ).all(run_id);
}

export function getRunSummary(run_id: string) {
  if (!db) return [];
  return db.prepare('SELECT * FROM summaries WHERE run_id=?').all(run_id);
}

// Simple kv helpers
export function setSavingEverything(_context: vscode.ExtensionContext, on: boolean) {
  if (!db) { console.warn('[storage] setSavingEverything before init'); return; }
  db.prepare('INSERT OR REPLACE INTO kv(k,v) VALUES(\'saving\',?)').run(on ? '1' : '0');
}
export function isSavingEverything(): boolean {
  if (!db) return false;
  const row = db.prepare("SELECT v FROM kv WHERE k='saving'").get() as { v?: string } | undefined;
  return !!(row && row.v === '1');
}

export type LogInsert = {
  run_id: string;
  session_id?: string | null;
  level?: string | null;
  text: string;
  phase?: 'train'|'test'|'info'|null;
  step?: number | null;
  epoch?: number | null;
  ts?: number | null;
};

export function insertLog(row: LogInsert) {
  if (!db) { console.error('[storage] insertLog before init'); return; }
  db.prepare(
    'INSERT INTO logs(run_id,session_id,level,text,phase,step,epoch,ts) VALUES (?,?,?,?,?,?,?,?)'
  ).run(
    row.run_id,
    row.session_id ?? null,
    row.level ?? 'info',
    row.text,
    row.phase ?? 'info',
    Number.isFinite(row.step as number) ? (row.step as number) : null,
    Number.isFinite(row.epoch as number) ? (row.epoch as number) : null,
    row.ts ?? Date.now()
  );
}

export function getLogs(run_id: string, phase?: 'train'|'test'|'info') {
  if (!db) return [];
  if (phase) {
    return db.prepare(
      'SELECT id, run_id, session_id, level, text, phase, step, epoch, ts FROM logs WHERE run_id=? AND phase=? ORDER BY ts ASC, id ASC'
    ).all(run_id, phase);
  }
  return db.prepare(
    'SELECT id, run_id, session_id, level, text, phase, step, epoch, ts FROM logs WHERE run_id=? ORDER BY ts ASC, id ASC'
  ).all(run_id);
}

export function insertTestMetric(run_id: string, step: number, loss: number | null) {
  if (!db) { console.error('[storage] insertTestMetric before init'); return; }
  db.prepare(
    'INSERT INTO test_metrics(run_id, step, loss, ts) VALUES (?,?,?,?)'
  ).run(run_id, step, Number.isFinite(Number(loss)) ? Number(loss) : null, Date.now());
}

export function getTestRows(run_id: string) {
  if (!db) return [];
  return db.prepare(
    'SELECT step, loss FROM test_metrics WHERE run_id=? ORDER BY step ASC'
  ).all(run_id);
}

export type FinalModelRow = {
  session_id: string;
  run_id: string;
  ckpt_path: string;
  step: number;
  avg_test_loss?: number | null;
};

export function upsertFinalModel(row: FinalModelRow) {
  if (!db) { console.error('[storage] upsertFinalModel before init'); return; }
  const avgLoss = Number.isFinite(Number(row.avg_test_loss)) ? Number(row.avg_test_loss) : null;
  db.prepare(
    ''
      + 'INSERT INTO session_final_models(session_id, run_id, ckpt_path, step, avg_test_loss, updated_at) '
      + 'VALUES (?,?,?,?,?,?) '
      + 'ON CONFLICT(session_id) DO UPDATE SET '
      + '  run_id=excluded.run_id,'
      + '  ckpt_path=excluded.ckpt_path,'
      + '  step=excluded.step,'
      + '  avg_test_loss=excluded.avg_test_loss,'
      + '  updated_at=excluded.updated_at'
  ).run(
    row.session_id,
    row.run_id,
    row.ckpt_path,
    row.step,
    avgLoss,
    Date.now()
  );
}

export function getFinalModel(session_id: string) {
  if (!db) return null;
  const row = db.prepare(
    'SELECT session_id, run_id, ckpt_path, step, avg_test_loss, updated_at FROM session_final_models WHERE session_id=?'
  ).get(session_id) as { session_id: string; run_id: string; ckpt_path: string; step: number; avg_test_loss: number | null; updated_at: number } | undefined;
  return row || null;
}

export function listSessions() {
  if (!db) return [];
  return db.prepare(
    'SELECT session_id, MAX(ts) AS last_ts, COUNT(DISTINCT run_id) AS run_count FROM logs WHERE session_id IS NOT NULL GROUP BY session_id ORDER BY last_ts DESC'
  ).all() as Array<{ session_id: string; last_ts: number; run_count: number }>;
}

export function runsForSession(session_id: string) {
  if (!db) return [];
  return db.prepare(
    'SELECT DISTINCT run_id FROM logs WHERE session_id=? AND run_id IS NOT NULL'
  ).all(session_id).map((r: any) => String(r.run_id));
}

export function getLogsBySession(session_id: string, phase?: 'train'|'test'|'info') {
  if (!db) return [];
  if (phase) {
    return db.prepare(
      'SELECT id, run_id, session_id, level, text, phase, step, epoch, ts FROM logs WHERE session_id=? AND phase=? ORDER BY ts ASC, id ASC'
    ).all(session_id, phase);
  }
  return db.prepare(
    'SELECT id, run_id, session_id, level, text, phase, step, epoch, ts FROM logs WHERE session_id=? ORDER BY ts ASC, id ASC'
  ).all(session_id);
}

export function latestCheckpointForRun(run_id: string): { path: string; step: number } | null {
  if (!db) return null;
  const row = db.prepare(
    `SELECT path, step FROM checkpoints
     WHERE run_id = ?
     ORDER BY step DESC, created_at DESC
     LIMIT 1`
  ).get(run_id) as { path?: string; step?: number } | undefined;

  if (!row || !row.path) return null;
  return { path: String(row.path), step: Number(row.step || 0) };
}
