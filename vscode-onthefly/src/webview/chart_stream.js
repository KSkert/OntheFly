/* chart-stream.js
 * Shared streaming buffer for all metric charts (loss, val, runtime metrics).
 */
(function () {
  const CH = window.ChartCreation || {};

  const SERIES = Object.create(null);
  const PEND_SERIES = Object.create(null);
  const TRACKED = new Set(['loss', 'val_loss']);
  const charts = Object.create(null); // metric -> Chart instance

  const STATE = {
    labels: [],
    epochs: [],      // strictly increasing epoch numbers (summary)
    stepEpochs: [],  // per-step epoch reference (number or null)
    series: SERIES,
    loss: (SERIES.loss = []),
    val_loss: (SERIES.val_loss = []),
  };

  const PEND = {
    labels: [],
    epochs: [],      // pending epoch uniques
    stepEpochs: [],
    series: PEND_SERIES,
    loss: (PEND_SERIES.loss = []),
    val_loss: (PEND_SERIES.val_loss = []),
  };

  const MIN_DRAW_INTERVAL_MS = 120;
  const MAX_POINTS_PER_SERIES = 6000;
  const MAX_AXIS_TICKS = 8;
  let lastDrawTs = 0;
  let rafPending = false;
  const AXIS_STEP = 'step';
  const AXIS_EPOCH = 'epoch';
  let axisMode = AXIS_STEP;
  let epochTickLock = false; // once 8 epochs are seen, keep 8 ticks forever

  function ensureSeries(store, metric) {
    if (!store[metric]) store[metric] = [];
    return store[metric];
  }

  function backfillSeries(metric) {
    const main = ensureSeries(SERIES, metric);
    while (main.length < STATE.labels.length) main.push(NaN);
    const pend = ensureSeries(PEND_SERIES, metric);
    while (pend.length < PEND.labels.length) pend.push(NaN);
  }

  function trackMetrics(metrics) {
    if (!metrics) return;
    const list = Array.isArray(metrics) ? metrics : [metrics];
    for (const metric of list) {
      if (!metric || TRACKED.has(metric)) continue;
      TRACKED.add(metric);
      backfillSeries(metric);
    }
  }

  function registerChart(metric, chart) {
    if (!metric || !chart) return;
    trackMetrics(metric);
    charts[metric] = chart;
    applyMetric(metric);
  }

  function attachCharts({ lossChart, valLossChart, accuracyChart }) {
    if (lossChart) registerChart('loss', lossChart);
    if (valLossChart) registerChart('val_loss', valLossChart);
    if (accuracyChart) registerChart('accuracy', accuracyChart);
  }

  function sanitizeValue(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : NaN;
  }

  function appendEpochIfNew(list, epoch) {
    if (!Number.isFinite(epoch)) return;
    const last = list.length ? list[list.length - 1] : null;
    if (!Number.isFinite(last) || epoch > last) {
      list.push(epoch);
    }
  }

  function replaceArray(target, nextValues) {
    target.length = 0;
    for (let i = 0; i < nextValues.length; i += 1) {
      target.push(nextValues[i]);
    }
  }

  function rebuildEpochSummary(stepEpochs) {
    STATE.epochs.length = 0;
    let last = null;
    for (const epoch of stepEpochs) {
      if (!Number.isFinite(epoch)) continue;
      if (!Number.isFinite(last) || epoch > last) {
        STATE.epochs.push(epoch);
        last = epoch;
      }
    }
  }

  function trimStateToLimit(limit = MAX_POINTS_PER_SERIES) {
    const total = STATE.labels.length;
    if (!(limit > 0) || total <= limit) return false;

    const stride = Math.ceil(total / limit);
    const keepIdx = [];
    for (let idx = 0; idx < total; idx += stride) {
      keepIdx.push(idx);
    }
    const lastIndex = total - 1;
    if (keepIdx[keepIdx.length - 1] !== lastIndex) {
      keepIdx.push(lastIndex);
    }

    const nextLabels = keepIdx.map((idx) => STATE.labels[idx]);
    const nextStepEpochs = keepIdx.map((idx) => STATE.stepEpochs[idx]);

    replaceArray(STATE.labels, nextLabels);
    replaceArray(STATE.stepEpochs, nextStepEpochs);
    rebuildEpochSummary(nextStepEpochs);

    for (const metric of TRACKED) {
      const src = ensureSeries(SERIES, metric);
      if (!src.length) continue;
      const trimmed = keepIdx.map((idx) => src[idx]);
      replaceArray(src, trimmed);
    }

    return true;
  }

  function enforceStateLimit() {
    return trimStateToLimit(MAX_POINTS_PER_SERIES);
  }

  function pendPush(step, values, epoch) {
    if (!Number.isFinite(step)) return;
    const s = +step;
    const e = Number.isFinite(epoch) ? Number(epoch) : null;

    PEND.labels.push(s);
    PEND.stepEpochs.push(e);
    appendEpochIfNew(PEND.epochs, e);

    for (const metric of TRACKED) {
      const series = ensureSeries(PEND_SERIES, metric);
      const val = values ? sanitizeValue(values[metric]) : NaN;
      series.push(val);
    }
  }

  function flushPendingToState() {
    if (!PEND.labels.length) return;

    STATE.labels.push(...PEND.labels);
    STATE.stepEpochs.push(...PEND.stepEpochs);

    if (PEND.epochs.length) {
      for (const epoch of PEND.epochs) {
        appendEpochIfNew(STATE.epochs, epoch);
      }
    }

    PEND.labels.length = 0;
    PEND.stepEpochs.length = 0;
    PEND.epochs.length = 0;

    for (const metric of TRACKED) {
      const pendSeries = ensureSeries(PEND_SERIES, metric);
      const main = ensureSeries(SERIES, metric);
      if (pendSeries.length) {
        main.push(...pendSeries);
        pendSeries.length = 0;
      }
    }

    enforceStateLimit();
  }

  function syncLossYScale(lossData) {
    const chart = charts.loss;
    if (!chart?.options?.scales) return;
    const vals = (lossData || []).filter(Number.isFinite);
    if (!vals.length) return;
    const minL = Math.min(...vals);
    const maxL = Math.max(...vals);
    const padL = (maxL - minL) * 0.05 || 1e-6;
    chart.options.scales.y.min = minL - padL;
    chart.options.scales.y.max = maxL + padL;
  }

  function labelCacheFromState() {
    const stepLabels = STATE.labels.map(Number);
    return {
      steps: stepLabels,
      stepEpochs: STATE.stepEpochs.slice(),
    };
  }

  // --- Helpers for Epoch Tick Logic ---
  // Derive epoch labels from sparse epoch markers so we can keep the step-axis tick
  // positions stable while simply relabeling them in epoch units.

  function collectEpochAnchors(steps, stepEpochs) {
    const anchors = [];
    if (!Array.isArray(steps) || !Array.isArray(stepEpochs)) return anchors;
    const n = Math.min(steps.length, stepEpochs.length);
    for (let i = 0; i < n; i += 1) {
      const ep = stepEpochs[i];
      if (!Number.isFinite(ep)) continue;
      const step = Number(steps[i]);
      if (!Number.isFinite(step)) continue;
      if (anchors.length && anchors[anchors.length - 1].step === step && anchors[anchors.length - 1].epoch === ep) {
        continue;
      }
      anchors.push({ step, epoch: ep });
    }
    return anchors;
  }

  function median(values) {
    if (!Array.isArray(values) || !values.length) return null;
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    if (sorted.length % 2) return sorted[mid];
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function estimateStepsPerEpoch(anchors) {
    if (!Array.isArray(anchors) || anchors.length < 2) return null;
    const ratios = [];
    for (let i = 1; i < anchors.length; i += 1) {
      const prev = anchors[i - 1];
      const cur = anchors[i];
      const dEpoch = cur.epoch - prev.epoch;
      const dStep = cur.step - prev.step;
      if (dEpoch > 0 && dStep > 0) {
        ratios.push(dStep / dEpoch);
      }
    }
    if (!ratios.length) return null;
    const m = median(ratios);
    return (m && m > 0 && Number.isFinite(m)) ? m : null;
  }

  function distinctEpochCount(stepEpochs) {
    if (!Array.isArray(stepEpochs) || !stepEpochs.length) return 0;
    const set = new Set();
    for (const ep of stepEpochs) {
      if (Number.isFinite(ep)) set.add(Math.round(ep));
    }
    return set.size;
  }

  // Robust lookup: if stepEpochs[index] is null, search neighbors
  function findNearestKnownEpoch(index, stepEpochs) {
    if (!stepEpochs || index < 0 || index >= stepEpochs.length) return null;

    // Direct hit?
    if (Number.isFinite(stepEpochs[index])) return stepEpochs[index];

    // Scan radius (snap to closest available data)
    const scanLimit = 150;
    for (let r = 1; r < scanLimit; r++) {
      const left = index - r;
      const right = index + r;
      const leftValid = left >= 0;
      const rightValid = right < stepEpochs.length;

      if (!leftValid && !rightValid) break;

      // Prefer left (past data) slightly
      if (leftValid && Number.isFinite(stepEpochs[left])) return stepEpochs[left];
      if (rightValid && Number.isFinite(stepEpochs[right])) return stepEpochs[right];
    }
    return null;
  }

  function findNearestStepIndex(stepValue, steps) {
    if (!Array.isArray(steps) || !steps.length || !Number.isFinite(stepValue)) return -1;
    let bestIdx = -1;
    let bestDist = Infinity;
    for (let i = 0; i < steps.length; i += 1) {
      const val = Number(steps[i]);
      if (!Number.isFinite(val)) continue;
      const dist = Math.abs(val - stepValue);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
        if (dist === 0) break;
      }
    }
    return bestIdx;
  }

  function formatEpochLabel(epoch) {
    if (!Number.isFinite(epoch)) return '';
    return String(Math.round(epoch));
  }

  function estimateEpochFromStep(stepVal, anchors, stepsPerEpoch) {
    if (!anchors?.length || !Number.isFinite(stepVal)) return null;

    const hasRatio = Number.isFinite(stepsPerEpoch) && stepsPerEpoch > 0;
    let base = anchors[0];
    for (let i = anchors.length - 1; i >= 0; i -= 1) {
      if (anchors[i].step <= stepVal) { base = anchors[i]; break; }
    }

    if (!hasRatio) return base?.epoch ?? null;
    if (stepVal <= base.step) return base.epoch;

    let next = anchors[anchors.length - 1];
    for (let i = 0; i < anchors.length; i += 1) {
      if (anchors[i].step >= stepVal) { next = anchors[i]; break; }
    }

    const approx = base.epoch + (stepVal - base.step) / stepsPerEpoch;
    if (next && next.step !== base.step) {
      const lo = Math.min(base.epoch, next.epoch);
      const hi = Math.max(base.epoch, next.epoch);
      return Math.min(hi, Math.max(lo, approx));
    }
    return approx;
  }

  function makeStepAlignedEpochFormatter(steps, stepEpochs) {
    if (!Array.isArray(steps) || !steps.length) return null;
    const anchors = collectEpochAnchors(steps, stepEpochs);
    const stepsPerEpoch = estimateStepsPerEpoch(anchors);

    return function epochTickFormatter(value, index, ticks) {
      const tick = ticks && ticks[index];
      let stepVal = Number(tick?.value);
      if (!Number.isFinite(stepVal)) stepVal = Number(value);
      if (!Number.isFinite(stepVal) && index >= 0 && index < steps.length) {
        stepVal = Number(steps[index]);
      }
      const nearestIndex = findNearestStepIndex(stepVal, steps);
      if (!Number.isFinite(stepVal) && nearestIndex >= 0) {
        stepVal = Number(steps[nearestIndex]);
      }

      const inferred = estimateEpochFromStep(stepVal, anchors, stepsPerEpoch);
      const fallback = findNearestKnownEpoch(nearestIndex >= 0 ? nearestIndex : index, stepEpochs);
      const epoch = Number.isFinite(inferred) ? inferred : fallback;

      if (!Number.isFinite(epoch)) return '';
      return formatEpochLabel(epoch);
    };
  }

  function applyMetric(metric, labelCache) {
    const chart = charts[metric];
    if (!chart) return;

    const cache = labelCache || labelCacheFromState();
    const labels = cache.steps; // step numbers (sorted)
    const data = ensureSeries(SERIES, metric).slice(0, labels.length);

    if (chart.data) {
      chart.data.labels = labels;
      if (chart.data.datasets?.[0]) {
        const points = labels.map((step, idx) => {
          const x = Number(step);
          const y = data[idx];
          return Number.isFinite(x) ? { x, y } : { x: idx, y };
        });
        chart.data.datasets[0].data = points;
      }
    }

    if (metric === 'loss') {
      syncLossYScale(data);
    } else {
      CH.autoScale?.(chart, metric);
    }

    const xScale = chart.options?.scales?.x;
    if (xScale) {
      const ticks = xScale.ticks || (xScale.ticks = {});
      const grid = xScale.grid || (xScale.grid = {});

      // Helps prevent sub-pixel shimmer on some canvases/DPIs (Chart.js v4).
      xScale.alignToPixels = true;

      if (grid.__origColor === undefined) {
        const chartDefault =
          (Chart?.defaults?.scales?.category?.grid?.color) ??
          (Chart?.defaults?.scales?.linear?.grid?.color);
        grid.__origColor = (typeof grid.color === 'string') ? grid.color : (chartDefault || '#e5e7eb');
      }

      if (axisMode === AXIS_EPOCH) {
        const stepEpochs = cache.stepEpochs.slice();
        const formatter = makeStepAlignedEpochFormatter(labels, stepEpochs);
        ticks.callback = formatter || undefined;

        ticks.autoSkip = true;
        const epochCount = distinctEpochCount(stepEpochs);
        if (epochCount >= MAX_AXIS_TICKS) epochTickLock = true;
        const cap = epochTickLock ? MAX_AXIS_TICKS : Math.min(MAX_AXIS_TICKS, Math.max(1, epochCount || 0));
        ticks.maxTicksLimit = cap;

        grid.display = true;
        grid.color = grid.__origColor || '#e5e7eb';
      } else {
        ticks.callback = undefined;
        ticks.autoSkip = true;
        ticks.maxTicksLimit = MAX_AXIS_TICKS;

        grid.display = true;
        grid.color = grid.__origColor || '#e5e7eb';
      }
    }

    chart.update('none');
  }

  function applyStateToCharts() {
    const cache = labelCacheFromState();
    for (const metric of Object.keys(charts)) {
      applyMetric(metric, cache);
    }
  }

  function scheduleChartsRedraw() {
    const now = performance.now();
    if (now - lastDrawTs < MIN_DRAW_INTERVAL_MS) return;
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      flushPendingToState();
      applyStateToCharts();
      lastDrawTs = performance.now();
    });
  }

  function reset() {
    STATE.labels.length = 0;
    STATE.epochs.length = 0;
    STATE.stepEpochs.length = 0;

    for (const metric of TRACKED) {
      ensureSeries(SERIES, metric).length = 0;
      ensureSeries(PEND_SERIES, metric).length = 0;
    }

    PEND.labels.length = 0;
    PEND.stepEpochs.length = 0;
    PEND.epochs.length = 0;

    lastDrawTs = 0;
    rafPending = false;
    epochTickLock = false;
    applyStateToCharts();
  }

  function ingestRows(rows) {
    reset();
    if (!Array.isArray(rows)) return;

    for (const row of rows) {
      const step = Number(row?.step);
      if (!Number.isFinite(step)) continue;

      STATE.labels.push(step);

      const epoch = Number(row?.epoch);
      const normalized = Number.isFinite(epoch) ? epoch : null;
      STATE.stepEpochs.push(normalized);
      appendEpochIfNew(STATE.epochs, normalized);

      for (const metric of TRACKED) {
        const series = ensureSeries(SERIES, metric);
        series.push(sanitizeValue(row?.[metric]));
      }
    }

    enforceStateLimit();
    applyStateToCharts();
  }

  function flushIfPending() {
    if (!PEND.labels.length) return false;
    flushPendingToState();
    return true;
  }

  function setAxisMode(mode) {
    const next = mode === AXIS_EPOCH ? AXIS_EPOCH : AXIS_STEP;
    const changed = axisMode !== next;
    if (changed) {
      axisMode = next;
    }
    const hadPending = flushIfPending();
    if (!changed && !hadPending) return;
    applyStateToCharts();
  }

  function getAxisMode() {
    return axisMode;
  }

  window.ChartStream = {
    STATE,
    PEND,
    attachCharts,
    registerMetricChart: registerChart,
    trackMetrics,
    pendPush,
    flushPendingToState,
    applyStateToCharts,
    applyMetric,
    scheduleChartsRedraw,
    ingestRows,
    reset,
    setAxisMode,
    getAxisMode,
  };

  // Back-compat globals for historical scripts
  window.pendPush = pendPush;
  window.scheduleChartsRedraw = scheduleChartsRedraw;
})();
