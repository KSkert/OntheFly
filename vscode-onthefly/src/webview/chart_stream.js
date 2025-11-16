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
    epochs: [],      // strictly increasing epoch numbers (optional summary)
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
  let lastDrawTs = 0;
  let rafPending = false;
  const AXIS_STEP = 'step';
  const AXIS_EPOCH = 'epoch';
  let axisMode = AXIS_STEP;

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

    // epoch here is effectiveEpoch from the webview (either a number, or null)
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

  // Build a simple cache: step labels + per-step epochs by index
  function labelCacheFromState() {
    const stepLabels = STATE.labels.map(Number);
    return {
      steps: stepLabels,
      stepEpochs: STATE.stepEpochs.slice(), // number or null per step
    };
  }

  // In epoch mode, the x-axis is category indexed. For each tick, we:
  // - find its label index
  // - read stepEpochs[index]
  // - if it's a number, show it; otherwise show nothing
  function makeEpochTickFormatter(stepEpochs) {
    if (!Array.isArray(stepEpochs) || !stepEpochs.length) return null;

    return function epochTickFormatter(value, index, ticks) {
      const tick = ticks && ticks[index];

      // For a category scale, tick.value is the index into labels[]
      const labelIndex =
        tick && typeof tick.value === 'number'
          ? tick.value
          : index;

      const epoch = stepEpochs[labelIndex];

      // No epoch recorded for this step? Show nothing.
      if (!Number.isFinite(epoch)) return '';

      // Backend only emits epoch when it changes, so we can just render it.
      return String(epoch);
    };
  }

  function applyMetric(metric, labelCache) {
    const chart = charts[metric];
    if (!chart) return;

    const cache = labelCache || labelCacheFromState();
    const labels = cache.steps;
    const data = ensureSeries(SERIES, metric).slice(0, labels.length);

    if (chart.data) {
      chart.data.labels = labels;
      if (chart.data.datasets?.[0]) {
        chart.data.datasets[0].data = data;
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
      const grid  = xScale.grid  || (xScale.grid  = {});

      // Cache a baseline grid color once (string or Chart default)
      if (grid.__origColor === undefined) {
        // Prefer an explicit string already set, otherwise Chart default, otherwise a light gray
        const chartDefault =
          (Chart?.defaults?.scales?.category?.grid?.color) ??
          (Chart?.defaults?.scales?.linear?.grid?.color);
        grid.__origColor = (typeof grid.color === 'string') ? grid.color : (chartDefault || '#e5e7eb');
      }

      if (axisMode === AXIS_EPOCH) {
        const stepEpochs = cache.stepEpochs.slice();
        const formatter = makeEpochTickFormatter(stepEpochs);
        ticks.callback = formatter || undefined;

        // Epoch mode: show every step tick so we can decide which ones to draw
        ticks.autoSkip = false;
        if (Object.prototype.hasOwnProperty.call(ticks, 'maxTicksLimit')) {
          delete ticks.maxTicksLimit;
        }

        grid.display = true;
        // Use a scriptable color that only draws at epoch boundaries
        const visibleColor = grid.__origColor || '#e5e7eb';
        grid.color = function (ctx) {
          const tick = ctx.tick;
          const idx = (tick && typeof tick.value === 'number') ? tick.value : ctx.index;
          const ep = stepEpochs[idx];
          return Number.isFinite(ep) ? visibleColor : 'transparent';
        };

      } else {
        // Step mode: normal behavior with sparse grid & limited ticks
        ticks.callback = undefined;
        ticks.autoSkip = true;
        // Always enforce our 8 verticals
        ticks.maxTicksLimit = 8;

        grid.display = true;
        // IMPORTANT: explicitly restore to a concrete color
        grid.color = grid.__origColor || '#e5e7eb';
        // (Optional) if you previously tweaked other grid props, reset them here too:
        // grid.lineWidth = 1; grid.drawTicks = true; grid.drawOnChartArea = true;
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
