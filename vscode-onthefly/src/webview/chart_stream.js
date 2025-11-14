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
    series: SERIES,
    loss: (SERIES.loss = []),
    val_loss: (SERIES.val_loss = []),
  };

  const PEND = {
    labels: [],
    series: PEND_SERIES,
    loss: (PEND_SERIES.loss = []),
    val_loss: (PEND_SERIES.val_loss = []),
  };

  const MIN_DRAW_INTERVAL_MS = 120;
  let lastDrawTs = 0;
  let rafPending = false;

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

  function attachCharts({ lossChart, valLossChart }) {
    if (lossChart) registerChart('loss', lossChart);
    if (valLossChart) registerChart('val_loss', valLossChart);
  }

  function sanitizeValue(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : NaN;
  }

  function pendPush(step, values) {
    if (!Number.isFinite(step)) return;
    const s = +step;
    PEND.labels.push(s);
    for (const metric of TRACKED) {
      const series = ensureSeries(PEND_SERIES, metric);
      const val = values ? sanitizeValue(values[metric]) : NaN;
      series.push(val);
    }
  }

  function flushPendingToState() {
    if (!PEND.labels.length) return;
    STATE.labels.push(...PEND.labels);
    PEND.labels.length = 0;
    for (const metric of TRACKED) {
      const pendSeries = ensureSeries(PEND_SERIES, metric);
      const main = ensureSeries(SERIES, metric);
      if (pendSeries.length) {
        main.push(...pendSeries);
        pendSeries.length = 0;
      }
    }
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

  function applyMetric(metric, labelsCache) {
    const chart = charts[metric];
    if (!chart) return;
    const labels = labelsCache || STATE.labels.map(Number);
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
    chart.update('none');
  }

  function applyStateToCharts() {
    const labels = STATE.labels.map(Number);
    for (const metric of Object.keys(charts)) {
      applyMetric(metric, labels);
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
    for (const metric of TRACKED) {
      ensureSeries(SERIES, metric).length = 0;
      ensureSeries(PEND_SERIES, metric).length = 0;
    }
    PEND.labels.length = 0;
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
      for (const metric of TRACKED) {
        const series = ensureSeries(SERIES, metric);
        series.push(sanitizeValue(row?.[metric]));
      }
    }
    applyStateToCharts();
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
  };

  // Back-compat globals for historical scripts
  window.pendPush = pendPush;
  window.scheduleChartsRedraw = scheduleChartsRedraw;
})();
