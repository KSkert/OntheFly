/* chart-utils.js
 * Small helpers shared by all chart code. No dependencies.
 */
(function () {
  const ids = (window.OnTheFlyExports && OnTheFlyExports.ids) || {};

  function byId(id) { return document.getElementById(id); }
  function on(el, ev, fn) { if (el) el.addEventListener(ev, fn, { passive: true }); }

  let defaultCanvasesSized = false;
  function prepareCanvasSizes() {
    if (defaultCanvasesSized) return;
    ['lossChart','valLossChart','lossDistChart'].forEach(id => {
      const c = byId(id);
      if (!c) return;
      c.style.width = '100%';
      c.style.height = '100%';
      c.removeAttribute('width');
      c.removeAttribute('height');
    });
    defaultCanvasesSized = true;
  }

  function chartToPngDataURL(chart) {
    if (!chart || !chart.canvas) return null;
    const src = chart.canvas;
    const w = src.width, h = src.height;

    const bg = getComputedStyle(src.closest('.chartWrap') || document.body).backgroundColor || '#ffffff';
    const out = document.createElement('canvas');
    out.width = w; out.height = h;
    const ctx = out.getContext('2d');
    ctx.save();
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(src, 0, 0);
    ctx.restore();
    return out.toDataURL('image/png');
  }

  function exportChartViaExtension(chart, suggestedName) {
    const dataUrl = chartToPngDataURL(chart);
    if (!dataUrl) { (window.notify||console.warn)('Chart not ready to export.', 'warn'); return; }
    window.vscode && vscode.postMessage({ command: 'exportChart', filename: suggestedName, dataUrl });
  }

  function wireExportButton(btnId, getChart, namePrefix) {
    const btn = byId(btnId);
    on(btn, 'click', () => {
      const ch = (typeof getChart === 'function') ? getChart() : getChart;
      const fname = `${namePrefix}_${Date.now()}.png`;
      exportChartViaExtension(ch, fname);
    });
  }

  // KDE-ish helper used by the report (extracted so others can reuse)
  function computeLossHistogram(values, numBins = 30) {
    if (!Array.isArray(values) || values.length === 0) {
      return { bars: [], line: [], xmin: undefined, xmax: undefined, edges: [] };
    }
    const min = Math.min(...values);
    const max = Math.max(...values);
    const width = (max - min) || 1e-9;
    const step = width / numBins;

    const edges = Array.from({ length: numBins + 1 }, (_, i) => min + i * step);
    const counts = new Array(numBins).fill(0);
    for (const v of values) {
      let idx = Math.floor((v - min) / step);
      if (idx >= numBins) idx = numBins - 1;
      if (idx < 0) idx = 0;
      counts[idx]++;
    }
    const centers = counts.map((_, i) => edges[i] + step * 0.5);

    const n = values.length;
    const mean = values.reduce((a, b) => a + b, 0) / n;
    const variance = values.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, (n - 1));
    const std = Math.sqrt(Math.max(variance, 0));
    let h = 1.06 * (std || (step / 1.06)) * Math.pow(n, -1 / 5);
    if (!Number.isFinite(h) || h <= 1e-12) h = step;
    h = Math.max(0.6 * step, Math.min(3 * step, h));

    const sampleCount = 160;
    const xs = Array.from({ length: sampleCount }, (_, i) => min + (i / (sampleCount - 1)) * (max - min));
    const invNorm = 1 / (Math.sqrt(2 * Math.PI) * h);
    const ySmooth = xs.map(x => {
      let dens = 0;
      for (let i = 0; i < centers.length; i++) {
        const u = (x - centers[i]) / h;
        dens += counts[i] * invNorm * Math.exp(-0.5 * u * u);
      }
      return dens * step;
    });

    return {
      bars: centers.map((x, i) => ({ x, y: counts[i] })),
      line: xs.map((x, i) => ({ x, y: ySmooth[i] })),
      xmin: min,
      xmax: max,
      edges
    };
  }

  window.ChartUtils = {
    byId, on, prepareCanvasSizes, chartToPngDataURL, exportChartViaExtension, wireExportButton, computeLossHistogram
  };
})();


(function () {
  if (window.__CHART_THEME_WIRED__) return; // idempotent
  window.__CHART_THEME_WIRED__ = true;

  function applyChartTheme() {
    var cs = getComputedStyle(document.documentElement);
    var text = (cs.getPropertyValue('--chart-text') || cs.getPropertyValue('--fg')).trim();
    var grid = (cs.getPropertyValue('--chart-grid') || cs.getPropertyValue('--border')).trim();
    if (window.Chart) {
      Chart.defaults.color = text;
      Chart.defaults.borderColor = grid;
      Chart.defaults.plugins = Chart.defaults.plugins || {};
      Chart.defaults.plugins.legend = Chart.defaults.plugins.legend || {};
      Chart.defaults.plugins.legend.labels = Chart.defaults.plugins.legend.labels || {};
      Chart.defaults.plugins.legend.labels.color = text;
      Chart.defaults.plugins.tooltip = Chart.defaults.plugins.tooltip || {};
      Chart.defaults.plugins.tooltip.titleColor = text;
      Chart.defaults.plugins.tooltip.bodyColor = text;
      Chart.defaults.scales = Chart.defaults.scales || {};
      ['linear','logarithmic','category','time','timeseries'].forEach(function (type) {
        Chart.defaults.scales[type] = Chart.defaults.scales[type] || {};
        Chart.defaults.scales[type].grid = Chart.defaults.scales[type].grid || {};
        Chart.defaults.scales[type].grid.color = grid;
        Chart.defaults.scales[type].ticks = Chart.defaults.scales[type].ticks || {};
        Chart.defaults.scales[type].ticks.color = text;
      });
    }
  }

  // run now (Chart is already loaded) + keep in sync with theme changes
  applyChartTheme();
  var mql = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
  if (mql && mql.addEventListener) mql.addEventListener('change', applyChartTheme);
  else if (mql && mql.addListener) mql.addListener(applyChartTheme);
  window.addEventListener('DOMContentLoaded', applyChartTheme);

  // expose for diagnostics, if needed
  (window.ChartUtils ||= {}).applyChartTheme = applyChartTheme;
})();
