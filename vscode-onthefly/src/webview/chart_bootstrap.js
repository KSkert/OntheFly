/* chart_bootstrap.js
 * Boots the default loss/val/hist charts once Chart.js is present.
 */
(function () {
  const ids = (window.OnTheFlyExports && OnTheFlyExports.ids) || {};
  const { wireExportButton } = window.ChartUtils || {};

  let bootstrapped = false;

  function hideChartsWhenMissing() {
    ['lossChart', 'valLossChart', 'lossDistChart'].forEach((id) => {
      const el = document.getElementById(id);
      if (el && el.parentElement) el.parentElement.style.display = 'none';
    });
  }

  function init() {
    if (bootstrapped) return;
    if (typeof Chart === 'undefined') {
      const started = Date.now();
      const tick = () => {
        if (typeof Chart !== 'undefined') { init(); return; }
        if (Date.now() - started > 3000) {
          (window.log || console.log)('Chart.js not loaded. Ensure chart.js is injected.');
          hideChartsWhenMissing();
        } else {
          setTimeout(tick, 50);
        }
      };
      tick();
      return;
    }

    bootstrapped = true;

    ChartCreation.ensurePluginsRegistered();
    ChartCreation.initDefaultCanvases();

    const lossChart = ChartCreation.createLineChart({
      name: 'loss',
      canvasId: ids.lossChart,
      label: 'Loss'
    });
    const val_lossChart = ChartCreation.createLineChart({
      name: 'val_loss',
      canvasId: ids.valLossChart,
      label: 'VAL'
    });
    const lossDistChart = ChartCreation.createHistogramChart({
      name: 'loss_dist',
      canvasId: ids.lossDistChart
    });

    if (typeof wireExportButton === 'function') {
      wireExportButton(ids.exportLossBtn, () => lossChart, 'loss_chart');
      wireExportButton(ids.exportValLossBtn, () => val_lossChart, 'val_loss');
      wireExportButton(ids.exportLossDistBtn, () => lossDistChart, 'loss_distribution');
    }

    ChartStream.attachCharts({ lossChart, valLossChart: val_lossChart });

    window.lossChart = lossChart;
    window.val_lossChart = val_lossChart;
    window.lossDistChart = lossDistChart;

    try {
      if (typeof window.enableDragOnReportChart === 'function') {
        window.enableDragOnReportChart();
      }
    } catch (e) {
      /* noop */
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.ChartBootstrap = { init };
})();
