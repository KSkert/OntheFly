/* chart_plugins.js
 * Custom Chart.js plugins plus a tiny registration helper so they can be
 * tested/loaded outside of the full dashboard.
 */
(function () {
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  let forkStateProvider = () => {
    const selection = window.ChartReportSelection?.getState?.();
    if (selection) return selection;
    const sel = window.regionSel || {};
    return {
      enabled: Boolean(window.regionSelectMode && sel?.active),
      sel
    };
  };

  let markerProvider = () => {
    try {
      const currentRun = typeof window.currentPageRunId === 'function' ? window.currentPageRunId() : null;
      const keyFn = typeof window.keyOf === 'function'
        ? window.keyOf
        : (id) => (id == null ? '' : String(id));
      const runKey = keyFn(currentRun);
      const store = window.AF_MARKERS;
      if (store && typeof store.get === 'function') {
        return store.get(runKey) || [];
      }
    } catch (e) {
      /* noop */
    }
    return [];
  };

  function forkSelectionPlugin(getState = forkStateProvider) {
    return {
      id: 'forkSelection',
      afterDraw(chart) {
        if (!chart || chart.canvas.id !== 'lossDistChart') return;
        const state = typeof getState === 'function' ? getState() : null;
        const enabled = Boolean(state?.enabled);
        const sel = state?.sel;

        if (!enabled || !sel?.active || sel.aVal == null || sel.bVal == null) return;

        const scaleX = chart.scales?.x;
        const area = chart.chartArea;
        if (!scaleX || !area) return;

        const aX = clamp(scaleX.getPixelForValue(sel.aVal), area.left, area.right);
        const bX = clamp(scaleX.getPixelForValue(sel.bVal), area.left, area.right);
        const left = Math.min(aX, bX);
        const right = Math.max(aX, bX);

        const ctx = chart.ctx;
        ctx.save();
        ctx.fillStyle = 'rgba(100, 149, 237, 0.18)';
        ctx.fillRect(left, area.top, right - left, area.bottom - area.top);

        ctx.lineWidth = 2;
        ctx.strokeStyle = '#efe444ff';
        ctx.beginPath();
        ctx.moveTo(aX, area.top);
        ctx.lineTo(aX, area.bottom);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(bX, area.top);
        ctx.lineTo(bX, area.bottom);
        ctx.stroke();

        const cap = 6;
        ctx.fillStyle = '#efe444ff';
        ctx.fillRect(aX - 3, area.top - cap, 6, cap);
        ctx.fillRect(bX - 3, area.top - cap, 6, cap);
        ctx.restore();

        if (typeof window.updateSelectedCountPill === 'function') {
          window.updateSelectedCountPill();
        }
      }
    };
  }

  function autoForkMarkerPlugin(getMarkers = markerProvider) {
    return {
      id: 'autoForkMarkers',
      afterDatasetsDraw(chart) {
        const canvasId = chart?.canvas?.id;
        if (!canvasId || (canvasId !== 'lossChart' && canvasId !== 'valLossChart' && canvasId !== 'accuracyChart')) return;

        const marks = typeof getMarkers === 'function' ? getMarkers() : null;
        if (!marks || !marks.length) return;

        const scaleX = chart.scales?.x;
        const area = chart.chartArea;
        if (!scaleX || !area) return;

        const labels = chart.data?.labels || [];
        const pixelForStep = (step) => {
          let idx = labels.findIndex((v) => Number(v) === Number(step));
          if (idx >= 0) return scaleX.getPixelForTick(idx);
          let best = -1;
          let dBest = Infinity;
          for (let i = 0; i < labels.length; i++) {
            const d = Math.abs(Number(labels[i]) - Number(step));
            if (d < dBest) {
              dBest = d;
              best = i;
            }
          }
          return (best >= 0) ? scaleX.getPixelForTick(best) : NaN;
        };

        const ctx = chart.ctx;
        ctx.save();
        ctx.setLineDash([6, 5]);
        ctx.lineWidth = 1;

        for (const mark of marks) {
          const x = pixelForStep(mark.step);
          if (!Number.isFinite(x)) continue;

          ctx.strokeStyle = (mark.kind === 'executed') ? '#22c55e' : '#94a3b8';
          ctx.beginPath();
          ctx.moveTo(x + 0.5, area.top);
          ctx.lineTo(x + 0.5, area.bottom);
          ctx.stroke();
        }

        ctx.restore();
      }
    };
  }

  function ensureRegistered(overrides = {}) {
    const getState = overrides.getForkSelectionState || forkStateProvider;
    const getMarkers = overrides.getMarkers || markerProvider;
    if (typeof Chart === 'undefined' || !Chart?.registry?.plugins) return false;

    if (!Chart.registry.plugins.get('forkSelection')) {
      Chart.register(forkSelectionPlugin(getState));
    }
    if (!Chart.registry.plugins.get('autoForkMarkers')) {
      Chart.register(autoForkMarkerPlugin(getMarkers));
    }
    return true;
  }

  function configure(opts = {}) {
    if (typeof opts.getForkSelectionState === 'function') {
      forkStateProvider = opts.getForkSelectionState;
    }
    if (typeof opts.getMarkers === 'function') {
      markerProvider = opts.getMarkers;
    }
  }

  window.ChartPlugins = {
    forkSelectionPlugin,
    autoForkMarkerPlugin,
    ensureRegistered,
    configure
  };
})();
