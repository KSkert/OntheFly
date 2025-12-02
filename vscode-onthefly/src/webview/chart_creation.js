/* chart-creation.js
 * A tiny "factory" that creates charts in one consistent way.
 * Produces line charts with {x,y} points and linear X scales by default.
 */
(function(){
  const { byId, prepareCanvasSizes } = window.ChartUtils || {};

  const registry = Object.create(null); // name -> Chart
  const PRIMARY_COLOR = '#3b82f6';
  const DEFAULT_TOOLTIP_THRESHOLD = 18;

  function formatMetricValue(value, metric) {
    if (!Number.isFinite(value)) return '—';
    if (metric === 'lr' && value !== 0) {
      return value.toExponential(2);
    }
    const rounded = Math.round(value * 1000) / 1000;
    if (Math.abs(rounded) >= 1000) {
      return rounded.toFixed(0);
    }
    return rounded.toFixed(3);
  }

  const SCALE_LIMITS = {
    accuracy: { min: 0, max: 1 },
    activation_zero_frac: { min: 0, max: 1 },
    gpu_util: { min: 0, max: 100 },
  };

  function autoScaleMetricChart(chart, metricName) {
    const yScale = chart?.options?.scales?.y;
    const data = chart?.data?.datasets?.[0]?.data || [];
    const values = data
      .map((pt) => (pt && typeof pt === 'object' ? pt.y ?? pt : pt))
      .map((v) => Number(v))
      .filter((v) => Number.isFinite(v));

    if (!values.length || !yScale) {
      if (yScale) {
        delete yScale.min;
        delete yScale.max;
      }
      return;
    }

    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
      const pad = Math.max(Math.abs(min) * 0.1, 1e-6);
      min -= pad;
      max += pad;
    } else {
      const pad = (max - min) * 0.1;
      min -= pad;
      max += pad;
    }

    const bounds = SCALE_LIMITS[metricName];
    if (bounds) {
      if (typeof bounds.min === 'number') min = Math.max(bounds.min, min);
      if (typeof bounds.max === 'number') max = Math.min(bounds.max, max);
      if (min >= max) {
        const spanMin = typeof bounds.min === 'number' ? bounds.min : min;
        const spanMax = typeof bounds.max === 'number' ? bounds.max : max;
        const mid = (spanMin + spanMax) / 2;
        const epsilon = Math.max(Math.abs(mid) * 0.01, 1e-4);
        min = Math.max(spanMin, mid - epsilon);
        max = Math.min(spanMax, mid + epsilon);
      }
    }
    yScale.min = min;
    yScale.max = max;
  }

  const ProximityTooltipPlugin = {
    id: 'otfProximityTooltip',
    afterEvent(chart, args) {
      if (!chart || chart.config?.type !== 'line') return;
      const tooltip = chart.tooltip;
      if (!tooltip || typeof tooltip.setActiveElements !== 'function') return;
      const event = args.event;
      if (!event) return;
      const type = event.type;
      if (type === 'mouseout') {
        tooltip.setActiveElements([], { x: 0, y: 0 });
        return;
      }
      if (!['mousemove', 'pointermove', 'touchmove'].includes(type)) return;
      const threshold =
        chart.options?.plugins?.tooltip?.proximityThreshold ?? DEFAULT_TOOLTIP_THRESHOLD;
      if (!threshold || threshold <= 0) return;
      const active = typeof tooltip.getActiveElements === 'function'
        ? tooltip.getActiveElements()
        : chart.getActiveElements();
      if (!active?.length) return;
      const element = active[0]?.element;
      if (!element) return;
      const { x, y } = element.getProps(['x', 'y'], true);
      const ex = event.x;
      const ey = event.y;
      if (!Number.isFinite(ex) || !Number.isFinite(ey)) return;
      const dx = ex - x;
      const dy = ey - y;
      if ((dx * dx + dy * dy) > threshold * threshold) {
        tooltip.setActiveElements([], { x: ex, y: ey });
        chart.update('none');
      }
    }
  };

  if (typeof Chart !== 'undefined' && !window.__OTF_PROX_PLUGIN__) {
    Chart.register(ProximityTooltipPlugin);
    window.__OTF_PROX_PLUGIN__ = true;
  }

  function ensurePluginsRegistered() {
    if (typeof Chart === 'undefined') return;
    try {
      window.ChartPlugins?.ensureRegistered?.();
    } catch(e) { /* noop */ }
  }

  function commonOpts({ yTitle } = {}) {
    return {
      animation: false,
      animations: { colors: false, x: false, y: false },
      parsing: false,                // overridden per chart when needed
      normalized: true,
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false },
      devicePixelRatio: 1,
      plugins: {
        legend: { display: false },
        decimation: { enabled: true, algorithm: 'lttb', samples: 300 }
      },
      elements: { point: { radius: 0, hitRadius: 0 } },
      scales: {
        x: { type: 'linear', ticks: { autoSkip: true } },  // match dynamic metrics
        y: {
          type: 'linear',
          position: 'left',
          title: yTitle ? { display: true, text: yTitle } : { display:false },
          ticks: { display: true }
        }
      }
    };
  }


  function createLineChart({ name, canvasId, label, yTitle }) {
    const ctx = byId(canvasId)?.getContext('2d');
    if (!ctx) return null;

    const palette = {
      loss: PRIMARY_COLOR,
      val_loss: PRIMARY_COLOR
    };

    const opts = commonOpts({ yTitle });
    opts.parsing = true;
    opts.scales.x = {
      type: 'linear',
      bounds: 'data',
      offset: false,
      ticks: { maxTicksLimit: 8 }
    };

    opts.scales.y = opts.scales.y || {};
    opts.scales.y.ticks = opts.scales.y.ticks || {};
    opts.scales.y.ticks.display = true;
    opts.scales.y.ticks.callback = (value) => formatMetricValue(value, name);

    opts.plugins.tooltip = opts.plugins.tooltip || {};
    opts.plugins.tooltip.callbacks = opts.plugins.tooltip.callbacks || {};
    opts.plugins.tooltip.proximityThreshold = DEFAULT_TOOLTIP_THRESHOLD;
    opts.plugins.tooltip.callbacks.label = (ctx) => {
      const label = ctx.dataset?.label ? `${ctx.dataset.label}: ` : '';
      return label + formatMetricValue(ctx.parsed?.y, name);
    };

    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        datasets: [{
          label: label || name,
          data: [],
          tension: 0.25,
          borderWidth: 2,
          pointRadius: 0,
          borderColor: palette[name] || PRIMARY_COLOR,
          // transparent gaps when points skip
          segment: { borderColor: (c)=> (c.p0.skip || c.p1.skip) ? 'transparent' : undefined }
        }]
      },
      options: opts
    });

    autoScaleMetricChart(chart, name);
    window.ChartStream?.registerMetricChart?.(name, chart);
    registry[name] = chart;
    requestAnimationFrame(() => chart.update('none'));
    return chart;
  }

  function createHistogramChart({ name, canvasId }) {
    const ctx = byId(canvasId)?.getContext('2d');
    if (!ctx) return null;
    const opts = commonOpts({});
    // disable decimation for mixed bar+line
    opts.plugins.decimation = { enabled: true };
    opts.scales.x = { type: 'linear', ticks: { maxTicksLimit: 8 }, offset: false };
    opts.scales.y = { type: 'linear', beginAtZero: true };

    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        datasets: [
          { type: 'bar',  label: 'Loss frequency', data: [], parsing: false, borderWidth: 1, backgroundColor: 'rgba(128, 0, 128, 0.15)' },
          { type: 'line', label: 'Loss density (smooth)', data: [], parsing: false, pointRadius: 0, tension: 0.3, borderWidth: 2, borderColor: 'purple' }
        ]
      },
      options: opts
    });

    registry[name] = chart;
    requestAnimationFrame(() => chart.update('none'));
    return chart;
  }

  function initDefaultCanvases() {
    prepareCanvasSizes && prepareCanvasSizes();
  }

  function get(name){ return registry[name] || null; }
  function all(){ return { ...registry }; }

  function destroy(name){
    const ch = registry[name];
    if (!ch) return;
    try { ch.destroy(); } catch {}
    delete registry[name];
  }
  function destroyAll(){
    Object.keys(registry).forEach(destroy);
  }

  window.ChartCreation = {
    ensurePluginsRegistered,
    createLineChart,
    createHistogramChart,
    initDefaultCanvases,
    get,
    all,
    destroy,
    destroyAll,
    autoScale: autoScaleMetricChart,
    formatMetricValue,
  };
})();


// ===== Metrics dropdown + dynamic charts (migrated) =====
(function(){
  const ids = (window.OnTheFlyExports && OnTheFlyExports.ids) || {};
  const select = document.getElementById(ids.metricsSelect || 'metricsSelect');
  const stack  = document.getElementById(ids.metricsStack  || 'metricsStack');
  if (!select || !stack) return;

  const METRIC_META = {
    lr: { label: 'Learning Rate', help: 'Optimizer learning rate (avg across parameter groups).' },
    grad_norm: { label: 'Gradient Norm', help: 'Global L2 norm of gradients after each backward pass.' },
    weight_norm: { label: 'Weight Norm', help: 'Global L2 norm of model weights.' },
    activation_zero_frac: { label: 'Activation Sparsity', help: 'Fraction of tracked activations that are exactly zero (0 = dense, 1 = all zero).' },
    throughput: { label: 'Throughput', help: 'Estimated samples processed per second (batch size ÷ step duration).' },
    mem_vram: { label: 'Memory', help: 'Approximate GPU memory consumption (MB) from the device monitor.' },
    gpu_util: { label: 'GPU Util', help: 'Reported GPU utilization percentage.' },
  };

  function labelFor(metric) {
    return METRIC_META[metric]?.label || metric;
  }

  const charts = Object.create(null);
  function idSafe(s){ return String(s).toLowerCase().replace(/[^a-z0-9_]+/g,'_'); }

  function makeChart(metric){
    const existing = charts[metric];
    if (existing) {
      const wrap = existing.canvas?.closest('.chartWrap');
      if (wrap && wrap.isConnected) {
        wrap.style.display = '';
        // bring back to the top to match the behaviour of new charts
        if (stack.firstChild !== wrap) stack.insertBefore(wrap, stack.firstChild);
        wrap.scrollIntoView({behavior:'smooth', block:'center'});
        window.ChartStream?.applyMetric?.(metric);
        return existing;
      }
    }

    // wrapper
    const wrap = document.createElement('div');
    wrap.className = 'chartWrap metricWrap';
    wrap.dataset.metric = metric;

    // close button
    const closeBtn = document.createElement('button');
    closeBtn.className = 'chartCloseBtn';
    closeBtn.type = 'button';
    closeBtn.title = 'Close';
    closeBtn.setAttribute('aria-label', 'Close '+labelFor(metric)+' chart');
    closeBtn.textContent = '✕';
    closeBtn.addEventListener('click', function(){
      wrap.style.display = 'none';
    });
    wrap.appendChild(closeBtn);

    // title
    const title = document.createElement('div');
    title.className = 'chartTitle';
    const titleText = document.createElement('span');
    titleText.textContent = labelFor(metric);
    title.appendChild(titleText);
    const helpText = METRIC_META[metric]?.help;
    if (helpText) {
      title.classList.add('fs-helpWrap');
      const helpMark = document.createElement('span');
      helpMark.className = 'fs-helpMark';
      helpMark.dataset.tip = helpText;
      helpMark.textContent = '?';
      title.appendChild(helpMark);
    }
    wrap.appendChild(title);

    // canvas
    const canvas = document.createElement('canvas');
    canvas.id = 'chart_'+idSafe(metric);
    canvas.className = 'metricCanvas';
    wrap.appendChild(canvas);

    // insert at top of stack
    stack.insertBefore(wrap, stack.firstChild);

    // build chart via the unified factory (identical defaults to train/val)
    const chart = ChartCreation.createLineChart({
      name: metric,
      canvasId: canvas.id,
      label: labelFor(metric)
    });

    if (window.MetricHistory?.requestBackfill) {
      try {
        const hasHistory = typeof window.MetricHistory.hasData === 'function'
          ? window.MetricHistory.hasData(metric)
          : false;
        if (!hasHistory) window.MetricHistory.requestBackfill();
      } catch (err) {
        console.warn('[onthefly] metric history backfill request failed', err);
      }
    }

    if (window.MetricHydrator?.hydrateMetric) {
      window.MetricHydrator.hydrateMetric(metric);
    }

    charts[metric] = chart;

    if (window.MetricHistory?.hydrate) {
      window.MetricHistory.hydrate(metric);
    } else {
      window.ChartStream?.applyMetric?.(metric);
    }

    return chart;
  }

  const METRIC_KEYS = Object.keys(METRIC_META);

  // dropdown → create/open metric charts
  select.addEventListener('change', function(){
    const metric = select.value;
    if (!metric) return;

    if (metric === 'show_all') {
      METRIC_KEYS.forEach(makeChart);
      select.selectedIndex = 0;
      select.blur();
      return;
    }

    // special cases: show built-in loss charts if hidden
    if (metric === 'loss' || metric === 'valloss' || metric === 'accuracy') {
      const wrap = document.querySelector('.chartWrap[data-chart="'+metric+'"]');
      if (wrap) {
        wrap.style.display = '';
        wrap.scrollIntoView({behavior:'smooth', block:'center'});
      }
      select.selectedIndex = 0;
      return;
    }

    makeChart(metric);
    // reset placeholder so user can add more
    select.selectedIndex = 0;
    select.blur();
  });

  // close buttons for built-in loss/valloss wrappers
  document.addEventListener('click', function(e){
    const btn = e.target.closest('[data-close-chart]');
    if (!btn) return;
    const chartType = btn.dataset.closeChart;
    const wrap = document.querySelector('.chartWrap[data-chart="'+chartType+'"]');
    if (wrap) wrap.style.display = 'none';
  });

  // host API (same as before)
  window.MetricsUI = {
    open: makeChart,
    close: function(metric){
      const wrap = stack.querySelector('.metricWrap[data-metric="'+metric+'"]');
      if (wrap) wrap.querySelector('.chartCloseBtn').click();
    },
    push: function(metric){
      charts[metric] ||= makeChart(metric);
      window.ChartStream?.applyMetric?.(metric);
    },
    replace: function(metric){
      charts[metric] ||= makeChart(metric);
      window.ChartStream?.applyMetric?.(metric);
    },
    charts
  };
})();
