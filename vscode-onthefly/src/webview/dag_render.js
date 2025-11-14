/* dag_render.js
 * Renders the DAG overlay using layout info from dag_layout.js.
 * Exposes DagRender.render(opts) which expects dashboard-driven data/maps.
 */
(function (global) {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';

  function _canvasHasLayout(cnv) {
    if (!cnv) return false;
    const r = cnv.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }

  function _deferUntilLayout(chart, cb, tries = 20) {
    if (!chart?.canvas) { cb(); return; }
    if (_canvasHasLayout(chart.canvas)) { chart.resize?.(); cb(); return; }
    if (tries <= 0) { cb(); return; }
    requestAnimationFrame(() => _deferUntilLayout(chart, cb, tries - 1));
  }

  function computeLayers(runsIndex, parentsOf) {
    const depth = new Map();
    const ids = Array.from(runsIndex?.keys?.() || []);
    const parentLookup = (id) => {
      if (parentsOf?.get) return parentsOf.get(id) || [];
      if (Array.isArray(parentsOf?.[id])) return parentsOf[id];
      return [];
    };

    const getDepth = (id, stack = new Set()) => {
      if (depth.has(id)) return depth.get(id);
      if (stack.has(id)) { depth.set(id, 0); return 0; }
      stack.add(id);
      const ps = Array.from(parentLookup(id) || []);
      const d = ps.length ? (Math.max(...ps.map(p => getDepth(p, stack))) + 1) : 0;
      depth.set(id, d);
      stack.delete(id);
      return d;
    };

    ids.forEach(id => getDepth(id));

    const byDepth = new Map();
    ids.forEach(id => {
      const d = depth.get(id) || 0;
      if (!byDepth.has(d)) byDepth.set(d, []);
      byDepth.get(d).push(id);
    });

    const depths = Array.from(byDepth.keys()).sort((a, b) => a - b);
    depths.forEach(d => byDepth.get(d).sort((a, b) => a.localeCompare(b)));
    return { depth, depths, byDepth };
  }

  function wrapSvgTextIntoTspans(textEl, raw, maxWidthPx, maxLines = 2) {
    const svg = textEl?.ownerSVGElement;
    const label = String(raw || '').trim();
    if (!svg || !label) {
      if (textEl) textEl.textContent = label || '';
      return;
    }

    const cs = getComputedStyle(textEl);
    const fontSize = parseFloat(cs.fontSize) || 12;
    const lineHeight = Math.round(fontSize * 1.2);

    let meas = svg.__measureTextEl;
    if (!meas) {
      meas = document.createElementNS(SVG_NS, 'text');
      meas.setAttribute('x', '-9999');
      meas.setAttribute('y', '-9999');
      meas.style.visibility = 'hidden';
      svg.appendChild(meas);
      svg.__measureTextEl = meas;
    }
    const mcs = meas.style;
    mcs.font = cs.font;
    meas.setAttribute('font-family', cs.fontFamily);
    meas.setAttribute('font-weight', cs.fontWeight);
    meas.setAttribute('font-size', cs.fontSize);

    const widthOf = (s) => {
      meas.textContent = s;
      return meas.getComputedTextLength();
    };

    const words = label.split(/\s+/).filter(Boolean);
    const lines = [];
    let cur = '';

    while (words.length && lines.length < maxLines) {
      if (!cur) {
        if (widthOf(words[0]) > maxWidthPx) {
          let w = words.shift();
          let lo = 1, hi = w.length, best = 1;
          while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            const slice = w.slice(0, mid) + '…';
            if (widthOf(slice) <= maxWidthPx) { best = mid; lo = mid + 1; }
            else { hi = mid - 1; }
          }
          lines.push(w.slice(0, best) + '…');
          break;
        }
        cur = words.shift();
      } else {
        const candidate = cur + ' ' + words[0];
        if (widthOf(candidate) <= maxWidthPx) {
          cur = candidate; words.shift();
        } else {
          lines.push(cur);
          cur = '';
        }
      }
    }
    if (cur && lines.length < maxLines) lines.push(cur);

    if (words.length) {
      let last = lines.pop() || '';
      while (last && widthOf(last + '…') > maxWidthPx) last = last.slice(0, -1);
      lines.push((last || '…') + '…');
    }

    while (textEl.firstChild) textEl.removeChild(textEl.firstChild);

    const x = textEl.getAttribute('x') || '0';
    const yCenter = Number(textEl.getAttribute('y') || 0);
    const totalH = (lines.length - 1) * lineHeight;
    const yStart = yCenter - totalH / 2;

    for (let i = 0; i < lines.length; i++) {
      const tspan = document.createElementNS(SVG_NS, 'tspan');
      tspan.setAttribute('x', x);
      tspan.setAttribute('y', String(yStart + i * lineHeight));
      tspan.textContent = lines[i];
      textEl.appendChild(tspan);
    }
  }

  function ensureDagLayers(svg) {
    if (!svg) return null;
    if (!svg.__dagLayers) svg.__dagLayers = {};
    const layers = svg.__dagLayers;

    if (!layers.defs) {
      const defs = document.createElementNS(SVG_NS, 'defs');
      defs.id = 'dag-defs';
      defs.innerHTML = `
        <marker id="arrowHead" markerWidth="12" markerHeight="10"
                viewBox="0 0 12 10" refX="12" refY="5" orient="auto"
                markerUnits="userSpaceOnUse">
          <path class="dagEdgeArrow" d="M0,0 L12,5 L0,10 Z"></path>
        </marker>`;
      svg.appendChild(defs);
      layers.defs = defs;
    }

    if (!layers.viewport) {
      const viewport = document.createElementNS(SVG_NS, 'g');
      viewport.id = 'dagViewport';
      svg.appendChild(viewport);
      layers.viewport = viewport;

      if (typeof global.enablePanZoom === 'function') {
        layers.panzoom = global.enablePanZoom(svg, viewport, {
          minScale: 0.2,
          maxScale: 5,
          step: 0.14,
        });
      }
    }

    if (!layers.root) {
      const root = document.createElementNS(SVG_NS, 'g');
      root.id = 'dagRoot';
      layers.viewport.appendChild(root);
      layers.root = root;
    }

    return layers;
  }

  function render(options = {}) {
    const {
      svg,
      runsIndex,
      edges,
      selectedForMerge = new Set(),
      onPrimarySelect = () => {},
      updateMergeUi = () => {},
      emptyStateLabel = 'No runs yet',
    } = options;

    if (!svg || !runsIndex) return;
    if (typeof global.layoutDAG !== 'function') {
      console.warn('[DagRender] layoutDAG() missing; skipping render.');
      return;
    }

    const layers = ensureDagLayers(svg);
    if (!layers) return;
    const { root, panzoom } = layers;

    while (root.firstChild) root.removeChild(root.firstChild);

    const nodes = Array.from(runsIndex.keys()).map(id => ({ id }));
    const dagEdges = (Array.isArray(edges) ? edges : []).filter(e =>
      e && e.source && e.target && runsIndex.has(e.source) && runsIndex.has(e.target)
    );

    const cs = global.getComputedStyle?.(document.documentElement) || {
      getPropertyValue: () => '',
    };
    const readCssNumber = (prop, fallback) => {
      const v = parseFloat(cs.getPropertyValue(prop));
      return Number.isFinite(v) ? v : fallback;
    };

    const MIN_W = readCssNumber('--dag-minW', 960);
    const MIN_H = readCssNumber('--dag-minH', 560);

    if (!nodes.length) {
      svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
      svg.setAttribute('viewBox', `0 0 ${MIN_W} ${MIN_H}`);
      svg.setAttribute('width', '100%');
      svg.setAttribute('height', '100%');

      const text = document.createElementNS(SVG_NS, 'text');
      text.setAttribute('x', '50%');
      text.setAttribute('y', '50%');
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('fill', '#fff');
      text.textContent = emptyStateLabel;
      root.appendChild(text);
      return;
    }

    const nodeW = 150;
    const nodeH = 42;

    const { pos, routed, size, initial } = global.layoutDAG(nodes, dagEdges, {
      nodeW,
      nodeH,
      margin: 48,
      rankSep: 200,
      nodeSep: 28,
      iterations: 6,
    });

    if (initial?.viewBox) {
      const { x, y, width, height } = initial.viewBox;
      const vbW = Math.max(width, MIN_W);
      const vbH = Math.max(height, MIN_H);
      const adjX = x - (vbW - width) / 2;
      const adjY = y - (vbH - height) / 2;

      svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
      svg.setAttribute('viewBox', `${adjX} ${adjY} ${vbW} ${vbH}`);
      svg.setAttribute('width', '100%');
      svg.setAttribute('height', '100%');
      root.removeAttribute('transform');
      try { panzoom?.set?.({ x: 0, y: 0, k: 1 }); } catch (_) { /* noop */ }
    } else {
      const W = size.W, H = size.H;
      const vbW = Math.max(W, MIN_W);
      const vbH = Math.max(H, MIN_H);

      svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
      svg.setAttribute('viewBox', `0 0 ${vbW} ${vbH}`);
      svg.setAttribute('width', '100%');
      svg.setAttribute('height', '100%');

      const offsetX = (vbW - W) / 2;
      const offsetY = (vbH - H) / 2;
      root.setAttribute('transform', `translate(${offsetX}, ${offsetY})`);
    }

    pos.forEach(({ x, y }, id) => {
      const isSelected = selectedForMerge.has(id);
      const g = document.createElementNS(SVG_NS, 'g');
      g.setAttribute('class', `dagNode${isSelected ? ' selected' : ''}`);
      g.setAttribute('transform', `translate(${x},${y})`);
      g.style.cursor = 'pointer';

      const label = (runsIndex.get(id)?.name || id);
      g.setAttribute('role', 'button');
      g.setAttribute('tabindex', '0');
      g.setAttribute('aria-label', label);
      g.setAttribute('aria-pressed', String(isSelected));

      const rect = document.createElementNS(SVG_NS, 'rect');
      rect.setAttribute('width', String(nodeW));
      rect.setAttribute('height', String(nodeH));
      rect.setAttribute('rx', '10');
      rect.setAttribute('ry', '10');
      g.appendChild(rect);

      const fo = document.createElementNS(SVG_NS, 'foreignObject');
      fo.setAttribute('x', '0');
      fo.setAttribute('y', '0');
      fo.setAttribute('width', String(nodeW));
      fo.setAttribute('height', String(nodeH));
      fo.style.pointerEvents = 'none';

      const outer = document.createElement('div');
      outer.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
      Object.assign(outer.style, {
        display: 'flex',
        width: '100%',
        height: '100%',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '4px 8px',
      });

      const inner = document.createElement('div');
      Object.assign(inner.style, {
        display: '-webkit-box',
        WebkitBoxOrient: 'vertical',
        WebkitLineClamp: '2',
        overflow: 'hidden',
        textAlign: 'center',
        lineHeight: '1.2',
        fontWeight: '600',
        fontSize: '12px',
        overflowWrap: 'anywhere',
        wordBreak: 'break-word',
      });
      inner.textContent = label;

      outer.appendChild(inner);
      fo.appendChild(outer);
      g.appendChild(fo);

      const title = document.createElementNS(SVG_NS, 'title');
      title.textContent = label;
      g.appendChild(title);

      g.addEventListener('pointerdown', (evt) => {
        evt.stopPropagation();
      });

      function toggleMergeSelection() {
        if (selectedForMerge.has(id)) selectedForMerge.delete(id);
        else selectedForMerge.add(id);
        const selected = selectedForMerge.has(id);
        g.classList.toggle('selected', selected);
        g.setAttribute('aria-pressed', String(selected));
        updateMergeUi();
      }

      function handlePrimarySelect() {
        onPrimarySelect(id);
      }

      g.addEventListener('click', (evt) => {
        if (evt.shiftKey) toggleMergeSelection();
        else handlePrimarySelect();
      });
      g.addEventListener('keydown', (evt) => {
        if (evt.key === 'Enter' || evt.key === ' ') {
          evt.preventDefault();
          if (evt.shiftKey) toggleMergeSelection();
          else handlePrimarySelect();
        }
      });

      root.appendChild(g);
    });

    routed.forEach(e => {
      const path = document.createElementNS(SVG_NS, 'path');
      path.setAttribute('class', 'dagEdge');
      path.setAttribute('d', e.d);
      path.setAttribute('marker-end', 'url(#arrowHead)');
      root.appendChild(path);
    });
  }

  global.DagRender = {
    render,
    ensureDagLayers,
    wrapSvgTextIntoTspans,
    computeLayers,
    canvasHasLayout: _canvasHasLayout,
    deferUntilLayout: _deferUntilLayout,
  };
})(typeof window !== 'undefined' ? window : globalThis);
