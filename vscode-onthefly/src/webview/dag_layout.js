(function (global) {
  const SVG_NS = 'http://www.w3.org/2000/svg';

  // -------------------- PAN / ZOOM: lightweight helper --------------------
  // Usage:
  //   const { reset, set, get } = enablePanZoom(svgEl, viewportG, { minScale: 0.2, maxScale: 6, step: 0.12 });
  //
  // - Wheel to zoom (anchored at cursor)
  // - Drag (mouse/touch/pen) to pan
  // - Double-click to zoom in at cursor
  // - Press "0" to reset
  function enablePanZoom(svg, viewport, opts = {}) {
    const minScale = opts.minScale ?? 0.25;
    const maxScale = opts.maxScale ?? 6;
    const step     = Math.max(0.01, opts.step ?? 0.1); // zoom step per wheel tick
    // NOTE: ensure pointer events work (prevents touch panning the page instead of the svg)
    svg.style.touchAction = 'none';

    const state = { x: 0, y: 0, k: 1 }; // translate (x,y), scale k
    let dragging = false;
    let dragStart = { x: 0, y: 0, sx: 0, sy: 0 };

    function apply() {
      viewport.setAttribute('transform', `translate(${state.x} ${state.y}) scale(${state.k})`);
    }

    function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

    function svgPointFromClient(clientX, clientY) {
      const r = svg.getBoundingClientRect();
      // convert screen point -> local graph space (before transform)
      const px = clientX - r.left;
      const py = clientY - r.top;
      const gx = (px - state.x) / state.k;
      const gy = (py - state.y) / state.k;
      return { gx, gy, px, py };
    }

    function zoomAt(clientX, clientY, dz) {
      const { gx, gy, px, py } = svgPointFromClient(clientX, clientY);
      const kOld = state.k;
      const kNew = clamp(kOld * (dz > 0 ? (1 - step) : (1 + step)), minScale, maxScale);

      if (kNew === kOld) return;
      // keep (gx,gy) under the cursor after zoom:
      // new screen pos: px' = (gx * kNew + x'), we want px' == px => x' = px - gx*kNew
      // same for y
      state.x = px - gx * kNew;
      state.y = py - gy * kNew;
      state.k = kNew;
      apply();
    }

    // Wheel to zoom (trackpads and standard wheels)
    const onWheel = (e) => {
      // prevent page scroll/zoom (especially in VS Code webview)
      e.preventDefault();
      // Prefer deltaY sign; ctrlKey may indicate pinch zoom on some platforms
      const dy = e.deltaY;
      zoomAt(e.clientX, e.clientY, dy);
    };

    // Pointer drag to pan
    const onPointerDown = (e) => {
      // Only primary button or touch/pen
      if (e.button !== undefined && e.button !== 0) return;
      dragging = true;
      svg.setPointerCapture?.(e.pointerId);
      dragStart = { x: e.clientX, y: e.clientY, sx: state.x, sy: state.y };
    };

    const onPointerMove = (e) => {
      if (!dragging) return;
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      state.x = dragStart.sx + dx;
      state.y = dragStart.sy + dy;
      apply();
    };

    const onPointerUp = (e) => {
      dragging = false;
      try { svg.releasePointerCapture?.(e.pointerId); } catch {}
    };

    const onDblClick = (e) => {
      // Zoom in by one "step" at cursor
      zoomAt(e.clientX, e.clientY, -1);
    };

    const onKeyDown = (e) => {
      if (e.key === '0') { reset(); }
      else if ((e.key === '+' || e.key === '=') && !e.ctrlKey && !e.metaKey) {
        zoomAt(svg.getBoundingClientRect().left + svg.clientWidth/2,
               svg.getBoundingClientRect().top + svg.clientHeight/2, -1);
      } else if (e.key === '-' && !e.ctrlKey && !e.metaKey) {
        zoomAt(svg.getBoundingClientRect().left + svg.clientWidth/2,
               svg.getBoundingClientRect().top + svg.clientHeight/2, +1);
      }
    };

    // Passive must be false to call preventDefault on wheel
    svg.addEventListener('wheel', onWheel, { passive: false });
    svg.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    svg.addEventListener('dblclick', onDblClick);
    window.addEventListener('keydown', onKeyDown);

    function reset() {
      state.x = 0; state.y = 0; state.k = 1; apply();
    }
    function set(next) {
      if (typeof next.k === 'number') state.k = clamp(next.k, minScale, maxScale);
      if (typeof next.x === 'number') state.x = next.x;
      if (typeof next.y === 'number') state.y = next.y;
      apply();
    }
    function get() { return { ...state }; }
    function destroy() {
      svg.removeEventListener('wheel', onWheel);
      svg.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      svg.removeEventListener('dblclick', onDblClick);
      window.removeEventListener('keydown', onKeyDown);
    }

    // initial paint
    apply();
    return { reset, set, get, destroy };
  }

  // -------------------- LAYOUT --------------------
  function layoutDAG(nodes, edges, opts = {}) {
    const nodeW   = opts.nodeW   ?? 150;
    const nodeH   = opts.nodeH   ?? 42;
    const margin  = opts.margin  ?? 48;
    const rankSep = opts.rankSep ?? 200;
    const nodeSep = opts.nodeSep ?? 28;

    // ---------- adjacency ----------
    const idSet = new Set(nodes.map(n => n.id));
    const inEdges  = new Map();
    const outEdges = new Map();
    nodes.forEach(n => { inEdges.set(n.id, []); outEdges.set(n.id, []); });
    edges.forEach(e => {
      if (!idSet.has(e.source) || !idSet.has(e.target)) return;
      inEdges.get(e.target).push(e.source);
      outEdges.get(e.source).push(e.target);
    });

    // ---------- ranks (x = depth by longest path) ----------
    const depth = new Map();
    const vis = new Set();
    const getDepth = (id) => {
      if (depth.has(id)) return depth.get(id);
      if (vis.has(id)) { depth.set(id, 0); return 0; }
      vis.add(id);
      const ps = inEdges.get(id) || [];
      const d = ps.length ? Math.max(...ps.map(getDepth)) + 1 : 0;
      vis.delete(id);
      depth.set(id, d);
      return d;
    };
    nodes.forEach(n => getDepth(n.id));

    const byDepth = new Map(); // d -> ids[]
    nodes.forEach(n => {
      const d = depth.get(n.id);
      if (!byDepth.has(d)) byDepth.set(d, []);
      byDepth.get(d).push(n.id);
    });
    const depths = Array.from(byDepth.keys()).sort((a,b)=>a-b);
    depths.forEach(d => byDepth.get(d).sort((a,b)=>a.localeCompare(b)));

    // ---------- primary parent (used only for layout) ----------
    const primaryParent = new Map();     // child -> parent
    const primaryChildren = new Map();   // parent -> [children]
    for (const d of depths) {
      const prev = byDepth.get(d-1) || [];
      for (const id of byDepth.get(d)) {
        const ps = (inEdges.get(id) || []).slice();
        if (!ps.length) continue;
        ps.sort((a,b) => (prev.indexOf(a) - prev.indexOf(b)) || a.localeCompare(b));
        primaryParent.set(id, ps[0]);
        if (!primaryChildren.has(ps[0])) primaryChildren.set(ps[0], []);
        primaryChildren.get(ps[0]).push(id);
      }
    }
    for (const [p, kids] of primaryChildren) {
      const d = (depth.get(p) || 0) + 1;
      const nextOrder = byDepth.get(d) || [];
      kids.sort((a,b)=> (nextOrder.indexOf(a) - nextOrder.indexOf(b)) || a.localeCompare(b));
    }

    // ---------- tidy lanes (y) ----------
    const lane = new Map();  // id -> float lane
    const seen = new Set();
    let nextLane = 0;

    function assign(id) {
      if (seen.has(id)) return lane.get(id);
      seen.add(id);
      const kids = (primaryChildren.get(id) || []).filter(k => depth.get(k) === depth.get(id) + 1);
      if (!kids.length) {
        const y = nextLane++;
        lane.set(id, y);
        return y;
      }
      const ys = [];
      for (const k of kids) ys.push(assign(k));
      const y = ys.reduce((a,b)=>a+b,0) / ys.length;
      lane.set(id, y);
      return y;
    }

    const roots = depths[0] != null
      ? byDepth.get(depths[0]).filter(id => !primaryParent.has(id))
      : [];
    for (const r of roots) assign(r);
    for (const id of nodes.map(n=>n.id)) if (!lane.has(id)) assign(id);

    // ---------- compact per column ----------
    const minGap = nodeH + nodeSep;
    const lanesUsed = Math.max(1, nextLane);
    const H = margin * 2 + (lanesUsed - 1) * minGap + nodeH;

    const pos = new Map(); // id -> {x,y,col,row}
    for (const d of depths) {
      const ids = byDepth.get(d) || [];
      if (!ids.length) continue;

      const targets = ids.map(id => ({ id, t: margin + (lane.get(id) || 0) * minGap }));
      targets.sort((a,b)=> (a.t - b.t) || a.id.localeCompare(b.id));

      let y = margin;
      for (const s of targets) {
        const top = Math.max(y, Math.min(s.t, H - margin - nodeH));
        s.y = top;
        y = top + minGap;
      }
      const overflow = (targets.length ? targets[targets.length-1].y + nodeH : margin) - (H - margin);
      if (overflow > 0) for (const s of targets) s.y -= overflow;

      const x = margin + d * (nodeW + rankSep);
      targets.forEach((s, idx) => pos.set(s.id, { x, y: s.y, col: d, row: idx }));
    }

    // ---------- helpers for obstacle clearance ----------
    const colLeft   = (col) => margin + col * (nodeW + rankSep);
    const colCenter = (col) => colLeft(col) + nodeW / 2;
    const colRight  = (col) => colLeft(col) + nodeW;

    const PAD = 10;      // extra clearance beyond box
    const EPS = 6;       // waypoint inset from a column's left/right edge

    function gapsForColumn(col) {
      // Build vertical gaps (inclusive) within [margin, H - margin]
      const nodesHere = (byDepth.get(col) || [])
        .map(id => ({ top: pos.get(id).y - PAD, bot: pos.get(id).y + nodeH + PAD }))
        .sort((a,b)=> a.top - b.top);

      const gaps = [];
      let cur = margin;
      for (const n of nodesHere) {
        const top = Math.max(margin, n.top);
        if (top - cur > 4) gaps.push([cur, top]); // [yMin, yMax)
        cur = Math.max(cur, n.bot);
      }
      if ((H - margin) - cur > 4) gaps.push([cur, H - margin]); // tail gap
      return gaps;
    }

    function pickYFromGaps(gaps, yPref) {
      // If yPref falls in any gap, keep it; else snap to nearest gap center.
      for (const [a,b] of gaps) if (yPref >= a && yPref <= b) return yPref;
      let best = gaps[0] ? (gaps[0][0] + gaps[0][1]) / 2 : yPref;
      let bestD = Infinity;
      for (const [a,b] of gaps) {
        const c = (a + b) / 2;
        const d = Math.abs(c - yPref);
        if (d < bestD) { bestD = d; best = c; }
      }
      return best;
    }

    // ---------- edge routing through safe waypoints ----------
    function routeEdge(pid, cid) {
      const P = pos.get(pid), C = pos.get(cid);
      if (!P || !C) return '';

      const x1 = P.x + nodeW, y1 = P.y + nodeH / 2;
      const x2 = C.x,         y2 = C.y + nodeH / 2;

      const colP = P.col, colC = C.col;

      // Start at the source anchor
      const pts = [{ x: x1, y: y1 }];

      // prefer to keep y near previous column’s choice to avoid jitter
      let lastY = y1;
      const STICK = 0.7; // 0..1: higher = stickier to previous y

      for (let k = colP + 1; k < colC; k++) {
        const gaps = gapsForColumn(k);

        // one y per column: choose at column center, then reuse for L/C/R
        const xL = colLeft(k) + EPS;
        const xC = colCenter(k);
        const xR = colRight(k) - EPS;

        const tC = (xC - x1) / (x2 - x1);
        const yLinear = y1 + (y2 - y1) * tC;

        // bias toward lastY to keep a consistent lane, then snap into a safe gap
        const yPref = STICK * lastY + (1 - STICK) * yLinear;
        const ySafe = pickYFromGaps(gaps, yPref);
        lastY = ySafe;

        // same y across the whole column eliminates in-column bumps
        pts.push({ x: xL, y: ySafe }, { x: xC, y: ySafe }, { x: xR, y: ySafe });
      }

      // End at the target anchor
      pts.push({ x: x2, y: y2 });

      // Build piecewise cubics; keep control-point y equal to endpoints' y
      // so each segment stays inside the horizontal strip defined by its endpoints.
      let d = `M ${pts[0].x} ${pts[0].y}`;
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], b = pts[i + 1];
        const dx = Math.max(40, (b.x - a.x) * 0.5);
        const c1x = a.x + dx, c1y = a.y;
        const c2x = b.x - dx, c2y = b.y;
        d += ` C ${c1x} ${c1y}, ${c2x} ${c2y}, ${b.x} ${b.y}`;
      }
      return d;
    }

    const routed = edges.map(e => ({ ...e, d: routeEdge(e.source, e.target) }));
    const W = margin * 2 + (depths.length ? (depths.length - 1) * (nodeW + rankSep) + nodeW : nodeW);

    return { pos, layers: byDepth, depth, routed, size: { W, H }, metrics: { nodeW, nodeH } };
  }

  layoutDAG.SVG_NS = SVG_NS;
  global.layoutDAG = layoutDAG;
  global.enablePanZoom = enablePanZoom;
})(window || globalThis);
