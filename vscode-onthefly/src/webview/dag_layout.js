(function (global) {
  'use strict';
  const SVG_NS = 'http://www.w3.org/2000/svg';

  // =====================================================================
  // PAN / ZOOM (pointer + wheel + keyboard), cursor-centric zoom (CTM-based)
  // API: enablePanZoom(svg, viewport, opts?) -> { set, reset, get, destroy }
  // =====================================================================
  function enablePanZoom(svg, viewport, opts = {}) {
    if (!svg || svg.tagName?.toLowerCase?.() !== 'svg') {
      throw new Error('enablePanZoom: first arg must be an <svg>');
    }
    if (!viewport || typeof viewport.setAttribute !== 'function') {
      throw new Error('enablePanZoom: second arg must be an SVG graphics element (e.g. <g>)');
    }

    const minScale = opts.minScale ?? 0.25;
    const maxScale = opts.maxScale ?? 6;
    const zoomSpeed = opts.zoomSpeed ?? 0.006; // Lower = smoother

    svg.style.touchAction = 'none';

    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    const state = { x: 0, y: 0, k: 1 };
    let dragging = false;
    let dragStart = { x: 0, y: 0, sx: 0, sy: 0 };

    function apply() {
      // Keep original order; math below matches translate then scale
      viewport.setAttribute('transform', `translate(${state.x} ${state.y}) scale(${state.k})`);
    }

    // --- CTM helpers: robust screen<->graph mapping ----------------------
    function getCTM() {
      // CTM mapping object coords -> screen coords for the <g> viewport
      // Use current transform matrix from the SVG DOM; fallback to identity
      return viewport.getCTM?.() || svg.createSVGMatrix();
    }

    function clientToGraph(clientX, clientY) {
      const pt = svg.createSVGPoint();
      pt.x = clientX;
      pt.y = clientY;
      const inv = getCTM().inverse();
      const p = pt.matrixTransform(inv);
      return { gx: p.x, gy: p.y };
    }

    function zoomAt(clientX, clientY, deltaY) {
      const r = svg.getBoundingClientRect();

      const inRect = (x, y) =>
        Number.isFinite(x) && Number.isFinite(y) &&
        x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;

      const useX = inRect(clientX, clientY) ? clientX : (r.left + r.width / 2);
      const useY = inRect(clientX, clientY) ? clientY : (r.top + r.height / 2);

      // Graph-space coords under the cursor BEFORE changing k
      const { gx, gy } = clientToGraph(useX, useY);

      const kOld = state.k;
      const factor = Math.exp(-deltaY * zoomSpeed); // smooth multiplicative zoom
      const kNew = clamp(kOld * factor, minScale, maxScale);
      if (Math.abs(kNew - kOld) < 1e-4) return;

      // Incremental update keeps (gx,gy) fixed on screen
      state.x += (kOld - kNew) * gx;
      state.y += (kOld - kNew) * gy;
      state.k = kNew;
      apply();
    }

    function zoomIncrementAt(clientX, clientY, factor) {
      const r = svg.getBoundingClientRect();
      const useX = Number.isFinite(clientX) ? clientX : (r.left + r.width / 2);
      const useY = Number.isFinite(clientY) ? clientY : (r.top + r.height / 2);

      const { gx, gy } = clientToGraph(useX, useY);

      const kOld = state.k;
      const kNew = clamp(kOld * factor, minScale, maxScale);
      if (Math.abs(kNew - kOld) < 1e-4) return;

      state.x += (kOld - kNew) * gx;
      state.y += (kOld - kNew) * gy;
      state.k = kNew;
      apply();
    }

    const onWheel = (e) => {
      e.preventDefault();
      // Normalize delta across devices (pixels/lines/pages)
      let dy = e.deltaY;
      if (e.deltaMode === 1) dy *= 15;       // lines -> px
      else if (e.deltaMode === 2) dy *= 120; // pages -> px
      zoomAt(e.clientX, e.clientY, dy);
    };

    const onPointerDown = (e) => {
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
      e.preventDefault?.();
      zoomIncrementAt(e.clientX, e.clientY, 1.5); // Zoom in by 50%
    };

    const onKeyDown = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable)) return;
      if (e.key === '0') { reset(); }
      else if ((e.key === '+' || e.key === '=') && !e.ctrlKey && !e.metaKey) {
        zoomIncrementAt(null, null, 1.2);
      } else if (e.key === '-' && !e.ctrlKey && !e.metaKey) {
        zoomIncrementAt(null, null, 1 / 1.2);
      }
    };

    svg.addEventListener('wheel', onWheel, { passive: false });
    svg.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    svg.addEventListener('dblclick', onDblClick);
    window.addEventListener('keydown', onKeyDown);

    function reset() { state.x = 0; state.y = 0; state.k = 1; apply(); }
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

    apply();
    return { set, reset, get, destroy };
  }

  // =====================================================================
  // GRID DAG LAYOUT (discrete 20x20 by default; squares 4x node size)
  // API: layoutDAG(nodes, edges, opts?) -> { pos, depth, routed, size, grid, initial }
  // =====================================================================
  function layoutDAG(nodes, edges, opts = {}) {
    // ---- Geometry & grid ----
    const nodeW      = opts.nodeW   ?? 150;
    const nodeH      = opts.nodeH   ?? 42;
    const gridCols   = Math.max(1, opts.gridCols ?? 20);
    const gridRows   = Math.max(1, opts.gridRows ?? 20);
    const cellW      = (opts.cellW ?? (nodeW * 4));   // squares are 4x node size (wide)
    const cellH      = (opts.cellH ?? (nodeH * 4));   // squares are 4x node size (tall)
    const margin     = opts.margin  ?? 24;            // outer padding around whole grid

    // column 0 is "depth 0". We place the FIRST node at the center column,
    // and we only grow to the RIGHT (left half remains empty forever).
    const centerCol  = Math.floor(gridCols / 2);
    const centerRow  = Math.floor(gridRows / 2);

    // ---- Input sanitation ----
    const idList = (nodes || []).filter(n => n && typeof n.id === 'string').map(n => n.id);
    const idSet  = new Set(idList);
    const E = (edges || []).filter(e => e && idSet.has(e.source) && idSet.has(e.target));

    // ---- Build adjacency ----
    const parents = new Map(); // id -> array of parent ids
    const kids    = new Map(); // id -> array of child ids
    for (const id of idSet) { parents.set(id, []); kids.set(id, []); }
    for (const e of E) { parents.get(e.target).push(e.source); kids.get(e.source).push(e.target); }

    // ---- Depths (strictly increasing to the right)
    // Use 1 + MAX(parentDepth) to ensure all edges go left->right.
    const depth = new Map();
    const visiting = new Set();
    const roots = [];
    function getDepth(id) {
      if (depth.has(id)) return depth.get(id);
      if (visiting.has(id)) { depth.set(id, 0); return 0; } // cycle guard
      visiting.add(id);
      const ps = parents.get(id);
      const d = ps.length ? Math.max(...ps.map(getDepth)) + 1 : 0;
      visiting.delete(id);
      depth.set(id, d);
      return d;
    }
    for (const id of idSet) if (getDepth(id) === 0) roots.push(id);

    // ---- Grid occupancy --------------------------------------------------
    // states: 0 = empty, 1 = node, 2 = corridor (edge passes through)
    const occ = Array.from({ length: gridCols }, () => new Array(gridRows).fill(0));
    const pos = new Map(); // id -> { col,row,x,y }
    const colNodes = new Map(); // depth -> ids (stable order)
    for (const id of idSet) {
      const d = depth.get(id);
      if (!colNodes.has(d)) colNodes.set(d, []);
      colNodes.get(d).push(id);
    }
    // deterministic order inside columns
    for (const [d, arr] of colNodes) arr.sort();

    // ---- Helpers ---------------------------------------------------------
    const colToX = (c) => margin + c * cellW + (cellW - nodeW) / 2;
    const rowToY = (r) => margin + r * cellH + (cellH - nodeH) / 2;

    const clampRow = (r) => Math.max(0, Math.min(gridRows - 1, r));
    const clampCol = (c) => Math.max(0, Math.min(gridCols - 1, c));

    function nearestFreeRow(col, targetRow) {
      targetRow = clampRow(targetRow);
      if (occ[col][targetRow] === 0) return targetRow;
      for (let k = 1; k < gridRows; k++) {
        const up = targetRow - k, dn = targetRow + k;
        if (up >= 0 && occ[col][up] === 0) return up;
        if (dn < gridRows && occ[col][dn] === 0) return dn;
      }
      return targetRow; // fallback (will be marked anyway)
    }

    // Even/odd fan-out offsets:
    // N odd: [0, -1, +1, -2, +2, ...]
    // N even: [-1, +1, -2, +2, ...]
    function fanOffsets(count) {
      const arr = [];
      if (count <= 0) return arr;
      if (count % 2 === 1) {
        arr.push(0);
        for (let k = 1; arr.length < count; k++) {
          arr.push(-k, +k);
        }
      } else {
        for (let k = 1; arr.length < count; k++) {
          arr.push(-k, +k);
        }
      }
      return arr;
    }

    // Rasterize a straight line across columns, picking one integer row per intermediate column.
    // Returns { ok, cells } where cells = [{col,row}, ...] for intermediate columns.
    function straightCorridor(colA, rowA, colB, rowB) {
      const leftCol = Math.min(colA, colB);
      const rightCol = Math.max(colA, colB);
      const a = (colA < colB) ? { c: colA, r: rowA } : { c: colB, r: rowB };
      const b = (colA < colB) ? { c: colB, r: rowB } : { c: colA, r: rowA };
      const dx = b.c - a.c;
      const dy = b.r - a.r;

      const cells = [];
      for (let c = a.c + 1; c < b.c; c++) {
        const t = (c - a.c) / dx;
        const r = Math.round(a.r + dy * t);
        if (r < 0 || r >= gridRows) return { ok: false, cells: [] };
        if (occ[c][r] === 1) return { ok: false, cells: [] }; // cannot pass through a node
        cells.push({ col: c, row: r });
      }
      return { ok: true, cells };
    }

    function reserveCorridor(cells) {
      for (const { col, row } of cells) {
        if (occ[col][row] === 0) occ[col][row] = 2;
      }
    }

    // For multi-parent children, place at the rounded vertical midpoint of parents' rows.
    function midpointRowOfParents(pars) {
      if (!pars.length) return centerRow;
      const rs = pars.map(p => pos.get(p)?.row).filter(r => r !== undefined);
      if (!rs.length) return centerRow;
      const avg = rs.reduce((a,b)=>a+b,0) / rs.length;
      return Math.round(avg);
    }

    // ---- Column coordinates (depth -> grid column) -----------------------
    // Place depth 0 at centerCol, then depth d at centerCol + d. (strictly rightward)
    function depthToCol(d) { return clampCol(centerCol + d); }

    // ---- Stage 1: place depth 0 ------------------------------------------
    // If there are multiple depth-0 nodes, stack around centerRow using the same offset rule.
    if (colNodes.has(0)) {
      const rootsInCol = colNodes.get(0);
      const offs = fanOffsets(rootsInCol.length);
      const col0 = depthToCol(0);
      for (let i = 0; i < rootsInCol.length; i++) {
        const id = rootsInCol[i];
        const rWanted = centerRow + offs[i];
        const r = nearestFreeRow(col0, rWanted);
        occ[col0][r] = 1;
        pos.set(id, { col: col0, row: r, x: colToX(col0), y: rowToY(r) });
      }
    }

    // ---- Precompute per-parent child offsets for single-step children ----
    // This satisfies the slope-magnitude rule for 1/2/3/4/... (evens/odds pattern).
    const childOrder = new Map(); // parentId -> Map(childId -> offset)
    for (const id of idSet) {
      const ks = kids.get(id).slice().sort();
      const offs = fanOffsets(ks.length);
      const map = new Map();
      for (let i = 0; i < ks.length; i++) map.set(ks[i], offs[i]);
      childOrder.set(id, map);
    }

    // ---- Stage 2: place remaining depths left->right ---------------------
    const maxDepth = Math.max(...Array.from(depth.values()));
    for (let d = 1; d <= maxDepth; d++) {
      const idsHere = (colNodes.get(d) || []).slice();
      const col = depthToCol(d);

      // Order: place nodes with multiple parents first (so single-parent ones can fan around them cleanly)
      idsHere.sort((a, b) => (parents.get(b).length - parents.get(a).length) || a.localeCompare(b));

      for (const id of idsHere) {
        const ps = parents.get(id).slice();
        // Desired row:
        let rDesired;
        if (ps.length === 1) {
          const p = ps[0];
          const pPos = pos.get(p);
          const base = pPos ? pPos.row : centerRow;
          // If this edge is a single-step (parent depth == d-1), use the fan offset.
          if (depth.get(p) === d - 1) {
            const off = childOrder.get(p)?.get(id) ?? 0;
            rDesired = base + off;
          } else {
            // Long jump: aim to keep the same row by default.
            rDesired = base;
          }
        } else {
          // Multi-parent: place at midpoint of parents (halfway vertically)
          rDesired = midpointRowOfParents(ps);
        }
        rDesired = clampRow(rDesired);

        // Find nearest free row for the node itself
        let rPlaced = nearestFreeRow(col, rDesired);

        // Ensure corridors from ALL parents are clean; if not, adjust row until all are clean
        // Try rows expanding around rDesired until corridors are valid
        function corridorsOKForRow(testRow) {
          for (const p of ps) {
            const pPos = pos.get(p);
            if (!pPos) continue; // parent not yet placed; unlikely but guard
            const { ok } = straightCorridor(pPos.col, pPos.row, col, testRow);
            if (!ok) return false;
          }
          return true;
        }
        if (!corridorsOKForRow(rPlaced)) {
          // scan up/down for a row that allows clean corridors through intermediate layers
          let found = false;
          for (let k = 1; k < gridRows; k++) {
            const up = rDesired - k;
            const dn = rDesired + k;
            if (up >= 0 && occ[col][up] === 0 && corridorsOKForRow(up)) { rPlaced = up; found = true; break; }
            if (dn < gridRows && occ[col][dn] === 0 && corridorsOKForRow(dn)) { rPlaced = dn; found = true; break; }
          }
          if (!found) {
            // As last resort, keep the nearest free row even if corridors will later get nudged (we still reserve)
          }
        }

        // Reserve node cell
        occ[col][rPlaced] = 1;
        pos.set(id, { col, row: rPlaced, x: colToX(col), y: rowToY(rPlaced) });

        // Reserve corridors from each parent to this child
        for (const p of ps) {
          const pPos = pos.get(p);
          if (!pPos) continue;
          const cr = straightCorridor(pPos.col, pPos.row, col, rPlaced);
          if (cr.ok) reserveCorridor(cr.cells);
          // If not ok, we leave routing to later, but edges still won't be allowed through node cells.
        }
      }
    }

    // ---- Edge routing: strict left->right from right-mid (src) to left-mid (tgt)
    // Path is a simple polyline with a single straight segment that matches the grid corridor.
    const routed = [];
    for (const e of E) {
      const s = pos.get(e.source);
      const t = pos.get(e.target);
      if (!s || !t) continue;
      const x1 = s.x + nodeW;         // rightmost mid of source
      const y1 = s.y + nodeH / 2;
      const x2 = t.x;                  // leftmost mid of target
      const y2 = t.y + nodeH / 2;

      // For aesthetics, we keep it as a straight line matching the grid corridor.
      const dPath = `M ${x1} ${y1} L ${x2} ${y2}`;
      routed.push({ ...e, d: dPath, points: [{x:x1,y:y1},{x:x2,y:y2}] });
    }

    // ---- Output canvas size (entire grid extents)
    const W = margin * 2 + gridCols * cellW;
    const H = margin * 2 + gridRows * cellH;

    // ---- Initial zoom/window (center on the occupied graph) --------------
    // Frame a configurable CxR region centered on the middle of the occupied
    // columns/rows. Defaults to 1x1 (very close). Override via opts.initialCols/Rows.
    const initCols = Math.max(1, opts.initialCols ?? 1);
    const initRows = Math.max(1, opts.initialRows ?? 1);

    let initial = null;
    if (pos.size > 0) {
      // Compute occupied column/row extents from placed nodes
      let minCol = Infinity, maxCol = -Infinity, minRow = Infinity, maxRow = -Infinity;
      pos.forEach(({ col, row }) => {
        if (col < minCol) minCol = col;
        if (col > maxCol) maxCol = col;
        if (row < minRow) minRow = row;
        if (row > maxRow) maxRow = row;
      });

      // Center (can be fractional: e.g., 1.5 means "between col 1 and 2")
      const centerColF = (minCol + maxCol) / 2;
      const centerRowF = (minRow + maxRow) / 2;

      // Convert grid center to absolute coords (allowing fractional centers)
      const centerX = margin + centerColF * cellW + cellW / 2;
      const centerY = margin + centerRowF * cellH + cellH / 2;

      const viewW = initCols * cellW;
      const viewH = initRows * cellH;

      const x = centerX - viewW / 2;
      const y = centerY - viewH / 2;

      initial = {
        viewBox: { x, y, width: viewW, height: viewH },
        panZoom: { x: -x, y: -y, k: 1 }
      };
    }

    // Build a simple debug grid description (optional for renderers)
    const grid = {
      cols: gridCols,
      rows: gridRows,
      cellW, cellH,
      centerCol, centerRow,
      occupancy: occ // 0 empty, 1 node, 2 corridor
    };

    return {
      pos,                // Map(id -> { col,row,x,y })
      depth,              // Map(id -> depth)
      routed,             // edges with SVG path 'd'
      size: { W, H },     // full canvas size
      grid,               // grid info if the renderer wants it
      metrics: { nodeW, nodeH, cellW, cellH, margin },
      initial             // suggested initial viewport (centered on full graph)
    };
  }

  // Expose
  layoutDAG.SVG_NS = SVG_NS;
  global.layoutDAG = layoutDAG;
  global.enablePanZoom = enablePanZoom;

})(window || globalThis);
