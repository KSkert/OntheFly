/**
 * run_state.js
 * ------------------------------------------------------------------
 * Shared lineage + navigation store for the dashboard and any other
 * webviews that want to reuse the run bookkeeping layer.
 */

(function attachRunState(global) {
  const parentsOf = new Map();        // child -> Set(parents)
  const childrenOf = new Map();       // parent -> Set(children)
  const edges = [];                   // flat list of { source, target }
  const edgeSet = new Set();          // "parent→child" dedupe set
  const runsIndex = new Map();        // run_id -> normalized row
  const AF_MARKERS = new Map();       // runKey -> [{ step, kind }]
  const RUN_STATE = new Map();        // runKey -> { running, paused }
  const LAST_PAUSED_STEP = new Map(); // runKey -> step

  let NAV_LIST = [];                  // newest -> oldest run ids
  let PAGE_INDEX = 0;                 // 0 = newest (left-most)
  let CURRENT_LIVE_RUN = null;        // run id we consider "live"
  let FOLLOW_ACTIVE = true;           // whether follow-live is enabled
  let _followResetTimer = null;
  let lastRowsRunKey = '';

  const keyOf = (id) => (id == null ? '' : String(id));
  const runIdOf = (r) => keyOf(r?.run_id ?? r?.id ?? r?.runId ?? r?.uuid ?? r?.uid);

  function ensureRunRow(rowLike) {
    const id = runIdOf(rowLike);
    if (!id) return null;
    const created = rowLike?.created_at ?? rowLike?.createdAt ?? rowLike?.created ?? rowLike?.timestamp ?? rowLike?.ts ?? 0;
    const name = rowLike?.name ?? runsIndex.get(id)?.name ?? id;
    const merged = Array.isArray(rowLike?.merged) ? rowLike.merged.map(keyOf).filter(Boolean) : [];
    runsIndex.set(id, { ...runsIndex.get(id), ...rowLike, name, created_at: Number(created) || 0, merged });
    return id;
  }

  function addEdge(parentId, childId) {
    const p = keyOf(parentId);
    const c = keyOf(childId);
    if (!p || !c || p === c) return false;
    const k = `${p}→${c}`;
    if (edgeSet.has(k)) return false;

    edgeSet.add(k);
    edges.push({ source: p, target: c });

    if (!parentsOf.has(c)) parentsOf.set(c, new Set());
    parentsOf.get(c).add(p);

    if (!childrenOf.has(p)) childrenOf.set(p, new Set());
    childrenOf.get(p).add(c);
    return true;
  }

  function addRun({ id, parents = [], ...rest }) {
    if (!id) return;
    ensureRunRow({ id, ...rest });
    const ps = parents.map(keyOf).filter(Boolean);
    for (const p of ps) addEdge(p, id);
  }

  function clampPage(i) {
    if (!NAV_LIST.length) return 0;
    return Math.max(0, Math.min(NAV_LIST.length - 1, Number(i) || 0));
  }

  function currentPageRunId() {
    if (!NAV_LIST.length) return null;
    return NAV_LIST[clampPage(PAGE_INDEX)] || null;
  }

  function gotoPageByIndex(i) {
    PAGE_INDEX = clampPage(i);
    return currentPageRunId();
  }

  function gotoPageByRunId(runId) {
    const idx = NAV_LIST.indexOf(String(runId));
    if (idx === -1) return null;
    PAGE_INDEX = idx;
    return currentPageRunId();
  }

  // function rebuildNavListFromRows(rows) {
  //   const items = (rows || [])
  //     .map(r => {
  //       const id = runIdOf(r);
  //       const created = r?.created_at ?? r?.createdAt ?? r?.created ?? r?.timestamp ?? r?.ts ?? 0;
  //       return id ? { id: String(id), created_at: Number(created) || 0 } : null;
  //     })
  //     .filter(Boolean);

  //   items.sort((a, b) => b.created_at - a.created_at);
  //   NAV_LIST = items.map(x => x.id);
  //   PAGE_INDEX = clampPage(PAGE_INDEX);
  //   return NAV_LIST;
  // }

  function rebuildNavListFromRows(rows) {
    const isArr = Array.isArray(rows);
    const arr = isArr ? rows : [];
    const withIds = arr.filter(r => !!runIdOf(r));

    if (!isArr) {
      console.log('[RunState] rebuildNavListFromRows: rows is not an array:', typeof rows, rows);
    } else if (arr.length === 0) {
      console.log('[RunState] rebuildNavListFromRows: rows[] is EMPTY – keeping NAV_LIST as-is (len=%d)', NAV_LIST.length);
      PAGE_INDEX = clampPage(PAGE_INDEX);
      return NAV_LIST;
    } else if (withIds.length === 0) {
      console.log('[RunState] rebuildNavListFromRows: no run ids in rows. first row keys:',
        typeof arr[0] === 'object' ? Object.keys(arr[0]) : typeof arr[0], 'example row:', arr[0]);
    } else {
      console.log('[RunState] rebuildNavListFromRows: %d/%d rows have ids. example id=%o row=%o',
        withIds.length, arr.length, runIdOf(withIds[0]), withIds[0]);
    }

    // Build items from whatever we got (only rows that yield an id)
    let items = withIds.map(r => {
      const id = runIdOf(r);
      const created = r?.created_at ?? r?.createdAt ?? r?.created ?? r?.timestamp ?? r?.ts ?? 0;
      return { id: String(id), created_at: Number(created) || 0 };
    });

    // Fallback: if no items but we have an index, rebuild from runsIndex
    if (items.length === 0 && runsIndex.size) {
      console.log('[RunState] rebuildNavListFromRows: falling back to runsIndex (size=%d)', runsIndex.size);
      items = Array.from(runsIndex.entries()).map(([id, r]) => ({
        id: String(id),
        created_at: Number(r?.created_at ?? r?.createdAt ?? r?.created ?? r?.timestamp ?? r?.ts ?? 0) || 0
      }));
    }

    if (items.length === 0) {
      // Still nothing: keep current NAV_LIST, don’t blow away selection
      console.log('[RunState] rebuildNavListFromRows: no items after processing – keeping NAV_LIST (len=%d)', NAV_LIST.length);
      PAGE_INDEX = clampPage(PAGE_INDEX);
      return NAV_LIST;
    }

    items.sort((a, b) => b.created_at - a.created_at);
    NAV_LIST = items.map(x => x.id);
    PAGE_INDEX = clampPage(PAGE_INDEX);
    return NAV_LIST;
  }



  function updateModelNav() {
    const idx = clampPage(PAGE_INDEX);
    const hasPrev = idx > 0;
    const hasNext = idx < NAV_LIST.length - 1;
    return {
      hasPrev,
      hasNext,
      prevVal: hasPrev ? NAV_LIST[idx - 1] : '',
      nextVal: hasNext ? NAV_LIST[idx + 1] : '',
      idx,
      length: NAV_LIST.length,
    };
  }

  function setLiveRun(id) {
    CURRENT_LIVE_RUN = keyOf(id) || null;
    return CURRENT_LIVE_RUN;
  }

  function getLiveRun() {
    return CURRENT_LIVE_RUN;
  }

  function streamTargetRunId() {
    const live = keyOf(CURRENT_LIVE_RUN);
    if (FOLLOW_ACTIVE && live) return live;
    return keyOf(currentPageRunId());
  }

  function isFollowActive() {
    return FOLLOW_ACTIVE;
  }

  function setFollowActive(flag) {
    FOLLOW_ACTIVE = !!flag;
    if (FOLLOW_ACTIVE && _followResetTimer) {
      clearTimeout(_followResetTimer);
      _followResetTimer = null;
    }
  }

  function followTemporarilyOff(timeoutMs = 10000) {
    FOLLOW_ACTIVE = false;
    if (_followResetTimer) clearTimeout(_followResetTimer);
    _followResetTimer = setTimeout(() => { FOLLOW_ACTIVE = true; }, timeoutMs);
  }

  function getPageIndex() {
    return clampPage(PAGE_INDEX);
  }

  function getNavList() {
    return NAV_LIST;
  }

  function setLastRowsRunKey(id) {
    lastRowsRunKey = keyOf(id);
  }

  function getLastRowsRunKey() {
    return lastRowsRunKey;
  }

  function resetLineage() {
    parentsOf.clear();
    childrenOf.clear();
    runsIndex.clear();
    edges.length = 0;
    edgeSet.clear();
    AF_MARKERS.clear();
    RUN_STATE.clear();
    LAST_PAUSED_STEP.clear();
    NAV_LIST = [];
    PAGE_INDEX = 0;
    CURRENT_LIVE_RUN = null;
    FOLLOW_ACTIVE = true;
    if (_followResetTimer) {
      clearTimeout(_followResetTimer);
      _followResetTimer = null;
    }
    lastRowsRunKey = '';
  }

  global.RunState = {
    parentsOf,
    childrenOf,
    edges,
    runsIndex,
    AF_MARKERS,
    RUN_STATE,
    LAST_PAUSED_STEP,

    keyOf,
    runIdOf,

    ensureRunRow,
    addEdge,
    addRun,
    rebuildNavListFromRows,
    updateModelNav,

    currentPageRunId,
    gotoPageByIndex,
    gotoPageByRunId,
    getNavList,
    getPageIndex,

    setLiveRun,
    getLiveRun,
    streamTargetRunId,
    isFollowActive,
    setFollowActive,
    followTemporarilyOff,

    setLastRowsRunKey,
    getLastRowsRunKey,

    resetLineage,
  };
})(typeof window !== 'undefined' ? window : globalThis);
