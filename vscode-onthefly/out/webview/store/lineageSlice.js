"use strict";
/**
 * store/lineageSlice.ts
 * Parent/child lineage maps + rebuild helper from rows.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.initialLineageSlice = void 0;
exports.pickPrimaryParent = pickPrimaryParent;
exports.rebuildLineageFromRows = rebuildLineageFromRows;
const runSlice_js_1 = require("./runSlice.js");
exports.initialLineageSlice = {
    parentsOf: new Map(),
    parentOf: new Map(),
    childrenOf: new Map(),
    runsIndex: new Map(),
    extraParents: new Map(),
};
function runIdOf(r) {
    return (0, runSlice_js_1.keyOf)(r?.run_id ?? r?.id ?? r?.runId ?? r?.uuid ?? r?.uid);
}
function pickPrimaryParent(childId, parentsOf, runsIndex) {
    const ps = Array.from(parentsOf.get(childId) || []);
    if (!ps.length)
        return null;
    ps.sort((a, b) => (runsIndex.get(b)?.created_at || 0) - (runsIndex.get(a)?.created_at || 0));
    return ps[0] || null;
}
function rebuildLineageFromRows(state, rows) {
    state.parentsOf.clear();
    state.parentOf.clear();
    state.childrenOf.clear();
    state.runsIndex.clear();
    if (!Array.isArray(rows) || rows.length === 0)
        return;
    for (const r of rows) {
        const id = runIdOf(r);
        if (!id)
            continue;
        const created = r.created_at ?? r.createdAt ?? r.created ?? r.timestamp ?? r.ts ?? 0;
        state.runsIndex.set(id, { ...r, created_at: Number(created) });
        const rowParents = Array.isArray(r.parents)
            ? r.parents
            : [r.parent ?? r.parent_run ?? r.parent_run_id ?? r.parentId ?? null];
        const learned = state.extraParents.get(id);
        const ps = new Set(rowParents.map(runSlice_js_1.keyOf).filter((p) => p && p !== id));
        if (learned)
            for (const p of learned)
                if (p && p !== id)
                    ps.add((0, runSlice_js_1.keyOf)(p));
        if (ps.size) {
            state.parentsOf.set(id, ps);
            for (const p of ps) {
                if (!state.childrenOf.has(p))
                    state.childrenOf.set(p, new Set());
                state.childrenOf.get(p).add(id);
            }
        }
    }
    // derive a primary parent for navigation arrows
    for (const id of state.runsIndex.keys()) {
        const p = pickPrimaryParent(id, state.parentsOf, state.runsIndex);
        if (p && p !== id)
            state.parentOf.set(id, p);
    }
}
//# sourceMappingURL=lineageSlice.js.map