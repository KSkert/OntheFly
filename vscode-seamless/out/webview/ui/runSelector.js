"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.keyOf = keyOf;
exports.runIdOf = runIdOf;
exports.createRunSelector = createRunSelector;
// ui/runSelector.ts
const dom_js_1 = require("./utils/dom.js");
function keyOf(id) { return id == null ? "" : String(id); }
function runIdOf(r) { return keyOf(r?.run_id ?? r?.id ?? r?.runId ?? r?.uuid ?? r?.uid); }
function createRunSelector(opts) {
    const sel = (0, dom_js_1.byId)("runSel");
    const prev = (0, dom_js_1.byId)("btnPrevModel");
    const next = (0, dom_js_1.byId)("btnNextModel");
    function setArrow(btn, enabled, targetId) {
        if (!btn)
            return;
        btn.dataset.target = enabled ? keyOf(targetId) : "";
        btn.disabled = !enabled;
        btn.style.opacity = enabled ? "" : "0.4";
        btn.style.pointerEvents = enabled ? "" : "none";
    }
    function updateArrows() {
        if (!sel || !prev || !next)
            return;
        const idx = sel.selectedIndex;
        const hasPrev = idx > 0;
        const hasNext = idx >= 0 && idx < sel.options.length - 1;
        const prevVal = hasPrev ? keyOf(sel.options[idx - 1].value) : "";
        const nextVal = hasNext ? keyOf(sel.options[idx + 1].value) : "";
        setArrow(prev, hasPrev, prevVal);
        setArrow(next, hasNext, nextVal);
    }
    function wireArrows() {
        if (!sel)
            return;
        if (prev && !prev._wired) {
            (0, dom_js_1.on)(prev, "click", () => {
                const t = (prev.dataset.target || "").trim();
                if (!t || t === sel.value)
                    return;
                const idx = Array.from(sel.options).findIndex(o => (o.value || "").trim() === t);
                if (idx >= 0) {
                    sel.selectedIndex = idx;
                    sel.dispatchEvent(new Event("change", { bubbles: true }));
                    updateArrows();
                }
            });
            prev._wired = true;
        }
        if (next && !next._wired) {
            (0, dom_js_1.on)(next, "click", () => {
                const t = (next.dataset.target || "").trim();
                if (!t || t === sel.value)
                    return;
                const idx = Array.from(sel.options).findIndex(o => (o.value || "").trim() === t);
                if (idx >= 0) {
                    sel.selectedIndex = idx;
                    sel.dispatchEvent(new Event("change", { bubbles: true }));
                    updateArrows();
                }
            });
            next._wired = true;
        }
    }
    (0, dom_js_1.on)(sel, "change", () => { if (sel?.value)
        opts.onChange(sel.value); updateArrows(); });
    function fillRunSel(rows) {
        if (!sel)
            return { activeChanged: false, activeId: null };
        const items = (rows || []).map(r => ({ id: runIdOf(r), row: r })).filter(x => x.id);
        const prevVal = keyOf(sel.value);
        sel.innerHTML = "";
        items.forEach(({ id, row }) => {
            const opt = document.createElement("option");
            opt.value = id;
            opt.textContent = String(row?.name || id);
            sel.appendChild(opt);
        });
        if (!items.length) {
            updateArrows();
            return { activeChanged: true, activeId: null };
        }
        const firstId = items[0].id;
        const target = items.some(x => x.id === prevVal) ? prevVal : firstId;
        const changed = keyOf(sel.value) !== target;
        sel.value = target;
        updateArrows();
        return { activeChanged: changed, activeId: target };
    }
    function current() { return sel?.value || null; }
    return { fillRunSel, updateArrows, wireArrows, current };
}
//# sourceMappingURL=runSelector.js.map