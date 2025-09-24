"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createCompareView = createCompareView;
// ui/compareView.ts
const dom_js_1 = require("./utils/dom.js");
function createCompareView(opts) {
    const grid = (0, dom_js_1.byId)("forgeGrid");
    const wCompare = (0, dom_js_1.byId)("w-compare");
    const compareEl = (wCompare && (wCompare.closest(".widget") || wCompare)) || null;
    compareEl?.classList.add("is-compare-card");
    const columnsGrid = (0, dom_js_1.byId)("columnsGrid");
    const btnAddColumn = (0, dom_js_1.byId)("btnAddColumn");
    const btnClearColumns = (0, dom_js_1.byId)("btnClearColumns");
    const compareRail = (0, dom_js_1.byId)("compareRail");
    const tabCreate = (0, dom_js_1.byId)("tabCreate");
    const tabCompare = (0, dom_js_1.byId)("tabCompare");
    const layoutBar = (0, dom_js_1.byId)("customBar") || document.querySelector(".customBar");
    let view = localStorage.getItem("fs.compare.view") || "train";
    const COL_STATE = new Map();
    const __AGG = "__aggregate__";
    const SUMMARY_CACHE = window.__SUMMARY_CACHE__ || (window.__SUMMARY_CACHE__ = new Map());
    (function installCompareCSS() {
        if (document.getElementById("compareCSS"))
            return;
        const style = document.createElement("style");
        style.id = "compareCSS";
        style.textContent = `
      #forgeGrid .widget.is-compare-card { display: none !important; }
      #forgeGrid[data-mode="compare"] .widget { display: none !important; }
      #forgeGrid[data-mode="compare"] .widget.is-compare-card { display: block !important; }
    `;
        document.head.appendChild(style);
    })();
    function setMode(mode) {
        const isCompare = mode === "compare";
        grid?.setAttribute("data-mode", isCompare ? "compare" : "create");
        tabCreate?.setAttribute("aria-selected", String(!isCompare));
        tabCompare?.setAttribute("aria-selected", String(isCompare));
        if (layoutBar)
            layoutBar.style.display = isCompare ? "none" : "";
        localStorage.setItem("fs.tab", isCompare ? "compare" : "create");
        if (isCompare) {
            ensureDefaultCompareColumn();
            refreshAllColumns();
        }
    }
    (0, dom_js_1.on)(tabCreate, "click", () => setMode("create"));
    (0, dom_js_1.on)(tabCompare, "click", () => setMode("compare"));
    setMode(localStorage.getItem("fs.tab") === "compare" ? "compare" : "create");
    function ensureAddTile() { if (columnsGrid && btnAddColumn)
        columnsGrid.appendChild(btnAddColumn); }
    function setColumnSubtitle(col, v) {
        const sub = col.querySelector(".capSub");
        if (sub)
            sub.textContent = `${v} summary`;
    }
    async function fillSummary(col, sessionId, runId, v) {
        const key = `${String(sessionId)}|${String(runId)}:${String(v)}`;
        const cached = SUMMARY_CACHE.get(key);
        const pre = col.querySelector(".summaryBox");
        if (typeof cached === "string" && cached.trim()) {
            pre.textContent = cached;
            return;
        }
        pre.textContent = "(loading…)";
        const txt = await opts.getSummary({ type: "session", sessionId, runId, view: v });
        if ((txt || "").trim())
            SUMMARY_CACHE.set(key, txt);
        else
            SUMMARY_CACHE.delete(key);
        pre.textContent = (txt || "").trim() ? txt : "(loading…)";
    }
    function makeColumnForSession(session) {
        if (!columnsGrid || !btnAddColumn)
            return null;
        const el = document.createElement("article");
        el.className = "modelCol";
        el.dataset.sessionId = session.id;
        const head = document.createElement("header");
        head.className = "modelCap";
        head.tabIndex = 0;
        head.title = "Change session";
        const cap = document.createElement("div");
        cap.className = "capInner";
        cap.innerHTML = `
      <span class="capName">${(0, dom_js_1.escapeHtml)(session.label || session.id)}</span>
      <span class="capSub">${(0, dom_js_1.escapeHtml)(view)} summary</span>
    `;
        const copyBtn = makeCopyBtn();
        head.append(cap, copyBtn);
        const pre = document.createElement("pre");
        pre.className = "summaryBox";
        pre.textContent = "(loading…)";
        el.append(head, pre);
        columnsGrid.insertBefore(el, btnAddColumn);
        ensureAddTile();
        COL_STATE.set(el, { sessionId: session.id, runId: __AGG });
        setColumnSubtitle(el, view);
        fillSummary(el, session.id, __AGG, view);
        const other = view === "train" ? "test" : "train";
        opts.getSummary({ type: "session", sessionId: session.id, runId: __AGG, view: other }); // warm cache
        return el;
    }
    function makeCopyBtn() {
        const b = document.createElement("button");
        b.type = "button";
        b.title = "Copy summary";
        b.className = "exportBtn";
        b.setAttribute("data-copy-summary", "");
        b.innerHTML = `
      <svg class="exportIcon" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
        <path d="M16 3H5a2 2 0 0 0-2 2v11h2V5h11V3zm3 4H9a2 2 0 0 0-2 2v12h12a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2zm0 14H9V9h10v12z" fill="currentColor"/>
      </svg>`;
        return b;
    }
    (0, dom_js_1.on)(compareRail, "click", (e) => {
        const btn = e.target.closest(".railTab");
        if (!btn)
            return;
        const v = btn.dataset.view;
        if (!v || v === view)
            return;
        compareRail?.querySelectorAll(".railTab").forEach(b => b.setAttribute("aria-selected", b === btn ? "true" : "false"));
        view = v;
        localStorage.setItem("fs.compare.view", view);
        refreshAllColumns();
    });
    (0, dom_js_1.on)(btnClearColumns, "click", () => {
        if (!columnsGrid)
            return;
        columnsGrid.querySelectorAll(".modelCol").forEach(n => n.remove());
        ensureAddTile();
    });
    document.addEventListener("click", async (e) => {
        const btn = e.target.closest("[data-copy-summary]");
        if (!btn)
            return;
        const col = (btn.closest(".modelCol") || document.getElementById("singleSessionCol"));
        const pre = col?.querySelector(".summaryBox");
        const text = (pre?.textContent || "").trim();
        if (!text || /^\s*(\(loading…\)|\(loading\.\.\.\)|loading|—|-|no data|n\/a)\s*$/i.test(text))
            return;
        try {
            await navigator.clipboard.writeText(text);
        }
        catch {
            const ta = document.createElement("textarea");
            ta.value = text;
            ta.style.position = "fixed";
            ta.style.top = "-10000px";
            document.body.appendChild(ta);
            ta.select();
            document.execCommand("copy");
            ta.remove();
        }
    });
    async function ensureDefaultCompareColumn() {
        if (!grid || grid.getAttribute("data-mode") !== "compare")
            return;
        if (!columnsGrid)
            return;
        if (columnsGrid.querySelector(".modelCol"))
            return;
        // simplest default: ask the extension to provide the “current run’s session”
        // you can enhance this by passing a current session id into this module
        opts.requestRunList?.();
    }
    function refreshAllColumns() {
        if (!columnsGrid)
            return;
        columnsGrid.querySelectorAll(".modelCol[data-session-id]").forEach(async (col) => {
            const st = COL_STATE.get(col);
            if (!st)
                return;
            setColumnSubtitle(col, view);
            await fillSummary(col, st.sessionId, "__aggregate__", view);
        });
        columnsGrid.querySelectorAll(".modelCol[data-run-id]").forEach(async (col) => {
            const runId = col.dataset.runId;
            const key = `${runId}:${view}`;
            const pre = col.querySelector(".summaryBox");
            const cached = SUMMARY_CACHE.get(key);
            const ready = (typeof cached === "string") && cached.trim();
            pre.textContent = ready ? cached : "(loading…)";
            if (!ready) {
                const txt = await opts.getSummary({ type: "run", runId, view });
                if ((txt || "").trim())
                    SUMMARY_CACHE.set(key, txt);
                else
                    SUMMARY_CACHE.delete(key);
                pre.textContent = (txt || "").trim() ? txt : "(loading…)";
            }
            setColumnSubtitle(col, view);
        });
    }
    return {
        makeColumnForSession,
        refreshAllColumns,
        ensureDefaultCompareColumn,
        setView(v) { view = v; refreshAllColumns(); },
    };
}
//# sourceMappingURL=compareView.js.map