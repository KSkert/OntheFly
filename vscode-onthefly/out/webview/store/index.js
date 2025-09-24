"use strict";
// src/webview/store/index.ts
Object.defineProperty(exports, "__esModule", { value: true });
exports.createStore = createStore;
exports.createAppStore = createAppStore;
function createStore(initial) {
    let state = Object.freeze({ ...initial });
    const listeners = new Set();
    function getState() { return state; }
    function setState(updater, _action) {
        const prev = state;
        const patch = typeof updater === "function"
            ? updater(prev)
            : updater;
        const next = Object.freeze({ ...prev, ...patch });
        if (next === prev)
            return;
        state = next;
        for (const l of listeners) {
            try {
                l(state, prev);
            }
            catch (e) {
                console.error("[store]", e);
            }
        }
    }
    function subscribe(fn) {
        listeners.add(fn);
        return () => listeners.delete(fn);
    }
    return { getState, setState, subscribe };
}
const runSlice_js_1 = require("./runSlice.js");
const reportSlice_js_1 = require("./reportSlice.js");
const seriesSlice_js_1 = require("./seriesSlice.js");
const lineageSlice_js_1 = require("./lineageSlice.js");
const compareSlice_js_1 = require("./compareSlice.js");
function createAppStore() {
    const initial = {
        ...runSlice_js_1.initialRunSlice,
        ...reportSlice_js_1.initialReportSlice,
        ...seriesSlice_js_1.initialSeriesSlice,
        ...lineageSlice_js_1.initialLineageSlice,
        ...compareSlice_js_1.initialCompareSlice,
    };
    return createStore(initial);
}
//# sourceMappingURL=index.js.map