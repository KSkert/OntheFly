"use strict";
/**
 * host.ts
 * ------------------------------------------------------------
 * Thin wrapper around the VS Code webview messaging API.
 *
 * - Strongly typed router: handler for "rows" gets the "rows" shape, etc.
 * - Framework-agnostic; no DOM/store deps here.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.createHost = createHost;
function getVSCode() {
    // VS Code injects acquireVsCodeApi into the webview environment.
    try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any, no-undef
        const fn = globalThis.acquireVsCodeApi;
        if (typeof fn === "function")
            return fn();
    }
    catch {
        /* noop */
    }
    // Fallback shim so local dev in a plain browser doesn't explode.
    const shim = {
        postMessage: (data) => {
            // eslint-disable-next-line no-console
            console.warn("[VSCodeShim] postMessage (dev shim):", data);
        },
        getState: () => ({}),
        setState: (_) => { },
    };
    // eslint-disable-next-line no-console
    console.warn("[host] acquireVsCodeApi() not found. Using dev shim.");
    return shim;
}
function createHost() {
    const vscode = getVSCode();
    // Fan-out list for all low-level subscribers:
    const subscribers = new Set();
    // One delegated onmessage for the whole app:
    const onWindowMessage = (ev) => {
        const msg = ev?.data;
        if (!msg || typeof msg !== "object")
            return;
        for (const fn of subscribers) {
            try {
                fn(msg);
            }
            catch (err) {
                // eslint-disable-next-line no-console
                console.error("[host] subscriber error:", err);
            }
        }
    };
    window.addEventListener("message", onWindowMessage);
    const api = {
        send(payload) {
            try {
                vscode.postMessage(payload);
            }
            catch (err) {
                // eslint-disable-next-line no-console
                console.error("[host] send error:", err, payload);
            }
        },
        onMessage(fn) {
            subscribers.add(fn);
            // Return unsubscribe
            return () => subscribers.delete(fn);
        },
        route(table) {
            // Wrap into a single low-level subscriber; narrow per type at callsite.
            const handler = (m) => {
                const t = m.type;
                const specific = table[t];
                if (specific)
                    specific(m);
                if (table["*"])
                    table["*"](m);
            };
            return api.onMessage(handler);
        },
        vscode,
    };
    // Helpful default: log unhandled errors coming from the host
    api.route({
        error: (m) => {
            // eslint-disable-next-line no-console
            console.error("[host error]", m);
        },
    });
    return api;
}
//# sourceMappingURL=host.js.map