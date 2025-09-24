"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ForkOverlay = void 0;
// ui/charts/forkOverlay.ts
const dom_js_1 = require("../utils/dom.js");
class ForkOverlay {
    el = null;
    countPill = null;
    getSelectedCount;
    onSubmit;
    onCancel;
    constructor(opts) {
        this.getSelectedCount = opts.getSelectedCount;
        this.onSubmit = opts.onSubmit;
        this.onCancel = opts.onCancel;
    }
    ensure() {
        if (this.el)
            return;
        const wrap = document.createElement("div");
        Object.assign(wrap.style, {
            position: "static",
            background: "#334155",
            border: "1px solid #ddd",
            borderRadius: "10px",
            padding: "10px",
            boxShadow: "0 6px 20px rgba(0,0,0,0.15)",
            marginTop: "12px",
            display: "none",
        });
        const chartWrap = (0, dom_js_1.byId)("lossDistChart")?.closest(".chartWrap");
        (chartWrap?.parentNode || document.body).insertBefore(wrap, chartWrap?.nextSibling || null);
        wrap.innerHTML = `
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <span id="forkCountPill"
          style="padding:4px 10px;border-radius:9999px;background:var(--btn-bg);color:var(--btn-fg);
                 font-weight:600;font-size:12px;box-shadow:var(--btn-shadow);user-select:none;">
          Total Samples Selected: 0
        </span>
        <label>LR <input id="forkLR" type="number" step="0.0001" value="0.001" style="width:90px"></label>
        <label>Batch <input id="forkBS" type="number" step="1" value="32" style="width:80px"></label>
        <label>Patience <input id="forkPat" type="number" step="1" value="5" style="width:80px"></label>
        <button id="forkOk">Okay, fork</button>
        <button id="forkCancel">Cancel</button>
      </div>
    `;
        this.el = wrap;
        this.countPill = (0, dom_js_1.byId)("forkCountPill");
        (0, dom_js_1.on)((0, dom_js_1.byId)("forkOk"), "click", () => {
            const a = Number(((0, dom_js_1.byId)("forkLR")?.value || "0.001"));
            const b = Number(((0, dom_js_1.byId)("forkBS")?.value || "32"));
            const p = Number(((0, dom_js_1.byId)("forkPat")?.value || "5"));
            this.onSubmit({ minLoss: NaN, maxLoss: NaN }, { lr: a, batch_size: b, patience: p });
        });
        (0, dom_js_1.on)((0, dom_js_1.byId)("forkCancel"), "click", () => this.onCancel());
    }
    show() { this.ensure(); if (this.el)
        this.el.style.display = "block"; }
    hide() { if (this.el)
        this.el.style.display = "none"; }
    setCount(a, b) {
        if (!this.countPill)
            return;
        const n = Number.isFinite(a) && Number.isFinite(b) ? this.getSelectedCount(a, b) : 0;
        this.countPill.textContent = `Total Samples Selected: ${n}`;
    }
}
exports.ForkOverlay = ForkOverlay;
//# sourceMappingURL=forkOverlay.js.map