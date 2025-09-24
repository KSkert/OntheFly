"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LogPanel = void 0;
// ui/logPanel.ts
const dom_js_1 = require("./utils/dom.js");
class LogPanel {
    ring;
    start = 0;
    len = 0;
    flushTimer = null;
    max;
    textarea;
    constructor(textareaId = "log", maxLines = 2000) {
        this.max = maxLines;
        this.ring = new Array(this.max);
        this.textarea = (0, dom_js_1.byId)(textareaId);
    }
    push(s) {
        const idx = (this.start + this.len) % this.max;
        this.ring[idx] = s;
        if (this.len < this.max)
            this.len++;
        else
            this.start = (this.start + 1) % this.max;
    }
    flushNow() {
        if (!this.textarea)
            return;
        const out = new Array(this.len);
        for (let i = 0; i < this.len; i++)
            out[i] = this.ring[(this.start + i) % this.max];
        const atBottom = (this.textarea.scrollTop + this.textarea.clientHeight) >= (this.textarea.scrollHeight - 4);
        this.textarea.value = out.join("\n");
        if (atBottom)
            this.textarea.scrollTop = this.textarea.scrollHeight;
    }
    log(s) {
        if (!this.textarea)
            return;
        this.push(String(s));
        if (this.flushTimer == null) {
            this.flushTimer = window.setTimeout(() => {
                this.flushTimer = null;
                this.flushNow();
            }, 200);
        }
    }
}
exports.LogPanel = LogPanel;
//# sourceMappingURL=logPanel.js.map