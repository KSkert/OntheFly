"use strict";
// ui/utils/dom.ts
Object.defineProperty(exports, "__esModule", { value: true });
exports.byId = byId;
exports.on = on;
exports.escapeHtml = escapeHtml;
exports.cssAttr = cssAttr;
exports._set = _set;
exports._num = _num;
exports._bool = _bool;
/** Get any element (HTML or SVG) by id. */
function byId(id) {
    // getElementById is declared as HTMLElement|null, so widen to Element.
    return document.getElementById(id);
}
function on(el, evt, fn) {
    if (el?.addEventListener) {
        el.addEventListener(evt, fn);
    }
}
function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (m) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[m]));
}
function cssAttr(s) {
    return String(s).replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}
function _set(el, v) {
    if (!el)
        return;
    if ('type' in el && el.type === 'checkbox') {
        el.checked = !!v;
    }
    else {
        el.value = String(v);
    }
}
function _num(el) {
    const n = el ? Number(el.value) : NaN;
    return Number.isFinite(n) ? n : NaN;
}
function _bool(el) {
    return !!(el && 'checked' in el && el.checked);
}
//# sourceMappingURL=dom.js.map