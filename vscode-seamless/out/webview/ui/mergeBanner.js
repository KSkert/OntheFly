"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ensureMergeBanner = ensureMergeBanner;
exports.showMergeBanner = showMergeBanner;
exports.hideMergeBanner = hideMergeBanner;
// ui/mergeBanner.ts
let el = null;
let timer = null;
function ensureMergeBanner() {
    if (el)
        return el;
    const d = document.createElement("div");
    d.id = "mergeBanner";
    Object.assign(d.style, {
        position: "fixed",
        right: "16px",
        bottom: "16px",
        zIndex: "9999",
        display: "none",
        alignItems: "center",
        gap: "8px",
        padding: "10px 12px",
        borderRadius: "10px",
        background: "var(--btn-bg, #1f2937)",
        color: "var(--btn-fg, #fff)",
        boxShadow: "var(--btn-shadow, 0 6px 20px rgba(0,0,0,0.18))",
        fontWeight: "600",
        fontSize: "12px",
        maxWidth: "42ch",
        lineHeight: "1.25",
    });
    document.body.appendChild(d);
    el = d;
    return d;
}
function showMergeBanner(text, kind = "info", { sticky = false } = {}) {
    const d = ensureMergeBanner();
    d.textContent = "";
    const icon = document.createElement("span");
    icon.textContent = kind === "error" ? "⚠️" : (kind === "busy" ? "⏳" : "ℹ️");
    d.append(icon, document.createTextNode(text));
    d.style.display = "flex";
    if (!sticky) {
        if (timer)
            clearTimeout(timer);
        timer = window.setTimeout(() => { d.style.display = "none"; }, 3500);
    }
}
function hideMergeBanner() {
    const d = ensureMergeBanner();
    d.style.display = "none";
}
//# sourceMappingURL=mergeBanner.js.map