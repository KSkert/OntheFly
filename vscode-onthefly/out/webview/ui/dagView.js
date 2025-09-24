"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createDagView = createDagView;
// ui/dagView.ts
const dom_js_1 = require("./utils/dom.js");
const SVG_NS = "http://www.w3.org/2000/svg";
function createDagView(opts) {
    const overlay = (0, dom_js_1.byId)("dagOverlay");
    const svg = (0, dom_js_1.byId)("dagSvg");
    const btnClose = (0, dom_js_1.byId)("dagClose");
    const btnMerge = (0, dom_js_1.byId)("dagMergeBtn");
    const strategySel = (0, dom_js_1.byId)("dagStrategy");
    const selected = new Set();
    function updateMergeUi() {
        if (!btnMerge)
            return;
        const n = selected.size;
        btnMerge.disabled = (n !== 2);
        btnMerge.textContent = (n === 2) ? "Merge selected" : "Pick 2 to merge";
    }
    (0, dom_js_1.on)(btnMerge, "click", () => {
        if (selected.size !== 2)
            return;
        const parents = Array.from(selected);
        const strategy = (strategySel && strategySel.value) || "swa";
        opts.onMergeClick(parents, strategy);
    });
    function ensureArrowDefs(svgEl) {
        let defs = svgEl.querySelector("#dag-defs");
        if (defs)
            return;
        defs = document.createElementNS(SVG_NS, "defs");
        defs.id = "dag-defs";
        defs.innerHTML = `
      <marker id="arrowHead" markerWidth="12" markerHeight="10"
              viewBox="0 0 12 10" refX="12" refY="5" orient="auto"
              markerUnits="userSpaceOnUse">
        <path class="dagEdgeArrow" d="M0,0 L12,5 L0,10 Z"></path>
      </marker>`;
        svgEl.appendChild(defs);
    }
    function render() {
        if (!svg)
            return;
        while (svg.firstChild)
            svg.removeChild(svg.firstChild);
        ensureArrowDefs(svg);
        const root = document.createElementNS(SVG_NS, "g");
        root.setAttribute("id", "dagRoot");
        svg.appendChild(root);
        const data = opts.getData();
        const nodeW = 150, nodeH = 42;
        if (!data.nodes.length) {
            svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
            const MIN_W = 960, MIN_H = 560;
            svg.setAttribute("viewBox", `0 0 ${MIN_W} ${MIN_H}`);
            svg.setAttribute("width", "100%");
            svg.setAttribute("height", "100%");
            const t = document.createElementNS(SVG_NS, "text");
            t.setAttribute("x", "50%");
            t.setAttribute("y", "50%");
            t.setAttribute("text-anchor", "middle");
            t.setAttribute("fill", "#fff");
            t.textContent = "No runs yet";
            root.appendChild(t);
            return;
        }
        // Use external layout (window.layoutDAG) already in your webview
        // @ts-ignore
        const { pos, routed, size } = window.layoutDAG(data.nodes, data.edges, { nodeW, nodeH, margin: 48, rankSep: 200, nodeSep: 28, iterations: 6 });
        const MIN_W = 960, MIN_H = 560;
        const W = size.W, H = size.H;
        const vbW = Math.max(W, MIN_W), vbH = Math.max(H, MIN_H);
        svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
        svg.setAttribute("viewBox", `0 0 ${vbW} ${vbH}`);
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", "100%");
        const offsetX = (vbW - W) / 2, offsetY = (vbH - H) / 2;
        root.setAttribute("transform", `translate(${offsetX}, ${offsetY})`);
        routed.forEach((e) => {
            const path = document.createElementNS(SVG_NS, "path");
            path.setAttribute("class", "dagEdge");
            path.setAttribute("d", e.d);
            path.setAttribute("marker-end", "url(#arrowHead)");
            root.appendChild(path);
        });
        pos.forEach(({ x, y }, id) => {
            const g = document.createElementNS(SVG_NS, "g");
            g.setAttribute("class", "dagNode" + (selected.has(id) ? " selected" : ""));
            g.setAttribute("transform", `translate(${x},${y})`);
            g.style.cursor = "pointer";
            const label = data.nodes.find(n => n.id === id)?.label || id;
            g.setAttribute("role", "button");
            g.setAttribute("tabindex", "0");
            g.setAttribute("aria-label", label);
            g.setAttribute("aria-pressed", String(selected.has(id)));
            const rect = document.createElementNS(SVG_NS, "rect");
            rect.setAttribute("width", String(nodeW));
            rect.setAttribute("height", String(nodeH));
            rect.setAttribute("rx", "10");
            rect.setAttribute("ry", "10");
            g.appendChild(rect);
            const fo = document.createElementNS(SVG_NS, "foreignObject");
            fo.setAttribute("x", "0");
            fo.setAttribute("y", "0");
            fo.setAttribute("width", String(nodeW));
            fo.setAttribute("height", String(nodeH));
            fo.style.pointerEvents = "none";
            const outer = document.createElement("div");
            outer.setAttribute("xmlns", "http://www.w3.org/1999/xhtml");
            Object.assign(outer.style, {
                display: "flex", width: "100%", height: "100%",
                alignItems: "center", justifyContent: "center", padding: "4px 8px",
            });
            const inner = document.createElement("div");
            Object.assign(inner.style, {
                display: "-webkit-box", WebkitBoxOrient: "vertical", WebkitLineClamp: "2",
                overflow: "hidden", textAlign: "center", lineHeight: "1.2",
                fontWeight: "600", fontSize: "12px", overflowWrap: "anywhere", wordBreak: "break-word",
            });
            inner.textContent = label;
            outer.appendChild(inner);
            fo.appendChild(outer);
            g.appendChild(fo);
            const title = document.createElementNS(SVG_NS, "title");
            title.textContent = label;
            g.appendChild(title);
            function toggle() {
                if (selected.has(id))
                    selected.delete(id);
                else
                    selected.add(id);
                updateMergeUi();
                render();
            }
            g.addEventListener("click", (evt) => { if (evt.shiftKey)
                toggle();
            else
                opts.onPickRun(id); });
            g.addEventListener("keydown", (evt) => {
                if (evt.key === "Enter" || evt.key === " ") {
                    evt.preventDefault();
                    (evt.shiftKey ? toggle() : opts.onPickRun(id));
                }
            });
            root.appendChild(g);
        });
    }
    (0, dom_js_1.on)(btnClose, "click", () => api.close());
    const api = {
        render,
        open() { overlay?.classList.add("show"); render(); },
        close() { overlay?.classList.remove("show"); selected.clear(); updateMergeUi(); },
        setData(_d) { },
        getSelectedForMerge() { return Array.from(selected); },
        clearSelection() { selected.clear(); updateMergeUi(); render(); },
    };
    return api;
}
//# sourceMappingURL=dagView.js.map