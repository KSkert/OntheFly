"use strict";
/**
 * utils/math.ts
 * ------------------------------------------------------------
 * Pure math helpers for the webview. No DOM, no VS Code APIs.
 *
 * What’s here:
 *  - computeLossHistogram(): turns an array of per-sample losses into
 *    a simple bar histogram + a smoothed density curve (KDE-ish).
 *
 * Design notes:
 *  - Behavior matches the legacy dashboard.js implementation so the
 *    UI continues to render the same chart.
 *  - Handles empty/degenerate input gracefully.
 *  - Stable and deterministic; no randomness.
 *
 * Migration:
 *  - Preferred: import { computeLossHistogram } from "../utils/math";
 *  - Temporary (until bundling is wired): this module also attaches
 *    computeLossHistogram to the global scope so existing dashboard.js
 *    calls will keep working if you include the built JS before it.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.computeLossHistogram = computeLossHistogram;
/**
 * Gaussian kernel density over histogram counts (fast + smooth).
 * We convolve bin counts positioned at bin centers with a Gaussian.
 */
function smoothCountsWithGaussian(centers, counts, h, xs) {
    const inv = 1 / (Math.sqrt(2 * Math.PI) * h);
    const out = new Array(xs.length).fill(0);
    for (let j = 0; j < xs.length; j++) {
        const x = xs[j];
        let dens = 0;
        for (let i = 0; i < centers.length; i++) {
            const u = (x - centers[i]) / h;
            dens += counts[i] * inv * Math.exp(-0.5 * u * u);
        }
        out[j] = dens;
    }
    return out;
}
/**
 * Compute a bar histogram (counts) and a smoothed curve approximating
 * a kernel density estimate, using the same strategy as the original
 * dashboard.js:
 *  - fixed number of bins
 *  - bandwidth h ~ 1.06 * std * n^(-1/5), clamped to [0.6..3] * step
 */
function computeLossHistogram(values, numBins = 30) {
    if (!Array.isArray(values) || values.length === 0) {
        return { bars: [], line: [], xmin: undefined, xmax: undefined, edges: [] };
    }
    // Filter to finite numbers only (robustness)
    const data = values.filter(Number.isFinite);
    if (data.length === 0) {
        return { bars: [], line: [], xmin: undefined, xmax: undefined, edges: [] };
    }
    const min = Math.min(...data);
    const max = Math.max(...data);
    const width = (max - min) || 1e-9;
    const step = width / numBins;
    // Edges and counts
    const edges = Array.from({ length: numBins + 1 }, (_, i) => min + i * step);
    const counts = new Array(numBins).fill(0);
    for (const v of data) {
        let idx = Math.floor((v - min) / step);
        if (idx >= numBins)
            idx = numBins - 1; // right-edge inclusion
        if (idx < 0)
            idx = 0;
        counts[idx]++;
    }
    const centers = counts.map((_, i) => edges[i] + step * 0.5);
    // Bandwidth selection (Silverman-like; matches legacy behavior)
    const n = data.length;
    const mean = data.reduce((a, b) => a + b, 0) / n;
    const variance = data.reduce((a, b) => a + (b - mean) * (b - mean), 0) / Math.max(1, (n - 1));
    const std = Math.sqrt(Math.max(variance, 0));
    let h = 1.06 * (std || (step / 1.06)) * Math.pow(n, -1 / 5);
    if (!Number.isFinite(h) || h <= 1e-12)
        h = step;
    h = Math.max(0.6 * step, Math.min(3 * step, h));
    // Smooth line sampled across range
    const sampleCount = 160;
    const xs = Array.from({ length: sampleCount }, (_, i) => min + (i / (sampleCount - 1)) * (max - min));
    const ySmooth = smoothCountsWithGaussian(centers, counts, h, xs).map(y => y * step);
    return {
        bars: centers.map((x, i) => ({ x, y: counts[i] })),
        line: xs.map((x, i) => ({ x, y: ySmooth[i] })),
        xmin: min,
        xmax: max,
        edges
    };
}
/* ------------------------------------------------------------------ */
/* Back-compat global export (can be removed once everything imports) */
/* ------------------------------------------------------------------ */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
;
globalThis.computeLossHistogram = computeLossHistogram;
//# sourceMappingURL=math.js.map