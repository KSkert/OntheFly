"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AF_DEFAULTS = void 0;
exports.setAutoModeUI = setAutoModeUI;
exports.prefillAutoforkUi = prefillAutoforkUi;
exports.readAutoForkConfig = readAutoForkConfig;
exports.renderAutoForkPlan = renderAutoForkPlan;
exports.wireAfTabs = wireAfTabs;
// ui/autoforkPanel.ts
const dom_js_1 = require("./utils/dom.js");
exports.AF_DEFAULTS = {
    rules: {
        enabled: true,
        loss_plateau_patience: 200,
        loss_plateau_delta: 1e-4,
        per_sample_window: 5000,
        kmeans_k: 5,
        dead_cluster_z: 1.0,
        high_loss_quantile: 0.85,
        spike_sigma: 3.0,
        ema_decay: 0.98,
        max_parallel_children: 2,
        fork_cooldown_steps: 1000,
        gate_epochs: 30,
        merge_on_plateau: true,
        reunify_every_steps: 0,
        min_child_steps_before_merge: 0,
        merge_uplift_threshold: 0.0,
        inter_fork_improvement: 0.0,
        fork_backpressure_alpha: 0.0,
        merge_method: "swa",
    },
    sampling: {
        psl_every: 200,
        psl_budget: 4000,
        mirror_train: true,
        amp_for_psl: true,
        compute_margins: true,
        compute_embeddings: false,
        embed_max_dim: 256,
    },
};
function setAutoModeUI(isAuto) {
    const btnManualFork = (0, dom_js_1.byId)("btnManualFork");
    const btnAutoForkExec = (0, dom_js_1.byId)("btnAutoForkExec");
    [btnManualFork, btnAutoForkExec].forEach((b) => {
        if (!b)
            return;
        b.style.display = isAuto ? "none" : "";
        b.disabled = !!isAuto;
    });
    const plan = (0, dom_js_1.byId)("afPlan")?.closest(".card");
    plan?.classList.toggle("readonly", isAuto);
}
function prefillAutoforkUi() {
    const g = (id) => (0, dom_js_1.byId)(id);
    (0, dom_js_1._set)(g("afEnabled"), exports.AF_DEFAULTS.rules.enabled);
    (0, dom_js_1._set)(g("afLossPlateauPatience"), exports.AF_DEFAULTS.rules.loss_plateau_patience);
    (0, dom_js_1._set)(g("afLossPlateauDelta"), exports.AF_DEFAULTS.rules.loss_plateau_delta);
    (0, dom_js_1._set)(g("afPerSampleWindow"), exports.AF_DEFAULTS.rules.per_sample_window);
    (0, dom_js_1._set)(g("afKmeansK"), exports.AF_DEFAULTS.rules.kmeans_k);
    (0, dom_js_1._set)(g("afDeadClusterZ"), exports.AF_DEFAULTS.rules.dead_cluster_z);
    (0, dom_js_1._set)(g("afHighLossQuantile"), exports.AF_DEFAULTS.rules.high_loss_quantile);
    (0, dom_js_1._set)(g("afSpikeSigma"), exports.AF_DEFAULTS.rules.spike_sigma);
    (0, dom_js_1._set)(g("afEmaDecay"), exports.AF_DEFAULTS.rules.ema_decay);
    (0, dom_js_1._set)(g("afMaxParallelChildren"), exports.AF_DEFAULTS.rules.max_parallel_children);
    (0, dom_js_1._set)(g("afForkCooldownSteps"), exports.AF_DEFAULTS.rules.fork_cooldown_steps);
    (0, dom_js_1._set)(g("afGateEpochs"), exports.AF_DEFAULTS.rules.gate_epochs);
    (0, dom_js_1._set)(g("afMergeOnPlateau"), exports.AF_DEFAULTS.rules.merge_on_plateau);
    (0, dom_js_1._set)(g("afReunifyEverySteps"), exports.AF_DEFAULTS.rules.reunify_every_steps);
    (0, dom_js_1._set)(g("afMinChildStepsBeforeMerge"), exports.AF_DEFAULTS.rules.min_child_steps_before_merge);
    (0, dom_js_1._set)(g("afMergeUpliftThreshold"), exports.AF_DEFAULTS.rules.merge_uplift_threshold);
    (0, dom_js_1._set)(g("afInterForkImprovement"), exports.AF_DEFAULTS.rules.inter_fork_improvement);
    (0, dom_js_1._set)(g("afForkBackpressureAlpha"), exports.AF_DEFAULTS.rules.fork_backpressure_alpha);
    (0, dom_js_1._set)(g("afMergeMethodSelector"), exports.AF_DEFAULTS.rules.merge_method);
    (0, dom_js_1._set)(g("afPslEvery"), exports.AF_DEFAULTS.sampling.psl_every);
    (0, dom_js_1._set)(g("afPslBudget"), exports.AF_DEFAULTS.sampling.psl_budget);
    (0, dom_js_1._set)(g("afMirrorTrain"), exports.AF_DEFAULTS.sampling.mirror_train);
    (0, dom_js_1._set)(g("afAmpForPsl"), exports.AF_DEFAULTS.sampling.amp_for_psl);
    (0, dom_js_1._set)(g("afComputeMargins"), exports.AF_DEFAULTS.sampling.compute_margins);
    (0, dom_js_1._set)(g("afComputeEmbeddings"), exports.AF_DEFAULTS.sampling.compute_embeddings);
    (0, dom_js_1._set)(g("afEmbedMaxDim"), exports.AF_DEFAULTS.sampling.embed_max_dim);
}
function readAutoForkConfig() {
    const g = (id) => (0, dom_js_1.byId)(id);
    return {
        rules: {
            enabled: (0, dom_js_1._bool)(g("afEnabled")),
            loss_plateau_patience: (0, dom_js_1._num)(g("afLossPlateauPatience")),
            loss_plateau_delta: (0, dom_js_1._num)(g("afLossPlateauDelta")),
            per_sample_window: (0, dom_js_1._num)(g("afPerSampleWindow")),
            kmeans_k: (0, dom_js_1._num)(g("afKmeansK")),
            dead_cluster_z: (0, dom_js_1._num)(g("afDeadClusterZ")),
            high_loss_quantile: (0, dom_js_1._num)(g("afHighLossQuantile")),
            spike_sigma: (0, dom_js_1._num)(g("afSpikeSigma")),
            ema_decay: (0, dom_js_1._num)(g("afEmaDecay")),
            max_parallel_children: (0, dom_js_1._num)(g("afMaxParallelChildren")),
            fork_cooldown_steps: (0, dom_js_1._num)(g("afForkCooldownSteps")),
            gate_epochs: (0, dom_js_1._num)(g("afGateEpochs")),
            merge_on_plateau: (0, dom_js_1._bool)(g("afMergeOnPlateau")),
            reunify_every_steps: (0, dom_js_1._num)(g("afReunifyEverySteps")),
            min_child_steps_before_merge: (0, dom_js_1._num)(g("afMinChildStepsBeforeMerge")),
            merge_uplift_threshold: (0, dom_js_1._num)(g("afMergeUpliftThreshold")),
            inter_fork_improvement: (0, dom_js_1._num)(g("afInterForkImprovement")),
            fork_backpressure_alpha: (0, dom_js_1._num)(g("afForkBackpressureAlpha")),
            merge: { method: (g("afMergeMethodSelector")?.value || "swa") },
        },
        sampling: {
            psl_every: (0, dom_js_1._num)(g("afPslEvery")),
            psl_budget: (0, dom_js_1._num)(g("afPslBudget")),
            mirror_train: (0, dom_js_1._bool)(g("afMirrorTrain")),
            amp_for_psl: (0, dom_js_1._bool)(g("afAmpForPsl")),
            compute_margins: (0, dom_js_1._bool)(g("afComputeMargins")),
            compute_embeddings: (0, dom_js_1._bool)(g("afComputeEmbeddings")),
            embed_max_dim: (0, dom_js_1._num)(g("afEmbedMaxDim")),
        },
    };
}
function renderAutoForkPlan(plan) {
    const box = (0, dom_js_1.byId)("afPlan");
    if (!box)
        return;
    if (!plan) {
        box.textContent = "(no plan yet)";
        return;
    }
    const v = Array.isArray(plan?.training_recipe?.variants) ? plan.training_recipe.variants : [];
    const summary = {
        reason: plan.reason,
        priority: plan.priority,
        selection: plan.selection,
        variants: v.map((x, i) => ({ i, ...x })),
        cooldown_steps: plan.cooldown_steps,
        analyzed_at_step: plan.at_step ?? plan.step ?? null,
    };
    box.textContent = JSON.stringify(summary, null, 2);
}
function wireAfTabs() {
    const LS_KEY = "fs.autofork.tab.v1";
    const btns = document.querySelectorAll("#afTabButtons .afTabBtn");
    const panels = document.querySelectorAll(".afTabPanel");
    const setActive = (tab) => {
        btns.forEach(b => b.setAttribute("aria-selected", b.dataset.tab === tab ? "true" : "false"));
        panels.forEach(p => p.toggleAttribute("hidden", p.getAttribute("data-tab") !== tab));
    };
    let initial = localStorage.getItem(LS_KEY) || "forking";
    const known = new Set(Array.from(btns).map(b => b.dataset.tab));
    if (!known.has(initial))
        initial = "forking";
    setActive(initial);
    btns.forEach(b => b.addEventListener("click", () => { const t = b.dataset.tab; setActive(t); localStorage.setItem(LS_KEY, t); }));
}
//# sourceMappingURL=autoforkPanel.js.map