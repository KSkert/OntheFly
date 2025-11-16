from __future__ import annotations
import os, json, warnings
from typing import Dict, Any, Optional, List
import torch
from torch.utils.data import DataLoader, Subset

from ..ids import _safe_component
from ..metrics_utils import _percentile_list
from ..kmeans_utils import _run_kmeans
from ..merging import (
    stochastic_weight_averaging,
    fisher_soup_merge,
    adapter_fuse_merge,
)

from ..data_explorer import (
    compute_per_sample_losses,
    compute_embeddings,
    cluster_embeddings,
    select_hard_clusters,
)


class RunManagementMixin:
    """Provides fork/merge helpers and feature construction for manual workflows."""

    # ---- simple run naming, deriving max values (e.g. max_fork=4 for next run being fork5) from session environemnt
    def _next_run_name(self, kind: str) -> str:
        """
        Allocate the next run name for a given kind.

        - "baseline" -> "baseline"
        - "fork"     -> "fork1", "fork2", "fork3", ...
        - "merge"    -> "merge1", "merge2", "merge3", ...
        """

        # Lazily seed counters from environment so the extension can
        # restore them after importing a session.
        if not hasattr(self, "_fork_counter"):
            try:
                self._fork_counter = int(os.environ.get("ONTHEFLY_FORK_COUNTER_INIT", "0"))
            except Exception:
                self._fork_counter = 0

        if not hasattr(self, "_merge_counter"):
            try:
                self._merge_counter = int(os.environ.get("ONTHEFLY_MERGE_COUNTER_INIT", "0"))
            except Exception:
                self._merge_counter = 0

        k = str(kind).lower()
        if k == "baseline":
            return "baseline"
        if k == "fork":
            self._fork_counter += 1
            return f"fork{self._fork_counter}"
        if k == "merge":
            self._merge_counter += 1
            return f"merge{self._merge_counter}"
        # Fallback for anything else
        return str(kind)

    # ---- runtime helpers -------------------------------------------------
    def _rebind_train_loader_to_subset(self, indices: Optional[List[int]]):
        if self._train_root_ds is None:
            self._active_subset_indices = None
            return
        bs = getattr(self.train_loader, "batch_size", 256)
        cf = getattr(self.train_loader, "collate_fn", None)
        shuffle = getattr(self.train_loader, "shuffle", True)
        drop_last = getattr(self.train_loader, "drop_last", False)
        if indices and len(indices) > 0:
            sub = Subset(self._train_root_ds, list(indices))
            self.train_loader = DataLoader(sub, batch_size=bs, shuffle=shuffle, drop_last=drop_last, collate_fn=cf)
            self._active_subset_indices = list(indices)
        else:
            self.train_loader = DataLoader(
                self._train_root_ds,
                batch_size=bs,
                shuffle=shuffle,
                drop_last=drop_last,
                collate_fn=cf,
            )
            self._active_subset_indices = None

    # ---- pure "merge weights" helper ------------------------------------
    def _merged_state_dict_from_checkpoints(self, paths: List[str], strategy: str = "swa"):
        models = []
        for p in paths:
            m = self._model_factory()
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            m.load_state_dict(ckpt["model"], strict=True)
            m.to(self.device).eval()
            models.append(m)

        s = (strategy or "swa").lower()

        # main three strategies
        if s == "swa":
            return stochastic_weight_averaging(models)

        if s == "fisher":
            # when you have Fisher info, pass fisher_mats here:
            # fisher_mats = ...
            # return fisher_soup_merge(models, fisher_mats=fisher_mats)
            return fisher_soup_merge(models)

        if s == "adapter":
            return adapter_fuse_merge(models)

        # optional: accept legacy aliases if they ever appear
        if s == "fisher_soup":
            return fisher_soup_merge(models)
        if s == "adapter_fuse":
            return adapter_fuse_merge(models)

        # fallback: default to SWA
        return stochastic_weight_averaging(models)

    # ---- merge with run creation + naming --------------------------------
    def _merge_from_checkpoints(
        self,
        paths: List[str],
        *,
        strategy: str = "swa",
        parents: Optional[List[str]] = None,
        new_name: Optional[str] = None,
    ) -> str:
        """
        Merge one or more checkpoints, choose a run name, and switch to the new run.

        Naming rules:
          - if new_name is provided → use that
          - otherwise → auto-allocate "merge1", "merge2", ... via _next_run_name("merge")
        """
        merged_sd = self._merged_state_dict_from_checkpoints(paths, strategy=strategy)

        # Centralized naming
        if new_name is not None:
            child_name = str(new_name)
        else:
            child_name = self._next_run_name("merge")

        self.model.load_state_dict(merged_sd, strict=False)

        new_id = self._switch_to_new_run(
            child_name,
            parents=parents or [],
            hparams={
                "merge": {
                    "strategy": strategy,
                    "parents": parents or None,
                    "paths": paths,
                }
            },
            meta={
                "kind": "merge",
                "strategy": strategy,
                "parents": parents or [],
                "paths": paths,
            },
        )

        return new_id


    def _switch_to_new_run(
        self,
        new_id: str,
        parents: List[str],
        *,
        hparams: Dict[str, Any] | None = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        hparams = hparams or {}
        if "lr_mul" in hparams:
            for pg in self.optimizer.param_groups:
                pg["lr"] *= float(hparams["lr_mul"])
        if "wd_mul" in hparams:
            for pg in self.optimizer.param_groups:
                if "weight_decay" in pg:
                    pg["weight_decay"] *= float(hparams["wd_mul"])

        prev_run = self.cfg.run_name
        prev_step = self.step

        # Human-facing name (extension sees this as run_id)
        display_name = str(new_id)

        # Filesystem id: sanitized but NO numeric suffixes anymore
        fs_id = _safe_component(hint=display_name)

        run_dir = os.path.join(self.cfg.save_dir, fs_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "run.json"), "w") as f:
            json.dump(
                {
                    "run_id": fs_id,
                    "display_name": display_name,
                    "parents": list(parents),
                    "spawned_from": prev_run,
                    "at_step": prev_step,
                    "hparams": hparams,
                    "meta": meta or {},
                },
                f,
            )

        self._event({"type": "runTransition", "from": prev_run, "to": fs_id, "prev_step": prev_step})
        self._event({"type": "finalizeRun", "run_id": prev_run, "next_run": fs_id, "last_step": prev_step})

        self._run_gen += 1
        self.cfg.run_name = fs_id
        self._event({"type": "log", "level": "info", "text": fs_id})

        self._ckpts.clear()
        self.epoch = 0
        self.step = 0

        # clear volatile state
        self._last_val_loss = None

        self._emit_new_run(
            fs_id,
            parents,
            {**(meta or {}), "display_name": display_name, "run_gen": self._run_gen},
        )
        return fs_id

    # -------------------- Fork execution (feature-aware) --------------------
    def _do_fork(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(payload.get("mode", "manual")).lower()
        allow_when_paused = bool(payload.get("allow_when_paused", mode == "manual"))
        if (self._paused or self._halt_evt.is_set()) and not allow_when_paused:
            self._event({"type": "log", "level": "info", "text": "Fork skipped: session is paused."})
            return {"new_run": None, "subset_indices": []}

        hparams = payload.get("hparams", {})
        selection = payload.get("selection")
        region = payload.get("region") or {}

        explicit_parent = payload.get("parent_run_id") or payload.get("owner_run_id")
        parent = str(explicit_parent or self.cfg.run_name)

        parent_ckpt = None
        explicit_parent_ckpt = payload.get("parent_ckpt_path")
        if explicit_parent_ckpt and os.path.exists(explicit_parent_ckpt):
            parent_ckpt = explicit_parent_ckpt
            self.step = self._load_checkpoint_into_state(parent_ckpt)
        elif parent != self.cfg.run_name:
            parent_ckpt = self._latest_ckpt_for_run(parent)
            if parent_ckpt:
                self.step = self._load_checkpoint_into_state(parent_ckpt)
            else:
                self._event(
                    {
                        "type": "log",
                        "level": "warn",
                        "text": f"No checkpoint found for parent '{parent}'. Forking with current weights.",
                    }
                )

        if payload.get("run_name"):
            # Manual override still allowed
            new_id = str(payload["run_name"])
        else:
            # Auto: fork1, fork2, fork3, ...
            new_id = self._next_run_name("fork")

        sel_indices: List[int] = []
        if (selection or region) and self._train_root_ds is not None:
            ds = self._train_root_ds
            cancel = (lambda: (not self._running)) if allow_when_paused else (
                lambda: (not self._running) or self._paused or self._halt_evt.is_set()
            )
            tr_losses, __ = compute_per_sample_losses(
                self.model,
                ds,
                getattr(self.train_loader, "collate_fn", None),
                self.raw_loss_fn,
                device=self.device,
                batch_size=getattr(self.train_loader, "batch_size", 256),
                indices=None,
                mirror_train_semantics=True,
                amp_enabled=bool(self.cfg.amp and "cuda" in self.device),
                should_stop=cancel,
            )
            if not tr_losses and (not self._running):
                self._event({"type": "log", "level": "warn", "text": "Fork cancelled during loss scan."})
                return {"new_run": None, "subset_indices": []}
            feature_cache: Dict[str, Any] = {"loss": tr_losses}

            if selection:
                kind = str(selection.get("kind"))
                if kind == "indices":
                    sel_indices = list(map(int, selection.get("ids") or []))
                elif kind == "quantile":
                    metric = str(selection.get("metric", "per_sample_loss"))
                    if metric != "per_sample_loss":
                        warnings.warn(
                            f"quantile selection metric '{metric}' is not supported; using per_sample_loss"
                        )
                    q_from = float(selection.get("from", 0.85))
                    q_to = float(selection.get("to", 1.0))
                    lo_th = _percentile_list(tr_losses, q_from)
                    hi_th = _percentile_list(tr_losses, q_to)
                    sel_indices = [
                        i
                        for i, L in enumerate(tr_losses)
                        if (L is not None and lo_th <= float(L) <= hi_th)
                    ]
                elif kind == "kmeans":
                    k = int(selection.get("k", 5))
                    targets = set(map(int, selection.get("target_clusters") or []))
                    feature_str = str(selection.get("feature", "loss"))
                    feats, _ = self._build_features_for_selection(
                        ds,
                        feature_str,
                        feature_cache,
                        batch_size=getattr(self.train_loader, "batch_size", 256),
                    )
                    if feats:
                        labels = _run_kmeans(feats, k)
                        sel_indices = [i for i, lab in enumerate(labels) if int(lab) in targets]
            if not sel_indices and region:
                lo = float(region.get("minLoss", float("-inf")))
                hi = float(region.get("maxLoss", float("inf")))
                sel_indices = [
                    i
                    for i, L in enumerate(tr_losses)
                    if (L is not None and lo <= float(L) <= hi)
                ]

        child_id = self._switch_to_new_run(
            new_id,
            parents=[parent],
            hparams=hparams,
            meta={
                "kind": "fork",
                "from": parent,
                "at_step": self.step,
                "init_from": parent_ckpt,
                "region": region,
                "subset_count": len(sel_indices),
                "mode": mode,
                "selection": (
                    selection
                    if selection
                    else (
                        {
                            "kind": "region",
                            "minLoss": region.get("minLoss"),
                            "maxLoss": region.get("maxLoss"),
                        }
                        if region
                        else None
                    )
                ),
            },
        )

        if sel_indices:
            bs = getattr(self.train_loader, "batch_size", 256)
            cf = getattr(self.train_loader, "collate_fn", None)
            shuffle = getattr(self.train_loader, "shuffle", True)
            drop_last = getattr(self.train_loader, "drop_last", False)
            sub = Subset(self._train_root_ds, list(sel_indices))
            self.train_loader = DataLoader(sub, batch_size=bs, shuffle=shuffle, drop_last=drop_last, collate_fn=cf)
            self._active_subset_indices = list(sel_indices)

        return {"new_run": child_id, "subset_indices": sel_indices}

    def _reconstruct_subset_indices_for_run(self, run_id: str) -> List[int]:
        if run_id == self.cfg.run_name and self._active_subset_indices:
            return list(self._active_subset_indices)

        run_dir = os.path.join(self.cfg.save_dir, run_id)
        meta_path = os.path.join(run_dir, "run.json")
        if not os.path.exists(meta_path):
            ds = self._train_root_ds
            if ds is None:
                raise RuntimeError("root training dataset not available; cannot reconstruct subset")
            return []
        with open(meta_path, "r") as f:
            info = json.load(f)
        meta = dict(info.get("meta") or {})
        selection = meta.get("selection")
        region = meta.get("region") or {}
        init_ckpt = meta.get("init_from")
        if selection is None and not region:
            ds_len = len(self._train_root_ds) if self._train_root_ds is not None else 0
            return list(range(ds_len))

        tmp_model = self._model_factory().to(self.device).eval()
        if init_ckpt and os.path.exists(init_ckpt):
            blob = torch.load(init_ckpt, map_location=self.device, weights_only=False)
            tmp_model.load_state_dict(blob["model"], strict=True)
        else:
            tmp_model.load_state_dict(self.model.state_dict(), strict=False)

        ds = self._train_root_ds
        if ds is None:
            raise RuntimeError("root training dataset not available; cannot reconstruct subset")
        bs = getattr(self.train_loader, "batch_size", 256)
        cf = getattr(self.train_loader, "collate_fn", None)

        losses, __ = compute_per_sample_losses(
            tmp_model,
            ds,
            cf,
            self.raw_loss_fn,
            device=self.device,
            batch_size=bs,
            indices=None,
            mirror_train_semantics=True,
            amp_enabled=bool(self.cfg.amp and "cuda" in self.device),
            should_stop=lambda: False,
        )

        if selection and str(selection.get("kind")) == "indices":
            ids = selection.get("ids") or []
            return list(map(int, ids))
        if selection and str(selection.get("kind")) == "quantile":
            q_from = float(selection.get("from", 0.85))
            q_to = float(selection.get("to", 1.0))
            lo_th = _percentile_list(losses, q_from)
            hi_th = _percentile_list(losses, q_to)
            return [
                i
                for i, L in enumerate(losses)
                if (L is not None and lo_th <= float(L) <= hi_th)
            ]
        if selection and str(selection.get("kind")) == "kmeans":
            k = int(selection.get("k", 5))
            targets = set(map(int, selection.get("target_clusters") or []))
            tmp_loader = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False, collate_fn=cf)
            embs = compute_embeddings(tmp_model, tmp_loader, self.device, hook_fn=self._embedding_hook_fn)
            cl = cluster_embeddings(embs, k=k)
            labels = cl["labels"]
            if not targets:
                import numpy as np

                labels_np = np.asarray(labels, dtype=int)
                losses_np = np.asarray(losses, dtype=float)
                targets = set(
                    select_hard_clusters(labels_np, losses_np, top_n=min(3, k))
                )
            return [i for i, lab in enumerate(labels) if int(lab) in targets]
        if region:
            lo = float(region.get("minLoss", float("-inf")))
            hi = float(region.get("maxLoss", float("inf")))
            return [
                i
                for i, L in enumerate(losses)
                if (L is not None and lo <= float(L) <= hi)
            ]
        return list(range(len(losses)))
