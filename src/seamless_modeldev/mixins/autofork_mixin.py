from __future__ import annotations
import os, json, time, math, random, warnings
from typing import Dict, Any, Optional, List, Tuple
import torch
from torch.utils.data import DataLoader, Subset

from ..ids import _safe_component, _unique_component
from ..ckpt_utils import _parse_step
from ..metrics_utils import _to_scalar_loss, _percentile_list
from ..kmeans_utils import _run_kmeans
from ..merging import weighted_average_merge, stochastic_weight_averaging, advanced_merge
from ..data_explorer import compute_per_sample_losses, compute_embeddings, cluster_embeddings, select_hard_clusters
from ..autofork import AutoForkRules, AutoForkEngine  # NOTE: uses 'autofork.py' (singular) per your tree

class AutoForkMixin:
    """
    Owns the AutoForkEngine and all fork/merge orchestration.
    """

    def set_auto_fork_rules(self, rules: AutoForkRules):
        """Replace rules + engine (history resets). Keeps current parent checkpoint if available."""
        parent_ckpt = self._ckpts[-1] if self._ckpts else None
        self._autofork_engine = AutoForkEngine(rules)
        if parent_ckpt:
            try:
                self._autofork_engine.set_parent_checkpoint(parent_ckpt)
            except Exception:
                pass

    # ---- runtime helpers ----
    def _rebind_train_loader_to_subset(self, indices: Optional[List[int]]):
        if self._train_root_ds is None:
            self._active_subset_indices = None
            return
        bs = getattr(self.train_loader, 'batch_size', 256)
        cf = getattr(self.train_loader, 'collate_fn', None)
        shuffle = getattr(self.train_loader, 'shuffle', True)
        drop_last = getattr(self.train_loader, 'drop_last', False)
        if indices and len(indices) > 0:
            sub = Subset(self._train_root_ds, list(indices))
            self.train_loader = DataLoader(sub, batch_size=bs, shuffle=shuffle, drop_last=drop_last, collate_fn=cf)
            self._active_subset_indices = list(indices)
        else:
            self.train_loader = DataLoader(self._train_root_ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last, collate_fn=cf)
            self._active_subset_indices = None

    def _merge_from_checkpoints(self, paths: List[str], strategy: str = "swa"):
        models = []
        for p in paths:
            m = self._model_factory()
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            m.load_state_dict(ckpt["model"], strict=True)
            m.to(self.device).eval()
            models.append(m)

        s = (strategy or "swa").lower()
        if s == "swa":
            return stochastic_weight_averaging(models)
        if s == "weighted":
            return weighted_average_merge(models, [1.0/len(models)]*len(models))
        if s in ("distill", "fisher_soup", "adapter_fuse"):
            try:
                return advanced_merge(models, method=s)
            except TypeError:
                return advanced_merge(models)
        if s in ("auto", "advanced"):
            return stochastic_weight_averaging(models)
        return stochastic_weight_averaging(models)

    def _switch_to_new_run(self, new_id: str, parents: List[str], *,
                           hparams: Dict[str, Any] = None,
                           meta: Optional[Dict[str, Any]] = None) -> str:
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
        self._save_ring_checkpoint()

        # Derive a short, filesystem-safe ID; keep the original text as a display name in metadata.
        display_name = str(new_id)
        fs_id_base = _safe_component(
            hint=display_name,
            extra_entropy=f"{prev_run}|{prev_step}|{time.time()}"
        )
        fs_id = _unique_component(self.cfg.save_dir, fs_id_base)

        run_dir = os.path.join(self.cfg.save_dir, fs_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "run.json"), "w") as f:
            json.dump({
                "run_id": fs_id,
                "display_name": display_name,
                "parents": list(parents),
                "spawned_from": prev_run,
                "at_step": prev_step,
                "hparams": hparams,
                "meta": meta or {},
                "region": getattr(self, "_pending_region", None),
            }, f)

        # Tell UI we're about to switch
        self._event({"type": "runTransition", "from": prev_run, "to": fs_id, "prev_step": prev_step})
        # Finalize previous run in UI
        self._event({"type": "finalizeRun", "run_id": prev_run, "next_run": fs_id, "last_step": prev_step})

        # Switch active run
        self._run_gen += 1
        self.cfg.run_name = fs_id
        self._event({"type": "log", "level": "info", "text": fs_id})

        if meta and meta.get("kind") == "fork":
            try:
                # Track child by the canonical filesystem-safe ID
                self._autofork_engine.register_child(fs_id, {"from_parent": (parents[0] if parents else None)})
                self._children_registered.add(fs_id)
                if parents:
                    self._child_parent[fs_id] = parents[0]
            except Exception:
                pass

        self._ckpts.clear()
        self.epoch = 0
        self.step = 0

        # clear volatile state
        self._pending_feature_feed  = None
        self._pending_feature_owner = None
        self._pending_feature_step  = None
        self._pending_feature_token = None
        self._clear_feature_queue()
        self._expected_token_by_owner.clear()
        self._last_val_loss = None

        # prime worker with a fresh snapshot
        try:
            token, ref = self._snapshots.push_from_model(owner=self.cfg.run_name, version=self.step, model=self.model)
            self._expected_token_by_owner[self.cfg.run_name] = token
            self._feature_worker.submit_snapshot(owner=self.cfg.run_name, version=self.step, token=token, snapshot=ref,
                                                 ckpt_path=None)
        except Exception:
            pass

        self._emit_new_run(fs_id, parents, {**(meta or {}), "display_name": display_name, "run_gen": self._run_gen})
        return fs_id

    # -------- fork/merge decision & execution ------------
    def _maybe_feed_autofork(self, batch_loss: float, grad_norm: float):
        feed = {"batch_loss": batch_loss, "grad_norm": grad_norm, "nan_flag": not math.isfinite(batch_loss)}
        if self._pending_feature_feed is not None:
            feed.update(self._pending_feature_feed)
            self._pending_feature_feed = None
            self._pending_feature_owner = None
            self._pending_feature_step = None
            self._pending_feature_token = None
        try:
            self._autofork_engine.observe_batch(**feed)
        except Exception:
            pass

        try:
            plan = self._autofork_engine.should_fork()
        except Exception:
            plan = None

        if not plan:
            return

        self._event({
            "type": "auto_fork_suggested",
            "plan": {**plan, "at_step": self.step, "init_from": (self._ckpts[-1] if self._ckpts else None)},
            "step": self.step,
            "run_id": self.cfg.run_name,
        })

        if not bool(self._af_rt.get("auto_execute", False)) or self._paused or self._halt_evt.is_set():
            return
        if self.cfg.run_name in self._children_registered:
            return

        owned_children = sum(1 for c, p in self._child_parent.items() if p == self.cfg.run_name)
        if owned_children >= int(self._autofork_engine.rules.max_parallel_children):
            return

        max_depth = int(self._af_rt.get("max_branch_depth", 1))
        if self._branch_depth(self.cfg.run_name) >= max_depth:
            return

        min_gap = int(self._af_rt.get("min_train_steps_between_autoforks", 0) or 0)
        last = self._last_autoexec_ckpt_step_by_owner.setdefault(self.cfg.run_name, self.step)
        if (self.step - last) < min_gap:
            return

        try:
            variant, vidx = self._choose_variant(plan)
            payload = {
                "selection": plan.get("selection"),
                "hparams": variant,
                "owner_run_id": self.cfg.run_name,
                "parent_ckpt_path": (self._ckpts[-1] if self._ckpts else None),
            }
            nm = str(self._af_rt.get("name_template") or "{parent}-auto@{step}")
            payload["run_name"] = self._dedupe_run_name(nm.format(parent=self.cfg.run_name, step=self.step, reason=plan.get("reason","auto")))
            res = self._do_fork(payload)
            self._last_autoexec_ckpt_step_by_owner[self.cfg.run_name] = self.step
            self._event({
                "type": "auto_fork_executed",
                "plan": {**plan, "at_step": self.step},
                "variant_index": int(vidx),
                "child_run": res.get("new_run"),
                "step": self.step,
            })
        except Exception as e:
            self._event({"type":"log","level":"error","text":f"Auto-exec failed: {e}"})

    def _choose_variant(self, plan) -> tuple[Dict[str, Any], int]:
        variants = list(plan.get("training_recipe", {}).get("variants") or [])
        if not variants:
            return {}, -1
        pol = str(self._af_rt.get("variant_policy","first"))
        if pol == "index":
            idx = int(self._af_rt.get("variant_index", 0))
            idx = max(0, min(idx, len(variants)-1))
        elif pol == "round_robin":
            idx = self._af_rr % len(variants)
            self._af_rr += 1
        elif pol == "random":
            idx = random.randrange(len(variants))
        else:
            idx = 0
        return dict(variants[idx]), idx

    def _maybe_suggest_or_do_merge(self):
        if self._paused or self._halt_evt.is_set():
            self._event({"type": "merge_gating", "reason": "paused"})
            return
        try:
            mplan = self._autofork_engine.should_merge()
        except Exception as e:
            self._event({"type": "merge_gating", "reason": "engine_error", "error": str(e)})
            return
        if not mplan:
            return

        self._event({"type": "auto_merge_suggested", "plan": mplan, "step": self.step, "run_id": self.cfg.run_name})
        if not bool(self._af_rt.get("auto_merge_execute", True)) or self._halt_evt.is_set():
            self._event({"type": "merge_gating", "reason": "auto_merge_disabled", "plan": mplan})
            return

        child_id = str(mplan.get("from_child"))
        parent_id = self._child_parent.get(child_id) or self.cfg.run_name
        child_ckpt  = self._latest_ckpt_for_run(child_id)
        parent_ckpt = self._latest_ckpt_for_run(parent_id)

        if not parent_ckpt or not child_ckpt:
            self._event({
                "type": "merge_gating",
                "reason": "awaiting_checkpoint",
                "have_parent_ckpt": bool(parent_ckpt),
                "have_child_ckpt": bool(child_ckpt),
                "parent_id": parent_id,
                "child_id": child_id,
            })
            if not child_ckpt:
                self._event({"type": "merge_gating", "reason": "saving_child_checkpoint", "child_id": child_id})
                child_ckpt = self._save_ring_checkpoint()
            if not parent_ckpt or not child_ckpt:
                return

        method = str(mplan.get("method") or "swa").lower()
        if method in ("auto", "advanced"):
            method = "swa"
        if method in ("none", "disabled", "disable"):
            self._event({"type": "merge_gating", "reason": "merge_method_disabled"})
            return
        strategy = method

        self._event({"type": "merge_gating", "reason": "merging", "parents": [parent_id, child_id], "method": method})
        merged_sd = self._merge_from_checkpoints([parent_ckpt, child_ckpt], strategy=strategy)
        self.model.load_state_dict(merged_sd, strict=False)

        new_id = self._switch_to_new_run(
            f"{parent_id}+{child_id}@merge",
            parents=[parent_id, child_id],
            hparams={"merge": {"strategy": strategy, "parents": [parent_id, child_id]}},
            meta={"kind": "merge", "strategy": strategy, "parents": [parent_id, child_id]}
        )

        try:
            self._autofork_engine.child_finished(child_id)
            self._children_registered.discard(child_id)
            self._child_parent.pop(child_id, None)
            self._autofork_engine.note_merge()
        except Exception:
            pass

        try:
            ck = self._save_ring_checkpoint()
            self._autofork_engine.set_parent_checkpoint(ck)
        except Exception:
            pass

        self._event({"type": "auto_merge_executed", "new_run": new_id, "merged": [parent_id, child_id]})

    def _spawn_feature_worker(self):
        import os
        aux_dev = os.environ.get("SEAMLESS_AUX_DEVICE", "cpu")
        import torch
        try:
            if aux_dev.startswith("cuda") and not torch.cuda.is_available():
                aux_dev = "cpu"
        except Exception:
            aux_dev = "cpu"
        gen = self._pause_gen
        self._feature_worker = self._FeatureWorkerCtor(
            model_ctor=self._model_factory,
            loss_fn=self.raw_loss_fn,
            train_dataset=self._train_root_ds,
            collate_fn=getattr(self.train_loader, 'collate_fn', None),
            batch_size=getattr(self.train_loader, 'batch_size', 256),
            aux_device=aux_dev,
            sampling_cfg=dict(self._af_cfg),
            embedding_hook=self._embedding_hook_fn,
            on_feature_pack=lambda pack, gen=gen: (
                None if (self._paused or gen != self._pause_gen or self._feature_queue.full())
                else self._feature_queue.put_nowait({**pack, "gen": gen})
            ),
            halt_fn=self._halt_evt.is_set,
        )
        self._feature_worker.start()

    def _branch_depth(self, run: str) -> int:
        d, cur = 0, run
        while cur in self._child_parent:
            cur = self._child_parent[cur]; d += 1
            if d > 32: break
        return d

    # -------------------- Fork execution (feature-aware) --------------------
    def _do_fork(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        allow_when_paused = bool(payload.get("allow_when_paused",
                                            str(payload.get("mode","")).lower() == "manual"))
        if (self._paused or self._halt_evt.is_set()) and not allow_when_paused:
            self._event({"type":"log","level":"info","text":"Fork skipped: session is paused."})
            return {"new_run": None, "subset_indices": []}

        self._pending_region = payload.get("region") or {}
        hparams = payload.get("hparams", {})

        selection = payload.get("selection")
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
                self._event({"type": "log", "level": "warn",
                            "text": f"No checkpoint found for parent '{parent}'. Forking with current weights."})

        try:
            if self._last_val_loss is not None:
                self._autofork_engine.observe_eval("val", {"loss": float(self._last_val_loss)})
        except Exception:
            pass

        if payload.get("run_name"):
            new_id = str(payload["run_name"])
        else:
            step_for_name = _parse_step(parent_ckpt) if parent_ckpt else self.step
            base_name = f"{parent}-fork@{step_for_name}"
            new_id = self._dedupe_run_name(base_name)

        sel_indices: List[int] = []
        region = self._pending_region or {}

        if (selection or region) and self._train_root_ds is not None:
            ds = self._train_root_ds
            cancel = (lambda: (not self._running)) if allow_when_paused else (lambda: (not self._running) or self._paused or self._halt_evt.is_set())
            tr_losses = compute_per_sample_losses(
                self.model, ds, getattr(self.train_loader, 'collate_fn', None), self.raw_loss_fn,
                device=self.device, batch_size=getattr(self.train_loader, 'batch_size', 256),
                indices=None, mirror_train_semantics=True,
                amp_enabled=bool(self.cfg.amp and "cuda" in self.device),
                should_stop=cancel,
            )
            if not tr_losses and (not self._running):
                self._event({"type":"log","level":"warn","text":"Fork cancelled during loss scan."})
                return {"new_run": None, "subset_indices": []}
            feature_cache: Dict[str, Any] = {"loss": tr_losses}

            if selection:
                kind = str(selection.get("kind"))
                if kind == "indices":
                    sel_indices = list(map(int, selection.get("ids") or []))
                elif kind == "quantile":
                    metric = str(selection.get("metric", "per_sample_loss"))
                    if metric != "per_sample_loss":
                        warnings.warn(f"quantile selection metric '{metric}' is not supported; using per_sample_loss")
                    q_from = float(selection.get("from", 0.85))
                    q_to   = float(selection.get("to", 1.0))
                    lo_th = _percentile_list(tr_losses, q_from)
                    hi_th = _percentile_list(tr_losses, q_to)
                    sel_indices = [i for i, L in enumerate(tr_losses) if (L is not None and lo_th <= float(L) <= hi_th)]
                elif kind == "kmeans":
                    k = int(selection.get("k", 5))
                    targets = set(map(int, selection.get("target_clusters") or []))
                    feature_str = str(selection.get("feature", "loss"))
                    feats, _ = self._build_features_for_selection(ds, feature_str, feature_cache,
                                                                  batch_size=getattr(self.train_loader, 'batch_size', 256))
                    if feats:
                        labels = _run_kmeans(feats, k)
                        sel_indices = [i for i, lab in enumerate(labels) if int(lab) in targets]
            if not sel_indices and region:
                lo = float(region.get("minLoss", float("-inf")))
                hi = float(region.get("maxLoss", float("inf")))
                sel_indices = [i for i, L in enumerate(tr_losses) if (L is not None and lo <= float(L) <= hi)]

        child_id = self._switch_to_new_run(
            new_id, parents=[parent], hparams=hparams,
            meta={
                "kind": "fork",
                "from": parent,
                "at_step": self.step,
                "init_from": parent_ckpt,
                "region": region,
                "subset_count": len(sel_indices),
                "mode": ("auto" if self._af_rt.get("auto_execute") else "manual"),
                "selection": (selection if selection else (
                    {"kind": "region", "minLoss": region.get("minLoss"), "maxLoss": region.get("maxLoss")}
                    if region else None
                )),
            }
        )

        if sel_indices:
            bs = getattr(self.train_loader, 'batch_size', 256)
            cf = getattr(self.train_loader, 'collate_fn', None)
            shuffle = getattr(self.train_loader, 'shuffle', True)
            drop_last = getattr(self.train_loader, 'drop_last', False)
            sub = Subset(self._train_root_ds, list(sel_indices))
            self.train_loader = DataLoader(sub, batch_size=bs, shuffle=shuffle, drop_last=drop_last, collate_fn=cf)
            self._active_subset_indices = list(sel_indices)

        ckpt_path = self._save_ring_checkpoint()
        try:
            self._autofork_engine.set_parent_checkpoint(ckpt_path)
        except Exception:
            pass

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

        losses = compute_per_sample_losses(
            tmp_model, ds, cf, self.raw_loss_fn,
            device=self.device, batch_size=bs,
            indices=None,
            mirror_train_semantics=True,
            amp_enabled=bool(self.cfg.amp and "cuda" in self.device),
            should_stop=lambda: False
        )

        sel_indices: List[int] = []
        if selection and str(selection.get("kind")) == "indices":
            ids = selection.get("ids") or []
            return list(map(int, ids))
        if selection and str(selection.get("kind")) == "quantile":
            q_from = float(selection.get("from", 0.85))
            q_to   = float(selection.get("to", 1.0))
            lo_th = _percentile_list(losses, q_from)
            hi_th = _percentile_list(losses, q_to)
            sel_indices = [i for i, L in enumerate(losses) if (L is not None and lo_th <= float(L) <= hi_th)]
        elif selection and str(selection.get("kind")) == "kmeans":
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
                targets = set(select_hard_clusters(labels_np, losses_np, top_n=min(3, k)))
            sel_indices = [i for i, lab in enumerate(labels) if int(lab) in targets]
        elif region:
            lo = float(region.get("minLoss", float("-inf")))
            hi = float(region.get("maxLoss", float("inf")))
            sel_indices = [i for i, L in enumerate(losses) if (L is not None and lo <= float(L) <= hi)]
        else:
            sel_indices = list(range(len(losses)))

        return list(map(int, sel_indices))
