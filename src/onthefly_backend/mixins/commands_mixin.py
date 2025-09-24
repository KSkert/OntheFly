# src/onthefly_backend/mixins/commands_mixin.py
from __future__ import annotations
import os, json, random
from typing import Any, Dict, List
import torch

from ..control import CommandRouter, serve_commands
from ..ckpt_utils import _parse_step
from ..data_explorer import export_subset_table, compute_per_sample_losses
from ..autofork import AutoForkRules


class CommandsMixin:
    """
    Registers command handlers on the router. Handlers delegate to mixin methods
    (fork/merge/export/rewind/ckpt/etc.) and avoid keeping heavy logic here.
    """
    _router: CommandRouter

    def _register_command_handlers(self):
        @self._router.on("pause")
        def _pause(_payload):
            self._paused = True
            self._halt_evt.set()
            self._auto_exec_before_pause  = bool(self._af_rt.get("auto_execute"))
            self._auto_merge_exec_before_pause = bool(self._af_rt.get("auto_merge_execute"))
            self._af_rt["auto_execute"] = False
            self._af_rt["auto_merge_execute"] = False
            self._pause_gen += 1

            path = self._save_ring_checkpoint()
            self._pause_ckpt_path = path

            self._clear_feature_queue()
            try:
                if getattr(self, "_feature_worker", None):
                    self._feature_worker.stop()
                    self._feature_worker.join(timeout=2.0)
            except Exception:
                pass

            self._event({"type": "paused", "step": self.step})
            return {"status": "paused", "step": self.step, "ckpt": path}

        @self._router.on("resume")
        def _resume(_payload):
            self._paused = False
            if self._auto_exec_before_pause is not None:
                if self._af_rt.get("auto_execute") is False:
                    self._af_rt["auto_execute"] = self._auto_exec_before_pause
                self._auto_exec_before_pause = None
            if self._auto_merge_exec_before_pause is not None:
                if self._af_rt.get("auto_merge_execute") is False:
                    self._af_rt["auto_merge_execute"] = self._auto_merge_exec_before_pause
                self._auto_merge_exec_before_pause = None
            self._halt_evt.clear()

            try:
                self._spawn_feature_worker()
                token, ref = self._snapshots.push_from_model(owner=self.cfg.run_name, version=self.step, model=self.model)
                self._expected_token_by_owner[self.cfg.run_name] = token
                self._feature_worker.submit_snapshot(
                    owner=self.cfg.run_name, version=self.step, token=token, snapshot=ref,
                    ckpt_path=(self._ckpts[-1] if self._ckpts else None)
                )
            except Exception:
                pass

            self._event({"type": "resumed", "step": self.step})
            return {"status": "resumed", "step": self.step}

        @self._router.on("stop")
        def _stop(_payload):
            self._running = False
            self._event({"type": "stopping"})
            return {"status": "stopping"}

        @self._router.on("save_ckpt")
        def _save(_payload):
            path = self._save_ring_checkpoint()
            self._event({"type": "checkpoint_saved", "path": path, "step": self.step})
            return {"path": path, "step": self.step}

        @self._router.on("load_ckpt")
        def _load(payload):
            path = payload.get("path")
            if not (path and os.path.exists(path)):
                raise RuntimeError(f"checkpoint not found: {path}")

            # Load into state (this sets self.step/epoch/_last_val_loss/scaler/etc.)
            step = self._load_checkpoint_into_state(path)

            # Optional: ensure self.step matches the ckpt (if _load… returns it)
            self.step = int(step)

            # Optional: event for logs/telemetry
            self._event({"type": "checkpoint_loaded", "path": path, "step": self.step})

            # REPLY to the RPC (this is what sendReq(...) resolves with)
            return {
                "path": path,
                "step": int(self.step),
                "epoch": int(getattr(self, "epoch", 0)),
                "last_val_loss": (float(self._last_val_loss)
                                if getattr(self, "_last_val_loss", None) is not None else None),
            }


        @self._router.on("rewind_steps")
        def _rewind(payload):
            steps = int(payload.get("steps", 0))
            path = self._find_ckpt_for_rewind(steps)
            if not path:
                raise RuntimeError("no checkpoint available to rewind")
            self.step = self._load_checkpoint_into_state(path)
            self._event({"type": "rewound", "path": path, "step": self.step})
            return {"path": path, "step": self.step}

        @self._router.on("fork")
        def _fork(payload):
            pd = dict(payload or {})
            mode = str(pd.get("mode", "manual")).lower()
            allow = bool(pd.get("allow_when_paused", (mode == "manual")))
            if (self._paused or self._halt_evt.is_set()) and not allow:
                self._event({"type":"log","level":"info","text":"Fork request ignored: session is paused."})
                return {"new_run": None, "subset_indices": []}
            pd["mode"] = mode
            pd["allow_when_paused"] = allow
            return self._do_fork(pd)

        @self._router.on("merge")
        def _merge(payload):
            parents = list(payload.get("parents") or [])
            strategy = payload.get("strategy", "swa")
            paths = list(payload.get("paths") or [])
            if not parents and not paths:
                raise RuntimeError("merge requires either 'parents' or explicit 'paths'")
            if parents and not paths:
                ckpts = []
                for run_name in parents:
                    p = self._latest_ckpt_for_run(run_name)
                    if not p:
                        raise RuntimeError(f"no checkpoint found for parent run: {run_name}")
                    ckpts.append(p)
                paths = ckpts

            merged_sd = self._merge_from_checkpoints(paths, strategy=strategy)

            child_name = (str(payload["new_name"]) if payload.get("new_name")
                        else f"{('+'.join(parents) if parents else self.cfg.run_name + '+merged')}@merge")

            self.model.load_state_dict(merged_sd, strict=False)
            new_id = self._switch_to_new_run(
                child_name,
                parents=parents,
                hparams={"merge": {"strategy": strategy, "parents": parents or None, "paths": paths}},
                meta={"kind": "merge", "strategy": strategy, "parents": parents, "paths": paths,
                    "mode": ("auto" if self._af_rt.get("auto_execute") else "manual")}
            )

            self._rebind_train_loader_to_subset(None)
            self._active_subset_indices = None

            ckpt_path = self._save_ring_checkpoint()
            try:
                self._autofork_engine.set_parent_checkpoint(ckpt_path)
            except Exception:
                pass

            for rn in (parents or []):
                if rn in self._children_registered:
                    try:
                        self._autofork_engine.child_finished(rn)
                    except Exception:
                        pass
                    self._children_registered.discard(rn)
                    self._child_parent.pop(rn, None)

            try:
                self._autofork_engine.note_merge()
            except Exception:
                pass

            return {"new_run": new_id, "parents": parents or None, "strategy": strategy, "paths": paths}

        @self._router.on("propose_subsets")
        def _subsets(_payload):
            clusters = []
            self._event({"type": "subset_proposals", "clusters": clusters})
            return {"clusters": clusters}

        @self._router.on("generate_report")
        def _gen_report(_payload):
            if not self._paused:
                raise RuntimeError("Model must be paused before generating report")

            import numpy as np
            owner = str(_payload.get("owner_run_id") or _payload.get("runId") or self.cfg.run_name)
            subset = _payload.get("subset_indices") or None
            subset_on = _payload.get("subset_on") or "val"
            req_id = _payload.get("reqId")

            if subset_on == "train" and self._train_root_ds is not None:
                ds = self._train_root_ds
                bs = getattr(self.train_loader, 'batch_size', 256)
                cf = getattr(self.train_loader, 'collate_fn', None)
                note = "train subset" if subset else "train split"
            else:
                ds = self.val_loader.dataset
                bs = getattr(self.val_loader, 'batch_size', 256)
                cf = getattr(self.val_loader, 'collate_fn', None)
                note = "validation subset" if subset else "validation split"

            cpu_state = torch.random.get_rng_state()
            np_state = np.random.get_state()
            py_state = random.getstate()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            prev_bench = torch.backends.cudnn.benchmark
            prev_det = torch.backends.cudnn.deterministic
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            random.seed(1337); np.random.seed(1337); torch.manual_seed(1337)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(1337)

            current_run = self.cfg.run_name
            owner_ckpt = None
            if owner == current_run:
                if self._pause_ckpt_path and os.path.exists(self._pause_ckpt_path):
                    owner_ckpt = self._pause_ckpt_path
                else:
                    expected = os.path.join(
                        self.cfg.save_dir, f"{self.cfg.project}__{owner}__step{self.step}.pt"
                    )
                    if os.path.exists(expected):
                        owner_ckpt = expected
            if owner_ckpt is None:
                owner_ckpt = self._latest_ckpt_for_run(owner)
            if not owner_ckpt:
                raise RuntimeError(f"No checkpoint found for requested run '{owner}'.")
            owner_step = _parse_step(owner_ckpt)

            try:
                # Build a fresh model on the report device and load the owner checkpoint
                report_model = self._model_factory().to(self.device)
                owner_blob = torch.load(owner_ckpt, map_location=self.device, weights_only=False)
                report_model.load_state_dict(owner_blob["model"], strict=True)

                # --- choose the true loss module (not the scalar wrapper) ---
                criterion = None
                if isinstance(getattr(self, "raw_loss_fn", None), torch.nn.Module):
                    criterion = self.raw_loss_fn
                elif isinstance(getattr(self, "criterion", None), torch.nn.Module):
                    criterion = self.criterion
                elif isinstance(getattr(self, "_criterion", None), torch.nn.Module):
                    criterion = self._criterion
                else:
                    # Last-resort: user supplied a plain callable; wrap it so the helper can use the fallback path.
                    fn = getattr(self, "raw_loss_fn", None) or getattr(self, "loss_fn", None)
                    if callable(fn):
                        class _CallableLoss(torch.nn.Module):
                            def __init__(self, f): super().__init__(); self.f = f
                            def forward(self, logits, target): return self.f(logits, target)
                        criterion = _CallableLoss(fn)
                    else:
                        raise RuntimeError("No usable loss found. Expected an nn.Module at self.raw_loss_fn or self.criterion.")

                # --- AMP decision (recovered behavior) ---
                cfg = getattr(self, "_af_cfg", {}) or {}
                want_amp = bool(cfg.get("amp_for_psl", True))
                dev_str  = str(getattr(self, "device", "cpu")).lower()
                amp_on   = bool(want_amp and torch.cuda.is_available() and ("cuda" in dev_str))
                # -----------------------------------------

                # Compute per-sample losses
                losses = compute_per_sample_losses(
                    model=report_model,
                    dataset=ds,
                    collate_fn=cf,
                    criterion=criterion,
                    device=self.device,
                    batch_size=bs,
                    indices=(subset or None),
                    mirror_train_semantics=bool(cfg.get("mirror_train", False)),
                    amp_enabled=amp_on,
                    should_stop=None,
                )

            finally:
                torch.random.set_rng_state(cpu_state)
                np.random.set_state(np_state)
                random.setstate(py_state)
                if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)
                torch.backends.cudnn.benchmark = prev_bench
                torch.backends.cudnn.deterministic = prev_det
                try: del report_model
                except Exception: pass
                if torch.cuda.is_available():
                    try: torch.cuda.empty_cache()
                    except Exception: pass


            return {
                "losses": losses,
                "owner_run_id": owner,
                "reqId": req_id,
                "meta": {
                    "note": note,
                    "samples": len(losses),
                    "owner_run_id": owner,
                    "subset_on": subset_on,
                    "subset_count": len(subset) if subset else 0,
                    "at_step": owner_step,
                    "at_epoch": self.epoch,
                    "owner_ckpt_path": owner_ckpt,
                }
            }

        @self._router.on("set_autofork_rules")
        def _set_af_rules(payload):
            from dataclasses import is_dataclass, replace
            r = payload.get("rules") or {}
            valid = set(AutoForkRules.__dataclass_fields__.keys())
            picked = {k: r[k] for k in r.keys() if k in valid}

            def _coerce_scalar(val, ref):
                try:
                    if isinstance(ref, bool):
                        return val if isinstance(val, bool) else str(val).lower() in ("1","true","yes","on")
                    if isinstance(ref, int):
                        return int(val)
                    if isinstance(ref, float):
                        return float(val)
                except Exception:
                    pass
                return val

            def _overlay_dataclass(curr_dc, incoming):
                if not is_dataclass(curr_dc) or not isinstance(incoming, dict):
                    return _coerce_scalar(incoming, curr_dc)
                updates = {}
                for k, v in incoming.items():
                    if k not in curr_dc.__dataclass_fields__:
                        continue
                    ref = getattr(curr_dc, k)
                    if is_dataclass(ref) and isinstance(v, dict):
                        updates[k] = _overlay_dataclass(ref, v)
                    else:
                        updates[k] = _coerce_scalar(v, ref)
                return replace(curr_dc, **updates)

            curr = self._autofork_engine.rules
            updates = {}
            for k, v in picked.items():
                ref = getattr(curr, k)
                updates[k] = _overlay_dataclass(ref, v)
            new_rules = replace(curr, **updates)

            parent_ckpt = self._ckpts[-1] if self._ckpts else None
            # engine swap is handled in AutoForkMixin/set_auto_fork_rules, but keep parity here:
            self._autofork_engine = type(self._autofork_engine)(new_rules)
            if parent_ckpt:
                try: self._autofork_engine.set_parent_checkpoint(parent_ckpt)
                except Exception: pass

            samp = payload.get("sampling") or {}
            self._af_cfg.update({k: samp[k] for k in (
                "psl_every","psl_budget","mirror_train","amp_for_psl",
                "compute_margins","compute_embeddings","embed_max_dim"
            ) if k in samp})
            try:
                self._feature_worker.update_sampling(dict(self._af_cfg))
            except Exception:
                pass

            rt = payload.get("runtime") or {}
            for k in ("auto_execute","auto_merge_execute","variant_policy","variant_index","name_template","min_train_steps_between_autoforks","max_branch_depth"):
                if k in rt: self._af_rt[k] = rt[k]

            out = {"rules": dict(new_rules.__dict__), "sampling": dict(self._af_cfg), "runtime": dict(self._af_rt)}
            self._event({"type": "autofork_rules_set", "config": out})
            return out

        @self._router.on("export_subset")
        def _export_subset(payload):
            run_id = str(payload.get("run_id") or self.cfg.run_name)
            fmt = str(payload.get("format") or "parquet").lower()
            if fmt not in ("parquet", "csv", "feather"):
                fmt = "parquet"

            ds = self._train_root_ds
            if ds is None:
                return {"ok": False, "error": "root training dataset not available", "run_id": run_id}

            export_all = False
            try:
                indices = self._reconstruct_subset_indices_for_run(run_id)
                if not indices:
                    export_all = True
                    indices = None
            except Exception as e:
                if "run.json not found" in str(e):
                    export_all = True
                    indices = None
                else:
                    self._event({"type": "log", "level": "error",
                                "text": f"export_subset({run_id}): {e}"})
                    return {"ok": False, "error": str(e), "run_id": run_id}

            run_dir = os.path.join(self.cfg.save_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            default_name = f"subset_indices.{('parquet' if fmt=='parquet' else ('feather' if fmt=='feather' else 'csv'))}"
            chosen = str(payload.get("out_path") or os.path.join(run_dir, default_name))
            out_path = os.path.abspath(chosen)

            result = export_subset_table(
                dataset=ds,
                indices=indices,
                out_path=out_path,
                fmt=fmt,
            )

            result_path = os.path.abspath(result.get("out_path") or out_path)
            rows = int(result.get("rows") or 0)
            if rows == 0 and export_all:
                try:
                    from collections.abc import Sized
                    if isinstance(ds, Sized):
                        rows = len(ds)
                except Exception:
                    pass

            self._event({
                "type": "artifact_created",
                "run_id": run_id,
                "path": result_path,
                "rows": rows
            })
            return {"ok": True, "run_id": run_id, "out_path": result_path, "rows": rows}
        
        @self._router.on("spill_all")
        def _spill_all(payload):
            """
            Spill latest (or all) snapshots to disk and return records that the
            extension will write into the SQLite 'checkpoints' table before
            building the bundle.

            Payload:
              { "dir": "<target dir>", "latest_only": true|false }

            Returns (list of dicts):
              [
                {
                  "ckpt_id": "<token>",
                  "owner":   "<run_id>",
                  "version": <int>,
                  "step":    <int>,       # == version
                  "path":    "<abs path to .pt>",
                  "bytes":   <int>
                },
                ...
              ]
            """
            if not hasattr(self, "_snapshots"):
                raise RuntimeError("snapshots manager not available on this session")

            d = str(payload.get("dir") or "").strip()
            latest_only = bool(payload.get("latest_only", True))

            if not d:
                # fall back to the configured save_dir if caller didn't pass a dir
                d = getattr(self.cfg, "save_dir", None) or os.getcwd()

            # ensure directory exists; spill() will create files under it
            os.makedirs(d, exist_ok=True)

            # do the spill
            recs = self._snapshots.spill_all(d, latest_only=latest_only)

            # IMPORTANT: we *do not* emit 'checkpoint_saved' events here to avoid
            # double-inserting rows on the extension side. The extension takes the
            # returned records and calls insertCheckpoint(...) itself before bundling.
            #
            # If you *do* want UI noise/logs, you can uncomment this loop:
            # for r in recs:
            #     try:
            #         self._event({"type": "log", "level": "info",
            #                      "text": f"Spilled {r['owner']}@{r['step']} → {r['path']}"})
            #     except Exception:
            #         pass

            return recs

        @self._router.on("prepare_export")
        def _prepare_export(payload):
            """
            One-shot: make sure the current run has something exportable, then
            spill snapshots for bundling.

            Why this exists:
              - If the user stopped at a weird step (e.g. 73) and never paused,
                they might have *no* ring ckpt and *no* snapshot in memory.
              - Export should still “just work”.

            Contract:
              - We *do not* pause. This is quick enough to do live.
              - We write a ring checkpoint for the *current* run (nice to have).
              - We push a snapshot for the *current* run so spill_all has at least one.
              - Then we call spill_all(latest_only) to materialize .pt files.

              Returns:
                {
                  "ckpt": {"path": str|None, "run": str, "step": int},
                  "snapshots": [ spill_all(...) records... ]
                }
            """
            if not hasattr(self, "_snapshots"):
                raise RuntimeError("snapshots manager not available on this session")

            d = str(payload.get("dir") or "").strip()
            latest_only = bool(payload.get("latest_only", True))
            if not d:
                d = getattr(self.cfg, "save_dir", None) or os.getcwd()
            os.makedirs(d, exist_ok=True)

            # 1) (nice-to-have) write a normal ring checkpoint for the *current* run
            ckpt_path = None
            try:
                ckpt_path = self._save_ring_checkpoint()
            except Exception:
                # don't break export if checkpointing fails for some reason
                ckpt_path = None

            # 2) make sure there's a snapshot in the ring for *this* owner@step
            try:
                token, _ = self._snapshots.push_from_model(
                    owner=self.cfg.run_name, version=self.step, model=self.model
                )
                self._expected_token_by_owner[self.cfg.run_name] = token
            except Exception:
                pass  # best-effort; spill_all will still return whatever exists

            # 3) now actually write snapshot .pt files (idempotent)
            recs = self._snapshots.spill_all(d, latest_only=latest_only)

            return {
                "ckpt": {"path": ckpt_path, "run": self.cfg.run_name, "step": int(self.step)},
                "snapshots": recs,
            }


