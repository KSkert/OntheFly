# src/onthefly_backend/mixins/commands_mixin.py
from __future__ import annotations
import os, random, time
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
import contextlib

import torch

from ..control import CommandRouter
from ..ckpt_utils import _parse_step
from ..data_explorer import export_subset_table, compute_per_sample_losses


def _device_type_from(device: Union[str, torch.device]) -> str:
    dt = str(getattr(device, "type", str(device))).lower()
    if "cuda" in dt:
        return "cuda"
    if "mps" in dt:
        return "mps"
    return "cpu"


@contextlib.contextmanager
def _inference_autocast(device, enabled: Optional[bool] = None):
    dev_type = _device_type_from(device)
    use_amp = enabled if enabled is not None else dev_type in {"cuda", "mps"}
    cm = contextlib.nullcontext()
    try:
        if hasattr(torch, "autocast"):
            cm = torch.autocast(device_type=dev_type, enabled=use_amp)
        elif dev_type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
            cm = torch.cuda.amp.autocast(enabled=use_amp)
    except Exception:
        cm = contextlib.nullcontext()
    with cm:
        yield


def _ddp_info() -> Dict[str, Any]:
    out = {"is_distributed": False}
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            out.update({
                "is_distributed": True,
                "backend": str(dist.get_backend()),
                "world_size": int(dist.get_world_size()),
                "rank": int(dist.get_rank()),
            })
    except Exception as e:
        out["error"] = str(e)
    return out


def _looks_like_fsdp(model: torch.nn.Module) -> bool:
    try:
        cn = type(model).__name__.lower()
        if "fullysharded" in cn or "fsdp" in cn:
            return True
        # Some projects wrap the root; check children
        for m in model.modules():
            cn2 = type(m).__name__.lower()
            if "fullysharded" in cn2 or "fsdp" in cn2:
                return True
    except Exception:
        pass
    return False


def _tensor_hash64(t: torch.Tensor) -> int:
    # Cheap content hash (64-bit) on CPU without syncing the whole tensor to Python
    if t.numel() == 0:
        return 0
    with torch.no_grad():
        x = t.detach().reshape(-1)
        if x.is_cuda:
            x = x.cpu()
        # sample 1024 elements evenly to keep it cheap
        n = x.numel()
        if n > 1024:
            idx = torch.linspace(0, n - 1, 1024, dtype=torch.long)
            x = x.index_select(0, idx)
        # reinterpret as bytes and compute a simple rolling hash
        b = x.numpy().tobytes()
        h = 1469598103934665603  # FNV-1a 64-bit offset basis
        for by in b[::4096]:  # stride to keep it light
            h ^= by
            h *= 1099511628211
            h &= (1 << 64) - 1
        return int(h)


def _metadata_hash(mod: torch.nn.Module) -> int:
    # FNV-1a 64 over stable, text-encoded records
    acc = 1469598103934665603
    def _mix(rec: str):
        nonlocal acc
        b = rec.encode("utf-8")
        for by in b:
            acc ^= by
            acc = (acc * 1099511628211) & ((1 << 64) - 1)

    for name, p in mod.named_parameters(recurse=True):
        _mix(f"P|{name}|{tuple(p.shape)}|{p.dtype}")
    for name, b in mod.named_buffers(recurse=True):
        _mix(f"B|{name}|{tuple(b.shape)}|{b.dtype}")
    return int(acc)



def _extract_xy(batch: Any) -> Tuple[Any, Any]:
    # Flexible (x, y) extraction with common patterns
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        for kx in ("input", "inputs", "x", "features"):
            if kx in batch:
                x = batch[kx]
                break
        else:
            # first tensor-ish value
            x = next((v for v in batch.values() if torch.is_tensor(v)), None)
        for ky in ("target", "targets", "label", "labels", "y"):
            if ky in batch:
                y = batch[ky]
                break
        else:
            y = None
        return x, y
    # last resort
    return batch, None

def _health_header_for(self, title: str) -> list[str]:
    """Standard header lines for all health reports."""
    return [
        f"Run: {self.cfg.run_name}",
        f"Step: {int(self.step)} | Epoch: {int(getattr(self, 'epoch', 0))}",
        f"{title}:",
    ]

def _yn(v: Optional[bool]) -> str:
    return "Yes" if v else "No"

def _pct(x: Optional[float]) -> str:
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "n/a"

def _jsonable_list(x):
    """Convert torch/np arrays to plain Python lists for JSON transport."""
    try:
        import numpy as _np
    except Exception:
        _np = None

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).tolist()
    if _np is not None and isinstance(x, _np.ndarray):
        return x.reshape(-1).tolist()
    return x  # already list-like or scalar


class CommandsMixin:
    """
    Registers command handlers on the router. Handlers delegate to mixin methods
    (pause/resume/export/ckpt/etc.) and expose new v2 *health* commands.
    """
    _router: CommandRouter

    # ------------------------------- Core runtime -------------------------------
    def _register_command_handlers(self):  # noqa: C901 (long but grouped by sections)

        # ----------------------------- lifecycle --------------------------------
        @self._router.on("pause")
        def _pause(_payload):
            self._paused = True
            # self._halt_evt.set()
            self._pause_gen += 1

            path = self._save_ring_checkpoint()
            self._pause_ckpt_path = path

            self._event({"type": "paused", "step": self.step})
            return {"status": "paused", "step": self.step, "ckpt": path}
        
        @self._router.on("resume")
        def _resume(_payload):
            self._paused = False

            self._event({"type": "resumed", "step": self.step})
            return {"status": "resumed", "step": self.step}
        
        @self._router.on("clean_disk")
        def _clean_disk(payload):
            root = getattr(self.cfg, "save_dir", None)
            if not root:
                return {"ok": False, "error": "cfg.save_dir not set"}

            root = os.path.abspath(root)
            removed: list[str] = []
            failed: list[dict[str, str]] = []

            try:
                if not os.path.isdir(root):
                    self._event({
                        "type": "log",
                        "level": "info",
                        "text": f"clean_disk: root does not exist or is not a directory: {root}",
                    })
                    return {"ok": True, "root": root, "removed": [], "failed": []}

                for name in os.listdir(root):
                    path = os.path.join(root, name)

                    # 1) Plain files (old behavior)
                    if os.path.isfile(path):
                        try:
                            os.remove(path)
                            removed.append(name)
                        except Exception as e:
                            failed.append({"path": path, "error": str(e)})
                        continue

                    # 2) Run directories: directory containing run.json
                    if os.path.isdir(path):
                        meta_path = os.path.join(path, "run.json")
                        if os.path.exists(meta_path):
                            try:
                                shutil.rmtree(path)
                                removed.append(name + "/")
                            except Exception as e:
                                failed.append({"path": path, "error": str(e)})

                self._event({
                    "type": "log",
                    "level": "info",
                    "text": f"clean_disk: removed={len(removed)}, failed={len(failed)}, root={root}",
                })

                ok = len(failed) == 0
                return {"ok": ok, "root": root, "removed": removed, "failed": failed}

            except Exception as e:
                self._event({
                    "type": "log",
                    "level": "error",
                    "text": f"clean_disk error: {e}",
                })
                return {"ok": False, "error": str(e), "root": root}
        
        @self._router.on("stop")
        def _stop(_payload):
            self._running = False
            self._event({"type": "stopping"})
            return {"status": "stopping"}

        # ----------------------------- checkpoints ------------------------------
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


        @self._router.on("test_now")
        def _test_now(payload):
            label = str(payload.get("label", "final") or "final").strip() or "final"
            # Delegate to the shared helper on TrainMixin
            return self._run_labeled_test_and_ckpt(label=label, source="manual")
        
        # ----------------------------- simple fork/merge ------------------------
        @self._router.on("fork")
        def _fork(payload):
            # Enforce the contract: fork only when paused
            if not self._paused:
                self._event({"type":"log","level":"warn","text":"Fork requires paused session."})
                return {"new_run": None, "subset_indices": []}
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

            new_name = str(payload["new_name"]) if payload.get("new_name") else None

            new_id = self._merge_from_checkpoints(
                paths,
                strategy=strategy,
                parents=parents,
                new_name=new_name,
            )

            self._rebind_train_loader_to_subset(None)
            self._active_subset_indices = None
            self._save_ring_checkpoint()

            return {"new_run": new_id, "parents": parents or None, "strategy": strategy, "paths": paths}

        # ----------------------------- reports/exports --------------------------
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
                if self.val_loader is None:
                    raise RuntimeError("val_loader is not configured; cannot generate validation reports.")
                ds = self.val_loader.dataset
                bs = getattr(self.val_loader, 'batch_size', 256)
                cf = getattr(self.val_loader, 'collate_fn', None)
                note = "validation subset" if subset else "validation split"

            # Save + set deterministic RNG
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
                if getattr(self, "_pause_ckpt_path", None) and os.path.exists(self._pause_ckpt_path):
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

                # Choose a loss module
                criterion = None
                if isinstance(getattr(self, "raw_loss_fn", None), torch.nn.Module):
                    criterion = self.raw_loss_fn
                elif isinstance(getattr(self, "criterion", None), torch.nn.Module):
                    criterion = self.criterion
                elif isinstance(getattr(self, "_criterion", None), torch.nn.Module):
                    criterion = self._criterion
                else:
                    fn = getattr(self, "raw_loss_fn", None) or getattr(self, "loss_fn", None)
                    if callable(fn):
                        class _CallableLoss(torch.nn.Module):
                            def __init__(self, f): super().__init__(); self.f = f
                            def forward(self, logits, target): return self.f(logits, target)
                        criterion = _CallableLoss(fn)
                    else:
                        raise RuntimeError("No usable loss found. Expected an nn.Module at self.raw_loss_fn or self.criterion.")

                with torch.inference_mode():
                    # Compute per-sample losses on the selected split
                    losses, sample_indices = compute_per_sample_losses(
                        model=report_model,
                        dataset=ds,
                        collate_fn=cf,
                        criterion=criterion,
                        device=self.device,
                        batch_size=bs,
                        indices=(subset or None),
                        mirror_train_semantics=False,
                        amp_enabled=_device_type_from(self.device) in {"cuda", "mps"},
                        should_stop=None,
                    )
                losses = _jsonable_list(losses)
                sample_indices = _jsonable_list(sample_indices)

            finally:
                # Restore RNG + cudnn knobs
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
                "sample_indices": sample_indices,
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

        @self._router.on("export_subset")
        def _export_subset(payload):
            run_id = str(payload.get("run_id") or self.cfg.run_name)
            fmt = str(payload.get("format") or "parquet").lower()
            if fmt not in ("parquet", "csv", "feather"):
                fmt = "parquet"

            ds = self._train_root_ds
            if ds is None:
                return {"ok": False, "error": "root training dataset not available", "run_id": run_id}

            # --- helpers -------------------------------------------------------------
            def _len_ds():
                try:
                    from collections.abc import Sized
                    return len(ds) if isinstance(ds, Sized) else None
                except Exception:
                    return None

            def _normalize_indices(obj, nmax=None):
                """
                Accepts list/tuple/set/numpy array/torch tensor -> returns sorted unique ints >=0 and < nmax (if given).
                """
                if obj is None:
                    return None
                try:
                    # handle numpy/torch
                    try:
                        import numpy as _np
                        if hasattr(obj, "tolist"):
                            obj = obj.tolist()
                        elif isinstance(obj, _np.ndarray):
                            obj = obj.astype("int64").tolist()
                    except Exception:
                        pass
                    try:
                        import torch as _torch
                        if isinstance(obj, _torch.Tensor):
                            obj = obj.detach().cpu().long().tolist()
                    except Exception:
                        pass
                    # now obj should be an iterable
                    ints = []
                    for x in obj:
                        if x is None:
                            continue
                        try:
                            v = int(x)
                        except Exception:
                            continue
                        if v < 0:
                            continue
                        if nmax is not None and v >= nmax:
                            continue
                        ints.append(v)
                    if not ints:
                        return []
                    # unique + sorted
                    ints = sorted(set(ints))
                    return ints
                except Exception:
                    return []
            # ------------------------------------------------------------------------

            ds_len = _len_ds()

            payload_indices = payload.get("subset_indices", None)
            have_payload_indices = payload_indices is not None  # distinguish “absent” vs “present but empty”

            indices = _normalize_indices(payload_indices, nmax=ds_len) if have_payload_indices else None

            # If caller explicitly provided indices but they normalize to empty -> don't silently export all.
            if have_payload_indices and (indices is None or len(indices) == 0):
                return {
                    "ok": False,
                    "error": "No valid subset_indices provided (after normalization). Aborting export to avoid exporting all rows.",
                    "run_id": run_id,
                    "received_indices": 0,
                    "format": fmt,
                }

            export_all = False

            # No indices provided in payload -> try reconstruct; else export-all fallback.
            if indices is None:
                try:
                    indices = self._reconstruct_subset_indices_for_run(run_id)
                    indices = _normalize_indices(indices, nmax=ds_len)
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

            if indices is None:
                # Final fallback: explicit export-all only if dataset is sized
                if ds_len is not None:
                    indices = list(range(ds_len))
                    export_all = True
                else:
                    return {"ok": False, "error": "Dataset has no known length; cannot export-all safely.", "run_id": run_id}

            run_dir = os.path.join(self.cfg.save_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            default_name = f"subset_indices.{('parquet' if fmt=='parquet' else ('feather' if fmt=='feather' else 'csv'))}"
            chosen = str(payload.get("out_path") or os.path.join(run_dir, default_name))
            out_path = os.path.abspath(chosen)

            # small trace log
            self._event({"type": "log", "level": "info",
                        "text": f"export_subset run={run_id} recv_payload={'yes' if have_payload_indices else 'no'} "
                                f"final_len={len(indices)} export_all={export_all} fmt={fmt} out={out_path}"})

            result = export_subset_table(
                dataset=ds,
                indices=indices,
                out_path=out_path,
                fmt=fmt,
            )

            result_path = os.path.abspath(result.get("out_path") or out_path)
            rows = int(result.get("rows") or 0)
            if rows == 0 and export_all and ds_len is not None:
                rows = ds_len

            self._event({"type": "artifact_created", "run_id": run_id, "path": result_path, "rows": rows})

            return {
                "ok": True,
                "run_id": run_id,
                "out_path": result_path,
                "rows": rows,
                "format": fmt,
                "received_indices": len(indices),
                "export_all": bool(export_all),
            }

        # ----------------------------- spill/export helpers ---------------------
        @self._router.on("spill_all")
        def _spill_all(payload):
            d = str(payload.get("dir") or "").strip()
            if not d:
                d = getattr(self.cfg, "save_dir", None) or os.getcwd()
            os.makedirs(d, exist_ok=True)
            return []

        @self._router.on("prepare_export")
        def _prepare_export(payload):
            d = str(payload.get("dir") or "").strip()
            if not d:
                d = getattr(self.cfg, "save_dir", None) or os.getcwd()
            os.makedirs(d, exist_ok=True)

            ckpt_path = None
            try:
                ckpt_path = self._save_ring_checkpoint()
            except Exception:
                ckpt_path = None

            return {
                "ckpt": {"path": ckpt_path, "run": self.cfg.run_name, "step": int(self.step)},
                "snapshots": [],
            }

        # ============================= v2 HEALTH COMMANDS =======================
        # Each command is intentionally separate so the UI can call them piecemeal.

        @self._router.on("dist_health")
        def _dist_health(payload):
            if not self._paused:
                raise RuntimeError("Model must be paused before running dist_health")

            require_all = bool(payload.get("require_all_ranks", False))
            sample_weights = int(payload.get("sample_weights", 0))

            info = _ddp_info()
            info["fsdp"] = bool(_looks_like_fsdp(self.model))

            all_ok = True
            notes: List[str] = []

            if not info.get("is_distributed"):
                text_lines = _health_header_for(self, "Distribution Health") + [
                    "  • torch.distributed not initialized; skipping parity checks",
                    f"  • backend: {info.get('backend')}",
                    "  • world_size: 1",
                    "  • rank: 0",
                    f"  • FSDP: {_yn(info['fsdp'])}",
                ]
                info.update({
                    "distributed": False,
                    "backend": None,
                    "world_size": 1,
                    "rank": 0,
                    "ok": True,
                    "notes": ["torch.distributed not initialized; skipping parity checks"],
                    "text": "\n".join(text_lines),
                })
                return info

            # metadata parity
            try:
                import torch.distributed as dist
                meta_hash = _metadata_hash(self.model)
                gathered: List[int] = [0 for _ in range(int(dist.get_world_size()))]
                dist.all_gather_object(gathered, int(meta_hash))
                same_meta = len(set(gathered)) == 1
                if not same_meta:
                    all_ok = False
                    notes.append("State metadata (names/shapes/dtypes) differ across ranks")
            except Exception as e:
                all_ok = False
                notes.append(f"metadata parity check failed: {e}")

            # optional weight-content parity (DDP only)
            if not info["fsdp"] and sample_weights > 0:
                try:
                    import torch.distributed as dist
                    params = [(n, p) for (n, p) in self.model.named_parameters() if p.requires_grad]
                    params = sorted(params, key=lambda kv: hash(kv[0]))
                    take = max(1, min(sample_weights, len(params)))
                    picks = [params[i * (len(params) // take)] for i in range(take)]
                    local_hash = 1469598103934665603
                    for _n, p in picks:
                        local_hash ^= _tensor_hash64(p.detach().float())
                        local_hash *= 1099511628211
                        local_hash &= (1 << 64) - 1
                    gathered: List[int] = [0 for _ in range(int(dist.get_world_size()))]
                    dist.all_gather_object(gathered, int(local_hash))
                    if len(set(gathered)) != 1:
                        all_ok = False
                        notes.append("Parameter content parity mismatch across ranks (DDP)")
                except Exception as e:
                    notes.append(f"content parity skipped/failed: {e}")

            # arrival protocol
            if require_all:
                try:
                    import torch.distributed as dist
                    token = random.randint(0, 2**31 - 1)
                    gathered: List[int] = [0 for _ in range(int(dist.get_world_size()))]
                    dist.all_gather_object(gathered, int(token))
                    arrived = len(gathered) == int(dist.get_world_size())
                    notes.append("all ranks arrived" if arrived else "not all ranks arrived")
                except Exception as e:
                    notes.append(f"arrival check failed: {e}")

            info.update({"ok": all_ok, "notes": notes})

            lines = _health_header_for(self, "Distribution Health") + [
                f"  • backend: {info.get('backend')}",
                f"  • world_size: {info.get('world_size')}",
                f"  • rank: {info.get('rank')}",
                f"  • FSDP: {_yn(info['fsdp'])}",
                f"  • status: {'OK' if all_ok else 'ISSUES'}",
            ]
            if notes:
                lines += ["Notes:"] + [f"  • {n}" for n in notes]
            info["text"] = "\n".join(lines)
            return info

        @self._router.on("activations_health")
        def _activations_health(payload):
            if not self._paused:
                raise RuntimeError("Model must be paused before running activations_health")

            budget_steps = max(1, int(payload.get("budget_steps", 2)))
            topk = max(1, int(payload.get("topk", 10)))
            eps = float(payload.get("eps", 1e-3))

            stats: Dict[str, Dict[str, float]] = {}
            hooks = []

            def _register(module: torch.nn.Module, name: str):
                def _hook(_mod, _inp, out):
                    try:
                        with torch.no_grad():
                            if not torch.is_tensor(out):
                                return
                            x = out.detach()
                            if not x.is_floating_point():
                                return
                            x = x.float()
                            if x.numel() == 0:
                                return
                            zero_frac = float((x == 0).float().mean().item())
                            mean = float(x.mean().item())
                            std = float(x.std(unbiased=False).item())
                            sat_frac = 0.0
                            cls = type(_mod).__name__.lower()
                            if "sigmoid" in cls:
                                sat_frac = float(((x < eps) | (x > 1 - eps)).float().mean().item())
                            elif "tanh" in cls:
                                sat_frac = float(((x < -1 + eps) | (x > 1 - eps)).float().mean().item())
                            dead_frac = zero_frac if "relu" in cls else 0.0
                            s = stats.setdefault(name, {"count": 0, "zero_frac": 0.0, "sat_frac": 0.0, "dead_frac": 0.0, "mean": 0.0, "std": 0.0})
                            c = s["count"] + 1; s["count"] = c
                            for k, v in (("zero_frac", zero_frac), ("sat_frac", sat_frac), ("dead_frac", dead_frac), ("mean", mean), ("std", std)):
                                if k in ("zero_frac", "sat_frac", "dead_frac"):
                                    s[k] = max(s[k], float(v))
                                else:
                                    s[k] = (s[k] * (c - 1) + float(v)) / c
                    except Exception:
                        pass
                hooks.append(module.register_forward_hook(_hook))

            for name, m in self.model.named_modules():
                nm = type(m).__name__.lower()
                if any(x in nm for x in ("relu", "gelu", "silu", "sigmoid", "tanh")):
                    _register(m, name)

            self.model.eval()
            dev = self.device
            if self.val_loader is None:
                raise RuntimeError("val_loader is not configured; cannot run activations_health.")
            it = iter(self.val_loader)
            steps = 0
            try:
                with torch.inference_mode(), _inference_autocast(dev):
                    while steps < budget_steps:
                        try:
                            batch = next(it)
                        except StopIteration:
                            break
                        x, _y = _extract_xy(batch)
                        if torch.is_tensor(x): x = x.to(dev, non_blocking=True)
                        elif isinstance(x, (list, tuple)): x = [t.to(dev, non_blocking=True) if torch.is_tensor(t) else t for t in x]
                        elif isinstance(x, dict): x = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in x.items()}
                        _ = self.model(x)
                        steps += 1
            finally:
                for h in hooks:
                    with contextlib.suppress(Exception):
                        h.remove()

            scored = []
            for name, s in stats.items():
                score = max(s.get("dead_frac", 0.0), s.get("sat_frac", 0.0), s.get("zero_frac", 0.0))
                s["score"] = float(score)
                scored.append((score, name, s))
            scored.sort(reverse=True, key=lambda t: t[0])
            top = [{"layer": n, **d} for (_sc, n, d) in scored[:topk]]

            out = {
                "budget_steps": steps,
                "top": top,
                "total_layers": len(stats),
                "notes": ["stats are max-over-batches for dead/saturation, mean-over-batches for mean/std"],
            }

            lines = _health_header_for(self, "Activation Health") + [
                f"  • batches probed: {steps}",
                f"  • layers scanned: {len(stats)}",
                f"  • reporting top-{len(top)} by max(dead/sat/zero)",
            ]
            if top:
                lines += ["Top offenders:"]
                for i, t in enumerate(top, 1):
                    lines.append(
                        f"  {i:>2}. {t['layer']} — score {_pct(t.get('score'))}, dead {_pct(t.get('dead_frac'))}, "
                        f"sat {_pct(t.get('sat_frac'))}, zero {_pct(t.get('zero_frac'))}, mean {t.get('mean', 0.0):.4f}, std {t.get('std', 0.0):.4f}"
                    )
            if out["notes"]:
                lines += ["Notes:"] + [f"  • {n}" for n in out["notes"]]
            out["text"] = "\n".join(lines)
            return out

        @self._router.on("numerics_health")
        def _numerics_health(payload):
            if not self._paused:
                raise RuntimeError("Model must be paused before running numerics_health")

            sample_layers = max(1, int(payload.get("sample_layers", 25)))

            def _sample(named_iter):
                items = list(named_iter)
                if not items:
                    return []
                items = sorted(items, key=lambda kv: hash(kv[0]))
                take = min(sample_layers, len(items))
                return [items[i * (len(items) // take)] for i in range(take)]

            probs: List[Dict[str, Any]] = []
            with torch.no_grad():
                for kind, iterator in (("param", self.model.named_parameters(recurse=True)),
                                    ("buffer", self.model.named_buffers(recurse=True))):
                    for name, t in _sample(iterator):
                        try:
                            x = t.detach()
                            if x.numel() == 0:
                                continue
                            flags = {
                                "has_nan": bool(torch.isnan(x).any().item()),
                                "has_inf": bool(torch.isinf(x).any().item()),
                            }
                            if flags["has_nan"] or flags["has_inf"]:
                                probs.append({"kind": kind, "name": name, **flags})
                        except Exception as e:
                            probs.append({"kind": kind, "name": name, "error": str(e)})

            amp_overflow = None
            scaler = getattr(self, "scaler", None) or getattr(self, "grad_scaler", None)
            try:
                if scaler is not None and hasattr(scaler, "_found_inf_per_device"):
                    s = 0.0
                    for _dev, tens in getattr(scaler, "_found_inf_per_device").items():
                        try:
                            s += float(tens.item())
                        except Exception:
                            pass
                    amp_overflow = s
            except Exception:
                pass

            out = {
                "problems": probs,
                "amp_overflow_sum": amp_overflow,
                "sample_layers": sample_layers,
                "note": "Budgeted sweep; set sample_layers high to force full scan (can be slow).",
            }

            lines = _health_header_for(self, "Numerics Health") + [
                f"  • layers sampled: {sample_layers}",
                f"  • AMP overflow sum: {('n/a' if amp_overflow is None else amp_overflow)}",
            ]
            if probs:
                lines += [f"  • Issues found: {len(probs)}", "Details:"]
                for i, p in enumerate(probs[:40], 1):
                    if "error" in p:
                        lines.append(f"    {i:>2}. [{p['kind']}] {p['name']} — error: {p['error']}")
                    else:
                        flags = []
                        if p.get("has_nan"):
                            flags.append("NaN")
                        if p.get("has_inf"):
                            flags.append("Inf")
                        lines.append(f"    {i:>2}. [{p['kind']}] {p['name']} — {', '.join(flags)}")
                if len(probs) > 40:
                    lines.append(f"    … and {len(probs) - 40} more")
            else:
                lines.append("  • No NaN/Inf detected in sampled params/buffers.")
            lines += ["Notes:", f"  • {out['note']}"]
            out["text"] = "\n".join(lines)
            return out

        @self._router.on("determinism_health")
        def _determinism_health(payload):
            """
            Report RNG capture and whether pause/resume determinism is guaranteed for *this* pause.
            """
            if not self._paused:
                raise RuntimeError("Model must be paused before running determinism_health")

            import numpy as np
            info: Dict[str, Any] = {"ok": True, "notes": []}

            # RNG capture probes
            try:
                cpu_state = torch.random.get_rng_state()
                info["torch_rng_len"] = int(cpu_state.numel())
            except Exception as e:
                info["ok"] = False; info["notes"].append(f"torch RNG capture failed: {e}")
            try:
                info["numpy_state_len"] = len(np.random.get_state()[1])
            except Exception as e:
                info["notes"].append(f"numpy RNG capture issue: {e}")
            try:
                _ = random.getstate()
            except Exception as e:
                info["notes"].append(f"python RNG capture issue: {e}")
            if torch.cuda.is_available():
                try:
                    _ = torch.cuda.get_rng_state_all()
                except Exception as e:
                    info["ok"] = False; info["notes"].append(f"CUDA RNG capture failed: {e}")

            # Sampler state (native) vs session-managed guard
            try:
                for name in ("train", "val"):
                    dl = getattr(self, f"{name}_loader", None)
                    if not dl: continue
                    sd = getattr(getattr(dl, "sampler", None), "state_dict", None)
                    if callable(sd):
                        st = sd()
                        info[f"{name}_sampler_has_state"] = True
                        set_sd = getattr(dl.sampler, "load_state_dict", None)
                        if callable(set_sd):
                            try: set_sd(st)
                            except Exception: info["notes"].append(f"{name} sampler load_state_dict failed (non-fatal)")
                    else:
                        info[f"{name}_sampler_has_state"] = False
            except Exception as e:
                info["notes"].append(f"sampler check failed: {e}")

            # Our policy & guard status
            data_policy = getattr(self, "_data_order_policy", "user")
            managed = bool(getattr(self, "_dl_determinism_managed", False))
            base_seed = getattr(self, "_det_seed", None)
            pause_gen = int(getattr(self, "_pause_gen", 0) or 0)
            cursor = int(getattr(self, "_epoch_batch_idx", 0) or 0)

            # We consider the pause "handled" if either native sampler state is present OR our guard is active
            handled = (
                info.get("train_sampler_has_state") or
                info.get("val_sampler_has_state") or
                managed
            )

            # Friendly “how” list (no framework jargon)
            reasons: List[str] = []
            if managed:
                seed_str = ("n/a" if base_seed is None else str(base_seed))
                reasons.append(f"fixed seeds for shuffling and data-loader workers (seed={seed_str})")
            if info.get("train_sampler_has_state") or info.get("val_sampler_has_state"):
                reasons.append("sampler exposes restart state")
            if cursor > 0:
                reasons.append(f"batch cursor preserved (next batch index ≈ {cursor})")

            # Assemble user-facing text
            lines = [
                f"Run: {self.cfg.run_name}",
                f"Step: {int(self.step)} | Epoch: {int(self.epoch)}",
                "Determinism Health:",
                f"  • torch RNG state length: {info.get('torch_rng_len', 'n/a')}",
                f"  • numpy RNG length:       {info.get('numpy_state_len', 'n/a')}",
                f"  • CUDA RNG available:     {'Yes' if torch.cuda.is_available() else 'No'}",
                f"  • train sampler stateful: {'Yes' if info.get('train_sampler_has_state') else 'No'}",
                f"  • val sampler stateful:   {'Yes' if info.get('val_sampler_has_state') else 'No'}",
                f"  • data_order_policy:      {data_policy}",
                f"  • session-managed order:  {'Yes' if managed else 'No'}",
                f"  • handled this pause:     {'Yes' if handled else 'No'}",
                f"  • pause generation:       {pause_gen}",
                f"  • train batch cursor:     {cursor}",
                f"  • base seed:              {('n/a' if base_seed is None else base_seed)}",
            ]
            if reasons:
                lines += ["  • ensured:"] + [f"    – {r}" for r in reasons]

            # Notes for anything else
            if not info.get("train_sampler_has_state"): info["notes"].append("train sampler has no state_dict(); order managed=" + str(bool(managed)).lower())
            if not info.get("val_sampler_has_state"):   info["notes"].append("val sampler has no state_dict(); order managed=" + str(bool(managed)).lower())

            lines += ["Notes:"] + [f"  • {n}" for n in (info['notes'] or ["(none)"])]

            info.update({
                "data_order_policy": data_policy,
                "session_managed": managed,
                "handled": handled,
                "pause_generation": pause_gen,
                "batch_cursor": cursor,
                "base_seed": base_seed,
                "text": "\n".join(lines),
            })
            return info


        @self._router.on("throughput_health")
        def _throughput_health(payload):
            if not self._paused:
                raise RuntimeError("Model must be paused before running throughput_health")

            budget_steps = max(1, int(payload.get("budget_steps", 2)))
            want_backward = bool(payload.get("micro_backward", False))

            self.model.eval()
            dev = self.device

            criterion = None
            if want_backward:
                if isinstance(getattr(self, "raw_loss_fn", None), torch.nn.Module):
                    criterion = self.raw_loss_fn
                elif isinstance(getattr(self, "criterion", None), torch.nn.Module):
                    criterion = self.criterion
                elif isinstance(getattr(self, "_criterion", None), torch.nn.Module):
                    criterion = self._criterion

            fwd_times: List[float] = []
            bwd_times: List[float] = []
            n_samples = 0

            if self.val_loader is None:
                raise RuntimeError("val_loader is not configured; cannot run throughput_health.")
            it = iter(self.val_loader)
            steps = 0
            sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

            while steps < budget_steps:
                try:
                    batch = next(it)
                except StopIteration:
                    break
                x, y = _extract_xy(batch)
                bs = 1
                if torch.is_tensor(x):
                    bs = int(x.shape[0]) if x.dim() > 0 else 1
                    x = x.to(dev, non_blocking=True)
                elif isinstance(x, (list, tuple)) and x and torch.is_tensor(x[0]):
                    bs = int(x[0].shape[0])
                    x = [t.to(dev, non_blocking=True) if torch.is_tensor(t) else t for t in x]
                elif isinstance(x, dict):
                    anyt = next((v for v in x.values() if torch.is_tensor(v) and v.dim() > 0), None)
                    bs = int(anyt.shape[0]) if anyt is not None else 1
                    x = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in x.items()}

                t0 = time.perf_counter(); sync()
                with torch.inference_mode(), _inference_autocast(dev):
                    out = self.model(x)
                sync(); fwd_times.append(max(0.0, time.perf_counter() - t0))

                if want_backward and (criterion is not None) and (y is not None):
                    try:
                        if torch.is_tensor(y): y = y.to(dev, non_blocking=True)
                        with _inference_autocast(dev, enabled=False):
                            for p in self.model.parameters():
                                if p.requires_grad: p.grad = None
                            logits = None
                            if torch.is_tensor(out): logits = out
                            elif isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]): logits = out[0]
                            elif isinstance(out, dict): logits = out.get("logits") or next((v for v in out.values() if torch.is_tensor(v)), None)
                            if (logits is not None) and torch.is_tensor(y):
                                sync(); t1 = time.perf_counter()
                                loss = criterion(logits, y); loss.backward()
                                sync(); bwd_times.append(max(0.0, time.perf_counter() - t1))
                        for p in self.model.parameters():
                            if p.grad is not None: p.grad = None
                    except Exception:
                        pass

                n_samples += int(bs or 0)
                steps += 1

            def _median(xs: List[float]) -> Optional[float]:
                if not xs:
                    return None
                xs2 = sorted(xs)
                return xs2[len(xs2)//2]

            fwd_batch_ms = (1000.0 * _median(fwd_times)) if fwd_times else None
            bwd_batch_ms = (1000.0 * _median(bwd_times)) if bwd_times else None
            per_batch_size = (n_samples // max(1, steps)) if steps else 1
            fwd_per_sample_ms = (1000.0 * (_median(fwd_times) / max(1, per_batch_size))) if fwd_times else None

            out = {
                "forward_only_ms_per_sample_med": fwd_per_sample_ms,
                "forward_batch_ms_med": fwd_batch_ms,
                "backward_batch_ms_med": bwd_batch_ms,
                "steps_probed": steps,
                "samples_seen": n_samples,
                "notes": [
                    "Forward timings exclude optimizer step; backward is a micro throwaway step",
                    "Medians over probed batches; warmup implicitly dropped",
                ],
            }

            lines = _health_header_for(self, "Throughput Health") + [
                f"  • batches probed: {steps}",
                f"  • samples seen:   {n_samples}",
                f"  • forward median per batch:   {('n/a' if fwd_batch_ms is None else f'{fwd_batch_ms:.2f} ms')}",
                f"  • forward median per sample:  {('n/a' if fwd_per_sample_ms is None else f'{fwd_per_sample_ms:.4f} ms')}",
                f"  • backward median per batch:  {('n/a' if bwd_batch_ms is None else f'{bwd_batch_ms:.2f} ms')}",
            ]
            if out["notes"]:
                lines += ["Notes:"] + [f"  • {n}" for n in out["notes"]]
            out["text"] = "\n".join(lines)
            return out
