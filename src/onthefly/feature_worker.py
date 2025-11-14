from __future__ import annotations
import math, time, threading
from typing import Any, Dict, List, Optional, Callable, Tuple
import torch
from torch.utils.data import DataLoader, Subset


from .snapshots import SnapshotRef
from .data_explorer import compute_per_sample_losses
from .control import send_event


# =============================================================
# Latest-only mailbox (for feature packs)
# =============================================================

class _LatestOnlyTask:
    """Thread-safe 'size=1' mailbox. Writers overwrite, reader consumes latest."""
    def __init__(self):
        self._lock = threading.Lock()
        self._task = None
        self._stop = False

    def submit(self, task):
        with self._lock:
            self._task = task  # overwrite (coalesce)

    def take(self):
        with self._lock:
            t = self._task
            self._task = None
            return t

    def stop(self):
        with self._lock:
            self._stop = True

    def should_stop(self):
        with self._lock:
            return self._stop

    def has_pending(self) -> bool:
        with self._lock:
            return self._task is not None


# =============================================================
# FeatureWorker (no engine inside; computes heavy features only)
# =============================================================

class FeatureWorker(threading.Thread):
    """
    Loads a snapshot, computes budgeted per-sample features, and returns a
    *feature pack* to the session. The main session consumes these features on
    the next train step; this worker only produces data.

    The pack schema mirrors the observe_batch(...) inputs handled by the session
    so it can feed them atomically along with its batch-level stats.
    """
    def __init__(
        self,
        *,
        model_ctor,
        loss_fn,
        train_dataset,
        collate_fn,
        batch_size,
        aux_device: str = "cpu",
        sampling_cfg: Dict[str, Any],
        embedding_hook: Optional[Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]],
        on_feature_pack: Callable[[Dict[str, Any]], None],
        halt_fn: Optional[Callable[[], bool]] = None,
    ):
        super().__init__(daemon=True)
        self._box = _LatestOnlyTask()
        self._model_ctor = model_ctor
        self._loss_fn = loss_fn
        self._ds = train_dataset
        self._collate = collate_fn
        self._bs = int(batch_size or 256)
        self._aux_device = aux_device or "cpu"
        self._samp = dict(sampling_cfg or {})
        self._hook = embedding_hook
        self._emit = on_feature_pack
        self._halt_fn = halt_fn or (lambda: False)

    # called by session
    def submit_snapshot(self, *, owner: str, version: int, token: str,
                        snapshot: SnapshotRef | Dict[str, Any] | None = None,
                        ckpt_path: Optional[str] = None):
        if not self._ds:
            return
        if snapshot is None and not ckpt_path:
            return
        payload = {"owner": owner, "version": int(version), "token": str(token),
                   "snapshot": snapshot, "ckpt": ckpt_path}
        self._box.submit(payload)

    def update_sampling(self, sampling_cfg: Dict[str, Any]):
        self._samp.update(sampling_cfg or {})

    def stop(self):
        self._box.stop()

    def _should_halt(self) -> bool:
        return self._box.should_stop() or self._halt_fn()

    def run(self):
        while not self._should_halt():
            task = self._box.take()
            if not task:
                time.sleep(0.05); continue
            try:
                if self._should_halt(): break
                pack = self._compute_feature_pack(task)
                if pack and not self._should_halt():
                    self._emit(pack)
            except Exception as e:
                send_event({"type":"log","level":"warn","text":f"[feature_worker] {e}"})

    # ---- internals ----
    def _compute_feature_pack(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._should_halt():
            return None
        ckpt_path = str(task.get("ckpt") or getattr(task.get("snapshot"), "path", "") or "")

        model = self._model_ctor().to(self._aux_device).eval()
        snap = task.get("snapshot")
        if isinstance(snap, SnapshotRef) and snap.state_dict is not None:
            model.load_state_dict(snap.state_dict, strict=True)
        elif isinstance(snap, dict) and "state_dict" in snap:
            model.load_state_dict(snap["state_dict"], strict=True)
        elif ckpt_path:
            blob = torch.load(ckpt_path, map_location=self._aux_device, weights_only=False)
            model.load_state_dict(blob["model"], strict=True)
        else:
            raise RuntimeError("No snapshot state provided to FeatureWorker")

        amp_on = bool(self._samp.get("amp_for_psl", True) and torch.cuda.is_available() and "cuda" in self._aux_device)

        # ---------- budget indices ----------
        N = len(self._ds) if hasattr(self._ds, "__len__") else 0
        budget = int(self._samp.get("psl_budget", 0) or 0)
        if budget and N > budget:
            stride = max(1, N // budget)
            indices = list(range(0, N, stride))[:budget]
        else:
            indices = None

        # per-sample losses (preempt if a newer snapshot arrives)
        losses, __ = compute_per_sample_losses(
            model, self._ds, self._collate, self._loss_fn,
            device=self._aux_device, batch_size=self._bs,
            indices=indices,
            mirror_train_semantics=bool(self._samp.get("mirror_train", True)),
            amp_enabled=amp_on,
            should_stop=lambda: self._should_halt() or self._box.has_pending()
        )

        num = [float(v) for v in losses if v is not None and math.isfinite(float(v))]
        bl = (sum(num) / max(1, len(num))) if num else float("nan")
        feed: Dict[str, Any] = {
            "batch_loss": bl,
            "grad_norm": 0.0,
            "nan_flag": not math.isfinite(bl),
            "sample_losses": losses,
        }
        if indices is not None:
            feed["sample_ids"] = indices

        # optional channels
        need_margin = bool(self._samp.get("compute_margins", True))
        need_embed  = bool(self._samp.get("compute_embeddings", False)) and (self._hook is not None)
        if need_margin or need_embed:
            margins, embeds = self._compute_margins_and_embeddings(
                model, device=self._aux_device, indices=indices,
                need_margin=need_margin, need_embed=need_embed,
                embed_max_dim=int(self._samp.get("embed_max_dim", 256)),
                batch_size=self._bs
            )
            if need_margin and margins: feed["sample_margins"] = margins
            if need_embed  and embeds:  feed["sample_embed"]   = embeds

        # attach provenance so session can enforce freshness
        feed.update({
            "owner_run_id": task["owner"],
            "at_step": int(task["version"]),
            "token": task["token"],
            "init_from": ckpt_path,
        })
        return feed

    def _compute_margins_and_embeddings(
            self,
            model: torch.nn.Module,
            *,
            device: str,
            indices: Optional[List[int]],
            need_margin: bool,
            need_embed: bool,
            embed_max_dim: int,
            batch_size: int,
        ) -> Tuple[List[float], List[List[float]]]:
            from .session import OnTheFlySession
            margins: List[float] = [] if need_margin else []
            embeds:  List[List[float]] = [] if need_embed else []
            loader = DataLoader(
                Subset(self._ds, indices) if indices is not None else self._ds,
                batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=self._collate
            )
            autocast = torch.cuda.amp.autocast if (torch.cuda.is_available() and "cuda" in device) else _noop_ctx
            with torch.no_grad():
                for batch in loader:
                    if self._should_halt() or self._box.has_pending():
                        break
                    x = batch[0].to(device)
                    with autocast():
                        logits = model(x)
                    if need_margin:
                        m = _top2_margin(logits).detach().cpu().tolist()
                        margins.extend([float(v) for v in m])
                    if need_embed and OnTheFlySession._embedding_hook_fn is not None:
                        try:
                            e = OnTheFlySession._embedding_hook_fn_static(model, x, logits)
                        except AttributeError:
                            e = None
                        if torch.is_tensor(e):
                            if e.dim() == 1: e = e.unsqueeze(1)
                            if e.size(1) > embed_max_dim:
                                e = e[:, :embed_max_dim]
                            el = e.detach().cpu().tolist()
                            embeds.extend([list(map(float, row)) for row in el])
            return margins, embeds



def _top2_margin(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 1:
        return logits.abs()
    if logits.dim() >= 2 and logits.size(1) >= 2:
        top2 = torch.topk(logits, k=2, dim=1).values
        return (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
    return logits.abs().flatten()

class _noop_ctx:
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __call__(self): return self
