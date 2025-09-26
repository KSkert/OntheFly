# src/onthefly/data_explorer.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception:
    MiniBatchKMeans = None


# -----------------------------
# Embeddings & simple clustering
# -----------------------------
def compute_embeddings(model, loader, device, hook_fn=None, max_batches=50) -> np.ndarray:
    model.eval()
    embs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            x = batch[0].to(device)
            if hook_fn is not None:
                e = hook_fn(model, batch)
                e = e if isinstance(e, np.ndarray) else e.detach().cpu().numpy()
            else:
                z = model(x)
                e = z.detach().cpu().numpy()
            embs.append(e)
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 8))


def cluster_embeddings(embs: np.ndarray, k: int = 10) -> Dict[str, Any]:
    if embs.size == 0:
        return {"labels": np.array([], dtype=int), "centers": np.zeros((0, embs.shape[-1] if embs.ndim else 0))}
    if MiniBatchKMeans is None:
        n = embs.shape[0]
        labels = np.random.randint(0, k, size=(n,))
        centers = np.zeros((k, embs.shape[-1]))
        return {"labels": labels, "centers": centers}
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=2048)
    labels = kmeans.fit_predict(embs)
    return {"labels": labels, "centers": kmeans.cluster_centers_}


def select_hard_clusters(labels: np.ndarray, losses: np.ndarray, top_n: int = 3) -> List[int]:
    clusters = np.unique(labels)
    means = [(c, float(losses[labels == c].mean())) for c in clusters]
    means.sort(key=lambda t: t[1], reverse=True)
    return [c for c, _ in means[:top_n]]


# -----------------------------
# Train-like context (dropout/BN)
# -----------------------------
class _BNState:
    __slots__ = ("mod", "running_mean", "running_var", "num_batches_tracked", "momentum")

    def __init__(self, mod: nn.modules.batchnorm._BatchNorm):
        self.mod = mod
        self.running_mean = mod.running_mean.detach().clone() if mod.running_mean is not None else None
        self.running_var = mod.running_var.detach().clone() if mod.running_var is not None else None
        self.num_batches_tracked = (
            mod.num_batches_tracked.detach().clone() if hasattr(mod, "num_batches_tracked") else None
        )
        self.momentum = mod.momentum

    def freeze_updates(self):
        self.mod.momentum = 0.0

    def restore(self):
        if self.running_mean is not None:
            self.mod.running_mean.copy_(self.running_mean)
        if self.running_var is not None:
            self.mod.running_var.copy_(self.running_var)
        if self.num_batches_tracked is not None:
            self.mod.num_batches_tracked.copy_(self.num_batches_tracked)
        self.mod.momentum = self.momentum


class _TrainLikeCtx:
    """Enable dropout + BN batch stats without mutating BN buffers."""
    def __init__(self, model: nn.Module):
        self.model = model
        self.was_training = model.training
        self.bn_states: List[_BNState] = []

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                st = _BNState(m)
                st.freeze_updates()
                self.bn_states.append(st)
        self.model.train()
        return self

    def __exit__(self, exc_type, exc, tb):
        for st in self.bn_states:
            st.restore()
        if not self.was_training:
            self.model.eval()
        return False


# -----------------------------
# Helpers for per-sample losses
# -----------------------------
def _flatten_nonbatch(x: torch.Tensor) -> torch.Tensor:
    """
    Return a 2D view [N, M] where N is the batch dim and M flattens everything else.
    Works for 0-D/1-D inputs by making M>=1.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if x.ndim == 0:
        return x.view(1, 1)
    return x.reshape(x.shape[0], -1)


def _build_effective_mask(criterion: nn.Module, target: Optional[torch.Tensor], shape_like: torch.Size) -> Optional[torch.Tensor]:
    """
    Return a boolean mask (True=included) using loss_fn.ignore_index if present and
    if the target tensor is available. If unknown, return None.
    """
    ignore = getattr(criterion, "ignore_index", None)
    if isinstance(ignore, int) and isinstance(target, torch.Tensor):
        try:
            return (target != ignore)
        except Exception:
            return None
    return None


# -----------------------------
# Per-sample loss computation
# -----------------------------
@torch.no_grad()
def compute_per_sample_losses(
    model: nn.Module,
    dataset,
    collate_fn,
    criterion: nn.Module,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 256,
    indices: Optional[List[int]] = None,
    *,
    mirror_train_semantics: bool = False,
    amp_enabled: Optional[bool] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Tuple[List[float], List[int]]:
    """
    Returns:
      (losses, sample_indices) where len(losses) == len(sample_indices)
    """
    model.to(device)

    # Build the iteration dataset and the *base* index mapping we’ll consume in order.
    if indices is not None and len(indices) > 0:
        ds = Subset(dataset, indices)
        base_indices = list(indices)  # original dataset indices, in order
    else:
        ds = dataset
        base_indices = list(range(len(dataset)))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    all_losses: List[float] = []
    all_indices: List[int] = []
    cursor = 0  # points into base_indices

    # AMP context
    if amp_enabled and ("cuda" in str(device).lower()):
        amp_ctx = torch.cuda.amp.autocast
    else:
        from contextlib import contextmanager
        @contextmanager
        def amp_ctx():
            yield

    # Choose model mode/context
    if mirror_train_semantics:
        outer_ctx = _TrainLikeCtx(model)
    else:
        from contextlib import ExitStack
        outer_ctx = ExitStack()
        was_training = model.training
        model.eval()
        outer_ctx.enter_context(torch.inference_mode())

    from contextlib import contextmanager
    @contextmanager
    def _tmp_attr(obj, name, value):
        had = hasattr(obj, name)
        old = getattr(obj, name, None)
        try:
            if had:
                setattr(obj, name, value)
            yield had
        finally:
            if had:
                try:
                    setattr(obj, name, old)
                except Exception:
                    pass

    def _consume_batch_indices(n: int):
        nonlocal cursor
        batch_idx = base_indices[cursor:cursor + n]
        all_indices.extend(int(i) for i in batch_idx)
        cursor += n

    with outer_ctx:
        for batch in loader:
            if should_stop and should_stop():
                return [], []
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise RuntimeError("Unexpected batch format; expected (inputs, targets, *...).")

            x, y = batch[0], batch[1]
            x = x.to(device)
            y = y.to(device) if isinstance(y, torch.Tensor) else y
            batch_size_now = int(x.shape[0])

            with amp_ctx():
                # Prefer reduction='none'
                with _tmp_attr(criterion, "reduction", "none") as could_set:
                    if could_set:
                        raw = criterion(x if getattr(criterion, "_expects_input", False) else model(x), y)
                        if not isinstance(raw, torch.Tensor):
                            logits = model(x)
                            raw = criterion(logits, y)
                    else:
                        logits = model(x)
                        try:
                            with _tmp_attr(criterion, "reduction", "none") as could:
                                if could:
                                    raw = criterion(logits, y)
                                else:
                                    from .utils import per_sample_loss
                                    loss_vec = per_sample_loss(criterion, logits, y).reshape(-1)
                                    all_losses.extend(loss_vec.detach().cpu().tolist())
                                    _consume_batch_indices(batch_size_now)
                                    continue
                        except Exception:
                            from .utils import per_sample_loss
                            loss_vec = per_sample_loss(criterion, logits, y).reshape(-1)
                            all_losses.extend(loss_vec.detach().cpu().tolist())
                            _consume_batch_indices(batch_size_now)
                            continue

                # Scalar fallback under 'none': replicate
                if not isinstance(raw, torch.Tensor):
                    out = float(torch.as_tensor(raw).item())
                    all_losses.extend([out] * batch_size_now)
                    _consume_batch_indices(batch_size_now)
                    continue

                # Mask & per-sample mean
                mask = _build_effective_mask(criterion, y if isinstance(y, torch.Tensor) else None, raw.shape)
                if mask is None:
                    mask = torch.ones_like(raw, dtype=torch.bool)

                raw_f = _flatten_nonbatch(raw)               # [N, M]
                m_f = _flatten_nonbatch(mask.to(raw.device)) # [N, M]

                num_i = (raw_f * m_f).sum(dim=1)             # [N]
                cnt_i = m_f.sum(dim=1)                       # [N]
                safe_cnt = cnt_i.clamp(min=1)
                per_sample = num_i / safe_cnt
                per_sample = torch.where(cnt_i > 0, per_sample, torch.zeros_like(per_sample))

                all_losses.extend(per_sample.detach().cpu().tolist())
                _consume_batch_indices(batch_size_now)

    if not mirror_train_semantics:
        try:
            if was_training:
                model.train()
        except NameError:
            pass

    return all_losses, all_indices

# -----------------------------
# Lightweight subset exporter
# -----------------------------
def _default_row_adapter(sample, idx: int) -> Dict[str, Any]:
    """
    Best-effort conversion of a dataset item into a flat row.
    Falls back to just sample_id when structure is unknown.
    """
    row = {"sample_id": int(idx)}
    try:
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            y = sample[1]
            if torch.is_tensor(y):
                if y.ndim == 0:
                    row["label"] = y.item()
                else:
                    row["label"] = y.detach().cpu().tolist()
            else:
                row["label"] = y
    except Exception:
        pass
    return row


def export_subset_table(
    dataset,
    indices: List[int],
    out_path: str,
    fmt: str = "parquet",
    row_fn: Optional[Callable[[Any, int], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build a simple table of rows for the given dataset indices and write it as
    Parquet/Feather/CSV. The default schema includes: sample_id and (if present) label.
    """
    import os
    import pandas as pd
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rf = row_fn or _default_row_adapter

    rows = []
    for idx in indices:
        try:
            sample = dataset[idx]
            row = rf(sample, int(idx))
            if not isinstance(row, dict):
                row = {"sample_id": int(idx)}
            rows.append(row)
        except Exception:
            # Skip unreadable samples (keeps export resilient)
            rows.append({"sample_id": int(idx)})

    df = pd.DataFrame(rows)
    fmt = (fmt or "parquet").lower()
    if fmt == "parquet":
        try:
            df.to_parquet(out_path, index=False)  # requires pyarrow or fastparquet
        except Exception:
            # Fallback to CSV if parquet engine isn't available
            out_path = os.path.splitext(out_path)[0] + ".csv"
            df.to_csv(out_path, index=False)
            fmt = "csv"
    elif fmt == "feather":
        try:
            df.to_feather(out_path)  # requires pyarrow
        except Exception:
            out_path = os.path.splitext(out_path)[0] + ".csv"
            df.to_csv(out_path, index=False)
            fmt = "csv"
    else:
        df.to_csv(out_path, index=False)

    return {"out_path": out_path, "rows": len(df), "format": fmt, "columns": list(df.columns)}


__all__ = [
    "compute_embeddings",
    "cluster_embeddings",
    "select_hard_clusters",
    "compute_per_sample_losses",
]


