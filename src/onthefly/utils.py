# src/onthefly/utils.py
from __future__ import annotations
from typing import Callable, Any
import contextlib
import torch
import numpy as np
import random

# Prefer torch.func.vmap (PyTorch 2.x), fall back to functorch.vmap, else None
try:
    from torch.func import vmap  # PyTorch >= 2.0
except Exception:
    try:
        from functorch import vmap  # older functorch
    except Exception:
        vmap = None

def _seed_worker(worker_id: int, base_seed: int, orig=None):
    """
    Worker init fn that sets Python, NumPy and torch RNGs based on a base seed
    plus worker_id, then optionally calls the original worker_init_fn.
    """
    s = int(base_seed) + int(worker_id)
    random.seed(s)
    try:
        np.random.seed(s)
    except Exception:
        pass
    torch.manual_seed(s)

    if callable(orig):
        orig(worker_id)



def _as_tensor_loss(x, device):
    """
    Normalize various loss return types to a Tensor on `device`.
    - If dict-like and has 'loss', use that.
    - If tuple/list, take the first element.
    - If scalar/number, wrap as tensor.
    - If Tensor on a different device, move it.
    """
    if isinstance(x, dict) and "loss" in x:
        x = x["loss"]
    if isinstance(x, (tuple, list)):
        x = x[0]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, device=device)
    else:
        if x.device.type != torch.device(device).type:
            x = x.to(device)
    return x


@contextlib.contextmanager
def _temporary_attr(obj: Any, name: str, value: Any):
    """Temporarily set `obj.name = value` if attribute exists; restore afterward."""
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


def _mean_over_nonbatch(x: torch.Tensor) -> torch.Tensor:
    """Average all dims except the leading batch dim."""
    if x.ndim == 1:
        return x
    return x.view(x.shape[0], -1).mean(dim=1)


def _masked_mean_over_nonbatch(raw: torch.Tensor, target: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """
    Average per-element `raw` over non-batch dims, excluding positions where
    `target == ignore_index`. Returns one scalar per sample [N].
    If a sample has 0 valid positions, returns NaN for that sample.
    """
    N = raw.shape[0]
    # Expect target to align with raw's non-batch geometry for elementwise losses
    if not isinstance(target, torch.Tensor) or target.shape[:1] != (N,):
        return _mean_over_nonbatch(raw)

    mask = (target != ignore_index)
    # Try to align shapes (e.g., CE returns raw with same non-batch shape as target)
    if mask.shape != raw.shape:
        # If mask has fewer dims (e.g., mask [N, L] vs raw [N, L]), broadcasting works.
        # If completely incompatible, fall back to unmasked mean.
        try:
            mask = mask.to(raw.device)
            # a dummy broadcasted op to verify compatibility
            _ = raw * mask.to(dtype=raw.dtype)
        except Exception:
            return _mean_over_nonbatch(raw)

    m = mask.to(dtype=raw.dtype, device=raw.device)
    num = (raw * m).view(N, -1).sum(dim=1)
    den = m.view(N, -1).sum(dim=1)
    out = num / den.clamp(min=1)
    out = torch.where(den > 0, out, torch.full_like(out, float("nan")))
    return out


def _reduce_per_sample_like_training(raw: torch.Tensor, target: Any, loss_fn: Callable) -> torch.Tensor:
    """
    Convert a per-element loss tensor `raw` (shape [N, ...]) to one scalar per
    sample, mirroring the loss' inclusion semantics when possible.

    - If the loss exposes `ignore_index` and the target is a Tensor with the
      same non-batch geometry: exclude ignored positions from the average.
    - Otherwise: average over all non-batch dims.
    """
    ignore = getattr(loss_fn, "ignore_index", None)
    if isinstance(ignore, int) and isinstance(target, torch.Tensor) and raw.ndim >= 1:
        return _masked_mean_over_nonbatch(raw, target, ignore)
    return _mean_over_nonbatch(raw)


def _vectorize_with_vmap(loss_fn: Callable, pred: torch.Tensor, target: Any) -> torch.Tensor:
    """
    Vectorize loss_fn over batch with vmap if available.
    We call the user's loss on a *single-sample batch* (unsqueezed) so any
    internal masking/weights/smoothing are applied exactly as defined.
    """
    if vmap is None:
        raise RuntimeError("vmap not available")

    N = pred.shape[0]
    target_is_tensor = isinstance(target, torch.Tensor)
    target_has_batch = target_is_tensor and target.shape[:1] == (N,)

    def _single(pr_i, tg_i):
        pr_b = pr_i.unsqueeze(0)  # reintroduce batch dim of 1
        tg_b = tg_i.unsqueeze(0) if isinstance(tg_i, torch.Tensor) else tg_i
        out = loss_fn(pr_b, tg_b)
        out = _as_tensor_loss(out, device=pr_i.device)
        # If scalar, that's the loss the user would backprop for this example
        if out.ndim == 0:
            return out
        # If it's still per-element, reduce like training would
        return _reduce_per_sample_like_training(out, tg_b if isinstance(tg_b, torch.Tensor) else tg_i, loss_fn)

    in_axes = (0, 0) if target_has_batch else (0, None)
    losses = vmap(_single, in_axes=in_axes)(pred, target)
    return losses.reshape(-1)


def _vectorize_with_loop(loss_fn: Callable, pred: torch.Tensor, target: Any) -> torch.Tensor:
    """Per-sample Python loop fallback when vmap isn't available."""
    N = pred.shape[0]
    out_list = []
    target_is_tensor = isinstance(target, torch.Tensor)
    target_has_batch = target_is_tensor and target.shape[:1] == (N,)

    for i in range(N):
        pr_i = pred[i].unsqueeze(0)  # single-sample batch
        tg_i = target[i] if target_has_batch else target
        tg_b = tg_i.unsqueeze(0) if isinstance(tg_i, torch.Tensor) else tg_i

        out = loss_fn(pr_i, tg_b)
        out = _as_tensor_loss(out, device=pred.device)
        if out.ndim == 0:
            out_list.append(out)
        else:
            out_list.append(_reduce_per_sample_like_training(out, tg_b if isinstance(tg_b, torch.Tensor) else tg_i, loss_fn))

    return torch.stack(out_list, dim=0).reshape(-1)


def per_sample_loss(loss_fn: Callable, pred: torch.Tensor, target: Any) -> torch.Tensor:
    """
    Return per-sample losses (shape [N]) that mirror the user's loss semantics:
      1) Prefer a single call with reduction='none' then reduce per sample
         using the loss' inclusion logic (e.g., ignore_index) to avoid any
         home-grown masking.
      2) If that isn't supported, call the loss on each example (batch size 1)
         so any internal masking/weights/smoothing apply exactly.
    """
    # Path 1: temporarily set reduction='none'
    with _temporary_attr(loss_fn, "reduction", "none") as could_set:
        if could_set:
            try:
                raw = loss_fn(pred, target)
                raw = _as_tensor_loss(raw, device=pred.device)
                if raw.ndim == 0 or raw.shape[0] != pred.shape[0]:
                    raise RuntimeError("loss did not return per-sample values under reduction='none'")
                return _reduce_per_sample_like_training(raw, target, loss_fn)
            except Exception:
                pass  # Fall through to vectorization if unsupported

    # Path 2: vectorize exact loss
    try:
        return _vectorize_with_vmap(loss_fn, pred, target)
    except Exception:
        return _vectorize_with_loop(loss_fn, pred, target)


__all__ = ["per_sample_loss", "_seed_worker"]
