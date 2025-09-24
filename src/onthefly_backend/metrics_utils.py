from __future__ import annotations
import math
from typing import Sequence, List

def _to_scalar_loss(loss, *, device="cpu"):
    import torch
    def _as_tensor(x):
        return x if torch.is_tensor(x) else torch.as_tensor(x, device=device, dtype=torch.float32)
    if isinstance(loss, dict):
        if "loss" in loss:
            loss = loss["loss"]
        else:
            cand = next((v for k, v in loss.items() if "loss" in k.lower()), None)
            loss = cand if cand is not None else sum(_as_tensor(v) for v in loss.values())
    if isinstance(loss, (list, tuple)):
        loss = sum(_as_tensor(x) for x in loss)
    if not hasattr(loss, "ndim"):
        loss = _as_tensor(loss)
    if getattr(loss, "ndim", 0) != 0:
        loss = loss.mean()
    return loss

def _grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item() ** 2)
    return total ** 0.5

def _percentile_list(values: Sequence[float], q: float) -> float:
    if not values:
        return float('nan')
    q = min(max(q, 0.0), 1.0)
    s = sorted(values)
    pos = (len(s)-1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1-frac) + s[hi] * frac)

def _ranks(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    order = sorted((v, i) for i, v in enumerate(values))
    ranks = [0.0] * len(values)
    for r, (_, i) in enumerate(order):
        ranks[i] = r / max(1, len(values) - 1)
    return ranks

def _top2_margin(logits):
    import torch
    if not torch.is_tensor(logits):
        logits = torch.as_tensor(logits)
    if logits.dim() == 1:
        return logits.abs()
    if logits.dim() >= 2 and logits.size(1) >= 2:
        top2 = torch.topk(logits, k=2, dim=1).values
        return (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
    return logits.abs().flatten()
