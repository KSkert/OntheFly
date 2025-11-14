# src/onthefly/merging.py
from __future__ import annotations
from typing import Iterable, Dict, List, Optional
import warnings
import torch


def weighted_average_merge(
    models: Iterable[torch.nn.Module],
    weights: Iterable[float],
) -> Dict[str, torch.Tensor]:
    """Internal helper; not used as a strategy name."""
    models = list(models)
    weights = list(weights)
    assert len(models) == len(weights) and len(models) > 0
    total = sum(weights)
    if total == 0:
        raise ValueError("Sum of weights must be non-zero")
    weights = [w / total for w in weights]

    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        ref_sd = models[0].state_dict()
        for k in ref_sd.keys():
            stacked = torch.stack(
                [m.state_dict()[k].float() * w for m, w in zip(models, weights)],
                dim=0,
            )
            out[k] = stacked.sum(dim=0).to(ref_sd[k].dtype)
    return out


def stochastic_weight_averaging(
    models: Iterable[torch.nn.Module],
) -> Dict[str, torch.Tensor]:
    """Equal-weight SWA."""
    ms = list(models)
    n = len(ms)
    if n == 0:
        raise ValueError("SWA received an empty model list")
    return weighted_average_merge(ms, [1.0 / n] * n)


def fisher_soup_merge(
    models: Iterable[torch.nn.Module],
    fisher_mats: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Fisher soup merge.

    If fisher_mats is None or mismatched, falls back to SWA.
    """
    ms = list(models)
    if not ms:
        raise ValueError("Fisher soup received an empty model list")

    if fisher_mats is None or len(fisher_mats) != len(ms):
        warnings.warn(
            "No valid Fisher info provided; Fisher soup falling back to SWA",
            RuntimeWarning,
        )
        return stochastic_weight_averaging(ms)

    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        ref_sd = ms[0].state_dict()
        for k in ref_sd.keys():
            params = [m.state_dict()[k].float() for m in ms]
            fishers_k = [fi.get(k) for fi in fisher_mats]

            if any(f is None for f in fishers_k):
                # Missing Fisher for this param -> equal average
                stacked = torch.stack(params, dim=0)
                merged = stacked.mean(dim=0)
                out[k] = merged.to(ref_sd[k].dtype)
                continue

            fisher_stack = torch.stack([f.float() for f in fishers_k], dim=0)
            denom = fisher_stack.sum(dim=0, keepdim=True) + 1e-8
            weights = fisher_stack / denom

            param_stack = torch.stack(params, dim=0)
            merged = (weights * param_stack).sum(dim=0)
            out[k] = merged.to(ref_sd[k].dtype)

    return out


def adapter_fuse_merge(models: Iterable[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    """
    Adapter fusion:

    - Base parameters taken from the first model.
    - Parameters whose name contains 'adapter' (case-insensitive)
      are averaged across all models.
    """
    ms = list(models)
    if not ms:
        raise ValueError("Adapter fuse received an empty model list")

    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        ref_sd = ms[0].state_dict()
        for k, ref_param in ref_sd.items():
            if "adapter" in k.lower():
                stacked = torch.stack(
                    [m.state_dict()[k].float() for m in ms],
                    dim=0,
                )
                merged = stacked.mean(dim=0).to(ref_param.dtype)
                out[k] = merged
            else:
                out[k] = ref_param.clone()
    return out
