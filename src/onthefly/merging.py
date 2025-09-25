# src/onthefly/merging.py
from __future__ import annotations
import torch
from typing import Iterable, Dict

def weighted_average_merge(models: Iterable[torch.nn.Module], weights: Iterable[float]) -> Dict[str, torch.Tensor]:
    models = list(models); weights = list(weights)
    assert len(models) == len(weights) and len(models) > 0
    total = sum(weights); weights = [w/total for w in weights]
    out = {}
    with torch.no_grad():
        for k in models[0].state_dict().keys():
            stacked = torch.stack([m.state_dict()[k].float() * w for m, w in zip(models, weights)], dim=0)
            out[k] = stacked.sum(dim=0).to(models[0].state_dict()[k].dtype)
    return out

def stochastic_weight_averaging(models: Iterable[torch.nn.Module]) -> Dict[str, torch.Tensor]:
    # Equal-weight SWA
    ms = list(models); n = len(ms)
    return weighted_average_merge(ms, [1.0/n]*n)

# Placeholder for “extremely sophisticated” mergers; keep the interface
def advanced_merge(models: Iterable[torch.nn.Module], meta=None) -> Dict[str, torch.Tensor]:
    # TODO: implement parameter-wise gate-less blending using fisher info / curvature estimates
    return stochastic_weight_averaging(models)
