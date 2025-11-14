from __future__ import annotations

import math
import sys
import threading
from typing import Any, Optional, Tuple

import torch


def _float_or_none(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        if torch.is_tensor(val):
            if val.numel() == 0:
                return None
            v = val.detach()
            if v.ndim > 0:
                v = v.reshape(-1)
            return float(v.mean().item())
        if isinstance(val, (list, tuple)):
            for item in val:
                out = _float_or_none(item)
                if out is not None:
                    return out
            return None
        v = float(val)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def current_learning_rate(optimizer) -> Optional[float]:
    if optimizer is None:
        return None
    try:
        groups = getattr(optimizer, "param_groups", None) or []
        vals = [float(g["lr"]) for g in groups if "lr" in g and math.isfinite(float(g["lr"]))]
        if not vals:
            return None
        return float(sum(vals) / len(vals))
    except Exception:
        return None


def estimate_batch_size(batch: Any) -> Optional[int]:
    if batch is None:
        return None
    if torch.is_tensor(batch):
        return int(batch.size(0)) if batch.ndim >= 1 else 1
    if isinstance(batch, (list, tuple)):
        for item in batch:
            bs = estimate_batch_size(item)
            if bs is not None:
                return bs
        return None
    if isinstance(batch, dict):
        for key in ("input", "inputs", "x", "features"):
            if key in batch:
                bs = estimate_batch_size(batch[key])
                if bs is not None:
                    return bs
        for value in batch.values():
            bs = estimate_batch_size(value)
            if bs is not None:
                return bs
        return None
    if hasattr(batch, "__len__") and not isinstance(batch, (str, bytes)):
        try:
            return int(len(batch))
        except Exception:
            return None
    return None


def batch_accuracy(logits: Any, targets: Any) -> Optional[float]:
    if logits is None or targets is None:
        return None
    if not (torch.is_tensor(logits) and torch.is_tensor(targets)):
        return None

    try:
        tgt = targets.detach()
        if tgt.ndim == 0:
            return None
        if tgt.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            return None
        preds = None
        with torch.no_grad():
            if logits.ndim >= 2 and logits.size(-1) > 1:
                preds = torch.argmax(logits.detach(), dim=-1)
            elif logits.ndim >= 1:
                preds = (logits.detach() > 0).to(dtype=torch.long)
        if preds is None:
            return None
        preds = preds.reshape(-1)
        tgt = tgt.to(dtype=torch.long).reshape(-1)
        if preds.numel() != tgt.numel():
            return None
        correct = (preds == tgt).float().mean().item()
        return float(correct) if math.isfinite(correct) else None
    except Exception:
        return None


def weight_norm(model: Optional[torch.nn.Module]) -> Optional[float]:
    if model is None:
        return None
    total = 0.0
    try:
        with torch.no_grad():
            for p in model.parameters():
                if p is None:
                    continue
                total += float(p.detach().float().pow(2).sum().item())
        return math.sqrt(total) if total > 0 else 0.0
    except Exception:
        return None


def _first_tensor(output: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(output, dict):
        for value in output.values():
            t = _first_tensor(value)
            if t is not None:
                return t
    return None


class ActivationZeroTracker:
    """
    Lightweight forward-hook tracker that approximates how sparse ReLU/GELU/Sigmoid/Tanh
    activations are. Captures per-forward zero fraction and exposes an averaged snapshot.
    """

    def __init__(self, model: Optional[torch.nn.Module], *, enabled: bool = True) -> None:
        self.enabled = bool(enabled and model is not None)
        self._hooks: list[Any] = []
        self._values: list[float] = []
        self._lock = threading.Lock()
        self._last: Optional[float] = None
        if self.enabled:
            self._attach(model)

    def _attach(self, model: Optional[torch.nn.Module]) -> None:
        if model is None:
            return
        for name, module in model.named_modules():
            if self._should_track(module):
                try:
                    hook = module.register_forward_hook(self._hook)
                    self._hooks.append(hook)
                except Exception:
                    continue

    @staticmethod
    def _should_track(module: torch.nn.Module) -> bool:
        cls = type(module).__name__.lower()
        return any(tok in cls for tok in ("relu", "gelu", "silu", "sigmoid", "tanh"))

    def _hook(self, _mod, _inp, out):  # noqa: ANN001
        try:
            tensor = _first_tensor(out)
            if tensor is None:
                return
            if not torch.is_floating_point(tensor) or tensor.numel() == 0:
                return
            with torch.no_grad():
                zero_frac = float((tensor == 0).float().mean().item())
            if math.isfinite(zero_frac):
                with self._lock:
                    self._values.append(zero_frac)
        except Exception:
            return

    def pop_recent(self) -> Optional[float]:
        with self._lock:
            if not self._values:
                return self._last
            avg = float(sum(self._values) / len(self._values))
            self._values.clear()
            self._last = avg
            return avg

    def close(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()


class DeviceStatsMonitor:
    """
    Samples process memory (CPU) or CUDA memory/utilization (GPU) per step.
    GPU utilization requires NVML; if unavailable we gracefully return None.
    """

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self._nvml = None
        self._nvml_handle = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                import pynvml  # type: ignore

                pynvml.nvmlInit()
                idx = torch.cuda.current_device() if self.device.index is None else int(self.device.index)
                self._nvml = pynvml
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(int(idx))
            except Exception:
                self._nvml = None
                self._nvml_handle = None

    @staticmethod
    def _process_memory_mb() -> Optional[float]:
        try:
            import psutil  # type: ignore

            rss = psutil.Process().memory_info().rss
            return float(rss) / (1024.0 ** 2)
        except Exception:
            try:
                import resource

                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform.startswith("darwin"):
                    return float(rss) / (1024.0 ** 2)
                return float(rss) / 1024.0
            except Exception:
                return None

    def snapshot(self) -> Tuple[Optional[float], Optional[float]]:
        mem_mb: Optional[float] = None
        gpu_util: Optional[float] = None

        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                mem_mb = float(torch.cuda.memory_allocated(self.device) / (1024.0 ** 2))
            except Exception:
                mem_mb = None
            if self._nvml and self._nvml_handle is not None:
                try:
                    stats = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    gpu_util = float(getattr(stats, "gpu", None))
                except Exception:
                    gpu_util = None
        else:
            mem_mb = self._process_memory_mb()
        return mem_mb, gpu_util

    def close(self) -> None:
        self._nvml = None
        self._nvml_handle = None
