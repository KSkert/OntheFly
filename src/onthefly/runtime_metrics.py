from __future__ import annotations

import contextlib
import math
import re
import sys
import threading
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from .metrics_utils import _grad_norm

_METRIC_ALIAS_MAP: Dict[str, Tuple[str, ...]] = {
    "accuracy": (
        "accuracy",
        "acc",
        "train_accuracy",
        "train_acc",
        "top1",
        "top-1",
        "top_1",
    ),
    "grad_norm": (
        "grad_norm",
        "gradient_norm",
        "gradnorm",
    ),
    "lr": (
        "lr",
        "learning_rate",
        "eta",
    ),
    "weight_norm": (
        "weight_norm",
    ),
    "activation_zero_frac": (
        "activation_zero_frac",
        "activation_zero_fraction",
        "zero_fraction",
        "zero_frac",
    ),
    "throughput": (
        "throughput",
        "samples_per_sec",
        "samples_per_second",
        "items_per_second",
        "tokens_per_sec",
        "tokens_per_second",
    ),
}


def _metric_tokens(name: str) -> Tuple[str, ...]:
    parts = re.split(r"[.\s:/\\_-]+", name.lower())
    return tuple(part for part in parts if part)


def canonicalize_metrics(*sources: Mapping[str, Any] | None) -> Dict[str, Any]:
    """
    Merge arbitrary metric dictionaries and expose canonical keys used by the dashboard.

    External integrations (Lightning, Accelerate, etc.) often surface metrics with
    framework-specific names. This helper collapses a list of such dictionaries into
    the subset OnTheFly understands (accuracy, grad_norm, lr, ...), applying simple
    alias heuristics so callers do not need to duplicate that logic.
    """
    merged: Dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key, value in source.items():
            if value is None:
                continue
            merged[str(key)] = value

    resolved: Dict[str, Any] = {}
    for canonical, aliases in _METRIC_ALIAS_MAP.items():
        for alias in aliases:
            alias_lower = alias.lower()
            for key, value in merged.items():
                key_lower = key.lower()
                if key_lower == alias_lower:
                    resolved[canonical] = value
                    break
                tokens = _metric_tokens(key_lower)
                alias_tokens = _metric_tokens(alias_lower)
                if alias_tokens and all(token in tokens for token in alias_tokens):
                    resolved[canonical] = value
                    break
            if canonical in resolved:
                break
    return resolved


def _extract_inputs_targets(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        return None, None
    if isinstance(batch, dict):
        input_keys = ("input", "inputs", "x", "features", "data")
        target_keys = ("target", "targets", "label", "labels", "y")
        x = next((batch.get(k) for k in input_keys if k in batch), None)
        y = next((batch.get(k) for k in target_keys if k in batch), None)
        return x, y
    return None, None


def _move_like(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_like(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_like(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_like(v, device) for v in obj)
    return obj


def _model_device(model: torch.nn.Module) -> Optional[torch.device]:
    if model is None:
        return None
    for collection in (model.parameters(), model.buffers()):
        with contextlib.suppress(StopIteration):
            tensor = next(collection)
            return tensor.device
    return torch.device("cpu")


def _compute_accuracy_from_model(model: Optional[torch.nn.Module], batch: Any) -> Optional[float]:
    if model is None or batch is None:
        return None
    inputs, targets = _extract_inputs_targets(batch)
    if inputs is None or targets is None:
        return None
    device = _model_device(model)
    if device is None:
        return None
    try:
        x = _move_like(inputs, device)
        y = _move_like(targets, device)
        with torch.no_grad():
            logits = model(x)
    except Exception:
        return None
    return batch_accuracy(logits, y)


def metric_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        if torch.is_tensor(val):
            if val.numel() == 0:
                return None
            data = val.detach()
            if data.ndim > 0:
                data = data.reshape(-1)
            out = float(data.mean().item())
            return out if math.isfinite(out) else None
        if isinstance(val, (list, tuple)):
            for item in val:
                out = metric_float(item)
                if out is not None:
                    return out
            return None
        out = float(val)
        return out if math.isfinite(out) else None
    except Exception:
        return None


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


def runtime_snapshot(
    metrics: Dict[str, Any],
    batch: Any,
    step_duration: float,
    *,
    model: Optional[torch.nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    activation_tracker: Optional[ActivationZeroTracker],
    device_monitor: Optional['DeviceStatsMonitor'],
    prev_lr: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    snapshot: Dict[str, Optional[float]] = {}

    accuracy = metric_float(metrics.get("accuracy"))
    if accuracy is None:
        accuracy = _compute_accuracy_from_model(model, batch)
    snapshot["accuracy"] = accuracy

    grad_norm = metric_float(metrics.get("grad_norm"))
    if grad_norm is None and model is not None:
        try:
            grad_norm = float(_grad_norm(model))
        except Exception:
            grad_norm = None
    snapshot["grad_norm"] = grad_norm

    lr = metric_float(metrics.get("lr"))
    if lr is None and optimizer is not None:
        lr = current_learning_rate(optimizer)
    if lr is None and scheduler is not None:
        try:
            if hasattr(scheduler, "get_last_lr"):
                last_lr_seq = scheduler.get_last_lr()
                if isinstance(last_lr_seq, (list, tuple)):
                    last_vals = [float(v) for v in last_lr_seq if math.isfinite(float(v))]
                    if last_vals:
                        lr = float(sum(last_vals) / len(last_vals))
                elif last_lr_seq is not None:
                    val = float(last_lr_seq)
                    if math.isfinite(val):
                        lr = val
        except Exception:
            pass
    if lr is None and prev_lr is not None:
        lr = float(prev_lr)
    snapshot["lr"] = lr

    weight = metric_float(metrics.get("weight_norm"))
    if weight is None and model is not None:
        weight = weight_norm(model)
    snapshot["weight_norm"] = weight

    zero_frac = metric_float(metrics.get("activation_zero_frac"))
    if zero_frac is None and activation_tracker is not None:
        try:
            zero_frac = activation_tracker.pop_recent()
        except Exception:
            zero_frac = None
    snapshot["activation_zero_frac"] = zero_frac

    throughput = metric_float(metrics.get("throughput"))
    if throughput is None:
        batch_size = estimate_batch_size(batch)
        if batch_size and step_duration > 0:
            throughput = float(batch_size) / max(step_duration, 1e-9)
    snapshot["throughput"] = throughput

    mem_vram = None
    gpu_util = None
    if device_monitor is not None:
        try:
            mem_vram, gpu_util = device_monitor.snapshot()
        except Exception:
            mem_vram, gpu_util = None, None
    snapshot["mem_vram"] = mem_vram
    snapshot["gpu_util"] = gpu_util

    return snapshot


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
