# src/onthefly/data_explorer.py
from __future__ import annotations
import os
import tempfile
import warnings
import inspect
import logging
from array import array
from contextlib import contextmanager, ExitStack, suppress
from typing import Dict, Any, Optional, List, Callable, Tuple, Sequence, Iterable, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader

from .runtime_metrics import estimate_batch_size, move_batch_like, _first_tensor

try:
    from sklearn.cluster import MiniBatchKMeans
except Exception:
    MiniBatchKMeans = None


logger = logging.getLogger(__name__)


def _brief(obj: Any) -> str:
    try:
        if torch.is_tensor(obj):
            return f"Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
        if isinstance(obj, np.ndarray):
            return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
        if isinstance(obj, dict):
            keys = list(obj.keys())
            return f"dict(keys={keys[:10]}{'...' if len(keys) > 10 else ''})"
        if isinstance(obj, (list, tuple)):
            return f"{type(obj).__name__}(len={len(obj)})"
        return type(obj).__name__
    except Exception as e:
        return f"{type(obj).__name__}(brief_error={e})"


def _embedding_variants(model: nn.Module, value: Any) -> List[Any]:
    variants: List[Any] = []
    if not isinstance(value, dict):
        return variants
    owners: List[Any] = [model]
    generator = getattr(model, "generator", None)
    if generator is not None and generator is not model:
        owners.append(generator)
    for owner in owners:
        fn = getattr(owner, "embedding_model", None)
        if callable(fn):
            try:
                adapted = fn(value)
                variants.append(adapted)
            except Exception:
                pass
    return variants


def model_input_candidates(model: nn.Module, first: Any, rest: Sequence[Any]) -> List[Any]:
    """Generate ordered guesses for how to feed a batch into `model`."""
    rest_items = list(rest or [])
    candidates: List[Any] = []

    def _extend(base: Any) -> None:
        candidates.append(base)
        if rest_items:
            candidates.append((base, rest_items[0]))
            candidates.append(tuple([base, *rest_items]))

    adapted_inputs = _embedding_variants(model, first)
    for adapted in adapted_inputs:
        _extend(adapted)
    _extend(first)

    # Remove obvious duplicates while preserving order
    pruned: List[Any] = []
    seen: List[int] = []
    for cand in candidates:
        marker = id(cand)
        if marker in seen:
            continue
        pruned.append(cand)
        seen.append(marker)
    return pruned or [first]


def should_retry_model_input(exc: Exception) -> bool:
    if not isinstance(exc, (ValueError, AttributeError, TypeError)):  # pragma: no cover - defensive
        return False
    msg = str(exc).lower()
    retry_tokens = (
        "too many values to unpack",
        "not enough values to unpack",
        "has no attribute 'device'",
        "has no attribute \"device\"",
        "expected tuple",
        "expected tensor",
    )
    return any(tok in msg for tok in retry_tokens)


def ensure_tensor_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    tensor = _first_tensor(output)
    if tensor is None:
        raise RuntimeError("Model output does not contain a Tensor; cannot continue.")
    return tensor


def _first_tensor_matching_shape(obj: Any, shape: torch.Size | Tuple[int, ...]) -> Optional[torch.Tensor]:
    tgt_shape = tuple(shape)
    if torch.is_tensor(obj):
        return obj if tuple(obj.shape) == tgt_shape else None
    if isinstance(obj, dict):
        for value in obj.values():
            match = _first_tensor_matching_shape(value, tgt_shape)
            if match is not None:
                return match
    if isinstance(obj, (list, tuple)):
        for value in obj:
            match = _first_tensor_matching_shape(value, tgt_shape)
            if match is not None:
                return match
    return None


def should_retry_target(exc: Exception) -> bool:
    if not isinstance(exc, (RuntimeError, ValueError, TypeError)):
        return False
    msg = str(exc).lower()
    tokens = (
        "size of tensor",
        "must match",
        "expects size",
        "expected size",
        "shapes of x and y",
        "not broadcastable",
        "size mismatch",
        "target size",
    )
    return any(tok in msg for tok in tokens)


def _prefers_shape_match(loss_obj: Any) -> bool:
    # Elementwise losses where "same shape" is the overwhelmingly common intent.
    return isinstance(
        loss_obj,
        (nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss, nn.HuberLoss, nn.BCELoss, nn.BCEWithLogitsLoss),
    )


def _extract_loss_value(val: Any) -> Any:
    # Accept common patterns: tensor, dict with 'loss', tuple (loss, logs), etc.
    if torch.is_tensor(val):
        return val
    if isinstance(val, dict):
        for k in ("per_sample", "per_sample_loss", "losses", "loss"):
            if k in val:
                return _extract_loss_value(val[k])
        t = _first_tensor(val)
        return t if t is not None else val
    if isinstance(val, (list, tuple)):
        # e.g. (loss, metrics_dict)
        t = _first_tensor(val)
        return t if t is not None else val
    return val


def _coerce_to_per_sample(val: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(val):
        t = val
        if t.ndim == 0 or t.numel() == 1:
            return t.reshape(1).expand(batch_size)
        if t.shape[0] == batch_size:
            return t.reshape(batch_size, -1).mean(dim=1)
        if t.numel() == batch_size:
            return t.reshape(batch_size)
        raise RuntimeError(f"Loss tensor shape {tuple(t.shape)} can't be coerced to per-sample for batch_size={batch_size}")
    if isinstance(val, (float, int)):
        return torch.full((batch_size,), float(val), device=device)
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (list, tuple)) and len(val) == batch_size:
        return torch.as_tensor(val, device=device, dtype=torch.float32)
    raise RuntimeError(f"Loss value type {type(val)} can't be coerced to per-sample for batch_size={batch_size}")



def _call_batch_loss(loss_fn: Any, model: nn.Module, batch: Any) -> Any:
    logger.debug(f"[otf] _call_batch_loss ENTER loss_fn={type(loss_fn).__name__} batch={_brief(batch)}")

    # -------------------------
    # 1) Try cached pattern first
    # -------------------------
    cached = getattr(loss_fn, "_otf_batch_call_cfg", None)
    if cached is not None:
        label, use_model, star_batch = cached
        logger.debug(f"[otf] _call_batch_loss using cached pattern {label}")

        try:
            args: list[Any] = []
            if use_model:
                args.append(model)
            if star_batch:
                if isinstance(batch, (tuple, list)):
                    args.extend(batch)
                else:
                    args.append(batch)
            else:
                args.append(batch)

            out = loss_fn(*args)
            logger.debug(f"[otf] _call_batch_loss SUCCESS (cached {label}) => out={_brief(out)}")
            return out
        except TypeError as exc:
            logger.debug(f"[otf] _call_batch_loss cached pattern TypeError: {exc}; clearing cache and falling back")
            try:
                delattr(loss_fn, "_otf_batch_call_cfg")
            except Exception:
                pass  # best-effort
            cached = None  # fall through to full search

    # -------------------------
    # 2) Full pattern search (first time, or cache invalid)
    # -------------------------
    call_patterns: List[Tuple[str, Tuple[Any, ...], Tuple[bool, bool]]] = [
        ("loss_fn(batch)", (batch,), (False, False)),
        ("loss_fn(model, batch)", (model, batch), (True, False)),
    ]

    # Only allow star-batch if explicitly opted-in (or already cached)
    allow_star = bool(getattr(loss_fn, "_otf_allow_star_batch", False))
    if allow_star and isinstance(batch, (tuple, list)):
        call_patterns.extend(
            [
                ("loss_fn(*batch)", tuple(batch), (False, True)),
                ("loss_fn(model, *batch)", (model, *batch), (True, True)),
            ]
        )

    last_exc: Optional[Exception] = None
    for label, args, meta in call_patterns:
        try:
            logger.debug(f"[otf] _call_batch_loss trying {label} args_len={len(args)}")
            out = loss_fn(*args)
            logger.debug(f"[otf] _call_batch_loss SUCCESS {label} => out={_brief(out)}")

            # 🔥 NEW: remember this pattern for future calls
            use_model, star_batch = meta
            try:
                setattr(loss_fn, "_otf_batch_call_cfg", (label, use_model, star_batch))
                logger.debug(f"[otf] _call_batch_loss cached pattern {label} as _otf_batch_call_cfg")
            except Exception as exc:
                logger.debug(f"[otf] _call_batch_loss WARNING: could not cache pattern: {type(exc).__name__}: {exc}")

            return out
        except TypeError as exc:
            logger.debug(f"[otf] _call_batch_loss TypeError on {label}: {exc}")
            last_exc = exc
            continue

    logger.debug(f"[otf] _call_batch_loss FAIL (raising last TypeError): {last_exc}")
    if last_exc is None:
        raise TypeError("Could not call batch loss function with any supported signature.")
    raise last_exc


_LOSS_WARNING_TOKENS = (
    "target size",
    "input size",
    "broadcast",
    "not broadcastable",
    "shapes of x and y",
    "size mismatch",
)


def _apply_loss_with_warning_guard(fn: Callable[[], Any]) -> Any:
    """Run `fn()` while treating PyTorch size/broadcast warnings as hard errors."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=Warning)
        warnings.simplefilter("always", category=UserWarning)
        result = fn()
    for warn in caught:
        if not issubclass(warn.category, Warning):  # pragma: no cover - defensive
            continue
        msg = str(warn.message).lower()
        if any(tok in msg for tok in _LOSS_WARNING_TOKENS):
            raise RuntimeError(msg)
    return result


class ChunkedArraySpool:
    """
    Spill scalar values to a temporary file in fixed-size chunks so we never keep
    the entire stream resident in Python lists.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 8192,
        typecode: str = "f",
        value_cast: Optional[Callable[[Any], Any]] = None,
    ):
        self._chunk_size = max(1, int(chunk_size))
        self._typecode = typecode
        self._value_cast = value_cast
        self._buffer = array(typecode)
        fd, path = tempfile.mkstemp(prefix="otf_spool_", suffix=".bin")
        os.close(fd)
        self._path = path
        self._fh = open(path, "wb+")
        self._closed = False
        self._count = 0

    def __enter__(self) -> "ChunkedArraySpool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cleanup()

    def __del__(self):
        with suppress(Exception):
            self.cleanup()

    def append(self, value: Any) -> None:
        if self._value_cast is not None:
            value = self._value_cast(value)
        self._buffer.append(value)
        if len(self._buffer) >= self._chunk_size:
            self._flush()

    def extend(self, values: Iterable[Any]) -> None:
        for value in values:
            self.append(value)

    def __len__(self) -> int:
        return self._count + len(self._buffer)

    def _flush(self) -> None:
        if not self._buffer:
            return
        self._fh.write(self._buffer.tobytes())
        self._count += len(self._buffer)
        self._buffer = array(self._typecode)

    def _ensure_closed(self) -> None:
        if self._closed:
            return
        self._flush()
        self._fh.flush()
        self._fh.close()
        self._closed = True

    def iter_values(self) -> Iterable[Any]:
        """
        Yield every value that has been appended so far.
        """
        self._ensure_closed()
        if self._path is None:
            return
        read_size = max(1, self._chunk_size) * array(self._typecode).itemsize
        with open(self._path, "rb") as fh:
            while True:
                data = fh.read(read_size)
                if not data:
                    break
                chunk = array(self._typecode)
                chunk.frombytes(data)
                for value in chunk:
                    yield value

    def finish(self) -> List[Any]:
        """
        Materialize a Python list for UI consumers once heavy GPU work is done.
        """
        try:
            values = list(self.iter_values())
        finally:
            self.cleanup()
        return values

    def cleanup(self) -> None:
        """
        Close and remove the backing file. Safe to call multiple times.
        """
        if not self._closed:
            with suppress(Exception):
                self._flush()
                self._fh.flush()
                self._fh.close()
            self._closed = True
        path, self._path = self._path, None
        if path:
            with suppress(Exception):
                os.remove(path)


def _extract_sampler_indices(loader: DataLoader) -> Optional[List[int]]:
    """
    Best-effort attempt to recover the sampler order from an existing DataLoader.
    Returns None if the sampler does not expose explicit indices.
    """

    def _candidate_indices(source) -> Optional[Iterable[Any]]:
        if source is None:
            return None
        if isinstance(source, (list, tuple, range)):
            return source
        try:
            import numpy as _np  # type: ignore
        except Exception:  # pragma: no cover - numpy optional
            _np = None
        if _np is not None and isinstance(source, _np.ndarray):
            return source.tolist()
        if isinstance(source, array):
            return list(source)
        return None

    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)
    for owner in (sampler, getattr(batch_sampler, "sampler", None)):
        for attr in ("indices", "_indices", "ids"):
            seq = _candidate_indices(getattr(owner, attr, None))
            if seq is None:
                continue
            try:
                length = len(seq)  # type: ignore[arg-type]
                if length > 50_000_000:
                    return None
            except Exception:
                pass
            try:
                return [int(i) for i in seq]
            except Exception:
                return None
    return None


# -----------------------------
# Embeddings & simple clustering
# -----------------------------
def compute_embeddings(model, loader, device, hook_fn=None, max_batches=50) -> np.ndarray:
    logger.debug("[otf] start of compute_embeddings")
    device = torch.device(device)
    model.eval()
    embs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            batch = move_batch_like(batch, device)
            first, rest = (batch, []) if not isinstance(batch, (list, tuple)) else (batch[0], list(batch[1:]))

            e = None
            if hook_fn is not None:
                try:
                    e = hook_fn(model, batch)
                    e = e if isinstance(e, np.ndarray) else ensure_tensor_output(e).detach().cpu().numpy()
                except Exception as exc:
                    # Fallback to generic candidate feeding instead of crashing
                    warnings.warn(f"[otf] embedding hook failed ({exc}); falling back to default forward.")
                    e = None

            if e is None:
                candidates = model_input_candidates(model, first, rest)
                last_exc: Optional[Exception] = None
                for cand in candidates:
                    try:
                        z = model(cand)
                        z = ensure_tensor_output(z)
                        e = z.detach().cpu().numpy()
                        break
                    except Exception as exc:
                        if should_retry_model_input(exc):
                            last_exc = exc
                            continue
                        raise
                else:
                    if last_exc is not None:
                        raise last_exc
                    raise RuntimeError("Could not prepare batch for embeddings.")

            embs.append(e)
    logger.debug("[otf] end of compute_embeddings")
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 8))


def cluster_embeddings(embs: np.ndarray, k: int = 10) -> Dict[str, Any]:
    logger.debug(
        f"[otf][debug] cluster_embeddings: type={type(embs)} shape={getattr(embs,'shape',None)} dtype={getattr(embs,'dtype',None)}",
    )

    if hasattr(embs, "shape"):
        n = int(embs.shape[0]) if len(embs.shape) > 0 else 0
        logger.debug(f"[otf][debug] cluster_embeddings: n={n}")
        if n < 0 or n > 5_000_000:
            raise RuntimeError(f"[otf] absurd embedding count n={n} shape={embs.shape} dtype={getattr(embs,'dtype',None)}")

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
        self.num_batches_tracked = mod.num_batches_tracked.detach().clone() if hasattr(mod, "num_batches_tracked") else None
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
            return target != ignore
        except Exception:
            return None
    return None


def _prepare_loss_loader(
    dataset,
    collate_fn,
    batch_size: int,
    indices: Optional[List[int]],
    data_loader: Optional[DataLoader],
):
    if data_loader is not None and not indices:
        loader = data_loader
        base_indices = _extract_sampler_indices(loader)
        return loader, base_indices

    if indices is not None and len(indices) > 0:
        ds = Subset(dataset, indices)
        base_indices = [int(i) for i in indices]
    else:
        ds = dataset
        base_indices = None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return loader, base_indices


def _build_amp_context_for_losses(device: torch.device, amp_enabled: Optional[bool]):
    if amp_enabled and ("cuda" in str(device).lower()):
        return torch.cuda.amp.autocast

    @contextmanager
    def _noop_amp():
        yield

    return _noop_amp


def _criterion_batch_hint(criterion: nn.Module) -> bool:
    return callable(criterion) and (
        (not isinstance(criterion, nn.Module)) or bool(getattr(criterion, "_otf_uses_batch", False))
    )


def _detect_batch_awareness(
    criterion: nn.Module,
    model: nn.Module,
    batch,
    amp_ctx,
    base_flag: bool,
) -> Tuple[bool, Any]:
    is_batch_aware = base_flag
    batch_loss_probe: Any = None

    if callable(criterion) and not is_batch_aware:
        try:
            with amp_ctx():
                batch_loss_probe = _apply_loss_with_warning_guard(lambda: _call_batch_loss(criterion, model, batch))
            is_batch_aware = True
            if not base_flag:
                try:
                    setattr(criterion, "_otf_uses_batch", True)
                except Exception:
                    pass
        except TypeError:
            batch_loss_probe = None

    return is_batch_aware, batch_loss_probe


def _process_batch_aware_loss(
    batch_size_now: int,
    batch_loss_probe: Any,
    criterion: nn.Module,
    model: nn.Module,
    batch,
    device: torch.device,
    amp_ctx,
    loss_spool: ChunkedArraySpool,
    consume_indices: Callable[[int], None],
) -> None:
    if batch_size_now <= 0:
        raise RuntimeError("Could not infer batch size for batch-aware loss_fn!")

    if batch_loss_probe is not None:
        raw = batch_loss_probe
    else:
        with amp_ctx():
            raw = _apply_loss_with_warning_guard(lambda: _call_batch_loss(criterion, model, batch))

    raw = _extract_loss_value(raw)
    vec = _coerce_to_per_sample(raw, batch_size_now, device)
    loss_spool.extend(vec.detach().cpu().tolist())
    consume_indices(batch_size_now)


def _process_non_batch_loss(
    batch,
    model: nn.Module,
    criterion: nn.Module,
    amp_ctx,
    device: torch.device,
    loss_spool: ChunkedArraySpool,
    consume_indices: Callable[[int], None],
    tmp_attr,
) -> None:
    elems = list(batch)
    x_raw, y = elems[0], elems[1]
    rest_batch = elems[2:]
    candidates = model_input_candidates(model, x_raw, rest_batch)

    last_exc: Optional[Exception] = None
    processed = False
    skip_batch = False
    logits_cache: Optional[torch.Tensor] = None
    raw_value: Any = None
    per_sample_tensor: Optional[torch.Tensor] = None
    loss_vec_tensor: Optional[torch.Tensor] = None
    effective_batch_size = 0

    try:
        for x in candidates:
            logits_cache = None
            batch_size_now = estimate_batch_size(x)
            if batch_size_now is None:
                batch_size_now = estimate_batch_size(y)
            if batch_size_now is None:
                raise RuntimeError("Could not infer batch size from inputs/targets")

            expects_input = bool(getattr(criterion, "_expects_input", False))

            def _obtain_logits() -> torch.Tensor:
                nonlocal logits_cache
                if logits_cache is not None:
                    return logits_cache
                if expects_input:
                    out = x
                else:
                    out_model = model(x)
                    out = ensure_tensor_output(out_model)
                logits_cache = out
                return out

            try:
                with amp_ctx():
                    _ = _obtain_logits()
            except Exception as exc:
                if should_retry_model_input(exc):
                    last_exc = exc
                    continue
                raise

            logits = _obtain_logits()
            fallback_target = _first_tensor_matching_shape(batch, logits.shape)
            if fallback_target is None:
                fallback_target = _first_tensor_matching_shape(x, logits.shape)

            prefer_shape = _prefers_shape_match(criterion)
            target_candidates: List[Any] = []
            if prefer_shape and fallback_target is not None:
                target_candidates.append(fallback_target)
            if y is not None:
                target_candidates.append(y)
            if (not prefer_shape) and fallback_target is not None and not any(
                fallback_target is cand for cand in target_candidates
            ):
                target_candidates.append(fallback_target)

            for target_option in target_candidates:
                target_value = target_option
                if target_value is None:
                    continue

                try:
                    if torch.is_tensor(target_value):
                        if target_value.device != logits.device:
                            target_value = target_value.to(device=logits.device)
                        if (
                            target_value.dtype.is_floating_point
                            and logits.dtype.is_floating_point
                            and target_value.dtype != logits.dtype
                        ):
                            target_value = target_value.to(dtype=logits.dtype)
                    else:
                        target_value = torch.as_tensor(target_value, device=logits.device)
                        if (
                            target_value.dtype.is_floating_point
                            and logits.dtype.is_floating_point
                            and target_value.dtype != logits.dtype
                        ):
                            target_value = target_value.to(dtype=logits.dtype)
                except Exception:
                    target_value = target_option

                if prefer_shape and torch.is_tensor(target_value):
                    if tuple(target_value.shape) != tuple(logits.shape):
                        continue

                try:
                    with amp_ctx():
                        with tmp_attr(criterion, "reduction", "none") as could_set:
                            preds = _obtain_logits()
                            if could_set:
                                raw_value = _apply_loss_with_warning_guard(lambda: criterion(preds, target_value))
                                if not isinstance(raw_value, torch.Tensor):
                                    raw_value = _apply_loss_with_warning_guard(lambda: criterion(preds, target_value))
                            else:
                                try:
                                    with tmp_attr(criterion, "reduction", "none") as could:
                                        if could:
                                            raw_value = _apply_loss_with_warning_guard(lambda: criterion(preds, target_value))
                                        else:
                                            from .utils import per_sample_loss

                                            loss_vec_tensor = per_sample_loss(criterion, preds, target_value).reshape(-1)
                                            loss_values = loss_vec_tensor.detach().cpu().tolist()
                                            loss_spool.extend(loss_values)
                                            loss_vec_tensor = None
                                            consume_indices(batch_size_now)
                                            processed = True
                                            skip_batch = True
                                            break
                                except Exception:
                                    from .utils import per_sample_loss

                                    loss_vec_tensor = per_sample_loss(criterion, preds, target_value).reshape(-1)
                                    loss_values = loss_vec_tensor.detach().cpu().tolist()
                                    loss_spool.extend(loss_values)
                                    loss_vec_tensor = None
                                    consume_indices(batch_size_now)
                                    processed = True
                                    skip_batch = True
                                    break

                    if skip_batch:
                        break

                    processed = True
                    effective_batch_size = batch_size_now
                    break

                except Exception as exc:
                    if should_retry_target(exc):
                        last_exc = exc
                        continue
                    if should_retry_model_input(exc):
                        last_exc = exc
                        break
                    raise

            if skip_batch or processed:
                break

        if not processed:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Could not prepare batch for per-sample losses.")

        if skip_batch:
            return

        if not isinstance(raw_value, torch.Tensor):
            out_scalar = float(torch.as_tensor(raw_value).item())
            loss_spool.extend([out_scalar] * effective_batch_size)
            consume_indices(effective_batch_size)
            return

        if raw_value.ndim == 0 or raw_value.numel() == 1:
            out_scalar = float(raw_value.item())
            loss_spool.extend([out_scalar] * effective_batch_size)
            consume_indices(effective_batch_size)
            return

        try:
            mask = _build_effective_mask(criterion, y if isinstance(y, torch.Tensor) else None, raw_value.shape)
        except Exception:
            mask = None

        if mask is None:
            mask = torch.ones_like(raw_value, dtype=torch.bool)

        raw_flat = _flatten_nonbatch(raw_value)
        mask_flat = _flatten_nonbatch(mask.to(raw_value.device))

        num_i = (raw_flat * mask_flat).sum(dim=1)
        cnt_i = mask_flat.sum(dim=1)
        safe_cnt = cnt_i.clamp(min=1)
        per_sample_tensor = num_i / safe_cnt
        per_sample_tensor = torch.where(cnt_i > 0, per_sample_tensor, torch.zeros_like(per_sample_tensor))

        loss_list = per_sample_tensor.detach().cpu().tolist()
        loss_spool.extend(loss_list)
        consume_indices(effective_batch_size)

    finally:
        for tensor_val in (loss_vec_tensor, per_sample_tensor, raw_value, logits_cache):
            try:
                if torch.is_tensor(tensor_val) and tensor_val.device.type == "cuda":
                    tensor_val = tensor_val.detach().cpu()
            except Exception:
                pass
        try:
            del elems
            del rest_batch
            del candidates
        except Exception:
            pass


# -----------------------------
# Per-sample loss computation
# -----------------------------

def _extract_sampler_indices(loader: DataLoader) -> Optional[List[int]]:
    """
    Best-effort attempt to recover the sampler order from an existing DataLoader.
    Returns None if the sampler does not expose explicit indices.
    """

    def _candidate_indices(source) -> Optional[Iterable[Any]]:
        if source is None:
            return None
        if isinstance(source, (list, tuple, range)):
            return source
        try:
            import numpy as _np  # type: ignore
        except Exception:  # pragma: no cover - numpy optional
            _np = None
        if _np is not None and isinstance(source, _np.ndarray):
            return source.tolist()
        if isinstance(source, array):
            return list(source)
        return None

    sampler = getattr(loader, "sampler", None)
    batch_sampler = getattr(loader, "batch_sampler", None)
    for owner in (sampler, getattr(batch_sampler, "sampler", None)):
        for attr in ("indices", "_indices", "ids"):
            seq = _candidate_indices(getattr(owner, attr, None))
            if seq is None:
                continue
            try:
                length = len(seq)  # type: ignore[arg-type]
                if length > 50_000_000:
                    return None
            except Exception:
                pass
            try:
                return [int(i) for i in seq]
            except Exception:
                return None
    return None


# -----------------------------
# Embeddings & simple clustering
# -----------------------------
def compute_embeddings(model, loader, device, hook_fn=None, max_batches=50) -> np.ndarray:
    logger.debug("[otf] start of compute_embeddings")
    device = torch.device(device)
    model.eval()
    embs = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            batch = move_batch_like(batch, device)
            first, rest = (batch, []) if not isinstance(batch, (list, tuple)) else (batch[0], list(batch[1:]))

            e = None
            if hook_fn is not None:
                try:
                    e = hook_fn(model, batch)
                    e = e if isinstance(e, np.ndarray) else ensure_tensor_output(e).detach().cpu().numpy()
                except Exception as exc:
                    # Fallback to generic candidate feeding instead of crashing
                    warnings.warn(f"[otf] embedding hook failed ({exc}); falling back to default forward.")
                    e = None

            if e is None:
                candidates = model_input_candidates(model, first, rest)
                last_exc: Optional[Exception] = None
                for cand in candidates:
                    try:
                        z = model(cand)
                        z = ensure_tensor_output(z)
                        e = z.detach().cpu().numpy()
                        break
                    except Exception as exc:
                        if should_retry_model_input(exc):
                            last_exc = exc
                            continue
                        raise
                else:
                    if last_exc is not None:
                        raise last_exc
                    raise RuntimeError("Could not prepare batch for embeddings.")

            embs.append(e)
    logger.debug("[otf] end of compute_embeddings")
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 8))


def cluster_embeddings(embs: np.ndarray, k: int = 10) -> Dict[str, Any]:
    logger.debug(
        f"[otf][debug] cluster_embeddings: type={type(embs)} shape={getattr(embs,'shape',None)} dtype={getattr(embs,'dtype',None)}",
    )

    if hasattr(embs, "shape"):
        n = int(embs.shape[0]) if len(embs.shape) > 0 else 0
        logger.debug(f"[otf][debug] cluster_embeddings: n={n}")
        if n < 0 or n > 5_000_000:
            raise RuntimeError(f"[otf] absurd embedding count n={n} shape={embs.shape} dtype={getattr(embs,'dtype',None)}")

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
        self.num_batches_tracked = mod.num_batches_tracked.detach().clone() if hasattr(mod, "num_batches_tracked") else None
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
            return target != ignore
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
    batch_size: int = 32,
    indices: Optional[List[int]] = None,
    *,
    mirror_train_semantics: bool = False,
    amp_enabled: Optional[bool] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    materialize: bool = True,
    data_loader: Optional[DataLoader] = None,
) -> Tuple[Union[List[float], ChunkedArraySpool], Union[List[int], ChunkedArraySpool]]:
    """
    Returns either Python lists or chunked spools depending on `materialize`. When
    `materialize=False`, the caller takes ownership of the spools and must call
    `.finish()` to obtain the lists (which also cleans up temp files).
    """
    import traceback

    def _desc(obj: Any) -> str:
        try:
            if torch.is_tensor(obj):
                return f"Tensor(shape={tuple(obj.shape)}, dtype={obj.dtype}, device={obj.device})"
            if isinstance(obj, np.ndarray):
                return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
            if isinstance(obj, dict):
                return f"dict(keys={list(obj.keys())[:10]}{'...' if len(obj.keys())>10 else ''})"
            if isinstance(obj, (list, tuple)):
                return f"{type(obj).__name__}(len={len(obj)})"
            return f"{type(obj).__name__}"
        except Exception as e:
            return f"{type(obj).__name__}(desc_error={e})"


    device = torch.device(device)
    model.to(device)

    loader, base_indices = _prepare_loss_loader(
        dataset,
        collate_fn,
        batch_size=batch_size,
        indices=indices,
        data_loader=data_loader,
    )

    chunk_hint = min(max(1024, batch_size * 4), 65536)
    loss_spool = ChunkedArraySpool(chunk_size=chunk_hint, typecode="f", value_cast=float)
    index_spool = ChunkedArraySpool(chunk_size=chunk_hint, typecode="q", value_cast=int)

    cursor = 0  # points into base_indices

    amp_ctx = _build_amp_context_for_losses(device, amp_enabled)

    was_training = model.training

    if mirror_train_semantics:
        outer_ctx = _TrainLikeCtx(model)
    else:
        outer_ctx = ExitStack()
        model.eval()
        outer_ctx.enter_context(torch.inference_mode())

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
                except Exception as exc:
                    pass
    def _consume_batch_indices(n: int):
        nonlocal cursor
        if base_indices is None:
            start = cursor
            index_spool.extend(range(start, start + n))
        else:
            batch_idx = base_indices[cursor:cursor + n]
            index_spool.extend(int(i) for i in batch_idx)
        cursor += n

    using_cuda = torch.cuda.is_available() and device.type == "cuda"

    result_losses: Union[List[float], ChunkedArraySpool]
    result_indices: Union[List[int], ChunkedArraySpool]

    try:
        with outer_ctx:

            for batch_num, batch in enumerate(loader):

                if should_stop and should_stop():
                    return [], []

                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                    raise RuntimeError("Unexpected batch format; expected (inputs, targets, *...).")

                try:
                    batch = move_batch_like(batch, device)

                    batch_size_now = estimate_batch_size(batch)

                    if batch_size_now is None:
                        t0 = _first_tensor(batch)
                        if torch.is_tensor(t0) and t0.ndim >= 1:
                            batch_size_now = int(t0.shape[0])
                        else:
                            batch_size_now = 0

                    batch_hint = _criterion_batch_hint(criterion)
                    is_batch_aware, batch_loss_probe = _detect_batch_awareness(
                        criterion,
                        model,
                        batch,
                        amp_ctx,
                        batch_hint,
                    )

                    if is_batch_aware:
                        _process_batch_aware_loss(
                            batch_size_now,
                            batch_loss_probe,
                            criterion,
                            model,
                            batch,
                            device,
                            amp_ctx,
                            loss_spool,
                            _consume_batch_indices,
                        )
                        continue

                    _process_non_batch_loss(
                        batch,
                        model,
                        criterion,
                        amp_ctx,
                        device,
                        loss_spool,
                        _consume_batch_indices,
                        _tmp_attr,
                    )
                finally:
                    if using_cuda:
                        with suppress(Exception):
                            torch.cuda.empty_cache()



        if not mirror_train_semantics and was_training:
            model.train()

        if materialize:
            result_losses = loss_spool.finish()
            result_indices = index_spool.finish()
        else:
            result_losses = loss_spool
            result_indices = index_spool

        loss_spool = None
        index_spool = None
        return result_losses, result_indices

    finally:
        if loss_spool is not None:
            loss_spool.cleanup()
        if index_spool is not None:
            index_spool.cleanup()


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
    "ChunkedArraySpool",
]
