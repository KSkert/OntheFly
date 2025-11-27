# src/onthefly/data_explorer.py
from __future__ import annotations
import os
import tempfile
import warnings
import inspect
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
    print(f"[otf] _call_batch_loss ENTER loss_fn={type(loss_fn).__name__} batch={_brief(batch)}", flush=True)

    # -------------------------
    # 1) Try cached pattern first
    # -------------------------
    cached = getattr(loss_fn, "_otf_batch_call_cfg", None)
    if cached is not None:
        label, use_model, star_batch = cached
        print(f"[otf] _call_batch_loss using cached pattern {label}", flush=True)

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
            print(f"[otf] _call_batch_loss SUCCESS (cached {label}) => out={_brief(out)}", flush=True)
            return out
        except TypeError as exc:
            print(f"[otf] _call_batch_loss cached pattern TypeError: {exc}; clearing cache and falling back", flush=True)
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
            print(f"[otf] _call_batch_loss trying {label} args_len={len(args)}", flush=True)
            out = loss_fn(*args)
            print(f"[otf] _call_batch_loss SUCCESS {label} => out={_brief(out)}", flush=True)

            # 🔥 NEW: remember this pattern for future calls
            use_model, star_batch = meta
            try:
                setattr(loss_fn, "_otf_batch_call_cfg", (label, use_model, star_batch))
                print(f"[otf] _call_batch_loss cached pattern {label} as _otf_batch_call_cfg", flush=True)
            except Exception as exc:
                print(f"[otf] _call_batch_loss WARNING: could not cache pattern: {type(exc).__name__}: {exc}", flush=True)

            return out
        except TypeError as exc:
            print(f"[otf] _call_batch_loss TypeError on {label}: {exc}", flush=True)
            last_exc = exc
            continue

    print(f"[otf] _call_batch_loss FAIL (raising last TypeError): {last_exc}", flush=True)
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
    print("[otf] start of compute_embeddings")
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
    print("[otf] end of compute_embeddings")
    return np.concatenate(embs, axis=0) if embs else np.zeros((0, 8))


def cluster_embeddings(embs: np.ndarray, k: int = 10) -> Dict[str, Any]:
    print(
        f"[otf][debug] cluster_embeddings: type={type(embs)} shape={getattr(embs,'shape',None)} dtype={getattr(embs,'dtype',None)}",
        flush=True,
    )

    if hasattr(embs, "shape"):
        n = int(embs.shape[0]) if len(embs.shape) > 0 else 0
        print(f"[otf][debug] cluster_embeddings: n={n}", flush=True)
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

    print("[otf] compute_per_sample_losses ENTER", flush=True)
    print(
        f"[otf] args: device={device!r} batch_size={batch_size} indices_len={(len(indices) if indices is not None else None)} "
        f"mirror_train_semantics={mirror_train_semantics} amp_enabled={amp_enabled} materialize={materialize} "
        f"data_loader={'yes' if data_loader is not None else 'no'}",
        flush=True,
    )
    print(f"[otf] torch.cuda.is_available={torch.cuda.is_available()} model.training={getattr(model,'training',None)}", flush=True)
    print(f"[otf] criterion={criterion} criterion_type={type(criterion)} callable={callable(criterion)}", flush=True)

    is_batch_aware_flag = bool(getattr(criterion, "_otf_uses_batch", False))

    print(f"[otf] global is_batch_aware_flag(by attr/type)={is_batch_aware_flag}", flush=True)

    device = torch.device(device)
    print(f"[otf] device parsed => {device} (type={device.type})", flush=True)

    print("[otf] calling model.to(device)", flush=True)
    model.to(device)
    print("[otf] returned from model.to(device)", flush=True)

    # Build the iteration dataset/loader and the *base* index mapping we’ll consume in order.
    base_indices: Optional[List[int]] = None
    loader: DataLoader

    print("[otf] building loader...", flush=True)
    if data_loader is not None and not indices:
        print("[otf] using provided data_loader (and indices is None/empty)", flush=True)
        loader = data_loader
        try:
            ds = loader.dataset
            print(f"[otf] loader.dataset => {_desc(ds)}", flush=True)
        except Exception as exc:
            print(f"[otf] loader.dataset threw: {type(exc).__name__}: {exc}", flush=True)
            print("[otf] traceback:\n" + traceback.format_exc(), flush=True)
            ds = dataset
        base_indices = _extract_sampler_indices(loader)
        print(
            f"[otf] base_indices extracted from existing loader: {'None' if base_indices is None else 'len='+str(len(base_indices))}",
            flush=True,
        )
    else:
        if indices is not None and len(indices) > 0:
            print(f"[otf] creating Subset(dataset, indices) with len(indices)={len(indices)}", flush=True)
            ds = Subset(dataset, indices)
            base_indices = [int(i) for i in indices]
            print(f"[otf] base_indices set from indices: len={len(base_indices)}", flush=True)
        else:
            print("[otf] using dataset directly (no indices subset)", flush=True)
            ds = dataset
        print(f"[otf] constructing DataLoader(drop_last=True, shuffle=False) batch_size={batch_size}", flush=True)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True,
        )
        print(f"[otf] DataLoader constructed: {loader}", flush=True)

    chunk_hint = min(max(1024, batch_size * 4), 65536)
    print(f"[otf] chunk_hint={chunk_hint}", flush=True)
    loss_spool = ChunkedArraySpool(chunk_size=chunk_hint, typecode="f", value_cast=float)
    index_spool = ChunkedArraySpool(chunk_size=chunk_hint, typecode="q", value_cast=int)
    print("[otf] spools created", flush=True)

    cursor = 0  # points into base_indices
    print(f"[otf] cursor initialized to {cursor}", flush=True)

    if amp_enabled and ("cuda" in str(device).lower()):
        print("[otf] amp_ctx = torch.cuda.amp.autocast (enabled)", flush=True)
        amp_ctx = torch.cuda.amp.autocast
    else:
        print("[otf] amp_ctx = no-op (disabled)", flush=True)

        @contextmanager
        def amp_ctx():
            yield

    was_training = model.training
    print(f"[otf] was_training captured as {was_training}", flush=True)

    if mirror_train_semantics:
        print("[otf] using _TrainLikeCtx(model)", flush=True)
        outer_ctx = _TrainLikeCtx(model)
    else:
        print("[otf] using inference_mode + model.eval()", flush=True)
        outer_ctx = ExitStack()
        model.eval()
        outer_ctx.enter_context(torch.inference_mode())

    @contextmanager
    def _tmp_attr(obj, name, value):
        had = hasattr(obj, name)
        old = getattr(obj, name, None)
        print(f"[otf] _tmp_attr enter: obj={type(obj).__name__} name={name!r} had={had} old={old!r} new={value!r}", flush=True)
        try:
            if had:
                setattr(obj, name, value)
            yield had
        finally:
            if had:
                try:
                    setattr(obj, name, old)
                    print(f"[otf] _tmp_attr restore: name={name!r} restored_to={old!r}", flush=True)
                except Exception as exc:
                    print(f"[otf] _tmp_attr restore failed: {type(exc).__name__}: {exc}", flush=True)

    def _consume_batch_indices(n: int):
        nonlocal cursor
        print(f"[otf] _consume_batch_indices(n={n}) cursor_before={cursor} base_indices={'None' if base_indices is None else 'len='+str(len(base_indices))}", flush=True)
        if base_indices is None:
            start = cursor
            index_spool.extend(range(start, start + n))
            print(f"[otf] _consume_batch_indices wrote synthetic indices [{start}, {start+n})", flush=True)
        else:
            batch_idx = base_indices[cursor:cursor + n]
            index_spool.extend(int(i) for i in batch_idx)
            print(f"[otf] _consume_batch_indices wrote mapped indices slice len={len(batch_idx)}", flush=True)
        cursor += n
        print(f"[otf] _consume_batch_indices cursor_after={cursor}", flush=True)

    using_cuda = torch.cuda.is_available() and device.type == "cuda"
    print(f"[otf] using_cuda={using_cuda}", flush=True)

    result_losses: Union[List[float], ChunkedArraySpool]
    result_indices: Union[List[int], ChunkedArraySpool]

    print("[otf] about to enter BIG TRY (outer)", flush=True)
    try:
        with outer_ctx:
            print("[otf] entered outer_ctx", flush=True)

            for batch_num, batch in enumerate(loader):
                print(f"[otf] --- BATCH LOOP START batch_num={batch_num} ---", flush=True)
                print(f"[otf] raw batch type={type(batch).__name__} desc={_desc(batch)}", flush=True)

                if should_stop and should_stop():
                    print("[otf] should_stop triggered => returning early [] []", flush=True)
                    return [], []

                if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                    print(f"[otf] BAD BATCH FORMAT: type={type(batch).__name__} value_desc={_desc(batch)}", flush=True)
                    raise RuntimeError("Unexpected batch format; expected (inputs, targets, *...).")

                logits_cache: Optional[torch.Tensor] = None
                raw_value: Any = None
                per_sample_tensor: Optional[torch.Tensor] = None
                loss_vec_tensor: Optional[torch.Tensor] = None
                batch_size_now = 0
                elems: List[Any] = []
                rest_batch: List[Any] = []
                candidates: List[Any] = []

                try:
                    print("[otf] move_batch_like(batch, device) about to run", flush=True)
                    batch = move_batch_like(batch, device)
                    print("[otf] move_batch_like finished", flush=True)
                    print(f"[otf] moved batch desc={_desc(batch)}", flush=True)

                    print("[otf] estimate_batch_size(batch) about to run", flush=True)
                    batch_size_now = estimate_batch_size(batch)
                    print(f"[otf] estimate_batch_size(batch) => {batch_size_now}", flush=True)

                    if batch_size_now is None:
                        print("[otf] batch_size_now is None, falling back to _first_tensor(batch)", flush=True)
                        t0 = _first_tensor(batch)
                        print(f"[otf] _first_tensor(batch) => {_desc(t0)}", flush=True)
                        if torch.is_tensor(t0) and t0.ndim >= 1:
                            batch_size_now = int(t0.shape[0])
                            print(f"[otf] inferred batch_size_now from first tensor => {batch_size_now}", flush=True)
                        else:
                            batch_size_now = 0
                            print("[otf] could not infer batch_size_now; set to 0", flush=True)

                    # PATCH (A): batch-aware detection + probe (works even if wrapper forgot _otf_uses_batch).
                    is_batch_aware_flag = callable(criterion) and (
                        (not isinstance(criterion, nn.Module)) or bool(getattr(criterion, "_otf_uses_batch", False))
                    )
                    print(f"[otf] is_batch_aware_flag(by attr/type)={is_batch_aware_flag}", flush=True)

                    batch_loss_probe: Any = None
                    is_batch_aware = is_batch_aware_flag

                    if callable(criterion) and not is_batch_aware:
                        print("[otf] attempting batch-aware auto-detect: probing _call_batch_loss(criterion, model, batch)", flush=True)
                        try:
                            with amp_ctx():
                                batch_loss_probe = _apply_loss_with_warning_guard(lambda: _call_batch_loss(criterion, model, batch))
                            print(f"[otf] batch-aware auto-detect PROBE SUCCESS => {_desc(batch_loss_probe)}", flush=True)
                            is_batch_aware = True
                            if not is_batch_aware_flag:
                                print("[otf] marking criterion as batch-aware (_otf_uses_batch=True)", flush=True)
                                try:
                                    setattr(criterion, "_otf_uses_batch", True)
                                except Exception as exc:
                                    print(f"[otf] WARNING: could not set _otf_uses_batch on criterion: {type(exc).__name__}: {exc}", flush=True)
                                is_batch_aware_flag = True
                        except TypeError as exc:
                            print(f"[otf] batch-aware auto-detect PROBE TypeError => treating as non-batch-aware: {exc}", flush=True)
                            batch_loss_probe = None
                        except Exception as exc:
                            # If we got here, the callable accepted a batch-aware signature
                            # but crashed for a real reason. Don't misclassify.
                            print(f"[otf] batch-aware auto-detect PROBE Exception (real failure): {type(exc).__name__}: {exc}", flush=True)
                            print("[otf] probe traceback:\n" + traceback.format_exc(), flush=True)
                            raise

                    print(f"[otf] is_batch_aware(final)={is_batch_aware}", flush=True)

                    if is_batch_aware:
                        if batch_size_now <= 0:
                            print("[otf] ERROR: batch-aware loss but batch_size_now <= 0", flush=True)
                            raise RuntimeError("Could not infer batch size for batch-aware loss_fn!")

                        if batch_loss_probe is not None:
                            raw = batch_loss_probe
                            print("[otf] batch-aware loss: using batch_loss_probe (no re-call)", flush=True)
                        else:
                            print("[otf] batch-aware loss: calling _call_batch_loss inside _apply_loss_with_warning_guard", flush=True)
                            with amp_ctx():
                                raw = _apply_loss_with_warning_guard(lambda: _call_batch_loss(criterion, model, batch))

                        print(f"[otf] batch-aware raw type={type(raw).__name__} desc={_desc(raw)}", flush=True)

                        raw = _extract_loss_value(raw)
                        print(f"[otf] batch-aware extracted loss value type={type(raw).__name__} desc={_desc(raw)}", flush=True)

                        vec = _coerce_to_per_sample(raw, batch_size_now, device)
                        print(f"[otf] batch-aware coerced per-sample vec => {_desc(vec)}", flush=True)

                        loss_spool.extend(vec.detach().cpu().tolist())
                        print(f"[otf] batch-aware wrote {batch_size_now} losses to spool (len(loss_spool)={len(loss_spool)})", flush=True)

                        _consume_batch_indices(batch_size_now)
                        print(f"[otf] batch-aware consumed indices (len(index_spool)={len(index_spool)})", flush=True)
                        print(f"[otf] --- BATCH LOOP END batch_num={batch_num} (batch-aware) ---", flush=True)
                        continue

                    print("[otf] non-batch-aware path: elems=list(batch)", flush=True)
                    elems = list(batch)
                    print(f"[otf] elems len={len(elems)} elems[0]={_desc(elems[0])} elems[1]={_desc(elems[1])}", flush=True)

                    x_raw, y = elems[0], elems[1]
                    print(f"[otf] x_raw={_desc(x_raw)} y={_desc(y)}", flush=True)

                    rest_batch = elems[2:]
                    print(f"[otf] rest_batch len={len(rest_batch)} rest_batch[0]={_desc(rest_batch[0]) if rest_batch else 'N/A'}", flush=True)

                    print("[otf] building candidates via model_input_candidates(...)", flush=True)
                    candidates = model_input_candidates(model, x_raw, rest_batch)
                    print(f"[otf] candidates built: len={len(candidates)}", flush=True)
                    for ci, cand in enumerate(candidates[:10]):
                        print(f"[otf] cand[{ci}] => {_desc(cand)}", flush=True)
                    if len(candidates) > 10:
                        print(f"[otf] (candidates truncated in log; total={len(candidates)})", flush=True)

                    last_exc: Optional[Exception] = None
                    processed = False
                    skip_batch = False

                    for cand_i, x in enumerate(candidates):
                        # Important correctness + debug: do not reuse logits across different candidate inputs.
                        logits_cache = None
                        print(f"[otf] --- CANDIDATE LOOP cand_i={cand_i} (logits_cache RESET) x={_desc(x)} ---", flush=True)

                        print("[otf] estimate_batch_size(x) about to run", flush=True)
                        batch_size_now = estimate_batch_size(x)
                        print(f"[otf] estimate_batch_size(x) => {batch_size_now}", flush=True)

                        if batch_size_now is None:
                            print("[otf] batch_size_now None from x; trying y", flush=True)
                            batch_size_now = estimate_batch_size(y)
                            print(f"[otf] estimate_batch_size(y) => {batch_size_now}", flush=True)

                        if batch_size_now is None:
                            print("[otf] ERROR: could not infer batch size from x or y", flush=True)
                            raise RuntimeError("Could not infer batch size from inputs/targets")

                        expects_input = bool(getattr(criterion, "_expects_input", False))
                        print(f"[otf] expects_input={expects_input} (criterion._expects_input)", flush=True)

                        print("[otf] defining _obtain_logits()", flush=True)

                        def _obtain_logits() -> torch.Tensor:
                            nonlocal logits_cache
                            if logits_cache is not None:
                                print("[otf] _obtain_logits: cache HIT", flush=True)
                                return logits_cache

                            print("[otf] _obtain_logits: cache MISS", flush=True)
                            print(f"[otf] _obtain_logits: about to compute out; x_desc={_desc(x)}", flush=True)

                            try:
                                if expects_input:
                                    print("[otf] _obtain_logits: expects_input True => out = x (no model forward)", flush=True)
                                    out = x
                                else:
                                    print("[otf] _obtain_logits: expects_input False => calling model(x)", flush=True)
                                    out_model = model(x)
                                    print(f"[otf] _obtain_logits: model(x) returned type={type(out_model).__name__} desc={_desc(out_model)}", flush=True)
                                    out = ensure_tensor_output(out_model)
                                    print(f"[otf] _obtain_logits: ensure_tensor_output(...) => {_desc(out)}", flush=True)
                            except Exception as exc:
                                print(f"[otf] _obtain_logits: EXCEPTION type={type(exc).__name__} msg={exc}", flush=True)
                                print("[otf] _obtain_logits: traceback:\n" + traceback.format_exc(), flush=True)
                                raise

                            logits_cache = out
                            print("[otf] _obtain_logits: cache set", flush=True)
                            return out

                        try:
                            with amp_ctx():
                                print("[otf] calling _obtain_logits() probe", flush=True)
                                _probe = _obtain_logits()
                                print(f"[otf] _obtain_logits() probe succeeded: {_desc(_probe)}", flush=True)
                        except Exception as exc:
                            print(f"[otf] exception during _obtain_logits probe; type={type(exc).__name__} msg={exc}", flush=True)
                            if should_retry_model_input(exc):
                                print("[otf] should_retry_model_input=True => trying next candidate", flush=True)
                                last_exc = exc
                                continue
                            print("[otf] should_retry_model_input=False => re-raising", flush=True)
                            raise

                        print("[otf] right before logits = _obtain_logits()", flush=True)
                        logits = _obtain_logits()
                        print(f"[otf] right after logits = _obtain_logits() logits={_desc(logits)}", flush=True)

                        print("[otf] computing fallback_target = _first_tensor_matching_shape(batch, logits.shape)", flush=True)
                        fallback_target = _first_tensor_matching_shape(batch, logits.shape)
                        print(f"[otf] fallback_target from batch => {_desc(fallback_target)}", flush=True)

                        if fallback_target is None:
                            print("[otf] fallback_target None; trying _first_tensor_matching_shape(x, logits.shape)", flush=True)
                            fallback_target = _first_tensor_matching_shape(x, logits.shape)
                            print(f"[otf] fallback_target from x => {_desc(fallback_target)}", flush=True)

                        prefer_shape = _prefers_shape_match(criterion)
                        print(f"[otf] prefer_shape={prefer_shape} (criterion class={type(criterion).__name__})", flush=True)

                        target_candidates: List[Any] = []
                        if prefer_shape and fallback_target is not None:
                            target_candidates.append(fallback_target)
                            print("[otf] target_candidates add: fallback_target (prefer_shape)", flush=True)
                        if y is not None:
                            target_candidates.append(y)
                            print("[otf] target_candidates add: y", flush=True)
                        if (not prefer_shape) and fallback_target is not None and not any(fallback_target is cand for cand in target_candidates):
                            target_candidates.append(fallback_target)
                            print("[otf] target_candidates add: fallback_target (not prefer_shape)", flush=True)

                        print(f"[otf] target_candidates final len={len(target_candidates)}", flush=True)
                        for ti, tc in enumerate(target_candidates):
                            print(f"[otf] target_candidate[{ti}] => {_desc(tc)}", flush=True)

                        for tgt_i, target_option in enumerate(target_candidates):
                            print(f"[otf] --- TARGET LOOP tgt_i={tgt_i} target_option={_desc(target_option)} ---", flush=True)
                            target_value = target_option
                            if target_value is None:
                                print("[otf] target_value is None; continue", flush=True)
                                continue

                            print("[otf] normalizing target_value to tensor/dtype/device if possible", flush=True)
                            try:
                                if torch.is_tensor(target_value):
                                    print(f"[otf] target_value is tensor pre: {_desc(target_value)}", flush=True)
                                    if target_value.device != logits.device:
                                        print("[otf] moving target_value to logits.device", flush=True)
                                        target_value = target_value.to(device=logits.device)
                                    if (
                                        target_value.dtype.is_floating_point
                                        and logits.dtype.is_floating_point
                                        and target_value.dtype != logits.dtype
                                    ):
                                        print("[otf] casting target_value dtype to logits.dtype", flush=True)
                                        target_value = target_value.to(dtype=logits.dtype)
                                    print(f"[otf] target_value tensor post: {_desc(target_value)}", flush=True)
                                else:
                                    print(f"[otf] target_value is non-tensor type={type(target_value).__name__}", flush=True)
                                    target_value = torch.as_tensor(target_value, device=logits.device)
                                    print(f"[otf] target_value as_tensor => {_desc(target_value)}", flush=True)
                                    if (
                                        target_value.dtype.is_floating_point
                                        and logits.dtype.is_floating_point
                                        and target_value.dtype != logits.dtype
                                    ):
                                        print("[otf] casting as_tensor target_value dtype to logits.dtype", flush=True)
                                        target_value = target_value.to(dtype=logits.dtype)
                                    print(f"[otf] target_value as_tensor post => {_desc(target_value)}", flush=True)
                            except Exception as exc:
                                print(f"[otf] target normalization EXCEPTION type={type(exc).__name__} msg={exc}", flush=True)
                                print("[otf] traceback:\n" + traceback.format_exc(), flush=True)
                                target_value = target_option

                            if prefer_shape and torch.is_tensor(target_value):
                                if tuple(target_value.shape) != tuple(logits.shape):
                                    print(
                                        f"[otf] prefer_shape=True but target.shape {tuple(target_value.shape)} != logits.shape {tuple(logits.shape)}; continue",
                                        flush=True,
                                    )
                                    continue

                            try:
                                with amp_ctx():
                                    print("[otf] entering _tmp_attr(criterion, 'reduction', 'none')", flush=True)
                                    with _tmp_attr(criterion, "reduction", "none") as could_set:
                                        print(f"[otf] _tmp_attr yielded could_set={could_set}", flush=True)

                                        preds = _obtain_logits()
                                        print(f"[otf] preds obtained => {_desc(preds)}", flush=True)

                                        if could_set:
                                            print("[otf] calling criterion(preds, target_value) with reduction possibly set", flush=True)
                                            raw_value = _apply_loss_with_warning_guard(lambda: criterion(preds, target_value))
                                            print(f"[otf] criterion call returned raw_value => {_desc(raw_value)}", flush=True)

                                            if not isinstance(raw_value, torch.Tensor):
                                                print("[otf] raw_value not a Tensor; calling criterion again (duplicate) for debug", flush=True)
                                                raw_value = _apply_loss_with_warning_guard(lambda: criterion(preds, target_value))
                                                print(f"[otf] criterion re-call raw_value => {_desc(raw_value)}", flush=True)
                                        else:
                                            print("[otf] criterion has no reduction attr; attempting per_sample_loss fallback", flush=True)
                                            try:
                                                with _tmp_attr(criterion, "reduction", "none") as could:
                                                    print(f"[otf] nested _tmp_attr yielded could={could}", flush=True)
                                                    if could:
                                                        print("[otf] criterion(preds, target_value) under nested _tmp_attr", flush=True)
                                                        raw_value = _apply_loss_with_warning_guard(lambda: criterion(preds, target_value))
                                                        print(f"[otf] criterion raw_value => {_desc(raw_value)}", flush=True)
                                                    else:
                                                        print("[otf] importing per_sample_loss and computing", flush=True)
                                                        from .utils import per_sample_loss

                                                        loss_vec_tensor = per_sample_loss(criterion, preds, target_value).reshape(-1)
                                                        print(f"[otf] per_sample_loss produced => {_desc(loss_vec_tensor)}", flush=True)
                                                        loss_values = loss_vec_tensor.detach().cpu().tolist()
                                                        print(f"[otf] writing loss_values len={len(loss_values)} (first3={loss_values[:3]})", flush=True)
                                                        loss_spool.extend(loss_values)
                                                        loss_vec_tensor = None
                                                        _consume_batch_indices(batch_size_now)
                                                        processed = True
                                                        skip_batch = True
                                                        print("[otf] per_sample_loss path processed=True skip_batch=True break target loop", flush=True)
                                                        break
                                            except Exception as exc:
                                                print(f"[otf] per_sample_loss fallback EXCEPTION type={type(exc).__name__} msg={exc}", flush=True)
                                                print("[otf] traceback:\n" + traceback.format_exc(), flush=True)
                                                from .utils import per_sample_loss

                                                loss_vec_tensor = per_sample_loss(criterion, preds, target_value).reshape(-1)
                                                print(f"[otf] per_sample_loss (outer except) produced => {_desc(loss_vec_tensor)}", flush=True)
                                                loss_values = loss_vec_tensor.detach().cpu().tolist()
                                                print(f"[otf] writing loss_values len={len(loss_values)} (first3={loss_values[:3]})", flush=True)
                                                loss_spool.extend(loss_values)
                                                loss_vec_tensor = None
                                                _consume_batch_indices(batch_size_now)
                                                processed = True
                                                skip_batch = True
                                                print("[otf] per_sample_loss outer-except path processed=True skip_batch=True break target loop", flush=True)
                                                break

                                if skip_batch:
                                    print("[otf] skip_batch True => breaking out of target loop", flush=True)
                                    break

                                processed = True
                                print("[otf] processed=True break out of target loop (criterion path)", flush=True)
                                break

                            except Exception as exc:
                                print(f"[otf] EXCEPTION during loss computation type={type(exc).__name__} msg={exc}", flush=True)
                                print("[otf] traceback:\n" + traceback.format_exc(), flush=True)

                                if should_retry_target(exc):
                                    print("[otf] should_retry_target=True => continue next target candidate", flush=True)
                                    last_exc = exc
                                    continue
                                if should_retry_model_input(exc):
                                    print("[otf] should_retry_model_input=True => break target loop and try next x candidate", flush=True)
                                    last_exc = exc
                                    break
                                print("[otf] retry flags false => re-raise", flush=True)
                                raise

                        if skip_batch:
                            print("[otf] skip_batch True => breaking out of candidate loop", flush=True)
                            break

                        if processed:
                            print("[otf] processed True => breaking out of candidate loop", flush=True)
                            break

                    print(
                        f"[otf] after candidate loop: processed={processed} skip_batch={skip_batch} last_exc={type(last_exc).__name__ if last_exc else None}",
                        flush=True,
                    )

                    if not processed:
                        if last_exc is not None:
                            print(f"[otf] not processed; raising last_exc type={type(last_exc).__name__} msg={last_exc}", flush=True)
                            raise last_exc
                        print("[otf] not processed; raising RuntimeError", flush=True)
                        raise RuntimeError("Could not prepare batch for per-sample losses.")

                    if skip_batch:
                        print("[otf] skip_batch True => continue to next batch", flush=True)
                        print(f"[otf] --- BATCH LOOP END batch_num={batch_num} (skip_batch) ---", flush=True)
                        continue

                    print(f"[otf] raw_value after processing => type={type(raw_value).__name__} desc={_desc(raw_value)}", flush=True)

                    if not isinstance(raw_value, torch.Tensor):
                        print("[otf] raw_value not tensor; converting to scalar and repeating for batch", flush=True)
                        out_scalar = float(torch.as_tensor(raw_value).item())
                        print(f"[otf] out_scalar={out_scalar}", flush=True)
                        loss_spool.extend([out_scalar] * batch_size_now)
                        print(f"[otf] wrote {batch_size_now} scalar losses; len(loss_spool)={len(loss_spool)}", flush=True)
                        _consume_batch_indices(batch_size_now)
                        raw_value = None
                        print(f"[otf] --- BATCH LOOP END batch_num={batch_num} (raw_value scalar) ---", flush=True)
                        continue

                    if raw_value.ndim == 0 or raw_value.numel() == 1:
                        print("[otf] raw_value is scalar tensor; repeating for batch", flush=True)
                        out_scalar = float(raw_value.item())
                        print(f"[otf] out_scalar={out_scalar}", flush=True)
                        loss_spool.extend([out_scalar] * batch_size_now)
                        print(f"[otf] wrote {batch_size_now} scalar-tensor losses; len(loss_spool)={len(loss_spool)}", flush=True)
                        _consume_batch_indices(batch_size_now)
                        raw_value = None
                        print(f"[otf] --- BATCH LOOP END batch_num={batch_num} (raw_value scalar tensor) ---", flush=True)
                        continue

                    print("[otf] building mask via _build_effective_mask", flush=True)
                    try:
                        mask = _build_effective_mask(criterion, y if isinstance(y, torch.Tensor) else None, raw_value.shape)
                        print(f"[otf] _build_effective_mask => {_desc(mask)}", flush=True)
                    except Exception as exc:
                        print(f"[otf] _build_effective_mask EXCEPTION type={type(exc).__name__} msg={exc}", flush=True)
                        print("[otf] traceback:\n" + traceback.format_exc(), flush=True)
                        mask = None

                    if mask is None:
                        print("[otf] mask None => using ones_like(raw_value, bool)", flush=True)
                        mask = torch.ones_like(raw_value, dtype=torch.bool)

                    print("[otf] flattening raw_value and mask", flush=True)
                    raw_flat = _flatten_nonbatch(raw_value)
                    mask_flat = _flatten_nonbatch(mask.to(raw_value.device))
                    print(f"[otf] raw_flat={_desc(raw_flat)} mask_flat={_desc(mask_flat)}", flush=True)

                    print("[otf] computing per-sample mean with mask", flush=True)
                    num_i = (raw_flat * mask_flat).sum(dim=1)
                    cnt_i = mask_flat.sum(dim=1)
                    safe_cnt = cnt_i.clamp(min=1)
                    per_sample_tensor = num_i / safe_cnt
                    per_sample_tensor = torch.where(cnt_i > 0, per_sample_tensor, torch.zeros_like(per_sample_tensor))
                    print(f"[otf] per_sample_tensor computed => {_desc(per_sample_tensor)}", flush=True)

                    print("[otf] writing per_sample_tensor to loss_spool", flush=True)
                    loss_list = per_sample_tensor.detach().cpu().tolist()
                    print(f"[otf] loss_list len={len(loss_list)} first3={loss_list[:3]}", flush=True)
                    loss_spool.extend(loss_list)
                    print(f"[otf] wrote per-sample losses; len(loss_spool)={len(loss_spool)}", flush=True)

                    _consume_batch_indices(batch_size_now)
                    print(f"[otf] wrote indices; len(index_spool)={len(index_spool)}", flush=True)

                    per_sample_tensor = None
                    raw_value = None
                    print(f"[otf] --- BATCH LOOP END batch_num={batch_num} (normal) ---", flush=True)

                finally:
                    print(f"[otf] finally cleanup for batch_num={batch_num}", flush=True)
                    try:
                        if torch.is_tensor(loss_vec_tensor):
                            print(f"[otf] cleanup loss_vec_tensor {_desc(loss_vec_tensor)}", flush=True)
                            if loss_vec_tensor.device.type == "cuda":
                                loss_vec_tensor = loss_vec_tensor.detach().cpu()
                            loss_vec_tensor = None
                    except Exception as exc:
                        print(f"[otf] cleanup loss_vec_tensor EXCEPTION {type(exc).__name__}: {exc}", flush=True)

                    try:
                        if torch.is_tensor(per_sample_tensor):
                            print(f"[otf] cleanup per_sample_tensor {_desc(per_sample_tensor)}", flush=True)
                            if per_sample_tensor.device.type == "cuda":
                                per_sample_tensor = per_sample_tensor.detach().cpu()
                            per_sample_tensor = None
                    except Exception as exc:
                        print(f"[otf] cleanup per_sample_tensor EXCEPTION {type(exc).__name__}: {exc}", flush=True)

                    try:
                        if torch.is_tensor(raw_value):
                            print(f"[otf] cleanup raw_value {_desc(raw_value)}", flush=True)
                            if raw_value.device.type == "cuda":
                                raw_value = raw_value.detach().cpu()
                            raw_value = None
                    except Exception as exc:
                        print(f"[otf] cleanup raw_value EXCEPTION {type(exc).__name__}: {exc}", flush=True)

                    try:
                        if torch.is_tensor(logits_cache):
                            print(f"[otf] cleanup logits_cache {_desc(logits_cache)}", flush=True)
                            if logits_cache.device.type == "cuda":
                                logits_cache = logits_cache.detach().cpu()
                            logits_cache = None
                    except Exception as exc:
                        print(f"[otf] cleanup logits_cache EXCEPTION {type(exc).__name__}: {exc}", flush=True)

                    try:
                        del elems
                        del rest_batch
                        del candidates
                        print("[otf] deleted elems/rest_batch/candidates", flush=True)
                    except Exception as exc:
                        print(f"[otf] del locals EXCEPTION {type(exc).__name__}: {exc}", flush=True)

                    x_raw = None
                    y = None

                    try:
                        del batch
                        print("[otf] deleted batch", flush=True)
                    except Exception as exc:
                        print(f"[otf] del batch EXCEPTION {type(exc).__name__}: {exc}", flush=True)

                    if using_cuda:
                        print("[otf] using_cuda True => torch.cuda.empty_cache()", flush=True)
                        with suppress(Exception):
                            torch.cuda.empty_cache()

            print("[otf] finished loader loop", flush=True)

        print("[otf] exited outer_ctx", flush=True)

        if not mirror_train_semantics and was_training:
            print("[otf] restoring model.train() because was_training=True and mirror_train_semantics=False", flush=True)
            model.train()

        print(f"[otf] materialize={materialize} len(loss_spool)={len(loss_spool)} len(index_spool)={len(index_spool)}", flush=True)
        if materialize:
            print("[otf] materializing spools via finish()", flush=True)
            result_losses = loss_spool.finish()
            result_indices = index_spool.finish()
            print(f"[otf] finish() done: result_losses_len={len(result_losses)} result_indices_len={len(result_indices)}", flush=True)
        else:
            print("[otf] returning spools (no materialize)", flush=True)
            result_losses = loss_spool
            result_indices = index_spool

        loss_spool = None
        index_spool = None
        print("[otf] compute_per_sample_losses RETURN", flush=True)
        return result_losses, result_indices

    finally:
        print("[otf] OUTER finally: spool cleanup", flush=True)
        if loss_spool is not None:
            print("[otf] OUTER finally: loss_spool.cleanup()", flush=True)
            loss_spool.cleanup()
        if index_spool is not None:
            print("[otf] OUTER finally: index_spool.cleanup()", flush=True)
            index_spool.cleanup()
        print("[otf] compute_per_sample_losses EXIT", flush=True)


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
