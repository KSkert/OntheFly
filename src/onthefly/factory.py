from __future__ import annotations
import copy
import threading
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import torch

_DEFAULT_STRIP_ATTRS = {
    "logger",
    "_logger",
    "loggers",
    "_loggers",
    "trainer",
    "_trainer",
    "callbacks",
    "_callbacks",
    "_logger_connector",
    "_callback_connector",
    "_checkpoint_connector",
    "_progress_bar_callback",
    "_fabric",
    "_strategy",
    "_results",
}

_DYNAMIC_STRIP_TOKENS = (
    "logger",
    "summarywriter",
    "tensorboard",
    "writer",
    "trainer",
    "callback",
    "connector",
    "profiler",
    "fabric",
    "strategy",
    "accelerator",
    "loop",
    "checkpoint",
)

_MODULE_STRIP_TOKENS = (
    "lightning",
    "tensorboard",
    "wandb",
)

_LOCK_TYPES = tuple({type(threading.Lock()), type(threading.RLock())})


def _deepcopy_lock_passthrough(_obj, _memo):
    return None


@contextmanager
def _patched_deepcopy_for_locks():
    dispatch = getattr(copy, "_deepcopy_dispatch", None)
    if dispatch is None:
        yield
        return
    replaced: Dict[type, Any] = {}
    try:
        for lock_type in _LOCK_TYPES:
            prev = dispatch.get(lock_type)
            replaced[lock_type] = prev
            dispatch[lock_type] = _deepcopy_lock_passthrough
        yield
    finally:
        for lock_type, prev in replaced.items():
            if prev is None:
                dispatch.pop(lock_type, None)
            else:
                dispatch[lock_type] = prev


def _restore_attrs(target: Any, stripped: Dict[str, Any]) -> None:
    """Put sanitized attributes back onto the source module."""
    namespace = getattr(target, "__dict__", None)
    for name, value in stripped.items():
        try:
            if isinstance(namespace, dict):
                namespace[name] = value
            else:
                setattr(target, name, value)
        except Exception:
            pass


def _strip_lock_prone_attrs(model: Any) -> Dict[str, Any]:
    """
    Remove attachments that commonly carry threading locks (TensorBoard loggers,
    live Trainer references, etc.) so deepcopy does not trip over them.
    """
    namespace = getattr(model, "__dict__", None)
    if not isinstance(namespace, dict):
        return {}
    stripped: Dict[str, Any] = {}

    def _strip(name: str) -> None:
        value = namespace.get(name)
        if value is None:
            return
        stripped[name] = value
        namespace[name] = None

    for name in list(namespace.keys()):
        if name in _DEFAULT_STRIP_ATTRS:
            _strip(name)

    for name, value in list(namespace.items()):
        if value is None or name in stripped:
            continue
        if isinstance(value, _LOCK_TYPES):
            _strip(name)
            continue
        value_type = type(value)
        type_name = (getattr(value_type, "__qualname__", "") or getattr(value_type, "__name__", "") or "").lower()
        module_name = (getattr(value_type, "__module__", "") or "").lower()
        haystack = " ".join(filter(None, (name.lower(), type_name, module_name)))
        if any(tok in haystack for tok in _DYNAMIC_STRIP_TOKENS):
            _strip(name)
            continue
        if any(tok in module_name for tok in _MODULE_STRIP_TOKENS):
            _strip(name)

    return stripped


def _ensure_module(obj: Any) -> torch.nn.Module:
    if isinstance(obj, torch.nn.Module):
        return obj
    raise TypeError(
        "model_factory must return a torch.nn.Module; "
        f"got {type(obj).__name__!s} instead."
    )


def _normalize_user_factory(user_factory: Any) -> Callable[[], Any] | None:
    if user_factory is None:
        return None
    if callable(user_factory):
        return lambda: _ensure_module(user_factory())

    def _wrap(fn: Callable[..., Any], args: Iterable[Any] | None = None, kwargs: Mapping[str, Any] | None = None):
        args_tuple = tuple(args or ())
        kwargs_dict = dict(kwargs or {})
        return lambda fn=fn, args=args_tuple, kwargs=kwargs_dict: _ensure_module(fn(*args, **kwargs))

    if isinstance(user_factory, (list, tuple)) and user_factory:
        fn = user_factory[0]
        if not callable(fn):
            return None
        args = ()
        kwargs = {}
        if len(user_factory) >= 2:
            candidate = user_factory[1]
            if candidate is not None:
                args = tuple(candidate) if isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes)) else (candidate,)
        if len(user_factory) >= 3:
            candidate_kwargs = user_factory[2]
            kwargs = dict(candidate_kwargs) if isinstance(candidate_kwargs, Mapping) else {}
        return _wrap(fn, args, kwargs)

    if isinstance(user_factory, Mapping):
        fn = (
            user_factory.get("factory")
            or user_factory.get("fn")
            or user_factory.get("callable")
        )
        if not callable(fn):
            return None
        args = user_factory.get("args")
        if args is not None and not isinstance(args, Iterable):
            args = (args,)
        kwargs = user_factory.get("kwargs")
        if kwargs is not None and not isinstance(kwargs, Mapping):
            kwargs = {}
        return _wrap(fn, args, kwargs)

    return None

def _build_model_factory(model, user_factory: Callable | None = None):
    """
    Build a robust factory that can respawn a model instance without requiring
    original constructor arguments. Prefers model-provided 'factory/build/new' if present,
    then tries zero-arg ctor, finally deepcopy as a fallback.
    """
    normalized = _normalize_user_factory(user_factory)
    if normalized is not None:
        return normalized

    for name in ("factory", "build", "make", "new", "spawn"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                import inspect
                sig = inspect.signature(fn)
                if all(
                    p.default != inspect._empty or
                    p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                    for p in sig.parameters.values()
                ):
                    return lambda fn=fn: fn()
            except Exception:
                return lambda fn=fn: fn()

    def _default_factory(model=model):
        try:
            return _ensure_module(type(model)())
        except Exception:
            pass

        stripped: Dict[str, Any] = {}
        try:
            return _ensure_module(copy.deepcopy(model))
        except TypeError as exc:
            if "_thread.lock" not in str(exc):
                raise
            stripped = _strip_lock_prone_attrs(model)
            with _patched_deepcopy_for_locks():
                return _ensure_module(copy.deepcopy(model))
        finally:
            if stripped:
                _restore_attrs(model, stripped)

    return _default_factory
