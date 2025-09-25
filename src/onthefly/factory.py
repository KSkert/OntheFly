from __future__ import annotations
from typing import Callable

def _build_model_factory(model, user_factory: Callable | None = None):
    """
    Build a robust factory that can respawn a model instance without requiring
    original constructor arguments. Prefers model-provided 'factory/build/new' if present,
    then tries zero-arg ctor, finally deepcopy as a fallback.
    """
    if callable(user_factory):
        return user_factory

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
        import copy
        try:
            return type(model)()
        except Exception:
            return copy.deepcopy(model)
    return _default_factory
