from __future__ import annotations

class _ScaledLossProxy:
    """
    Wrapper that ensures .backward() over non-scalar losses reduces to mean() by default.
    Compatible with native GradScaler.scale(loss) outputs.
    """
    def __init__(self, tensor):
        import torch
        self._t = tensor
        self._torch = torch

    def backward(self, *args, **kwargs):
        t = self._t
        if self._torch.is_tensor(t) and t.ndim != 0 and not args and "gradient" not in kwargs:
            t = t.mean()
        return t.backward(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._t, name)

class _SafeScaler:
    """
    Thin guard around torch.cuda.amp.GradScaler that preserves API when AMP is disabled.
    """
    def __init__(self, inner):
        self._inner = inner
    def scale(self, loss):
        s = self._inner.scale(loss) if hasattr(self._inner, "scale") else loss
        return _ScaledLossProxy(s)
    def unscale_(self, opt):
        return self._inner.unscale_(opt) if hasattr(self._inner, "unscale_") else None
    def step(self, opt):
        return self._inner.step(opt) if hasattr(self._inner, "step") else opt.step()
    def update(self, *a, **kw):
        return self._inner.update(*a, **kw) if hasattr(self._inner, "update") else None
    def state_dict(self):
        return self._inner.state_dict() if hasattr(self._inner, "state_dict") else {}
    def load_state_dict(self, sd):
        if hasattr(self._inner, "load_state_dict"): self._inner.load_state_dict(sd)
    def __getattr__(self, name):
        return getattr(self._inner, name)
