from __future__ import annotations
import os, json, torch, random, numpy as np
from typing import Any, Dict, Optional

# not implemented yet: allow a trusted fallback to full pickle for legacy ckpts
_ALLOW_UNSAFE = os.environ.get("ONTHEFLY_ALLOW_UNSAFE_PICKLE", "1") == "1"

def _np_state_to_safe(s):
    if not s:
        return None
    algo, keys, pos, has_gauss, cached_gaussian = s
    # keys is a numpy ndarray → convert to plain list
    return {
        "algo": str(algo),
        "keys": keys.tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }

def _np_state_from_safe(o):
    if not o:
        return None
    keys = np.array(o.get("keys", []), dtype=np.uint32)
    return (
        str(o.get("algo", "MT19937")),
        keys,
        int(o.get("pos", 0)),
        int(o.get("has_gauss", 0)),
        float(o.get("cached_gaussian", 0.0)),
    )

def save_checkpoint(path: str, state: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "step": int(state["step"]),
        # NEW: read these if provided by caller (TrainMixin will pass them)
        "epoch": int(state.get("epoch", 0)),
        "last_val_loss": (
            float(state.get("last_val_loss"))
            if state.get("last_val_loss") is not None else None
        ),

        "model": state["model"].state_dict(),
        "optim": state["optimizer"].state_dict(),
        "sched": state["scheduler"].state_dict() if state.get("scheduler") else None,

        # NEW: scaler is important for mixed-precision resumption
        "scaler": (state.get("scaler").state_dict()
                   if state.get("scaler") is not None and hasattr(state.get("scaler"), "state_dict")
                   else None),

        "rng": {
            "python": list(random.getstate()),  # tuple → list
            "numpy_safe": _np_state_to_safe(np.random.get_state()),
            "torch": torch.get_rng_state().cpu(),
            "cuda": [t.cpu() for t in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
        },
        "sampler_state": _sampler_state(state.get("train_loader")),
        "meta": state.get("meta", {}),
    }

    torch.save(payload, path)
    _write_sidecar(path + ".meta.json", {"path": path, "step": int(state["step"])})
    return path

def _torch_load_compat(path: str, map_location: Any):
    # 1) Try safe path first (PyTorch ≥2.6 default)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e_safe:
        # 2) Legacy shim: allowlist NumPy’s reconstruct (still “safe” mode)
        try:
            from torch.serialization import add_safe_globals
            add_safe_globals([np.core.multiarray._reconstruct])
            return torch.load(path, map_location=map_location, weights_only=True)
        except Exception:
            pass
        # 3) FINAL fallback: full pickle, only if we explicitly allow it
        if _ALLOW_UNSAFE:
            return torch.load(path, map_location=map_location, weights_only=False)
        raise e_safe  # refuse if unsafe fallback not allowed

def load_checkpoint(path: str, state: Dict[str, Any]) -> Dict[str, Any]:
    payload = _torch_load_compat(path, map_location=state["device"])

    # weights
    state["model"].load_state_dict(payload["model"], strict=True)
    state["optimizer"].load_state_dict(payload["optim"])
    if state.get("scheduler") is not None and payload.get("sched") is not None:
        state["scheduler"].load_state_dict(payload["sched"])

    # NEW: scaler
    if state.get("scaler") is not None and payload.get("scaler") is not None:
        try:
            state["scaler"].load_state_dict(payload["scaler"])
        except Exception:
            pass

    # RNG (unchanged)
    rng = payload.get("rng") or {}
    py_state = rng.get("python")
    if py_state is not None:
        try: random.setstate(tuple(py_state))
        except Exception: pass

    if rng.get("numpy_safe") is not None:
        try:
            np.seterr(all="ignore")
            np.random.set_state(_np_state_from_safe(rng["numpy_safe"]))
        except Exception: pass
    elif rng.get("numpy") is not None:
        try:
            np.seterr(all="ignore")
            np.random.set_state(rng["numpy"])
        except Exception: pass

    try: torch.set_rng_state(rng.get("torch"))
    except Exception: pass
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        try: torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception: pass

    _restore_sampler(state.get("train_loader"), payload.get("sampler_state"))

    # NEW: hand back resume counters to the caller
    return {
        "step": int(payload.get("step", 0)),
        "epoch": int(payload.get("epoch", 0)),
        "last_val_loss": payload.get("last_val_loss", None),
    }

def _sampler_state(loader) -> Optional[Dict[str, Any]]:
    try:
        return loader.sampler.state_dict() if hasattr(loader.sampler, "state_dict") else None
    except Exception:
        return None

def _restore_sampler(loader, s):
    try:
        if loader and s and hasattr(loader.sampler, "load_state_dict"):
            loader.sampler.load_state_dict(s)
    except Exception:
        pass

def _write_sidecar(path: str, obj: Dict[str, Any]):
    try:
        with open(path, "w") as f:
            json.dump(obj, f)
    except Exception:
        pass
