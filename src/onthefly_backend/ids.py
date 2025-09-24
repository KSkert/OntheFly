from __future__ import annotations
import os, re, hashlib

# --- filesystem-safe run-id helpers (short slug + hash) ---
_MAX_COMPONENT = 120  # below the 255-byte limit for a single path component
_slug_re = re.compile(r'[^A-Za-z0-9._-]+')

def _slug(s: str) -> str:
    s = s or ""
    return _slug_re.sub('-', s).strip('-_.') or "run"

def _short_hash(*parts: str, n: int = 10) -> str:
    h = hashlib.blake2b(("|".join(p or "" for p in parts)).encode("utf-8"), digest_size=8).hexdigest()
    return h[:n]

def _safe_component(hint: str, extra_entropy: str = "", max_len: int = _MAX_COMPONENT) -> str:
    base = _slug(hint)
    if extra_entropy:
        base = f"{base}-{_short_hash(extra_entropy)}"
    if len(base) > max_len:
        base = f"{base[:max_len-11]}-{_short_hash(base, n=10)}"
    return base

def _unique_component(root_dir: str, base: str) -> str:
    cand = base
    i = 2
    while os.path.exists(os.path.join(root_dir, cand)):
        suffix = f"-{i}"
        cut = max(1, _MAX_COMPONENT - len(suffix))
        cand = f"{base[:cut]}{suffix}"
        i += 1
    return cand
