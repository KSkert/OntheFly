from __future__ import annotations
import random
from typing import Sequence, List

# Optional sklearn for clustering replay; we degrade gracefully if unavailable.
try:
    from sklearn.cluster import KMeans as _SKKMeans  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    _SKKMeans = None
    _HAVE_SKLEARN = False

def _sqdist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))

def _kmeans_nd(feat: List[List[float]], k_: int, iters: int = 50) -> List[int]:
    if not feat:
        return []
    d = len(feat[0])
    centers = random.sample(feat, min(k_, len(feat)))
    labels = [0] * len(feat)
    for _ in range(max(5, iters)):
        for i, v in enumerate(feat):
            labels[i] = min(range(len(centers)), key=lambda j: _sqdist(v, centers[j]))
        changed = False
        for j in range(len(centers)):
            pts = [feat[i] for i in range(len(feat)) if labels[i] == j]
            if pts:
                newc = [sum(p[t] for p in pts)/len(pts) for t in range(d)]
                if _sqdist(newc, centers[j]) > 1e-9:
                    changed = True
                centers[j] = newc
        if not changed:
            break
    return labels

def _run_kmeans(feats: List[List[float]], k: int) -> List[int]:
    if _HAVE_SKLEARN and _SKKMeans is not None:
        km = _SKKMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(feats)
        return list(map(int, labels))
    return _kmeans_nd(feats, k)
