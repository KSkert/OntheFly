# src/onthefly/snapshots.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import os
import uuid
import threading
import hashlib
import torch


@dataclass(frozen=True)
class SnapshotRef:
    """
    What we pass around as a "snapshot":
      - Either contains a CPU state_dict (fast-path for planners/workers)
      - Or a path on disk after we "spill" it (saves RAM)
    We never mutate these; when spilling we create a new one and swap it in.
    """
    owner: str                 # run id (== cfg.run_name at the time)
    version: int               # monotonic version, usually the *training step*
    token: str                 # opaque freshness key owner:version:short-uuid
    state_dict: Optional[Dict[str, torch.Tensor]] = None  # present pre-spill
    path: Optional[str] = None                              # present post-spill
    bytes_estimate: int = 0                                 # rough size of SD


class SnapshotManager:
    """
    Tiny, thread-safe broker for (owner, version) snapshots.

    Why this exists:
      - Avoid saving to disk in the hot path just to run planning/analysis.
      - Keep a small per-owner ring (defaults to 3) so we can coalesce spammy
        publishes and still not explode memory.
      - Give the worker a *freshness token* so plans tie back to the exact
        weights they were computed from (no timestamp guessing).

    Key ideas:
      - keys are (owner, version, token); tokens are unique, short, random.
      - we store the newest tokens per owner in `_order[owner]`.
      - spill() converts an in-memory snapshot (state_dict) to a file and
        replaces the in-memory ref with a path-only ref.
      - attach_session(...) lets spill() embed optional resume metadata.
    """

    def __init__(self, keep_per_owner: int = 3, default_spill_dir: Optional[str] = None):
        self._keep = max(1, int(keep_per_owner))         # ring size per owner
        self._spill_dir = default_spill_dir               # default dir if spill() called without a path
        self._by_owner: Dict[str, Dict[str, SnapshotRef]] = {}  # owner -> token -> ref
        self._order: Dict[str, list[str]] = {}            # owner -> list[token] (oldest..newest)
        self._by_token: Dict[str, SnapshotRef] = {}       # token -> ref
        self._lock = threading.Lock()
        self._session_ref: Optional[Any] = None           # optional training session (for richer ckpts)

    # ---------- lifecycle wiring ----------

    def attach_session(self, session: Any) -> None:
        """
        Give us a pointer to the live training session.
        If present, spill() will embed a best-effort "contract" describing how to resume,
        plus optional optimizer/scheduler/amp/rng state. This is *not* required to load
        weights; it's sugar to make resuming less brittle later.
        """
        self._session_ref = session

    # ---------- producers push snapshots ----------

    @torch.no_grad()
    def push_from_model(self, *, owner: str, version: int, model: torch.nn.Module) -> Tuple[str, SnapshotRef]:
        """
        Fast path for callers that have a model handy:
          - grab a *CPU* state_dict (non-blocking copy off CUDA when we can)
          - register it under (owner, version)
        Note: this captures parameters only; optimizer/etc. is added at spill() time.
        """
        sd = model.state_dict()
        cpu_sd: Dict[str, torch.Tensor] = {}
        for k, t in sd.items():
            # Copy to CPU, try to avoid sync where possible; clone to detach grads
            if t.is_cuda:
                cpu_sd[k] = t.detach().to("cpu", non_blocking=True).clone()
            else:
                cpu_sd[k] = t.detach().clone()
        bytes_est = int(sum(t.numel() * t.element_size() for t in cpu_sd.values()))
        return self.push(owner=owner, version=version, state_dict=cpu_sd, bytes_estimate=bytes_est)

    def push(
        self,
        *,
        owner: str,
        version: int,
        state_dict: Dict[str, torch.Tensor],
        bytes_estimate: Optional[int] = None
    ) -> Tuple[str, SnapshotRef]:
        """
        Register a ready CPU state_dict; returns (token, ref).

        Token format: f"{owner}:{version}:{uuid8}"
        - token is what downstream planners echo back so we can assert freshness
        - version should be a monotonic step counter (we don't enforce it here)
        """
        token = f"{owner}:{version}:{uuid.uuid4().hex[:8]}"
        ref = SnapshotRef(
            owner=owner,
            version=int(version),
            token=token,
            state_dict=state_dict,
            path=None,
            bytes_estimate=int(bytes_estimate or 0),
        )
        with self._lock:
            owner_map = self._by_owner.setdefault(owner, {})
            order = self._order.setdefault(owner, [])
            owner_map[token] = ref
            order.append(token)
            self._by_token[token] = ref

            # enforce small ring, drop oldest first
            while len(order) > self._keep:
                old_tok = order.pop(0)
                old = owner_map.pop(old_tok, None)
                if old:
                    self._by_token.pop(old_tok, None)
        return token, ref

    # ---------- lookups ----------

    def latest(self, owner: str) -> Optional[SnapshotRef]:
        """
        Handy helper for "give me the newest token for this owner".
        """
        with self._lock:
            order = self._order.get(owner) or []
            if not order:
                return None
            return self._by_owner[owner][order[-1]]

    def get(self, token: str) -> Optional[SnapshotRef]:
        with self._lock:
            return self._by_token.get(token)

    # ---------- spill to disk (checkpoint) ----------

    def spill(self, token: str, dir_or_path: Optional[str] = None) -> SnapshotRef:
        """
        Make sure the given token exists on disk:
          - If it's already spilled (ref.path set and no state_dict), we return it as-is.
          - If it's in-memory only, we write a .pt with at least {"model": state_dict}.
          - If attach_session(...) was called, we also try to add contract/opt/sched/amp/rng.
        Returns the *updated* SnapshotRef (path-only) and swaps it into the indexes.
        """
        with self._lock:
            ref = self._by_token.get(token)
            if ref is None:
                raise KeyError(f"unknown snapshot token: {token}")
            if ref.path and (ref.state_dict is None):
                return ref  # already on disk

        target = dir_or_path or self._spill_dir
        if not target:
            # caller didn't give us a path and we don't have a default
            raise ValueError("spill() requires a path or a default_spill_dir set at init")

        if os.path.isdir(target):
            # keep filenames stable + semi-readable for debugging/import
            filename = f"{ref.owner}__v{ref.version}__{ref.token.split(':')[-1]}.pt"
            path = os.path.join(target, filename)
        else:
            # explicit file path
            path = target

        # Bare minimum payload. Weights go under "model".
        blob: Dict[str, Any] = {"model": ref.state_dict}

        # If we have a live session, try to enrich the blob.
        sess = self._session_ref
        if sess is not None:
            # "contract" is best-effort; never fail a spill if this breaks
            try:
                blob["contract"] = _contract_from_session(sess)
            except Exception:
                pass
            # Optional attachments — all best-effort as well
            try:
                if getattr(sess, "optimizer", None) is not None:
                    blob["optimizer"] = sess.optimizer.state_dict()
            except Exception:
                pass
            try:
                if getattr(sess, "scheduler", None) is not None and sess.scheduler is not None:
                    blob["scheduler"] = getattr(sess.scheduler, "state_dict", lambda: {})()
            except Exception:
                pass
            try:
                scaler = getattr(sess, "scaler", None)  # could be GradScaler or custom
                if scaler is not None and hasattr(scaler, "state_dict"):
                    blob["scaler"] = scaler.state_dict()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                blob["rng"] = {
                    "torch": torch.get_rng_state().tolist(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                }
            except Exception:
                pass

        # single write; if this ever needs atomicity across files, wrap in temp+rename
        torch.save(blob, path)

        # Swap the path-only ref into our indexes (drop the heavy state_dict)
        with self._lock:
            new_ref = SnapshotRef(
                owner=ref.owner,
                version=ref.version,
                token=ref.token,
                state_dict=None,
                path=path,
                bytes_estimate=ref.bytes_estimate,
            )
            self._by_token[token] = new_ref
            if ref.owner in self._by_owner and token in self._by_owner[ref.owner]:
                self._by_owner[ref.owner][token] = new_ref
            return new_ref

    # ---------- bulk helpers used by the extension before export ----------

    def owners(self) -> list[str]:
        """
        Returns the owners we currently know about (whether in RAM or already spilled).
        Note: This is bounded by whatever producers actually pushed — if a run never
        pushed a snapshot, it won't show up here.
        """
        with self._lock:
            return list(self._by_owner.keys())

    def tokens_for_owner(self, owner: str) -> list[str]:
        """
        Return this owner's tokens in chronological order (oldest..newest).
        We return a copy so callers don't trip on concurrent modifications.
        """
        with self._lock:
            return list(self._order.get(owner, []))

    def spill_all(
        self,
        dir_or_path: str,
        latest_only: bool = True,
    ) -> list[dict]:
        """
        Core export hook used by the VS Code extension:

        Contract:
          - The extension calls `spill_all(dir=tmp, latest_only=True)` *right before* building
            a bundle to guarantee there's at least one fresh on-disk checkpoint for every
            owner that has ever published a snapshot.
          - We return a list of metadata records; the extension inserts them into SQLite
            (checkpoints table) and then its bundler copies the files into the bundle.

        Important nuances:
          - This does *not* fabricate snapshots for runs that never pushed one. If you want
            “every run has a file”, make sure the session publishes at least *one* snapshot
            per run (we do that on resume and on some events), or call your own save routine
            first (pause/save_ckpt/etc.).
          - Idempotent: if the token is already spilled, we don't re-write the file; we just
            return the existing path, so you can call this multiple times without duplicating work.

        Returns list like:
          {
            "ckpt_id": "<token>",        # use as checkpoints.ckpt_id
            "owner":   "<run_id>",
            "version": <int>,            # == step
            "step":    <int>,
            "path":    "<abs path to .pt>",
            "bytes":   <int>
          }
        """
        if not dir_or_path:
            raise ValueError("spill_all requires a directory or file path")

        # If a directory is provided, ensure it exists; spill() creates files under it.
        if os.path.isdir(dir_or_path):
            try:
                os.makedirs(dir_or_path, exist_ok=True)
            except Exception:
                # leave it to spill() to raise if needed
                pass

        records: list[dict] = []

        # Grab a stable view of owners/tokens; producers might be pushing concurrently.
        owner_list = self.owners()
        for owner in owner_list:
            tokens = self.tokens_for_owner(owner)
            if not tokens:
                continue

            target_tokens = [tokens[-1]] if latest_only else tokens

            for tok in target_tokens:
                ref = self.get(tok)
                if ref is None:
                    # token raced and got dropped; skip quietly
                    continue
                try:
                    spilled = self.spill(tok, dir_or_path)
                except Exception:
                    # Don't let a single bad write stop others — export is best-effort.
                    continue

                records.append({
                    "ckpt_id": spilled.token,
                    "owner":   spilled.owner,
                    "version": int(spilled.version),
                    "step":    int(spilled.version),  # our DB speaks "step"
                    "path":    spilled.path,
                    "bytes":   int(getattr(spilled, "bytes_estimate", 0)),
                })

        return records

    def drop_old(self, owner: str, keep: Optional[int] = None) -> None:
        """
        Manual trim hook: shrink the per-owner ring to `keep` newest snapshots.
        We rarely need this directly (the ring auto-enforces size on push), but
        it's handy after big spills if you want to free memory aggressively.
        """
        k = max(1, int(keep or self._keep))
        with self._lock:
            order = self._order.get(owner) or []
            owner_map = self._by_owner.get(owner) or {}
            while len(order) > k:
                old_tok = order.pop(0)
                owner_map.pop(old_tok, None)
                self._by_token.pop(old_tok, None)


# ---------------------- helpers ----------------------

def _qname(obj: Any) -> str:
    """Fully-qualified-ish name for debug/telemetry."""
    try:
        m = getattr(obj, "__module__", "builtins")
        q = getattr(obj, "__qualname__", None)
        if q is None:
            q = getattr(obj.__class__, "__qualname__", str(obj.__class__.__name__))
        return f"{m}.{q}"
    except Exception:
        return str(obj)


def _hash_state_schema(sd: Dict[str, torch.Tensor]) -> str:
    """
    Tiny signature to catch “same model structure?” without storing the whole thing.
    Key list + shapes + dtypes hashed into 12 hex chars. Purely informational.
    """
    sig = "|".join(f"{k}:{tuple(v.size())}:{str(v.dtype)}" for k, v in sd.items())
    return hashlib.sha1(sig.encode()).hexdigest()[:12]


def _dataset_fingerprint(ds: Any) -> Optional[Dict[str, Any]]:
    """
    Best-effort dataset fingerprint for the "contract":
      - includes class name, length, and an index sample hash when we can.
      - entirely optional and never fails the emit.
    """
    try:
        n = len(ds)  # type: ignore[call-arg]
        idxs = getattr(ds, "indices", None)
        if isinstance(idxs, (list, tuple)) and len(idxs) > 0:
            probe = [idxs[i % len(idxs)] for i in range(min(256, len(idxs)))]
            h = hashlib.sha1((",".join(map(str, probe))).encode()).hexdigest()[:12]
        else:
            h = f"N{n}"
        return {"class": _qname(ds.__class__), "len": n, "hash": h}
    except Exception:
        return None


def _contract_from_session(session: Any) -> Dict[str, Any]:
    """
    Deterministic "training contract" — describes what's needed to resume.
    NB: *weights* live under "model" and are the only required part for loading.
        The contract is guard rails + breadcrumbs; we never require it to load.
    """
    import torch as _T

    sd = session.model.state_dict()
    contract: Dict[str, Any] = {
        "schema_version": "1.0",
        "lib": "onthefly",
        "torch": _T.__version__,
        "device_flags": {
            "cuda": _T.cuda.is_available(),
            "mps": hasattr(_T.backends, "mps") and _T.backends.mps.is_available(),
        },
        "project": getattr(session.cfg, "project", ""),
        "run_name": getattr(session.cfg, "run_name", ""),
        "last_step": getattr(session, "step", 0),
        "last_epoch": getattr(session, "epoch", 0),
        "model": {
            # use the factory when present; fallback to instance typename
            "factory": _qname(getattr(session, "_model_factory", session.model)),
            "state_schema": {"keys": sorted(list(sd.keys())), "hash": _hash_state_schema(sd)},
        },
        "optimizer": None,     # filled below if we can
        "scheduler": None,     # filled below if we can
        "loss_fn": _qname(getattr(session, "raw_loss_fn", None) or getattr(session, "loss_fn", None)),
        "amp": bool("cuda" in getattr(session, "device", "cpu") and getattr(session.cfg, "amp", False)),
        "grad_clip_norm": getattr(session.cfg, "grad_clip_norm", None),
        "datasets": {
            "train": _dataset_fingerprint(getattr(session, "train_loader", None) and session.train_loader.dataset),
            "val": _dataset_fingerprint(getattr(session, "val_loader", None) and session.val_loader.dataset),
            "test": _dataset_fingerprint(getattr(session, "test_loader", None) and session.test_loader.dataset),
        },
        "seed": 42,  # we don't plumb seeds yet; include something predictable
    }

    try:
        opt = getattr(session, "optimizer", None)
        if opt is not None:
            groups = []
            for g in getattr(opt, "param_groups", []):
                groups.append(sorted([k for k in g.keys() if k not in ("params", "params_tensor")]))
            contract["optimizer"] = {"class": _qname(opt.__class__), "param_groups": groups}
    except Exception:
        pass

    try:
        sch = getattr(session, "scheduler", None)
        if sch is not None:
            contract["scheduler"] = {"class": _qname(sch.__class__)}
    except Exception:
        pass

    return contract
