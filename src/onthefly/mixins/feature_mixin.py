from __future__ import annotations
import contextlib
import random
from typing import Callable, Dict, Any, Optional, List, Tuple, Union

import numpy as np
import warnings
import torch
from torch.utils.data import DataLoader, Subset

from ..metrics_utils import _top2_margin
from ..device_utils import _noop_ctx
from ..runtime_metrics import move_batch_like
from ..data_explorer import (
    compute_per_sample_losses,
    model_input_candidates,
    should_retry_model_input,
    ensure_tensor_output,
    ChunkedArraySpool,
)


class FeatureMixin:
    """
    Helpers for computing feature matrices used by manual selection flows.
    """

    def _loss_module_for_features(self) -> torch.nn.Module:
        """Return an nn.Module-compatible criterion for feature/loss scans."""
        if isinstance(getattr(self, "raw_loss_fn", None), torch.nn.Module):
            return self.raw_loss_fn  # type: ignore[return-value]
        for attr in ("criterion", "_criterion"):
            cand = getattr(self, attr, None)
            if isinstance(cand, torch.nn.Module):
                return cand
        fn = getattr(self, "raw_loss_fn", None) or getattr(self, "loss_fn", None)
        if callable(fn):
            class _CallableLoss(torch.nn.Module):
                def __init__(self, f):
                    super().__init__()
                    self._fn = f

                def forward(self, logits, target):  # type: ignore[override]
                    return self._fn(logits, target)

            return _CallableLoss(fn)
        raise RuntimeError("No usable loss function available for per-sample analysis.")

    @contextlib.contextmanager
    def _rng_guard(self, seed: Optional[int]):
        if seed is None:
            yield
            return
        cpu_state = torch.random.get_rng_state()
        np_state = np.random.get_state()
        py_state = random.getstate()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        prev_bench = torch.backends.cudnn.benchmark
        prev_det = torch.backends.cudnn.deterministic
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            torch.random.set_rng_state(cpu_state)
            np.random.set_state(np_state)
            random.setstate(py_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            torch.backends.cudnn.benchmark = prev_bench
            torch.backends.cudnn.deterministic = prev_det

    def _model_from_checkpoint(self, path: str) -> torch.nn.Module:
        if not path or not hasattr(self, "_model_factory"):
            raise RuntimeError("Cannot load model for report; checkpoint path or factory missing.")
        model = self._model_factory()
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "model_factory must return a torch.nn.Module; "
                f"got {type(model).__name__!s} instead."
            )
        try:
            model = model.to(self.device)
        except AttributeError as exc:
            raise TypeError(
                "model_factory returned an object without `.to(...)`; "
                f"type={type(model).__name__!s}. Make sure the factory "
                "creates and returns a torch.nn.Module."
            ) from exc
        blob = torch.load(path, map_location=self.device, weights_only=False)
        state = blob.get("model", blob)
        model.load_state_dict(state, strict=True)
        return model

    def _compute_subset_losses(
        self,
        *,
        model: torch.nn.Module,
        dataset,
        collate_fn,
        batch_size: int,
        indices: Optional[List[int]] = None,
        mirror_train_semantics: bool = False,
        amp_enabled: Optional[bool] = None,
        deterministic_seed: Optional[int] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        materialize: bool = True,
        data_loader: Optional[DataLoader] = None,
    ) -> Tuple[
        Union[List[float], ChunkedArraySpool],
        Union[List[int], ChunkedArraySpool],
    ]:
        criterion = self._loss_module_for_features()
        device_hint = str(getattr(self, "device", "cpu"))
        amp_flag = bool(amp_enabled if amp_enabled is not None else (self.cfg.amp and "cuda" in device_hint))
        with self._rng_guard(deterministic_seed):
            losses, sample_indices = compute_per_sample_losses(
                model=model,
                dataset=dataset,
                collate_fn=collate_fn,
                criterion=criterion,
                device=self.device,
                batch_size=batch_size,
                indices=indices,
                mirror_train_semantics=mirror_train_semantics,
                amp_enabled=amp_flag,
                should_stop=should_stop,
                materialize=materialize,
                data_loader=data_loader,
            )
        return losses, sample_indices

    def _compute_margins_and_embeddings(
        self,
        ds,
        *,
        indices: Optional[List[int]] = None,
        batch_size: int = 256,
        amp_enabled: bool = True,
        need_margin: bool = True,
        need_embed: bool = False,
        embed_max_dim: int = 256,
    ) -> Tuple[List[float], List[List[float]]]:
        margin_spool: Optional[ChunkedArraySpool] = None
        embed_spool: Optional[ChunkedArraySpool] = None
        embed_row_width: Optional[int] = None

        if need_margin:
            margin_chunk = min(max(1024, batch_size * 4), 65536)
            margin_spool = ChunkedArraySpool(chunk_size=margin_chunk, typecode="f", value_cast=float)
        use_embed = need_embed and (self._embedding_hook_fn is not None)
        if use_embed:
            embed_chunk = max(1024, batch_size * max(1, embed_max_dim))
            embed_chunk = min(embed_chunk, 262144)
            embed_spool = ChunkedArraySpool(chunk_size=embed_chunk, typecode="f", value_cast=float)

        self.model.eval()
        old_train = self.model.training
        autocast = torch.cuda.amp.autocast if (amp_enabled and torch.cuda.is_available()) else _noop_ctx
        loader = DataLoader(
            Subset(ds, indices) if indices is not None else ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=getattr(self.train_loader, 'collate_fn', None),
        )
        using_cuda = torch.cuda.is_available() and "cuda" in str(self.device).lower()

        try:
            with torch.no_grad():
                for batch in loader:
                    logits_t: Optional[torch.Tensor] = None
                    embed_tensor: Optional[torch.Tensor] = None
                    active_input: Optional[Any] = None
                    first: Any = None
                    rest: List[Any] = []
                    candidates_list: List[Any] = []
                    try:
                        batch = move_batch_like(batch, self.device)
                        first, rest = (
                            (batch, [])
                            if not isinstance(batch, (list, tuple))
                            else (batch[0], list(batch[1:]))
                        )
                        candidates_list = model_input_candidates(self.model, first, rest)
                        last_exc: Optional[Exception] = None
                        for cand in candidates_list:
                            try:
                                with autocast():
                                    logits = self.model(cand)
                                logits_t = ensure_tensor_output(logits)
                                active_input = cand
                                break
                            except Exception as exc:
                                if should_retry_model_input(exc):
                                    last_exc = exc
                                    continue
                                raise
                        else:
                            if last_exc is not None:
                                raise last_exc
                            raise RuntimeError("Could not prepare batch for margin/embedding scan")

                        if need_margin and margin_spool is not None and logits_t is not None:
                            margin_vals = _top2_margin(logits_t)
                            margin_spool.extend(float(v) for v in margin_vals.detach().cpu().tolist())

                        if use_embed and embed_spool is not None and self._embedding_hook_fn is not None:
                            try:
                                e = self._embedding_hook_fn(self.model, active_input, logits_t)
                                if not torch.is_tensor(e):
                                    raise RuntimeError("embedding_hook must return a Tensor[B,D]")
                                embed_tensor = e
                                if e.dim() == 1:
                                    e = e.unsqueeze(1)
                                if e.size(1) > embed_max_dim:
                                    e = e[:, :embed_max_dim]
                                row_block = e.detach().cpu().tolist()
                                for row in row_block:
                                    vals = [float(v) for v in row]
                                    if embed_row_width is None:
                                        embed_row_width = len(vals)
                                    if embed_row_width == 0:
                                        continue
                                    if len(vals) < embed_row_width:
                                        vals = vals + [0.0] * (embed_row_width - len(vals))
                                    elif len(vals) > embed_row_width:
                                        vals = vals[:embed_row_width]
                                    embed_spool.extend(vals)
                            except Exception as ex:
                                warnings.warn(f"embedding_hook failed: {ex}")
                                use_embed = False
                                if embed_spool is not None:
                                    embed_spool.cleanup()
                                    embed_spool = None
                    finally:
                        if torch.is_tensor(embed_tensor):
                            if embed_tensor.device.type == "cuda":
                                embed_tensor = embed_tensor.detach().cpu()
                            embed_tensor = None
                        if torch.is_tensor(logits_t):
                            if logits_t.device.type == "cuda":
                                logits_t = logits_t.detach().cpu()
                            logits_t = None
                        active_input = None
                        del first
                        del rest
                        del batch
                        del candidates_list
                        if using_cuda:
                            with contextlib.suppress(Exception):
                                torch.cuda.empty_cache()
        finally:
            self.model.train(old_train)

        def _finish_embeddings() -> List[List[float]]:
            if embed_spool is None or not embed_row_width or embed_row_width <= 0:
                if embed_spool is not None:
                    embed_spool.finish()
                return []
            flat = embed_spool.finish()
            width = embed_row_width
            return [flat[i:i + width] for i in range(0, len(flat), width)]

        try:
            margins = margin_spool.finish() if margin_spool is not None else []
            embeds = _finish_embeddings()
            return margins, embeds
        finally:
            if margin_spool is not None:
                margin_spool.cleanup()
            if embed_spool is not None:
                embed_spool.cleanup()

    def _build_features_for_selection(
        self,
        ds,
        feature_str: str,
        cache: Dict[str, Any],
        *,
        batch_size: int = 256,
    ) -> Tuple[List[List[float]], int]:
        from ..metrics_utils import _percentile_list, _ranks
        features: List[List[float]] = []
        dims = 0
        parts = [p.strip() for p in feature_str.split('|') if p.strip()]
        N = len(cache.get("loss", [])) if "loss" in cache else (len(ds) if hasattr(ds, '__len__') else 0)
        if N == 0:
            return [], 0
        features = [[ ] for _ in range(N)]
        if "loss" in parts:
            lossv = cache.get("loss")
            for i in range(N):
                features[i].append(float(lossv[i]))
            dims += 1
        if "margin" in parts or "embed" in parts:
            need_margin = ("margin" in parts)
            need_embed  = ("embed" in parts) and (self._embedding_hook_fn is not None)
            if (need_margin and "margin" not in cache) or (need_embed and "embed" not in cache):
                mlist, elist = self._compute_margins_and_embeddings(
                    ds,
                    indices=None,
                    batch_size=batch_size,
                    amp_enabled=bool(self._feature_sampling_cfg.get("amp_for_psl", True) and self.cfg.amp and "cuda" in self.device),
                    need_margin=need_margin,
                    need_embed=need_embed,
                    embed_max_dim=int(self._feature_sampling_cfg.get("embed_max_dim", 256)),
                )
                if need_margin: cache["margin"] = mlist
                if need_embed:  cache["embed"] = elist
            if "margin" in parts:
                margins = cache.get("margin", [0.0]*N)
                for i in range(N):
                    features[i].append(float(margins[i]))
                dims += 1
            if "embed" in parts:
                embeds = cache.get("embed", [[0.0]]*N)
                D = len(embeds[0]) if embeds and isinstance(embeds[0], list) else 0
                for i in range(N):
                    row = embeds[i] if i < len(embeds) else ([0.0] * D)
                    features[i].extend(list(map(float, row)))
                dims += D
        if dims == 1:
            vals = [r[0] for r in features]
            ranks = _ranks(vals)
            features = [[r] for r in ranks]
        else:
            for j in range(dims):
                col = [r[j] for r in features]
                q1 = _percentile_list(col, 0.25)
                q3 = _percentile_list(col, 0.75)
                iqr = (q3 - q1) + 1e-12
                for i in range(N):
                    features[i][j] = (features[i][j] - q1) / iqr
        return features, dims
