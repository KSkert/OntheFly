# src/onthefly/sampler_utils.py
from __future__ import annotations
from typing import Callable, Iterator, Dict, Any, Optional
import torch
from torch.utils.data import Sampler

class EpochSeededRandomSampler(Sampler[int]):
    """
    Reproducible shuffle with optional per-epoch reseed.
    - If fixed_order=True: uses seed+(0) every time (same order each epoch).
    - Otherwise: uses seed+epoch_fn() (fresh but reproducible order).
    Exposes state_dict/load_state_dict so we can resume mid-epoch (cursor).
    """
    def __init__(
        self,
        data_len: int,
        *,
        base_seed: int,
        epoch_fn: Callable[[], int],
        fixed_order: bool = False,
    ):
        self.data_len = int(data_len)
        self.base_seed = int(base_seed)
        self.epoch_fn = epoch_fn
        self.fixed_order = bool(fixed_order)
        self._cursor = 0  # where we are within this epoch’s ordering

    def __len__(self) -> int:
        return self.data_len

    def _order_for_epoch(self, epoch: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(self.base_seed + (0 if self.fixed_order else int(epoch)))
        return torch.randperm(self.data_len, generator=g)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "cursor": int(self._cursor),
            "base_seed": int(self.base_seed),
            "fixed_order": bool(self.fixed_order),
        }

    def load_state_dict(self, s: Dict[str, Any]) -> None:
        try:
            self._cursor = int(s.get("cursor", 0))
            self.base_seed = int(s.get("base_seed", self.base_seed))
            self.fixed_order = bool(s.get("fixed_order", self.fixed_order))
        except Exception:
            # never break training because of a resume mismatch
            self._cursor = 0

    def reset_cursor(self) -> None:
        self._cursor = 0

    def __iter__(self) -> Iterator[int]:
        # Order for current epoch; continue from cursor (resume mid-epoch)
        epoch = int(self.epoch_fn())
        order = self._order_for_epoch(epoch)
        # Slice from last cursor position
        cur = int(self._cursor)
        if cur < 0 or cur >= self.data_len:
            cur = 0
        # IMPORTANT: advance cursor as we yield (so a pause can resume precisely)
        for k in range(cur, self.data_len):
            self._cursor = k + 1
            yield int(order[k])
        # Next __iter__ starts at 0 again (new epoch or user called reset)
        self._cursor = 0
