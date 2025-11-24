from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..control import PauseGate


class FrameworkDelegate(ABC):
    """
    Abstract hook for plugging external training frameworks into the ConsoleAction.

    Implementations should attach to the target framework (model/trainer/datamodule)
    and ensure they emit batch-end metrics/events in the same shape the dashboard expects.
    Hooks should poll the provided PauseGate at natural batch boundaries (call
    `gate.should_block()` and then `gate.wait_until_resumed()` before the next batch
    is fetched) so the host loop can pause deterministically.
    """

    @abstractmethod
    def attach(self, model: Any, trainer: Any, datamodule: Any) -> None:
        """
        Bind this delegate into the external framework.

        This should configure hooks/callbacks but should not start training.
        """
        raise NotImplementedError

    @abstractmethod
    def install_batch_boundary_hook(self, gate: PauseGate) -> None:
        """
        Install whatever hook is needed to emit validation metrics or post-batch signals.

        Framework hooks should ensure they emit metrics after every training batch
        (or whatever granularity the dashboard expects) and honor `gate` by checking
        `gate.should_block()` before each new batch and waiting on `gate.wait_until_resumed()`.
        """
        raise NotImplementedError

    # Optional command hooks -------------------------------------------------

    def request_pause(self, reason: str) -> None:
        raise NotImplementedError

    def request_resume(self) -> None:
        raise NotImplementedError

    def request_stop(self) -> None:
        raise NotImplementedError

    def save_checkpoint(self, reason: Optional[str] = None) -> Optional[str]:
        raise NotImplementedError

    def load_checkpoint(self, path: str, step_hint: Optional[int] = None) -> int:
        raise NotImplementedError

    def restore_initial_state(self) -> bool:
        """
        Optional hook used by resetSession to revert the wrapped trainer back
        to its baseline weights/optimizer state.
        """
        return False
