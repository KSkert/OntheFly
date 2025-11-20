from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .session import OnTheFlySession
from .control import set_channel
from .dashboard_channel import SocketChannel


class Trainer:
    """
    High-level orchestrator that mirrors the ergonomics of Lightning's Trainer.
    Users instantiate it once per project/run family and call `.fit(...)`
    inside their scripts. When the VS Code dashboard is open, metrics stream
    live without the extension having to launch the script.
    """

    def __init__(
        self,
        *,
        project: str,
        run_name: Optional[str] = None,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        do_test_after: bool = False,
        val_every_n_epochs: Optional[int] = 1,
        dashboard_host: str = "127.0.0.1",
        dashboard_port: Optional[int] = None,
        auto_connect: bool = True,
        connect_timeout: float = 1.0,
        reconnect_interval: float = 2.0,
        backlog_limit: int = 20000,
        session_defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.project = project
        self._default_run_name = (run_name or "").strip() or None
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.do_test_after = bool(do_test_after)
        self._val_every = val_every_n_epochs
        self._session_defaults = dict(session_defaults or {})
        self._channel: Optional[SocketChannel] = None

        if auto_connect:
            self._channel = SocketChannel(
                host=dashboard_host,
                port=dashboard_port,
                connect_timeout=connect_timeout,
                reconnect_interval=reconnect_interval,
                backlog_limit=backlog_limit,
                auto_connect=True,
            )
            set_channel(self._channel)

    # ------------------------------------------------------------------ public API

    def fit(
        self,
        *,
        model,
        optimizer,
        loss_fn: Callable,
        train_loader,
        val_loader=None,
        test_loader=None,
        run_name: Optional[str] = None,
        scheduler=None,
        device: Optional[str] = None,
        amp: bool = True,
        grad_clip_norm: Optional[float] = 1.0,
        seed: int = 42,
        embedding_hook: Optional[Callable] = None,
        model_factory: Optional[Callable[[], Any]] = None,
        data_order_policy: str = "user",
        deterministic_pauses: bool = True,
        enforce_sampler_state: bool = True,
        val_every_n_epochs: Optional[int] = None,
        **session_overrides,
    ) -> OnTheFlySession:
        actual_run = (run_name or self._default_run_name)
        if not actual_run:
            raise ValueError("Trainer.fit requires a run_name (either passed to fit or set on the Trainer).")

        val_schedule = (
            val_every_n_epochs
            if val_every_n_epochs is not None
            else self._val_every
        )

        cfg: Dict[str, Any] = dict(self._session_defaults)
        cfg.update(session_overrides or {})
        cfg.setdefault("val_every_n_epochs", val_schedule)

        session = OnTheFlySession(
            project=self.project,
            run_name=actual_run,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            scheduler=scheduler,
            amp=amp,
            grad_clip_norm=grad_clip_norm,
            seed=seed,
            embedding_hook=embedding_hook,
            model_factory=model_factory,
            data_order_policy=data_order_policy,
            deterministic_pauses=deterministic_pauses,
            enforce_sampler_state=enforce_sampler_state,
            **cfg,
        )

        session.serve(
            max_steps=self.max_steps,
            max_epochs=self.max_epochs,
            do_test_after=self.do_test_after,
        )
        return session

    def close(self) -> None:
        if self._channel:
            try:
                self._channel.close()
            except Exception:
                pass
