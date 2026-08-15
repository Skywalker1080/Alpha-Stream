"""ModelProvisioner — the single owner of "is a forecast available for this ticker?".

It decides whether the parent or child for a ticker is provisioned, training, or
missing, and enqueues the training that makes it available. All outside-world I/O
(filesystem, task registry, training functions, prediction-cache priming) is
injected at construction, so callers and tests cross the same seam.

The task seam contract: `task_status(task_id) -> Optional[dict]` and
`start_task(task_id, fn, *args, chain_fn=None)` where `start_task` is an async
callable that schedules `fn(*args)` to run and run `chain_fn` (sync) on completion.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from src.config.pipeline_config import Config

PARENT_TASK_ID = "parent_training"
CHILD_TASK_PREFIX = "train_child_"


def child_task_id(ticker: str) -> str:
    return f"{CHILD_TASK_PREFIX}{ticker.upper()}"


def scaler_path(config: Config, ticker: str, model_type: str = "child") -> Path:
    """Location of the scaler artifact for a ticker. Single owner of this rule."""
    ticker = ticker.upper()
    if model_type == "parent":
        return Path(config.parent_dir) / f"{config.parent_ticker}_parent_scaler.pkl"
    return Path(config.workdir) / ticker / f"{ticker}_child_scaler.pkl"


class ProvisionState(str, Enum):
    PROVISIONED = "provisioned"
    TRAINING = "training"
    ENQUEUED = "enqueued"
    MISSING = "missing"


@dataclass(frozen=True)
class ProvisionResult:
    state: ProvisionState
    task_id: Optional[str] = None
    detail: Optional[str] = None


class ModelProvisioner:
    def __init__(
        self,
        config: Optional[Config] = None,
        *,
        task_status: Optional[Callable[[str], Optional[dict]]] = None,
        start_task: Optional[Callable] = None,
        train_parent: Optional[Callable] = None,
        train_child: Optional[Callable] = None,
        prime_child: Optional[Callable] = None,
    ):
        self.config = config or Config()
        self._task_status = task_status
        self._start_task = start_task
        self._train_parent = train_parent
        self._train_child = train_child
        self._prime_child = prime_child

    def scaler_path(self, ticker: str, model_type: str = "child") -> Path:
        return scaler_path(self.config, ticker, model_type)

    def is_provisioned(self, ticker: str, model_type: str = "child") -> bool:
        return self.scaler_path(ticker, model_type).exists()

    def status(self, ticker: str, model_type: str = "child") -> ProvisionResult:
        if model_type == "parent":
            return self._parent_status()
        return self._child_status(ticker.upper())

    async def ensure(self, ticker: str, model_type: str = "child") -> ProvisionResult:
        if model_type == "parent":
            return await self._ensure_parent()
        return await self._ensure_child(ticker.upper())

    def _parent_status(self) -> ProvisionResult:
        if self._parent_scaler_exists():
            return ProvisionResult(ProvisionState.PROVISIONED)
        if self._is_running(PARENT_TASK_ID):
            return ProvisionResult(ProvisionState.TRAINING, PARENT_TASK_ID)
        return ProvisionResult(ProvisionState.MISSING, PARENT_TASK_ID)

    def _child_status(self, ticker: str) -> ProvisionResult:
        if self.is_provisioned(ticker, "child"):
            return ProvisionResult(ProvisionState.PROVISIONED)
        child_task = child_task_id(ticker)
        if self._is_running(child_task):
            return ProvisionResult(ProvisionState.TRAINING, child_task)
        if not self._parent_scaler_exists():
            if self._is_running(PARENT_TASK_ID):
                return ProvisionResult(ProvisionState.TRAINING, PARENT_TASK_ID)
            return ProvisionResult(ProvisionState.MISSING, PARENT_TASK_ID)
        return ProvisionResult(ProvisionState.MISSING, child_task)

    async def _ensure_parent(self) -> ProvisionResult:
        if self._parent_scaler_exists():
            return ProvisionResult(ProvisionState.PROVISIONED)
        if self._is_running(PARENT_TASK_ID):
            return ProvisionResult(ProvisionState.TRAINING, PARENT_TASK_ID)
        self._require_seam("start_task")
        await self._start_task(PARENT_TASK_ID, self._train_parent)
        return ProvisionResult(ProvisionState.ENQUEUED, PARENT_TASK_ID)

    async def _ensure_child(self, ticker: str) -> ProvisionResult:
        if self.is_provisioned(ticker, "child"):
            return ProvisionResult(ProvisionState.PROVISIONED)
        child_task = child_task_id(ticker)
        if self._is_running(child_task):
            return ProvisionResult(ProvisionState.TRAINING, child_task)
        if not self._parent_scaler_exists():
            if self._is_running(PARENT_TASK_ID):
                return ProvisionResult(ProvisionState.TRAINING, PARENT_TASK_ID)
            self._require_seam("start_task")
            await self._start_task(
                PARENT_TASK_ID,
                self._train_parent,
                chain_fn=self._child_starter(ticker),
            )
            return ProvisionResult(ProvisionState.ENQUEUED, PARENT_TASK_ID)
        self._require_seam("start_task")
        await self._start_task(
            child_task,
            self._train_child,
            ticker,
            chain_fn=lambda: self._prime(ticker),
        )
        return ProvisionResult(ProvisionState.ENQUEUED, child_task)

    def _parent_scaler_exists(self) -> bool:
        return self.is_provisioned(self.config.parent_ticker, "parent")

    def _is_running(self, task_id: str) -> bool:
        if self._task_status is None:
            return False
        status = self._task_status(task_id)
        return bool(status and status.get("status") == "running")

    def _require_seam(self, name: str) -> None:
        if self._start_task is None:
            raise ValueError(f"ModelProvisioner needs a '{name}' adapter to act")

    def _prime(self, ticker: str) -> None:
        if self._prime_child is not None:
            self._prime_child(ticker)

    def _child_starter(self, ticker: str) -> Callable[[], None]:
        loop = asyncio.get_running_loop()

        def start_child() -> None:
            asyncio.run_coroutine_threadsafe(
                self._start_task(
                    child_task_id(ticker),
                    self._train_child,
                    ticker,
                    chain_fn=lambda: self._prime(ticker),
                ),
                loop,
            )

        return start_child