"""Context state for MLflow tracking."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any


_ACTIVE_RUN_ID: ContextVar[str | None] = ContextVar(
    "active_mlflow_run_id",
    default=None,
)
_ACTIVE_EXPERIMENT_NAME: ContextVar[str | None] = ContextVar(
    "active_mlflow_experiment_name",
    default=None,
)
_RUN_TERMINATION_SUSPENDED: ContextVar[bool] = ContextVar(
    "mlflow_run_termination_suspended",
    default=False,
)
_TRACKING_CAPTURE: ContextVar["TrackingCapture | None"] = ContextVar(
    "mlflow_tracking_capture",
    default=None,
)
_TRACKING_CAPTURE_STACK: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "mlflow_tracking_capture_stack",
    default=None,
)
_ARTIFACT_NAMESPACE: ContextVar[str | None] = ContextVar(
    "mlflow_artifact_namespace",
    default=None,
)


class MlflowTrackingContext:
    """Tracks the active MLflow run for the current workflow."""

    def __init__(
        self,
        run_id_token: Token[str | None] | None,
        started_fluent_run: bool = False,
    ) -> None:
        """Create context metadata for a bound MLflow run."""

        self.run_id_token = run_id_token
        self.started_fluent_run = started_fluent_run


@dataclass
class TrackingCapture:
    """Stores captured spans and artifacts for deterministic evaluation."""

    spans: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def bind_mlflow_run(run_id: str):
    """Temporarily bind one existing MLflow run to the current execution context."""

    run_id_token = _ACTIVE_RUN_ID.set(run_id)
    try:
        yield
    finally:
        _ACTIVE_RUN_ID.reset(run_id_token)


@contextmanager
def suspend_mlflow_run_termination():
    """Keep workflow finalization from closing a shared MLflow run.

    Use this when a workflow should log its final state but leave the active run
    open for additional work.
    """

    token = _RUN_TERMINATION_SUSPENDED.set(True)
    try:
        yield
    finally:
        _RUN_TERMINATION_SUSPENDED.reset(token)


@contextmanager
def capture_tracking_events():
    """Capture span payloads and logged artifacts for one execution block."""

    capture = TrackingCapture()
    capture_token = _TRACKING_CAPTURE.set(capture)
    stack_token = _TRACKING_CAPTURE_STACK.set([])
    try:
        yield capture
    finally:
        _TRACKING_CAPTURE.reset(capture_token)
        _TRACKING_CAPTURE_STACK.reset(stack_token)


@contextmanager
def bind_artifact_namespace(namespace: str | None):
    """Prefix MLflow artifact paths for the current execution block."""

    token = _ARTIFACT_NAMESPACE.set(namespace.strip("/") if namespace else None)
    try:
        yield
    finally:
        _ARTIFACT_NAMESPACE.reset(token)


@contextmanager
def bind_mlflow_experiment(experiment_name: str | None):
    """Temporarily bind the MLflow experiment used for new runs."""

    normalized_name = experiment_name.strip() if experiment_name else None
    token = _ACTIVE_EXPERIMENT_NAME.set(normalized_name)
    try:
        yield
    finally:
        _ACTIVE_EXPERIMENT_NAME.reset(token)


def get_capture_stack() -> list[dict[str, Any]]:
    """Return a copy of the current tracking capture stack."""

    return list(_TRACKING_CAPTURE_STACK.get() or [])


def set_capture_stack(stack: list[dict[str, Any]]) -> None:
    """Replace the active capture stack."""

    _TRACKING_CAPTURE_STACK.set(stack)
