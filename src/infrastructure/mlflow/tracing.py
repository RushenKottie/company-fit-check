"""MLflow tracing spans and trace-session metadata."""

from __future__ import annotations

from contextlib import contextmanager
from types import TracebackType
from typing import Any

import mlflow
from mlflow.entities.span import SpanType

from infrastructure.mlflow.common import (
    is_tracking_enabled,
    to_json_safe,
    safe_mlflow_call,
)
from infrastructure.mlflow.context import (
    _TRACKING_CAPTURE,
    get_capture_stack,
    set_capture_stack,
)
from logging_utils import get_logger

logger = get_logger(__name__)
SpanExitArgs = tuple[
    type[BaseException] | None,
    BaseException | None,
    TracebackType | None,
]


class ObservedSpan:
    """Wrap an MLflow span and keep an in-memory copy for deterministic evals."""

    def __init__(self, span: Any, event: dict[str, Any]) -> None:
        """Create a span wrapper backed by one captured event dictionary."""

        self._span = span
        self._event = event

    def set_inputs(self, inputs: Any) -> None:
        """Store span inputs in memory and forward them to MLflow."""

        self._event["inputs"] = to_json_safe(inputs)
        safe_mlflow_call("set span inputs", lambda: self._span.set_inputs(inputs), None)

    def set_outputs(self, outputs: Any) -> None:
        """Store span outputs in memory and forward them to MLflow."""

        self._event["outputs"] = to_json_safe(outputs)
        safe_mlflow_call(
            "set span outputs",
            lambda: self._span.set_outputs(outputs),
            None,
        )

    def set_attribute(self, key: str, value: Any) -> None:
        """Store one span attribute in memory and forward it to MLflow."""

        attributes = self._event.setdefault("attributes", {})
        attributes[str(key)] = to_json_safe(value)
        safe_mlflow_call(
            f"set span attribute {key}",
            lambda: self._span.set_attribute(key, value),
            None,
        )


def update_current_trace_session(
    *,
    session_id: str | None,
    request_preview: str | None = None,
    response_preview: str | None = None,
) -> None:
    """Annotate the active trace so MLflow can surface it in the Sessions view."""

    if not is_tracking_enabled() or not session_id:
        return

    metadata = {
        "mlflow.trace.session": session_id,
        "mlflow.sourceRun": session_id,
    }
    kwargs: dict[str, Any] = {"metadata": metadata}
    if request_preview:
        kwargs["request_preview"] = request_preview
    if response_preview:
        kwargs["response_preview"] = response_preview

    def _update_trace() -> None:
        """Update the active MLflow trace with version-compatible arguments."""

        try:
            mlflow.update_current_trace(**kwargs)
        except TypeError:
            # Older MLflow releases may not support preview keyword arguments yet.
            mlflow.update_current_trace(metadata=metadata)

    safe_mlflow_call("update current trace", _update_trace, None)


@contextmanager
def traced_operation(
    name: str,
    *,
    span_type: str = SpanType.UNKNOWN,
    inputs: Any | None = None,
    attributes: dict[str, Any] | None = None,
):
    """Create one MLflow tracing span around a logical workflow operation."""

    if not is_tracking_enabled():
        yield None
        return

    span_context = _start_span_context(name, span_type, attributes)
    if span_context is None:
        yield None
        return

    span = _enter_span_context(name, span_context)
    if span is None:
        yield None
        return

    event = _start_capture_event(
        name=name,
        span_type=span_type,
        inputs=inputs,
        attributes=attributes,
    )
    observed_span = ObservedSpan(span, event) if event is not None else span
    exit_args: SpanExitArgs = (None, None, None)
    try:
        if inputs is not None:
            observed_span.set_inputs(inputs)
        yield observed_span
    except BaseException as exc:
        exit_args = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        if event is not None:
            _finish_capture_event()
        _exit_span_context(name, span_context, exit_args)


def _start_span_context(
    name: str,
    span_type: str,
    attributes: dict[str, Any] | None,
) -> Any | None:
    """Start an MLflow span context, returning None if MLflow fails."""

    try:
        return mlflow.start_span(
            name=name,
            span_type=span_type,
            attributes=attributes,
        )
    except Exception:
        logger.exception("MLflow operation failed: start traced operation %s", name)
        return None


def _enter_span_context(name: str, span_context: Any) -> Any | None:
    """Enter an MLflow span context, returning None if MLflow fails."""

    try:
        return span_context.__enter__()
    except Exception:
        logger.exception("MLflow operation failed: enter traced operation %s", name)
        return None


def _exit_span_context(
    name: str,
    span_context: Any,
    exit_args: SpanExitArgs,
) -> None:
    """Exit an MLflow span context without letting MLflow errors escape."""

    safe_mlflow_call(
        f"exit traced operation {name}",
        lambda: span_context.__exit__(*exit_args),
        None,
    )


def _start_capture_event(
    *,
    name: str,
    span_type: str,
    inputs: Any | None,
    attributes: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Start an in-memory capture event for the active tracking capture."""

    capture = _TRACKING_CAPTURE.get()
    if capture is None:
        return None

    event = {
        "name": name,
        "span_type": str(span_type),
        "attributes": to_json_safe(attributes or {}),
        "inputs": to_json_safe(inputs),
        "outputs": None,
    }
    capture.spans.append(event)

    stack = get_capture_stack()
    stack.append(event)
    set_capture_stack(stack)
    return event


def _finish_capture_event() -> None:
    """Remove the current in-memory capture event from the capture stack."""

    stack = get_capture_stack()
    if not stack:
        return
    stack.pop()
    set_capture_stack(stack)
