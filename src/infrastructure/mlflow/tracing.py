"""MLflow tracing spans and trace-session metadata."""

from __future__ import annotations

from contextlib import contextmanager
from types import TracebackType
from typing import Any

import mlflow
from mlflow.entities.span import SpanType

from infrastructure.mlflow.common import (
    is_tracking_enabled,
    json_ready,
    safe_mlflow_call,
)
from infrastructure.mlflow.context import (
    _TRACKING_CAPTURE,
    get_capture_stack,
    set_capture_stack,
)
from logging_utils import get_logger

logger = get_logger(__name__)


class ObservedSpan:
    """Small proxy that mirrors span mutations into the active tracking capture."""

    def __init__(self, span: Any, event: dict[str, Any]) -> None:
        self._span = span
        self._event = event

    def set_inputs(self, inputs: Any) -> None:
        self._event["inputs"] = json_ready(inputs)
        safe_mlflow_call("set span inputs", lambda: self._span.set_inputs(inputs), None)

    def set_outputs(self, outputs: Any) -> None:
        self._event["outputs"] = json_ready(outputs)
        safe_mlflow_call("set span outputs", lambda: self._span.set_outputs(outputs), None)

    def set_attribute(self, key: str, value: Any) -> None:
        attributes = self._event.setdefault("attributes", {})
        attributes[str(key)] = json_ready(value)
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

    try:
        span_context = mlflow.start_span(
            name=name,
            span_type=span_type,
            attributes=attributes,
        )
    except Exception:
        logger.exception("MLflow operation failed: start traced operation %s", name)
        yield None
        return

    try:
        span = span_context.__enter__()
    except Exception:
        logger.exception("MLflow operation failed: enter traced operation %s", name)
        yield None
        return

    capture = _TRACKING_CAPTURE.get()
    event: dict[str, Any] | None = None
    observed_span: Any = span
    if capture is not None:
        event = {
            "name": name,
            "span_type": str(span_type),
            "attributes": json_ready(attributes or {}),
            "inputs": json_ready(inputs),
            "outputs": None,
        }
        capture.spans.append(event)
        stack = get_capture_stack()
        stack.append(event)
        set_capture_stack(stack)
        observed_span = ObservedSpan(span, event)

    exc_info: tuple[type[BaseException], BaseException, TracebackType] | None = None
    try:
        if inputs is not None:
            observed_span.set_inputs(inputs)
        yield observed_span
    except BaseException as exc:
        exc_info = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        if event is not None:
            stack = get_capture_stack()
            if stack:
                stack.pop()
                set_capture_stack(stack)
        if exc_info is None:
            safe_mlflow_call(
                f"exit traced operation {name}",
                lambda: span_context.__exit__(None, None, None),
                None,
            )
        else:
            safe_mlflow_call(
                f"exit traced operation {name}",
                lambda: span_context.__exit__(*exc_info),
                None,
            )
