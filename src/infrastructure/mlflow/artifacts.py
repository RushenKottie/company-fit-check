"""MLflow artifact, response, and metric logging."""

from __future__ import annotations

from typing import Any
import json
import re

from langchain_core.messages import BaseMessage
import mlflow

from infrastructure.mlflow.common import (
    get_mlflow_client,
    is_tracking_enabled,
    json_ready,
    safe_mlflow_call,
    timestamp_slug,
    utc_now_iso,
)
from infrastructure.mlflow.context import (
    _ACTIVE_RUN_ID,
    _ARTIFACT_NAMESPACE,
    _TRACKING_CAPTURE,
    get_capture_stack,
    set_capture_stack,
)
from logging_utils import get_logger

logger = get_logger(__name__)


def log_text_artifact(artifact_path: str, content: str) -> None:
    """Persist one text artifact for the active MLflow run."""

    run_id = _ACTIVE_RUN_ID.get()
    if not run_id or not is_tracking_enabled():
        return

    resolved_path = resolve_artifact_path(artifact_path)
    record_logged_artifact(run_id, resolved_path, content)
    safe_mlflow_call(
        f"log text artifact {resolved_path}",
        lambda: mlflow.log_text(content, artifact_file=resolved_path, run_id=run_id),
        None,
    )
    logger.info("Logged MLflow text artifact run_id=%s path=%s", run_id, resolved_path)


def log_json_artifact(artifact_path: str, payload: Any) -> None:
    """Persist one JSON artifact for the active MLflow run."""

    content = json.dumps(payload, ensure_ascii=True, indent=2)
    log_text_artifact(artifact_path, content)


def log_clarification_question(question: str, target: str | None) -> None:
    """Persist one assistant clarification question as a session artifact."""

    content = (
        f"timestamp_utc: {utc_now_iso()}\n"
        f"target: {target or 'unknown'}\n\n"
        f"{question.strip()}"
    )
    log_text_artifact(f"clarifications/question-{timestamp_slug()}.txt", content)


def log_clarification_answer(question: str | None, answer: str, target: str | None) -> None:
    """Persist one user clarification answer as a session artifact."""

    run_id = _ACTIVE_RUN_ID.get()
    if not run_id or not is_tracking_enabled():
        return
    _log_clarification_answer_for_run(run_id, question, answer, target)


def log_clarification_answer_for_run(
    run_id: str | None,
    question: str | None,
    answer: str,
    target: str | None,
) -> None:
    """Persist one user clarification answer for an explicit run id."""

    if not run_id or not is_tracking_enabled():
        return
    _log_clarification_answer_for_run(run_id, question, answer, target)


def log_assistant_response_for_run(
    run_id: str | None,
    message: str,
    *,
    status: str | None,
) -> None:
    """Persist one rendered assistant response for an explicit run id."""

    if not run_id or not is_tracking_enabled():
        return

    content = (
        f"timestamp_utc: {utc_now_iso()}\n"
        f"status: {status or 'unknown'}\n\n"
        f"{message.strip()}"
    )
    log_text_artifact_for_run(
        run_id,
        f"assistant/response-{timestamp_slug()}.txt",
        content,
    )


def log_guardrail_blocked_user_message(
    run_id: str | None,
    message: str,
    source: str | None,
) -> None:
    """Persist one user message that triggered a guardrail block."""

    if not run_id or not is_tracking_enabled():
        return
    content = (
        f"timestamp_utc: {utc_now_iso()}\n"
        f"source: {source or 'unknown'}\n\n"
        f"{message.strip()}"
    )
    log_text_artifact_for_run(
        run_id,
        f"guardrails/blocked-user-message-{timestamp_slug()}.txt",
        content,
    )


def _log_clarification_answer_for_run(
    run_id: str,
    question: str | None,
    answer: str,
    target: str | None,
) -> None:
    """Build and log the clarification answer payload."""

    question_text = (question or "").strip() or "None"
    content = (
        f"timestamp_utc: {utc_now_iso()}\n"
        f"target: {target or 'unknown'}\n\n"
        f"Agent question:\n{question_text}\n\n"
        f"User answer:\n{answer.strip()}"
    )
    log_text_artifact_for_run(
        run_id,
        f"clarifications/answer-{timestamp_slug()}.txt",
        content,
    )


def log_llm_prompt_artifact(prefix: str, messages: list[BaseMessage]) -> None:
    """Attach one LLM prompt payload to the current MLflow span."""

    if not is_tracking_enabled():
        return

    span = safe_mlflow_call(
        "get current active span",
        mlflow.get_current_active_span,
        None,
    )
    if span is None:
        return

    rendered_messages = [
        {
            "index": index,
            "role": getattr(message, "type", message.__class__.__name__),
            "content": (
                message.content
                if isinstance(message.content, str)
                else str(message.content)
            ),
        }
        for index, message in enumerate(messages, start=1)
    ]
    stack = get_capture_stack()
    if stack:
        stack[-1]["prompt_prefix"] = prefix
        stack[-1]["prompt_messages"] = json_ready(rendered_messages)
        set_capture_stack(stack)
    safe_mlflow_call(
        "set prompt span attribute",
        lambda: span.set_attribute("llm.prompt.prefix", prefix),
        None,
    )
    safe_mlflow_call(
        "set prompt span inputs",
        lambda: span.set_inputs({"messages": rendered_messages}),
        None,
    )
    logger.info("Attached LLM prompt payload to active MLflow span prefix=%s", prefix)


def log_text_artifact_for_run(run_id: str, artifact_path: str, content: str) -> None:
    """Persist one text artifact for an explicit run id."""

    if not is_tracking_enabled():
        return

    resolved_path = resolve_artifact_path(artifact_path)
    record_logged_artifact(run_id, resolved_path, content)
    safe_mlflow_call(
        f"log text artifact {resolved_path}",
        lambda: mlflow.log_text(content, artifact_file=resolved_path, run_id=run_id),
        None,
    )


def log_json_artifact_for_run(run_id: str, artifact_path: str, payload: Any) -> None:
    """Persist one JSON artifact for an explicit run id."""

    log_text_artifact_for_run(
        run_id,
        artifact_path,
        json.dumps(payload, ensure_ascii=True, indent=2),
    )


def log_metric_for_run(run_id: str, metric_name: str, value: float) -> None:
    """Persist one numeric metric for an explicit run id."""

    if not run_id or not is_tracking_enabled():
        return

    client = get_mlflow_client()
    if client is None:
        return
    safe_mlflow_call(
        f"log metric {metric_name}",
        lambda: client.log_metric(run_id, metric_name, value),
        None,
    )


def set_run_status_for_run(
    run_id: str,
    *,
    status: str,
    error: str | None = None,
) -> None:
    """Update one existing MLflow run to a completed or failed terminal status."""

    if not run_id or not is_tracking_enabled():
        return
    if status not in {"completed", "failed"}:
        return

    log_json_artifact_for_run(
        run_id,
        "workflow/final-status.json",
        {
            "session_status": status,
            "error": error,
            "finalized_at_utc": utc_now_iso(),
        },
    )
    if error:
        log_text_artifact_for_run(run_id, "workflow/error.txt", error)

    client = get_mlflow_client()
    if client is None:
        return
    safe_mlflow_call(
        "set run terminated",
        lambda: client.set_terminated(
            run_id,
            status="FINISHED" if status == "completed" else "FAILED",
        ),
        None,
    )


def set_run_name_for_run(run_id: str, run_name: str) -> None:
    """Persist one user-friendly display name for an explicit run id."""

    if not run_id or not is_tracking_enabled():
        return

    normalized = re.sub(r"[^a-z0-9]+", "_", run_name.lower()).strip("_")
    if not normalized:
        return

    client = get_mlflow_client()
    if client is None:
        return
    safe_mlflow_call(
        "set run name tag",
        lambda: client.set_tag(run_id, "mlflow.runName", normalized),
        None,
    )


def record_logged_artifact(run_id: str, artifact_path: str, content: str) -> None:
    """Mirror one logged artifact into the active capture for deterministic checks."""

    capture = _TRACKING_CAPTURE.get()
    if capture is None:
        return
    capture.artifacts.append(
        {
            "run_id": run_id,
            "artifact_path": artifact_path,
            "content": content,
        }
    )


def resolve_artifact_path(artifact_path: str) -> str:
    """Apply the current artifact namespace to one relative artifact path."""

    namespace = _ARTIFACT_NAMESPACE.get()
    if not namespace:
        return artifact_path
    clean_path = artifact_path.lstrip("/")
    return f"{namespace}/{clean_path}" if clean_path else namespace
