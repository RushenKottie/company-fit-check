"""MLflow-backed persistence for prompt logs and workflow artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
import json
import os
from typing import Any

from langchain_core.messages import BaseMessage
import mlflow
from mlflow.entities.span import SpanType
from mlflow import MlflowClient
from pydantic import BaseModel

from company_fit_check.config import get_azure_openai_settings, get_mlflow_settings
from company_fit_check.logging_utils import get_logger
from company_fit_check.models.state import CompanyFitState

logger = get_logger(__name__)
_ACTIVE_RUN_ID: ContextVar[str | None] = ContextVar("active_mlflow_run_id", default=None)
_ACTIVE_SESSION_ID: ContextVar[str | None] = ContextVar(
    "active_mlflow_session_id",
    default=None,
)
_MISSING_CONFIG_WARNING_EMITTED = False


class MlflowTrackingContext:
    """Opaque tokens for one active MLflow workflow context."""

    def __init__(
        self,
        run_id_token: Token[str | None] | None,
        session_id_token: Token[str | None] | None,
    ) -> None:
        self.run_id_token = run_id_token
        self.session_id_token = session_id_token


def activate_mlflow_tracking(state: CompanyFitState) -> MlflowTrackingContext | None:
    """Ensure a session run exists and bind it to the current execution context."""

    if not _is_tracking_enabled():
        return None

    run_id = state.get("mlflow_run_id") or _create_session_run(state)
    if not run_id:
        return None

    state["mlflow_run_id"] = run_id
    settings = get_mlflow_settings()
    mlflow.set_experiment(settings.experiment_name)
    _update_run_metadata(run_id, state)
    run_id_token = _ACTIVE_RUN_ID.set(run_id)
    session_id_token = _ACTIVE_SESSION_ID.set(state.get("debug_session_id"))
    return MlflowTrackingContext(
        run_id_token=run_id_token,
        session_id_token=session_id_token,
    )


def deactivate_mlflow_tracking(context: MlflowTrackingContext | None) -> None:
    """Restore the previous MLflow tracking context."""

    if context is None:
        return
    if context.run_id_token is not None:
        _ACTIVE_RUN_ID.reset(context.run_id_token)
    if context.session_id_token is not None:
        _ACTIVE_SESSION_ID.reset(context.session_id_token)


def finalize_mlflow_tracking(state: CompanyFitState) -> None:
    """Flush session artifacts and finalize the MLflow run when terminal."""

    run_id = state.get("mlflow_run_id") or _ACTIVE_RUN_ID.get()
    if not run_id or not _is_tracking_enabled():
        return

    _update_run_metadata(run_id, state)
    _flush_session_prompt_log(run_id, state.get("debug_session_id"))
    log_json_artifact(
        f"workflow/state-{_timestamp_slug()}.json",
        _serialize_workflow_state(state),
    )

    if state.get("error"):
        log_text_artifact("workflow/error.txt", state["error"])

    status = state.get("session_status")
    if status in {"completed", "failed"}:
        log_json_artifact(
            "workflow/final-status.json",
            {
                "session_status": status,
                "error": state.get("error"),
                "finalized_at_utc": _utc_now_iso(),
            },
        )
        client = _get_mlflow_client()
        if client is not None:
            mlflow_status = "FINISHED" if status == "completed" else "FAILED"
            client.set_terminated(run_id, status=mlflow_status)


def log_text_artifact(artifact_path: str, content: str) -> None:
    """Persist one text artifact for the active MLflow run."""

    run_id = _ACTIVE_RUN_ID.get()
    if not run_id or not _is_tracking_enabled():
        return

    mlflow.log_text(content, artifact_file=artifact_path, run_id=run_id)
    logger.info("Logged MLflow text artifact run_id=%s path=%s", run_id, artifact_path)


def log_json_artifact(artifact_path: str, payload: Any) -> None:
    """Persist one JSON artifact for the active MLflow run."""

    content = json.dumps(payload, ensure_ascii=True, indent=2)
    log_text_artifact(artifact_path, content)


def log_clarification_question(question: str, target: str | None) -> None:
    """Persist one assistant clarification question as a session artifact."""

    content = (
        f"timestamp_utc: {_utc_now_iso()}\n"
        f"target: {target or 'unknown'}\n\n"
        f"{question.strip()}"
    )
    log_text_artifact(f"clarifications/question-{_timestamp_slug()}.txt", content)


def log_clarification_answer(question: str | None, answer: str, target: str | None) -> None:
    """Persist one user clarification answer as a session artifact."""

    run_id = _ACTIVE_RUN_ID.get()
    if not run_id or not _is_tracking_enabled():
        return
    _log_clarification_answer_for_run(run_id, question, answer, target)


def log_clarification_answer_for_run(
    run_id: str | None,
    question: str | None,
    answer: str,
    target: str | None,
) -> None:
    """Persist one user clarification answer for an explicit run id."""

    if not run_id or not _is_tracking_enabled():
        return
    _log_clarification_answer_for_run(run_id, question, answer, target)


def _log_clarification_answer_for_run(
    run_id: str,
    question: str | None,
    answer: str,
    target: str | None,
) -> None:
    """Build and log the clarification answer payload."""

    question_text = (question or "").strip() or "None"
    content = (
        f"timestamp_utc: {_utc_now_iso()}\n"
        f"target: {target or 'unknown'}\n\n"
        f"Agent question:\n{question_text}\n\n"
        f"User answer:\n{answer.strip()}"
    )
    log_text_artifact_for_run(
        run_id,
        f"clarifications/answer-{_timestamp_slug()}.txt",
        content,
    )


def log_llm_prompt_artifact(prefix: str, messages: list[BaseMessage]) -> None:
    """Attach one LLM prompt payload to the current MLflow span."""

    if not _is_tracking_enabled():
        return

    span = mlflow.get_current_active_span()
    if span is None:
        return

    rendered_messages = [
        {
            "index": index,
            "role": getattr(message, "type", message.__class__.__name__),
            "content": message.content if isinstance(message.content, str) else str(message.content),
        }
        for index, message in enumerate(messages, start=1)
    ]
    span.set_attribute("llm.prompt.prefix", prefix)
    span.set_inputs({"messages": rendered_messages})
    logger.info("Attached LLM prompt payload to active MLflow span prefix=%s", prefix)


def log_text_artifact_for_run(run_id: str, artifact_path: str, content: str) -> None:
    """Persist one text artifact for an explicit run id."""

    if not _is_tracking_enabled():
        return

    mlflow.log_text(content, artifact_file=artifact_path, run_id=run_id)


def _create_session_run(state: CompanyFitState) -> str | None:
    """Create the initial MLflow run for one workflow session."""

    client = _get_mlflow_client()
    if client is None:
        return None

    experiment_id = _ensure_experiment(client)
    if experiment_id is None:
        return None

    session_id = state.get("debug_session_id") or "unknown-session"
    run = client.create_run(
        experiment_id=experiment_id,
        tags={
            "mlflow.runName": f"session-{session_id[:8]}",
            "app.name": "company-fit-check",
            "session_id": session_id,
            "created_at_utc": _utc_now_iso(),
        },
    )
    run_id = run.info.run_id

    user_input = state.get("input")
    if user_input is not None:
        client.log_param(run_id, "prompt_length", len(user_input.prompt))
        client.log_param(run_id, "cv_pdf_bytes", len(user_input.cv_pdf_bytes))
        log_text_artifact_for_run(run_id, "inputs/user-prompt.txt", user_input.prompt)

    azure_settings = get_azure_openai_settings()
    if azure_settings.deployment:
        client.log_param(run_id, "azure_openai_deployment", azure_settings.deployment)

    logger.info("Created MLflow session run run_id=%s session_id=%s", run_id, session_id)
    return run_id


def _update_run_metadata(run_id: str, state: CompanyFitState) -> None:
    """Refresh run tags that mirror the current workflow state."""

    client = _get_mlflow_client()
    if client is None:
        return

    client.set_tag(run_id, "session_status", state.get("session_status") or "unknown")
    client.set_tag(
        run_id,
        "clarification_target",
        state.get("clarification_target") or "none",
    )
    client.set_tag(
        run_id,
        "interpretation_clarification_iterations",
        str(state.get("interpretation_clarification_iterations", 0)),
    )
    client.set_tag(
        run_id,
        "company_search_clarification_iterations",
        str(state.get("company_search_clarification_iterations", 0)),
    )
    client.set_tag(run_id, "last_updated_at_utc", _utc_now_iso())


def _ensure_experiment(client: MlflowClient) -> str | None:
    """Create the configured MLflow experiment if needed and return its id."""

    settings = get_mlflow_settings()
    experiment = client.get_experiment_by_name(settings.experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    if not settings.artifact_root:
        logger.warning("MLflow artifact root is not configured; skipping experiment setup.")
        return None

    experiment_id = client.create_experiment(
        settings.experiment_name,
        artifact_location=settings.artifact_root,
    )
    logger.info(
        "Created MLflow experiment name=%s experiment_id=%s artifact_root=%s",
        settings.experiment_name,
        experiment_id,
        settings.artifact_root,
    )
    return experiment_id


def _get_mlflow_client() -> MlflowClient | None:
    """Return a configured MLflow client when tracking is enabled."""

    if not _is_tracking_enabled():
        return None

    settings = get_mlflow_settings()
    os.environ["AZURE_STORAGE_CONNECTION_STRING"] = (
        settings.azure_storage_connection_string or ""
    )
    mlflow.set_tracking_uri(settings.tracking_uri)
    return MlflowClient(tracking_uri=settings.tracking_uri)


def _is_tracking_enabled() -> bool:
    """Return whether MLflow + Blob persistence is configured."""

    global _MISSING_CONFIG_WARNING_EMITTED

    settings = get_mlflow_settings()
    if settings.is_configured:
        return True

    if not _MISSING_CONFIG_WARNING_EMITTED:
        logger.warning(
            "MLflow tracking is disabled because required settings are incomplete."
        )
        _MISSING_CONFIG_WARNING_EMITTED = True
    return False


def _flush_session_prompt_log(run_id: str, session_id: str | None) -> None:
    """No-op now that prompt payloads are captured via MLflow tracing spans."""

    return


def _serialize_workflow_state(state: CompanyFitState) -> dict[str, Any]:
    """Convert the workflow state into JSON-safe data without raw PDF bytes."""

    user_input = state.get("input")
    serialized: dict[str, Any] = {
        "masked_cv_text": state.get("masked_cv_text"),
        "pii_masking_status": state.get("pii_masking_status"),
        "simplified_cv_text": state.get("simplified_cv_text"),
        "company_search_criteria": _json_ready(state.get("company_search_criteria")),
        "axes": _json_ready(state.get("axes", [])),
        "companies": _json_ready(state.get("companies", [])),
        "company_scores": _json_ready(state.get("company_scores", [])),
        "pending_clarification_message": state.get("pending_clarification_message"),
        "latest_clarification_response": state.get("latest_clarification_response"),
        "clarification_target": state.get("clarification_target"),
        "debug_session_id": state.get("debug_session_id"),
        "mlflow_run_id": state.get("mlflow_run_id"),
        "interpretation_clarification_iterations": state.get(
            "interpretation_clarification_iterations"
        ),
        "company_search_clarification_iterations": state.get(
            "company_search_clarification_iterations"
        ),
        "session_status": state.get("session_status"),
        "error": state.get("error"),
    }
    if user_input is not None:
        serialized["input"] = {
            "prompt": user_input.prompt,
            "cv_pdf_bytes_length": len(user_input.cv_pdf_bytes),
        }
    return serialized


def _json_ready(value: Any) -> Any:
    """Recursively convert Pydantic models and containers into JSON-safe data."""

    if isinstance(value, BaseModel):
        return _json_ready(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _timestamp_slug() -> str:
    """Return a UTC timestamp formatted for artifact file names."""

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


@contextmanager
def traced_operation(
    name: str,
    *,
    span_type: str = SpanType.UNKNOWN,
    inputs: Any | None = None,
    attributes: dict[str, Any] | None = None,
):
    """Create one MLflow tracing span around a logical workflow operation."""

    if not _is_tracking_enabled():
        yield None
        return

    with mlflow.start_span(name=name, span_type=span_type, attributes=attributes) as span:
        if inputs is not None:
            span.set_inputs(inputs)
        yield span
