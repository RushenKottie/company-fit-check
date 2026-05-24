"""MLflow-backed persistence for prompt logs and workflow artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
import json
import os
import re
from typing import Any
import uuid

from langchain_core.messages import BaseMessage
import mlflow
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.entities.span import SpanType
from mlflow import MlflowClient
from pydantic import BaseModel

from config import get_azure_openai_settings, get_mlflow_settings
from logging_utils import get_logger
from models.input import UserInput
from models.state import CompanyFitState

logger = get_logger(__name__)
_ACTIVE_RUN_ID: ContextVar[str | None] = ContextVar("active_mlflow_run_id", default=None)
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
_TRACKING_CAPTURE_STACK: ContextVar[list[dict[str, Any]]] = ContextVar(
    "mlflow_tracking_capture_stack",
    default=[],
)
_ARTIFACT_NAMESPACE: ContextVar[str | None] = ContextVar(
    "mlflow_artifact_namespace",
    default=None,
)
_MISSING_CONFIG_WARNING_EMITTED = False


class MlflowTrackingContext:
    """Opaque tokens for one active MLflow workflow context."""

    def __init__(
        self,
        run_id_token: Token[str | None] | None,
        started_fluent_run: bool = False,
    ) -> None:
        self.run_id_token = run_id_token
        self.started_fluent_run = started_fluent_run


@dataclass
class TrackingCapture:
    """In-memory observation of spans and artifacts for deterministic evaluation."""

    spans: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)


class ObservedSpan:
    """Small proxy that mirrors span mutations into the active tracking capture."""

    def __init__(self, span: Any, event: dict[str, Any]) -> None:
        self._span = span
        self._event = event

    def set_inputs(self, inputs: Any) -> None:
        self._event["inputs"] = _json_ready(inputs)
        self._span.set_inputs(inputs)

    def set_outputs(self, outputs: Any) -> None:
        self._event["outputs"] = _json_ready(outputs)
        self._span.set_outputs(outputs)

    def set_attribute(self, key: str, value: Any) -> None:
        attributes = self._event.setdefault("attributes", {})
        attributes[str(key)] = _json_ready(value)
        self._span.set_attribute(key, value)


@contextmanager
def bind_mlflow_run(
    run_id: str,
):
    """Temporarily bind one existing MLflow run to the current execution context."""

    run_id_token = _ACTIVE_RUN_ID.set(run_id)
    try:
        yield
    finally:
        _ACTIVE_RUN_ID.reset(run_id_token)


@contextmanager
def suspend_mlflow_run_termination():
    """Prevent workflow finalization from terminating the current run."""

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
    """Temporarily override the MLflow experiment used for new runs."""

    normalized_name = experiment_name.strip() if experiment_name else None
    token = _ACTIVE_EXPERIMENT_NAME.set(normalized_name)
    try:
        yield
    finally:
        _ACTIVE_EXPERIMENT_NAME.reset(token)


def create_run_id(user_input: UserInput) -> str:
    """Return one workflow run id, creating an MLflow run when tracking is enabled."""

    if not _is_tracking_enabled():
        return uuid.uuid4().hex

    return _create_session_run(user_input)


def activate_mlflow_tracking(state: CompanyFitState) -> MlflowTrackingContext | None:
    """Ensure a session run exists and bind it to the current execution context."""

    if not _is_tracking_enabled():
        return None

    run_id = state.get("run_id")
    if not run_id:
        return None

    settings = get_mlflow_settings()
    mlflow.set_experiment(_get_effective_experiment_name(settings))
    _update_run_metadata(run_id, state)
    run_id_token = _ACTIVE_RUN_ID.set(run_id)
    started_fluent_run = False
    active_run = mlflow.active_run()
    if active_run is None:
        mlflow.start_run(run_id=run_id)
        started_fluent_run = True
    elif active_run.info.run_id != run_id:
        logger.warning(
            "MLflow tracking activated with mismatched active run current=%s requested=%s",
            active_run.info.run_id,
            run_id,
        )
    return MlflowTrackingContext(
        run_id_token=run_id_token,
        started_fluent_run=started_fluent_run,
    )


def deactivate_mlflow_tracking(context: MlflowTrackingContext | None) -> None:
    """Restore the previous MLflow tracking context."""

    if context is None:
        return
    if context.started_fluent_run:
        mlflow.end_run()
    if context.run_id_token is not None:
        _ACTIVE_RUN_ID.reset(context.run_id_token)


def finalize_mlflow_tracking(state: CompanyFitState) -> None:
    """Flush session artifacts and finalize the MLflow run when terminal."""

    run_id = state.get("run_id") or _ACTIVE_RUN_ID.get()
    if not run_id or not _is_tracking_enabled():
        return

    _update_run_metadata(run_id, state)
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
        if client is not None and not _RUN_TERMINATION_SUSPENDED.get():
            mlflow_status = "FINISHED" if status == "completed" else "FAILED"
            client.set_terminated(run_id, status=mlflow_status)


def update_current_trace_session(
    *,
    session_id: str | None,
    request_preview: str | None = None,
    response_preview: str | None = None,
) -> None:
    """Annotate the active trace so MLflow can surface it in the Sessions view."""

    if not _is_tracking_enabled() or not session_id:
        return

    kwargs: dict[str, Any] = {"session_id": session_id}
    if request_preview:
        kwargs["request_preview"] = request_preview
    if response_preview:
        kwargs["response_preview"] = response_preview

    try:
        mlflow.update_current_trace(**kwargs)
    except TypeError:
        # Older MLflow releases may not support the newer keyword arguments yet.
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})


def log_text_artifact(artifact_path: str, content: str) -> None:
    """Persist one text artifact for the active MLflow run."""

    run_id = _ACTIVE_RUN_ID.get()
    if not run_id or not _is_tracking_enabled():
        return

    resolved_path = _resolve_artifact_path(artifact_path)
    _record_logged_artifact(run_id, resolved_path, content)
    mlflow.log_text(content, artifact_file=resolved_path, run_id=run_id)
    logger.info("Logged MLflow text artifact run_id=%s path=%s", run_id, resolved_path)


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


def log_assistant_response_for_run(
    run_id: str | None,
    message: str,
    *,
    status: str | None,
) -> None:
    """Persist one rendered assistant response for an explicit run id."""

    if not run_id or not _is_tracking_enabled():
        return

    content = (
        f"timestamp_utc: {_utc_now_iso()}\n"
        f"status: {status or 'unknown'}\n\n"
        f"{message.strip()}"
    )
    log_text_artifact_for_run(
        run_id,
        f"assistant/response-{_timestamp_slug()}.txt",
        content,
    )


def log_guardrail_blocked_user_message(
    run_id: str | None,
    message: str,
    source: str | None,
) -> None:
    """Persist one user message that triggered a guardrail block."""

    if not run_id or not _is_tracking_enabled():
        return
    content = (
        f"timestamp_utc: {_utc_now_iso()}\n"
        f"source: {source or 'unknown'}\n\n"
        f"{message.strip()}"
    )
    log_text_artifact_for_run(
        run_id,
        f"guardrails/blocked-user-message-{_timestamp_slug()}.txt",
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
    stack = _TRACKING_CAPTURE_STACK.get()
    if stack:
        stack[-1]["prompt_prefix"] = prefix
        stack[-1]["prompt_messages"] = _json_ready(rendered_messages)
    span.set_attribute("llm.prompt.prefix", prefix)
    span.set_inputs({"messages": rendered_messages})
    logger.info("Attached LLM prompt payload to active MLflow span prefix=%s", prefix)


def log_text_artifact_for_run(run_id: str, artifact_path: str, content: str) -> None:
    """Persist one text artifact for an explicit run id."""

    if not _is_tracking_enabled():
        return

    resolved_path = _resolve_artifact_path(artifact_path)
    _record_logged_artifact(run_id, resolved_path, content)
    mlflow.log_text(content, artifact_file=resolved_path, run_id=run_id)


def log_json_artifact_for_run(run_id: str, artifact_path: str, payload: Any) -> None:
    """Persist one JSON artifact for an explicit run id."""

    log_text_artifact_for_run(
        run_id,
        artifact_path,
        json.dumps(payload, ensure_ascii=True, indent=2),
    )


def log_metric_for_run(run_id: str, metric_name: str, value: float) -> None:
    """Persist one numeric metric for an explicit run id."""

    if not run_id or not _is_tracking_enabled():
        return

    client = _get_mlflow_client()
    if client is None:
        return
    client.log_metric(run_id, metric_name, value)


def set_run_status_for_run(
    run_id: str,
    *,
    status: str,
    error: str | None = None,
) -> None:
    """Update one existing MLflow run to a completed or failed terminal status."""

    if not run_id or not _is_tracking_enabled():
        return
    if status not in {"completed", "failed"}:
        return

    log_json_artifact_for_run(
        run_id,
        "workflow/final-status.json",
        {
            "session_status": status,
            "error": error,
            "finalized_at_utc": _utc_now_iso(),
        },
    )
    if error:
        log_text_artifact_for_run(run_id, "workflow/error.txt", error)

    client = _get_mlflow_client()
    if client is None:
        return
    client.set_terminated(run_id, status="FINISHED" if status == "completed" else "FAILED")


def set_run_name_for_run(run_id: str, run_name: str) -> None:
    """Persist one user-friendly display name for an explicit run id."""

    if not run_id or not _is_tracking_enabled():
        return

    normalized = re.sub(r"[^a-z0-9]+", "_", run_name.lower()).strip("_")
    if not normalized:
        return

    client = _get_mlflow_client()
    if client is None:
        return
    client.set_tag(run_id, "mlflow.runName", normalized)


def ensure_case_dataset_for_run(
    run_id: str,
    *,
    case_id: int,
    case_name: str,
    case_payload: dict[str, Any],
) -> str | None:
    """Create or reuse one MLflow dataset for a regression case and link the run to it."""

    if not run_id or not _is_tracking_enabled():
        return None

    try:
        client = _get_mlflow_client()
        if client is None:
            return None

        experiment_id = _ensure_experiment(client)
        if experiment_id is None:
            return None

        dataset_name = _build_case_dataset_name(case_id=case_id, case_name=case_name)
        dataset = _get_or_create_case_dataset(
            client,
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            case_id=case_id,
            case_name=case_name,
        )
        client.set_dataset_tags(
            dataset.dataset_id,
            _build_case_dataset_tags(
                case_id=case_id,
                case_name=case_name,
                case_payload=case_payload,
            ),
        )
        client.log_inputs(
            run_id,
            datasets=[
                DatasetInput(
                    dataset=dataset._to_mlflow_entity(),
                    tags=[
                        InputTag(key="mlflow.data.context", value="evaluation"),
                        InputTag(key="case_id", value=str(case_id)),
                    ],
                )
            ],
        )
        logger.info(
            "Linked MLflow run to case dataset run_id=%s dataset_id=%s dataset_name=%s",
            run_id,
            dataset.dataset_id,
            dataset_name,
        )
        return dataset.dataset_id
    except Exception:
        logger.exception(
            "Failed to create or link MLflow case dataset run_id=%s case_id=%s",
            run_id,
            case_id,
        )
        return None


def _create_session_run(user_input: UserInput) -> str:
    """Create the initial MLflow run for one workflow session."""

    client = _get_mlflow_client()
    if client is None:
        return uuid.uuid4().hex

    experiment_id = _ensure_experiment(client)
    if experiment_id is None:
        return uuid.uuid4().hex

    run = client.create_run(
        experiment_id=experiment_id,
        tags={
            "mlflow.runName": "session-pending",
            "app.name": "company-fit-check",
            "created_at_utc": _utc_now_iso(),
        },
    )
    run_id = run.info.run_id
    client.set_tag(run_id, "mlflow.runName", f"session-{run_id[:8]}")

    client.log_param(run_id, "prompt_length", len(user_input.prompt))
    client.log_param(run_id, "cv_pdf_bytes", len(user_input.cv_pdf_bytes))
    log_text_artifact_for_run(run_id, "inputs/user-prompt.txt", user_input.prompt)

    azure_settings = get_azure_openai_settings()
    if azure_settings.deployment:
        client.log_param(run_id, "azure_openai_deployment", azure_settings.deployment)

    logger.info("Created MLflow session run run_id=%s", run_id)
    return run_id


def _build_case_dataset_name(*, case_id: int, case_name: str) -> str:
    """Return one stable MLflow dataset name for a regression case."""

    return f"regression_case_{case_id}"


def _get_or_create_case_dataset(
    client: MlflowClient,
    *,
    experiment_id: str,
    dataset_name: str,
    case_id: int,
    case_name: str,
):
    """Return one existing case dataset or create and associate it."""

    matches = client.search_datasets(
        filter_string=f"name = '{dataset_name}'",
        max_results=2,
    )
    if matches:
        dataset = matches[0]
        if experiment_id not in dataset.experiment_ids:
            dataset = client.add_dataset_to_experiments(dataset.dataset_id, [experiment_id])
        return dataset

    return client.create_dataset(
        name=dataset_name,
        experiment_id=[experiment_id],
        tags={
            "suite": "llm_regression",
            "case_id": str(case_id),
            "case_name": case_name,
        },
    )


def _build_case_dataset_tags(
    *,
    case_id: int,
    case_name: str,
    case_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return dataset tags that keep one readable case summary on the dataset object."""

    payload = _json_ready(case_payload)
    return {
        "suite": "llm_regression",
        "case_id": str(case_id),
        "case_name": case_name,
        "profession": str(payload.get("profession") or ""),
        "pdf_path": str(payload.get("pdf_path") or ""),
        "updated_at_utc": _utc_now_iso(),
    }


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
        "user_input_interpretation_clarification_iterations",
        str(state.get("user_input_interpretation_clarification_iterations", 0)),
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
    experiment_name = _get_effective_experiment_name(settings)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    if settings.artifact_root:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=settings.artifact_root,
        )
    else:
        experiment_id = client.create_experiment(experiment_name)
    logger.info(
        "Created MLflow experiment name=%s experiment_id=%s artifact_root=%s",
        experiment_name,
        experiment_id,
        settings.artifact_root,
    )
    return experiment_id


def _get_effective_experiment_name(settings) -> str:
    """Return the active experiment name, honoring any temporary override."""

    return _ACTIVE_EXPERIMENT_NAME.get() or settings.experiment_name


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
        "run_id": state.get("run_id"),
        "user_input_interpretation_clarification_iterations": state.get(
            "user_input_interpretation_clarification_iterations"
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


def _record_logged_artifact(run_id: str, artifact_path: str, content: str) -> None:
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


def _resolve_artifact_path(artifact_path: str) -> str:
    """Apply the current artifact namespace to one relative artifact path."""

    namespace = _ARTIFACT_NAMESPACE.get()
    if not namespace:
        return artifact_path
    clean_path = artifact_path.lstrip("/")
    return f"{namespace}/{clean_path}" if clean_path else namespace


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
        capture = _TRACKING_CAPTURE.get()
        event: dict[str, Any] | None = None
        observed_span: Any = span
        if capture is not None:
            event = {
                "name": name,
                "span_type": str(span_type),
                "attributes": _json_ready(attributes or {}),
                "inputs": _json_ready(inputs),
                "outputs": None,
            }
            capture.spans.append(event)
            stack = list(_TRACKING_CAPTURE_STACK.get())
            stack.append(event)
            _TRACKING_CAPTURE_STACK.set(stack)
            observed_span = ObservedSpan(span, event)
        try:
            if inputs is not None:
                observed_span.set_inputs(inputs)
            yield observed_span
        finally:
            if event is not None:
                stack = list(_TRACKING_CAPTURE_STACK.get())
                if stack:
                    stack.pop()
                    _TRACKING_CAPTURE_STACK.set(stack)
