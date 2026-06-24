"""MLflow run lifecycle for app sessions and workflow execution."""

from __future__ import annotations

from typing import Any
import uuid

import mlflow
from pydantic import BaseModel, Field

from config import get_azure_openai_settings, get_mlflow_settings
from infrastructure.mlflow.artifacts import (
    log_json_artifact,
    log_text_artifact,
    log_text_artifact_for_run,
)
from infrastructure.mlflow.common import (
    ensure_experiment,
    get_mlflow_client,
    is_tracking_enabled,
    to_json_safe,
    resolve_experiment_name,
    safe_mlflow_call,
    utc_now_iso,
    artifact_timestamp,
)
from infrastructure.mlflow.context import (
    _ACTIVE_RUN_ID,
    _RUN_TERMINATION_SUSPENDED,
    MlflowTrackingContext,
)
from logging_utils import get_logger
from models.input import UserInput
from models.state import (
    Axis,
    ClarificationTarget,
    CompanyCandidate,
    SESSION_STATUS_COMPLETED,
    TERMINAL_SESSION_STATUSES,
    CompanyScore,
    CompanyFitState,
    CompanySearchCriteria,
    FinalCompanyResult,
    SessionStatus,
)

logger = get_logger(__name__)


class WorkflowInputSnapshot(BaseModel):
    """Serializable user input metadata for an MLflow workflow snapshot."""

    prompt: str
    cv_pdf_bytes_length: int


class WorkflowStateSnapshot(BaseModel):
    """Serializable MLflow artifact snapshot of the workflow state."""

    masked_cv_text: str | None = None
    pii_masking_status: str | None = None
    simplified_cv_text: str | None = None
    company_search_criteria: CompanySearchCriteria | None = None
    axes: list[Axis] = Field(default_factory=list)
    companies: list[CompanyCandidate] = Field(default_factory=list)
    company_scores: list[CompanyScore] = Field(default_factory=list)
    final_results: list[FinalCompanyResult] = Field(default_factory=list)
    pending_clarification_message: str | None = None
    latest_clarification_response: str | None = None
    clarification_target: ClarificationTarget | None = None
    run_id: str | None = None
    user_input_interpretation_clarification_iterations: int | None = None
    session_status: SessionStatus | None = None
    error: str | None = None
    input: WorkflowInputSnapshot | None = None


def create_run_id(user_input: UserInput) -> str:
    """Return one workflow run id, creating an MLflow run when tracking is enabled."""

    if not is_tracking_enabled():
        return uuid.uuid4().hex

    return _create_session_run(user_input)


def activate_mlflow_tracking(state: CompanyFitState) -> MlflowTrackingContext | None:
    """Ensure a session run exists and bind it to the current execution context."""

    if not is_tracking_enabled():
        return None

    run_id = state.get("run_id")
    if not run_id:
        return None

    settings = get_mlflow_settings()
    safe_mlflow_call(
        "set MLflow experiment",
        lambda: mlflow.set_experiment(resolve_experiment_name(settings)),
        None,
    )
    update_run_metadata(run_id, state)
    run_id_token = _ACTIVE_RUN_ID.set(run_id)
    started_fluent_run = False
    active_run = safe_mlflow_call("get active run", mlflow.active_run, None)
    if active_run is None:
        started_fluent_run = bool(
            safe_mlflow_call(
                "start MLflow run",
                lambda: mlflow.start_run(run_id=run_id),
                None,
            )
        )
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
        safe_mlflow_call("end MLflow run", mlflow.end_run, None)
    if context.run_id_token is not None:
        _ACTIVE_RUN_ID.reset(context.run_id_token)


def finalize_mlflow_tracking(state: CompanyFitState) -> None:
    """Flush session artifacts and finalize the MLflow run when terminal."""

    run_id = state.get("run_id") or _ACTIVE_RUN_ID.get()
    if not run_id or not is_tracking_enabled():
        return

    update_run_metadata(run_id, state)
    log_json_artifact(
        f"workflow/state-{artifact_timestamp()}.json",
        serialize_workflow_state(state),
    )

    if state.get("error"):
        log_text_artifact("workflow/error.txt", state["error"])

    status = state.get("session_status")
    if status in TERMINAL_SESSION_STATUSES:
        log_json_artifact(
            "workflow/final-status.json",
            {
                "session_status": status,
                "error": state.get("error"),
                "finalized_at_utc": utc_now_iso(),
            },
        )
        client = get_mlflow_client()
        if client is not None and not _RUN_TERMINATION_SUSPENDED.get():
            mlflow_status = "FINISHED" if status == SESSION_STATUS_COMPLETED else "FAILED"
            safe_mlflow_call(
                "set run terminated",
                lambda: client.set_terminated(run_id, status=mlflow_status),
                None,
            )


def _create_session_run(user_input: UserInput) -> str:
    """Create the initial MLflow run for one workflow session."""

    client = get_mlflow_client()
    if client is None:
        return uuid.uuid4().hex

    experiment_id = safe_mlflow_call(
        "ensure MLflow experiment",
        lambda: ensure_experiment(client),
        None,
    )
    if experiment_id is None:
        return uuid.uuid4().hex

    run = safe_mlflow_call(
        "create MLflow session run",
        lambda: client.create_run(
            experiment_id=experiment_id,
            tags={
                "mlflow.runName": "session-pending",
                "app.name": "company-fit-check",
                "created_at_utc": utc_now_iso(),
            },
        ),
        None,
    )
    if run is None:
        return uuid.uuid4().hex

    run_id = run.info.run_id
    safe_mlflow_call(
        "set session run name",
        lambda: client.set_tag(run_id, "mlflow.runName", f"session-{run_id[:8]}"),
        None,
    )
    safe_mlflow_call(
        "log prompt length parameter",
        lambda: client.log_param(run_id, "prompt_length", len(user_input.prompt)),
        None,
    )
    safe_mlflow_call(
        "log CV byte length parameter",
        lambda: client.log_param(run_id, "cv_pdf_bytes", len(user_input.cv_pdf_bytes)),
        None,
    )
    log_text_artifact_for_run(run_id, "inputs/user-prompt.txt", user_input.prompt)

    azure_settings = get_azure_openai_settings()
    if azure_settings.deployment:
        safe_mlflow_call(
            "log Azure OpenAI deployment parameter",
            lambda: client.log_param(
                run_id,
                "azure_openai_deployment",
                azure_settings.deployment,
            ),
            None,
        )

    logger.info("Created MLflow session run run_id=%s", run_id)
    return run_id


def update_run_metadata(run_id: str, state: CompanyFitState) -> None:
    """Update MLflow tags with the latest workflow state."""

    client = get_mlflow_client()
    if client is None:
        return

    safe_mlflow_call(
        "set session status tag",
        lambda: client.set_tag(
            run_id,
            "session_status",
            state.get("session_status") or "unknown",
        ),
        None,
    )
    safe_mlflow_call(
        "set clarification target tag",
        lambda: client.set_tag(
            run_id,
            "clarification_target",
            state.get("clarification_target") or "none",
        ),
        None,
    )
    safe_mlflow_call(
        "set user-input clarification iteration tag",
        lambda: client.set_tag(
            run_id,
            "user_input_interpretation_clarification_iterations",
            str(state.get("user_input_interpretation_clarification_iterations", 0)),
        ),
        None,
    )
    safe_mlflow_call(
        "set last-updated tag",
        lambda: client.set_tag(run_id, "last_updated_at_utc", utc_now_iso()),
        None,
    )


def serialize_workflow_state(state: CompanyFitState) -> dict[str, Any]:
    """Convert the workflow state into JSON-safe data without raw PDF bytes."""

    user_input = state.get("input")
    snapshot = WorkflowStateSnapshot(
        masked_cv_text=state.get("masked_cv_text"),
        pii_masking_status=state.get("pii_masking_status"),
        simplified_cv_text=state.get("simplified_cv_text"),
        company_search_criteria=state.get("company_search_criteria"),
        axes=state.get("axes", []),
        companies=state.get("companies", []),
        company_scores=state.get("company_scores", []),
        final_results=state.get("final_results", []),
        pending_clarification_message=state.get("pending_clarification_message"),
        latest_clarification_response=state.get("latest_clarification_response"),
        clarification_target=state.get("clarification_target"),
        run_id=state.get("run_id"),
        user_input_interpretation_clarification_iterations=state.get(
            "user_input_interpretation_clarification_iterations"
        ),
        session_status=state.get("session_status"),
        error=state.get("error"),
        input=(
            WorkflowInputSnapshot(
                prompt=user_input.prompt,
                cv_pdf_bytes_length=len(user_input.cv_pdf_bytes),
            )
            if user_input is not None
            else None
        ),
    )
    return to_json_safe(snapshot)
