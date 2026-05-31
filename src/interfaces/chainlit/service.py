"""Thin Chainlit-to-backend workflow adapter."""

from dataclasses import dataclass

from graph.workflow import (
    apply_clarification,
    create_initial_state,
    run_workflow,
)
from interfaces.chainlit.presenters import (
    build_clarification_message,
    build_completion_message,
    build_failure_message,
)
from logging_utils import configure_logging, get_logger
from models.artifacts import GeneratedArtifact
from models.input import UserInput
from models.state import CompanyFitState
from services.mlflow_tracking import create_run_id, log_assistant_response_for_run
from services.result_exports import build_results_csv

logger = get_logger(__name__)


@dataclass(slots=True)
class SessionResult:
    """Backend response packaged for session-driven callers."""

    state: CompanyFitState
    assistant_message: str
    csv_artifact: GeneratedArtifact | None = None


def start_session(cv_pdf_bytes: bytes, prompt: str, run_id: str | None = None) -> SessionResult:
    """Start a new backend workflow session from a PDF CV and prompt."""

    configure_logging()
    logger.info(
        "Starting UI workflow session pdf_bytes=%s prompt_length=%s",
        len(cv_pdf_bytes),
        len(prompt),
    )
    user_input = UserInput(cv_pdf_bytes=cv_pdf_bytes, prompt=prompt)
    state = create_initial_state(
        user_input,
        run_id=run_id or create_run_id(user_input),
    )
    return _package_result(run_workflow(state))


def continue_session(
    state: CompanyFitState,
    user_response: str,
) -> SessionResult:
    """Resume an existing backend workflow session with user clarification."""

    configure_logging()
    logger.info(
        "Continuing UI workflow session target=%s response_length=%s",
        state.get("clarification_target"),
        len(user_response),
    )
    return _package_result(apply_clarification(state, user_response))


def _package_result(state: CompanyFitState) -> SessionResult:
    """Attach any backend-generated artifacts needed by the UI."""

    status = state.get("session_status")
    logger.info("Packaging workflow result status=%s", status)
    csv_artifact = None
    if status == "completed":
        logger.info("Building CSV artifact for completed workflow.")
        csv_artifact = build_results_csv(state)

    assistant_message = _build_assistant_message(state)
    log_assistant_response_for_run(
        state.get("run_id"),
        assistant_message,
        status=status,
    )
    return SessionResult(
        state=state,
        assistant_message=assistant_message,
        csv_artifact=csv_artifact,
    )


def _build_assistant_message(state: CompanyFitState) -> str:
    """Render the assistant-visible message for the current workflow state."""

    status = state.get("session_status")
    if status == "needs_clarification":
        return build_clarification_message(state)
    if status == "failed":
        return build_failure_message(state)
    return build_completion_message(state)
