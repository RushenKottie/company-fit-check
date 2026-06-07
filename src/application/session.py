"""backend entrypoint"""

from dataclasses import dataclass

from application.messages import build_assistant_message
from application.workflow_state import (
    apply_clarification_to_state,
    create_initial_state,
)
from graph.workflow import run_workflow
from logging_utils import configure_logging, get_logger
from models.artifacts import GeneratedArtifact
from models.input import UserInput
from models.state import CompanyFitState
from infrastructure.mlflow_tracking import (
    create_run_id,
    log_assistant_response_for_run,
    log_clarification_answer_for_run,
)
from application.result_exports import build_results_csv

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
        "Starting workflow session pdf_bytes=%s prompt_length=%s",
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
        "Continuing workflow session target=%s response_length=%s",
        state.get("clarification_target"),
        len(user_response),
    )
    log_clarification_answer_for_run(
        run_id=state.get("run_id"),
        question=state.get("pending_clarification_message"),
        answer=user_response,
        target=state.get("clarification_target"),
    )
    clarified_state = apply_clarification_to_state(state, user_response)
    return _package_result(run_workflow(clarified_state))


def _package_result(state: CompanyFitState) -> SessionResult:
    """Attach backend-generated artifacts needed by session callers."""

    status = state.get("session_status")
    logger.info("Packaging workflow result status=%s", status)
    csv_artifact = None
    if status == "completed":
        logger.info("Building CSV artifact for completed workflow.")
        csv_artifact = build_results_csv(state)

    assistant_message = build_assistant_message(state)
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
