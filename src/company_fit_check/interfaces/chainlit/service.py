"""Thin Chainlit-to-backend workflow adapter."""

from dataclasses import dataclass

from company_fit_check.graph.workflow import (
    apply_clarification,
    create_initial_state,
    run_workflow,
)
from company_fit_check.logging_utils import get_logger
from company_fit_check.models.artifacts import GeneratedArtifact
from company_fit_check.models.input import UserInput
from company_fit_check.models.state import CompanyFitState
from company_fit_check.services.result_exports import build_results_csv

logger = get_logger(__name__)


@dataclass(slots=True)
class UiWorkflowResult:
    """Backend response packaged for the Chainlit UI layer."""

    state: CompanyFitState
    csv_artifact: GeneratedArtifact | None = None


def start_session(cv_pdf_bytes: bytes, prompt: str) -> UiWorkflowResult:
    """Start a new backend workflow session from a PDF CV and prompt."""

    logger.info(
        "Starting UI workflow session pdf_bytes=%s prompt_length=%s",
        len(cv_pdf_bytes),
        len(prompt),
    )
    state = create_initial_state(
        UserInput(cv_pdf_bytes=cv_pdf_bytes, prompt=prompt),
    )
    return _package_result(run_workflow(state))


def continue_session(
    state: CompanyFitState,
    user_response: str,
) -> UiWorkflowResult:
    """Resume an existing backend workflow session with user clarification."""

    logger.info(
        "Continuing UI workflow session target=%s response_length=%s",
        state.get("clarification_target"),
        len(user_response),
    )
    return _package_result(apply_clarification(state, user_response))


def _package_result(state: CompanyFitState) -> UiWorkflowResult:
    """Attach any backend-generated artifacts needed by the UI."""

    logger.info("Packaging workflow result status=%s", state.get("session_status"))
    csv_artifact = None
    if state.get("session_status") == "completed":
        logger.info("Building CSV artifact for completed workflow.")
        csv_artifact = build_results_csv(state)
    return UiWorkflowResult(state=state, csv_artifact=csv_artifact)
