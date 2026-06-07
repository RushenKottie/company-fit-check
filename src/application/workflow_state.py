"""Application-owned workflow state construction and clarification updates."""

from logging_utils import get_logger
from models.input import UserInput
from models.state import CompanyFitState, CompanySearchCriteria

from application.workflow_policy import is_pending_guardrail_rephrase
from application.workflow_transitions import (
    increment_company_search_clarification_iterations,
    increment_user_input_clarification_iterations,
    mark_running,
)

logger = get_logger(__name__)


def create_initial_state(user_input: UserInput, *, run_id: str) -> CompanyFitState:
    """Create fresh in-memory state for one workflow run."""

    logger.info(
        "Creating initial workflow state prompt_length=%s pdf_bytes=%s",
        len(user_input.prompt),
        len(user_input.cv_pdf_bytes),
    )
    return CompanyFitState(
        input=user_input,
        masked_cv_text=None,
        pii_masking_status="not_started",
        simplified_cv_text=None,
        company_search_criteria=CompanySearchCriteria(),
        axes=[],
        companies=[],
        company_scores=[],
        pending_clarification_message=None,
        latest_clarification_response=None,
        clarification_target=None,
        latest_user_message_text=user_input.prompt,
        latest_user_message_kind="prompt",
        guardrail_rephrase_source=None,
        run_id=run_id,
        user_input_interpretation_clarification_iterations=0,
        company_search_clarification_iterations=0,
        session_status="running",
        error=None,
    )


def apply_clarification_to_state(
    state: CompanyFitState,
    user_response: str,
) -> CompanyFitState:
    """Apply one user clarification before resuming the workflow graph."""

    logger.info(
        "Applying clarification target=%s response_length=%s",
        state.get("clarification_target"),
        len(user_response),
    )
    is_guardrail_rephrase = is_pending_guardrail_rephrase(state)
    if state.get("clarification_target") == "company_search":
        increment_company_search_clarification_iterations(state)
    elif not is_guardrail_rephrase:
        increment_user_input_clarification_iterations(state)

    if is_guardrail_rephrase:
        state["input"].prompt = user_response.strip()
        state["latest_clarification_response"] = None
        state["latest_user_message_text"] = user_response.strip()
        state["latest_user_message_kind"] = "prompt"
    else:
        state["latest_clarification_response"] = build_clarification_context(
            state.get("pending_clarification_message"),
            user_response,
        )
        state["latest_user_message_text"] = user_response.strip()
        state["latest_user_message_kind"] = "clarification"

    mark_running(state)
    state["pending_clarification_message"] = None
    state["guardrail_rephrase_source"] = None
    return state


def build_clarification_context(
    assistant_question: str | None,
    user_response: str,
) -> str:
    """Combine the clarification question and answer into one scoped context block."""

    question = (assistant_question or "").strip()
    answer = user_response.strip()
    if not question:
        return answer

    return (
        "Previous information was not enough, so the model requested clarification.\n"
        f"Agent question: {question}\n"
        f"User answer: {answer}"
    )
