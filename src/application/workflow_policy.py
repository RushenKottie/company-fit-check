"""Workflow retry and clarification policy."""

from llm.client import (
    build_rephrase_retry_exhausted_message,
    is_guardrail_rephrase_message,
)
from models.state import CompanyFitState

MAX_CLARIFICATION_ITERATIONS = 5
USER_INPUT_CLARIFICATION_EXHAUSTED_MESSAGE = (
    "User-input interpretation clarification did not make progress after 5 attempts."
)
COMPANY_SEARCH_CLARIFICATION_EXHAUSTED_MESSAGE = (
    "Company-search clarification did not make progress after 5 attempts."
)


def user_input_clarification_limit_reached(state: CompanyFitState) -> bool:
    """Return whether user-input clarification attempts are exhausted."""

    return (
        state.get("user_input_interpretation_clarification_iterations", 0)
        >= MAX_CLARIFICATION_ITERATIONS
    )


def company_search_clarification_limit_reached(state: CompanyFitState) -> bool:
    """Return whether company-search clarification attempts are exhausted."""

    return (
        state.get("company_search_clarification_iterations", 0)
        >= MAX_CLARIFICATION_ITERATIONS
    )


def guardrail_rephrase_exhausted_message() -> str:
    """Return the terminal message for repeated guardrail rephrase failures."""

    return build_rephrase_retry_exhausted_message()


def is_pending_guardrail_rephrase(state: CompanyFitState) -> bool:
    """Return whether the current clarification is a guardrail rephrase request."""

    return (
        state.get("clarification_target") == "user_input_interpretation"
        and is_guardrail_rephrase_message(state.get("pending_clarification_message"))
    )


def guardrail_blocked_message_source(state: CompanyFitState) -> str:
    """Return whether the blocked user message came from the prompt or clarification."""

    kind = state.get("latest_user_message_kind")
    if kind in {"prompt", "clarification"}:
        return kind
    return "clarification" if state.get("latest_clarification_response") else "prompt"


def guardrail_blocked_message_text(state: CompanyFitState) -> str:
    """Return the raw user message that most likely triggered the guardrail."""

    latest = (state.get("latest_user_message_text") or "").strip()
    if latest:
        return latest
    if state.get("latest_clarification_response"):
        return str(state.get("latest_clarification_response")).strip()
    if state.get("input") is not None:
        return state["input"].prompt.strip()
    return ""
