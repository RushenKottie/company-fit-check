"""Routing decisions for the simplified graph."""

from graph.node_names import STOP_ROUTE, WorkflowNodeName
from models.state import (
    INTERRUPTED_SESSION_STATUSES,
    SESSION_STATUS_FAILED,
    SESSION_STATUS_RUNNING,
    TERMINAL_SESSION_STATUSES,
    CompanyFitState,
)


def route_from_entry(state: CompanyFitState) -> str:
    """Choose the first graph node for a new or resumed workflow state."""

    if state.get("session_status") in TERMINAL_SESSION_STATUSES:
        return STOP_ROUTE
    if state.get("latest_clarification_response"):
        return WorkflowNodeName.INTERPRET_USER_INPUT.value
    if state.get("session_status") == SESSION_STATUS_RUNNING:
        if state.get("clarification_target") == "user_input_interpretation":
            return WorkflowNodeName.INTERPRET_USER_INPUT.value
    if not state.get("masked_cv_text"):
        return WorkflowNodeName.EXTRACT_AND_MASK_CV.value
    return STOP_ROUTE


def route_after_privacy_check(state: CompanyFitState) -> str:
    """Route after privacy validation, stopping when masking failed."""

    if state.get("session_status") == SESSION_STATUS_FAILED:
        return STOP_ROUTE
    return WorkflowNodeName.SIMPLIFY_CV.value


def route_after_user_input_interpretation(state: CompanyFitState) -> str:
    """Route after interpretation, stopping when the session needs user action."""

    if state.get("session_status") in INTERRUPTED_SESSION_STATUSES:
        return STOP_ROUTE
    return WorkflowNodeName.VALIDATE_USER_INPUT_INTERPRETATION.value


def route_after_validation(state: CompanyFitState) -> str:
    """Route after interpretation validation, stopping on interrupted sessions."""

    if state.get("session_status") in INTERRUPTED_SESSION_STATUSES:
        return STOP_ROUTE
    return WorkflowNodeName.SEARCH_COMPANIES.value


def route_after_company_search(state: CompanyFitState) -> str:
    """Route after company search based on interruption status and search results."""

    if state.get("session_status") in INTERRUPTED_SESSION_STATUSES:
        return STOP_ROUTE
    if not state.get("companies"):
        return WorkflowNodeName.PREPARE_FINAL_RESULTS.value
    return WorkflowNodeName.SCORE_COMPANIES.value
