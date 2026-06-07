"""Routing decisions for the simplified graph."""

from graph.node_names import WorkflowNodeName, WorkflowRouteName
from models.state import CompanyFitState


def route_from_entry(state: CompanyFitState) -> str:
    if state.get("session_status") in {"completed", "failed"}:
        return WorkflowRouteName.STOP.value
    if state.get("latest_clarification_response"):
        if state.get("clarification_target") == "company_search":
            return WorkflowNodeName.REFINE_COMPANY_SEARCH.value
        return WorkflowNodeName.INTERPRET_USER_INPUT.value
    if state.get("session_status") == "running":
        if state.get("clarification_target") == "company_search":
            return WorkflowNodeName.REFINE_COMPANY_SEARCH.value
        if state.get("clarification_target") == "user_input_interpretation":
            return WorkflowNodeName.INTERPRET_USER_INPUT.value
    if not state.get("masked_cv_text"):
        return WorkflowNodeName.EXTRACT_AND_MASK_CV.value
    return WorkflowRouteName.STOP.value


def route_after_privacy_check(state: CompanyFitState) -> str:
    if state.get("session_status") == "failed":
        return WorkflowRouteName.STOP.value
    return WorkflowNodeName.SIMPLIFY_CV.value


def route_after_user_input_interpretation(state: CompanyFitState) -> str:
    if state.get("session_status") in {"failed", "needs_clarification"}:
        return WorkflowRouteName.STOP.value
    return WorkflowNodeName.VALIDATE_USER_INPUT_INTERPRETATION.value


def route_after_validation(state: CompanyFitState) -> str:
    if state.get("session_status") in {"failed", "needs_clarification"}:
        return WorkflowRouteName.STOP.value
    return WorkflowNodeName.SEARCH_COMPANIES.value


def route_after_company_search_refinement(state: CompanyFitState) -> str:
    if state.get("session_status") == "failed":
        return WorkflowRouteName.STOP.value
    return WorkflowNodeName.SEARCH_COMPANIES.value


def route_after_company_search(state: CompanyFitState) -> str:
    if state.get("session_status") in {"failed", "needs_clarification"}:
        return WorkflowRouteName.STOP.value
    return WorkflowNodeName.SCORE_COMPANIES.value
