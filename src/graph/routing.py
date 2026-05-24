"""Routing decisions for the simplified graph."""

from typing import Literal

from models.state import CompanyFitState


def route_from_entry(
    state: CompanyFitState,
) -> Literal[
    "extract_and_mask_cv",
    "interpret_user_input",
    "refine_company_search",
    "stop",
]:
    if state.get("session_status") in {"completed", "failed"}:
        return "stop"
    if state.get("latest_clarification_response"):
        if state.get("clarification_target") == "company_search":
            return "refine_company_search"
        return "interpret_user_input"
    if state.get("session_status") == "running":
        if state.get("clarification_target") == "company_search":
            return "refine_company_search"
        if state.get("clarification_target") == "user_input_interpretation":
            return "interpret_user_input"
    if not state.get("masked_cv_text"):
        return "extract_and_mask_cv"
    return "stop"


def route_after_privacy_check(
    state: CompanyFitState,
) -> Literal["simplify_cv", "stop"]:
    if state.get("session_status") == "failed":
        return "stop"
    return "simplify_cv"


def route_after_user_input_interpretation(
    state: CompanyFitState,
) -> Literal["validate_user_input_interpretation", "stop"]:
    if state.get("session_status") in {"failed", "needs_clarification"}:
        return "stop"
    return "validate_user_input_interpretation"


def route_after_validation(
    state: CompanyFitState,
) -> Literal["search_companies", "stop"]:
    if state.get("session_status") in {"failed", "needs_clarification"}:
        return "stop"
    return "search_companies"


def route_after_company_search_refinement(
    state: CompanyFitState,
) -> Literal["search_companies", "stop"]:
    if state.get("session_status") == "failed":
        return "stop"
    return "search_companies"


def route_after_company_search(
    state: CompanyFitState,
) -> Literal["score_companies", "stop"]:
    if state.get("session_status") in {"failed", "needs_clarification"}:
        return "stop"
    return "score_companies"
