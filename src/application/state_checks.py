"""Pure workflow state checks and summaries."""

from typing import Any

from models.state import CompanyFitState


def validate_search_prerequisites(state: CompanyFitState) -> str | None:
    """Return one deterministic error when company search prerequisites are missing."""

    if state.get("session_status") == "failed":
        return "Workflow is already in a failed state."
    if not (state.get("simplified_cv_text") or "").strip():
        return "Company search requires simplified CV text."
    if not state.get("axes"):
        return "Company search requires at least one matching axis."
    if state.get("company_search_criteria") is None:
        return "Company search requires initialized search criteria."
    return None


def validate_company_score_payload(state: CompanyFitState) -> str | None:
    """Return one deterministic error when the company score payload is malformed."""

    axes = state.get("axes", [])
    companies = state.get("companies", [])
    company_scores = state.get("company_scores", [])
    axis_names = [axis.name for axis in axes if axis.name.strip()]
    known_company_names = {company.name for company in companies}

    if not company_scores:
        return "Company scoring returned no company scores."

    for company_score in company_scores:
        if company_score.company_name not in known_company_names:
            return f"Company score references unknown company: {company_score.company_name}"

        score_by_axis = {
            axis_score.axis: axis_score.percentage
            for axis_score in company_score.axis_scores
        }
        if len(score_by_axis) != len(axis_names):
            return (
                f"Company {company_score.company_name} does not contain exactly one "
                "score for every axis."
            )

        for axis_name in axis_names:
            if axis_name not in score_by_axis:
                return f"Company {company_score.company_name} is missing axis score: {axis_name}"
            if not 0.0 <= score_by_axis[axis_name] <= 100.0:
                return (
                    f"Company {company_score.company_name} has out-of-range axis score: "
                    f"{axis_name}={score_by_axis[axis_name]}"
                )

        if not 0.0 <= company_score.overall_score <= 100.0:
            return (
                f"Company {company_score.company_name} has out-of-range overall score: "
                f"{company_score.overall_score}"
            )

    scored_company_names = {
        company_score.company_name for company_score in company_scores
    }
    if known_company_names != scored_company_names:
        missing = sorted(known_company_names - scored_company_names)
        return f"Missing company scores for: {', '.join(missing)}"

    return None


def build_case_result_summary(state: CompanyFitState) -> dict[str, Any]:
    """Return a compact JSON-safe state summary for deterministic eval artifacts."""

    return {
        "input": (
            {"prompt": state["input"].prompt}
            if state.get("input") is not None
            else None
        ),
        "session_status": state.get("session_status"),
        "error": state.get("error"),
        "pii_masking_status": state.get("pii_masking_status"),
        "clarification_target": state.get("clarification_target"),
        "user_input_interpretation_clarification_iterations": state.get(
            "user_input_interpretation_clarification_iterations",
            0,
        ),
        "company_search_clarification_iterations": state.get(
            "company_search_clarification_iterations",
            0,
        ),
        "masked_cv_text": state.get("masked_cv_text"),
        "simplified_cv_text": state.get("simplified_cv_text"),
        "axes": [axis.model_dump() for axis in state.get("axes", [])],
        "company_search_criteria": (
            state["company_search_criteria"].model_dump()
            if state.get("company_search_criteria") is not None
            else None
        ),
        "companies": [company.model_dump() for company in state.get("companies", [])],
        "company_scores": [
            score.model_dump() for score in state.get("company_scores", [])
        ],
        "pending_clarification_message": state.get("pending_clarification_message"),
        "latest_user_message_text": state.get("latest_user_message_text"),
        "latest_user_message_kind": state.get("latest_user_message_kind"),
        "guardrail_rephrase_source": state.get("guardrail_rephrase_source"),
    }
