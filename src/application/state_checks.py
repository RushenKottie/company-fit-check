"""Pure workflow state checks."""

from models.state import SESSION_STATUS_FAILED, CompanyFitState


def validate_search_prerequisites(state: CompanyFitState) -> str | None:
    """Return one deterministic error when company search prerequisites are missing."""

    if state.get("session_status") == SESSION_STATUS_FAILED:
        return "Workflow is already in a failed state."
    if not (state.get("simplified_cv_text") or "").strip():
        return "Company search requires simplified CV text."
    if not state.get("axes"):
        return "Company search requires at least one matching axis."
    if state.get("company_search_criteria") is None:
        return "Company search requires initialized search criteria."
    return None


def validate_company_score_payload(state: CompanyFitState) -> str | None:
    """Return one error when normalized company scores are incomplete."""

    axis_names = {axis.name for axis in state.get("axes", []) if axis.name.strip()}
    company_names = {company.name for company in state.get("companies", [])}
    company_scores = state.get("company_scores", [])

    if not company_scores:
        return "Company scoring returned no company scores."

    scored_company_names = {score.company_name for score in company_scores}
    unknown_companies = sorted(scored_company_names - company_names)
    missing_companies = sorted(company_names - scored_company_names)

    if unknown_companies:
        return f"Company score references unknown company: {unknown_companies[0]}"
    if missing_companies:
        return f"Missing company scores for: {', '.join(missing_companies)}"

    for score in company_scores:
        scored_axis_names = {axis_score.axis for axis_score in score.axis_scores}
        if scored_axis_names != axis_names:
            return (
                f"Company {score.company_name} does not contain one score "
                "for every axis."
            )

    return None
