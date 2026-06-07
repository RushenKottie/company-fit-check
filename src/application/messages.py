"""Shared user-facing workflow message rendering."""

from models.state import CompanyFitState


def build_clarification_message(state: CompanyFitState) -> str:
    """Return the backend clarification request for the user."""

    return state.get("pending_clarification_message") or (
        "More information is needed before the workflow can continue."
    )


def build_failure_message(state: CompanyFitState) -> str:
    """Return a user-facing workflow failure message."""

    error = state.get("error") or "Unknown workflow error."
    return f"Workflow failed.\n\n{error}"


def build_completion_message(state: CompanyFitState) -> str:
    """Return a concise final summary for a completed workflow."""

    companies = state.get("companies", [])
    company_scores = state.get("company_scores", [])
    axes = state.get("axes", [])

    top_scores = sorted(
        company_scores,
        key=lambda score: score.overall_score,
        reverse=True,
    )[:5]
    top_lines = [
        f"- {score.company_name}: {score.overall_score:.1f}%"
        for score in top_scores
    ]
    axes_text = ", ".join(axis.name for axis in axes) or "None"
    top_scores_text = "\n".join(top_lines) if top_lines else "- No scored companies"

    return (
        "Workflow completed.\n\n"
        f"Axes: {axes_text}\n"
        f"Discovered companies: {len(companies)}\n"
        f"Scored companies: {len(company_scores)}\n\n"
        "Top matches:\n"
        f"{top_scores_text}\n\n"
        "The CSV export is attached below."
    )


def build_assistant_message(state: CompanyFitState) -> str:
    """Return the rendered assistant message for the current workflow state."""

    status = state.get("session_status")
    if status == "needs_clarification":
        return build_clarification_message(state)
    if status == "failed":
        return build_failure_message(state)
    if status == "completed":
        return build_completion_message(state)
    return ""
