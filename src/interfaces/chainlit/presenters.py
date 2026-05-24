"""Presentation helpers for Chainlit chat messages."""

from models.state import CompanyFitState


def build_welcome_message() -> str:
    """Return the initial instruction text for the Chainlit UI."""

    return (
        "This service helps match your CV to companies that fit your goals, then "
        "scores the results against your priorities.\n\n"
        "Upload one PDF CV and include a prompt in your first message. A strong "
        "prompt should mention the kinds of companies, roles, industries, stages, "
        "locations, work modes, or constraints you care about.\n\n"
        "Please mask personal or sensitive information in your CV before uploading "
        "when possible. The app will also try to mask PII automatically before any "
        "LLM step.\n\n"
        "If the workflow needs clarification later, just reply in the same chat."
    )


def build_missing_initial_input_message() -> str:
    """Return guidance for an invalid first-turn submission."""

    return (
        "Please send a prompt and attach exactly one PDF CV in your first message. "
        "Your prompt should describe the company or role preferences you want this "
        "service to use for matching."
    )


def build_missing_clarification_message() -> str:
    """Return guidance for an empty clarification reply."""

    return "Please reply with the clarification needed to continue the workflow."


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
