"""Build user-facing messages for workflow results and follow-up prompts."""

from models.state import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_FAILED,
    SESSION_STATUS_NEEDS_CLARIFICATION,
    CompanyFitState,
)

SUPPORT_EMAIL = "support@example.com"


def build_generic_llm_failure_message() -> str:
    """Return the generic user-facing LLM failure message."""

    return (
        "Sorry, something went wrong. Please try again later or contact "
        f"{SUPPORT_EMAIL}."
    )


def build_restart_due_to_guardrail_message(stage: str) -> str:
    """Return the user-facing restart message for terminal guardrail failures."""

    return (
        "Sorry, I couldn't continue because the model blocked this request during "
        f"{stage}. Please start a new session with another prompt or CV."
    )


def build_rephrase_due_to_guardrail_message() -> str:
    """Return the user-facing rephrase message for recoverable guardrail failures."""

    return (
        "Please reformulate your message and try again so I can continue the workflow."
    )


def build_rephrase_retry_exhausted_message() -> str:
    """Return the user-facing message for repeated reformulation failures."""

    return (
        "Sorry, I still couldn't continue after repeated reformulation attempts. "
        "Please start a new session with another prompt or CV."
    )


def build_cv_cleanup_due_to_guardrail_message() -> str:
    """Return the user-facing message for CV-content guardrail failures."""

    return (
        "Your CV appears to contain trigger words or sensitive topics. Please remove "
        "topics like fraud, scam, abuse, violence, harassment, and similar content, "
        "then start a new session with the cleaned CV."
    )


def is_guardrail_rephrase_message(message: str | None) -> bool:
    """Return whether the clarification message is the fixed guardrail rephrase prompt."""

    return (message or "").strip() == build_rephrase_due_to_guardrail_message()


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
    if status == SESSION_STATUS_NEEDS_CLARIFICATION:
        return build_clarification_message(state)
    if status == SESSION_STATUS_FAILED:
        return build_failure_message(state)
    if status == SESSION_STATUS_COMPLETED:
        return build_completion_message(state)
    return ""
