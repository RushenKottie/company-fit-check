"""Central workflow state transitions."""

from models.state import CompanyFitState, ClarificationTarget


def mark_failed(state: CompanyFitState, message: str) -> None:
    """Move the workflow into a terminal failed state."""

    state["session_status"] = "failed"
    state["error"] = message


def mark_running(state: CompanyFitState) -> None:
    """Move the workflow back into active execution."""

    state["session_status"] = "running"


def mark_completed(state: CompanyFitState) -> None:
    """Move the workflow into a terminal completed state."""

    state["session_status"] = "completed"


def request_clarification(
    state: CompanyFitState,
    *,
    target: ClarificationTarget,
    message: str | None,
) -> None:
    """Pause the workflow until the user clarifies one target area."""

    state["pending_clarification_message"] = message
    state["clarification_target"] = target
    state["session_status"] = "needs_clarification"


def clear_clarification(state: CompanyFitState) -> None:
    """Clear any pending clarification metadata."""

    state["pending_clarification_message"] = None
    state["clarification_target"] = None
    state["guardrail_rephrase_source"] = None


def reset_user_input_clarification_iterations(state: CompanyFitState) -> None:
    """Reset user-input clarification progress."""

    state["user_input_interpretation_clarification_iterations"] = 0


def reset_company_search_clarification_iterations(state: CompanyFitState) -> None:
    """Reset company-search clarification progress."""

    state["company_search_clarification_iterations"] = 0


def increment_user_input_clarification_iterations(state: CompanyFitState) -> None:
    """Increment user-input clarification attempts."""

    state["user_input_interpretation_clarification_iterations"] = (
        state.get("user_input_interpretation_clarification_iterations", 0) + 1
    )


def increment_company_search_clarification_iterations(state: CompanyFitState) -> None:
    """Increment company-search clarification attempts."""

    state["company_search_clarification_iterations"] = (
        state.get("company_search_clarification_iterations", 0) + 1
    )
