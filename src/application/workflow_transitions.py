"""Central workflow state transitions."""

from models.state import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_FAILED,
    SESSION_STATUS_NEEDS_CLARIFICATION,
    SESSION_STATUS_RUNNING,
    CompanyFitState,
    ClarificationTarget,
)


def mark_failed(state: CompanyFitState, message: str) -> None:
    """Move the workflow into a terminal failed state."""

    state["session_status"] = SESSION_STATUS_FAILED
    state["error"] = message


def mark_running(state: CompanyFitState) -> None:
    """Move the workflow back into active execution."""

    state["session_status"] = SESSION_STATUS_RUNNING


def mark_completed(state: CompanyFitState) -> None:
    """Move the workflow into a terminal completed state."""

    state["session_status"] = SESSION_STATUS_COMPLETED


def request_clarification(
    state: CompanyFitState,
    *,
    target: ClarificationTarget,
    message: str | None,
) -> None:
    """Pause the workflow until the user clarifies one target area."""

    state["pending_clarification_message"] = message
    state["clarification_target"] = target
    state["session_status"] = SESSION_STATUS_NEEDS_CLARIFICATION


def clear_clarification(state: CompanyFitState) -> None:
    """Clear any pending clarification metadata."""

    state["pending_clarification_message"] = None
    state["clarification_target"] = None
    state["guardrail_rephrase_source"] = None


def reset_user_input_clarification_iterations(state: CompanyFitState) -> None:
    """Reset user-input clarification progress."""

    state["user_input_interpretation_clarification_iterations"] = 0


def increment_user_input_clarification_iterations(state: CompanyFitState) -> None:
    """Increment user-input clarification attempts."""

    state["user_input_interpretation_clarification_iterations"] = (
        state.get("user_input_interpretation_clarification_iterations", 0) + 1
    )
