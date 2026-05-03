"""Temporary Chainlit session helpers."""

import chainlit as cl

from company_fit_check.models.state import CompanyFitState

WORKFLOW_STATE_KEY = "workflow_state"


def get_workflow_state() -> CompanyFitState | None:
    """Return the current in-memory workflow state for this chat session."""

    return cl.user_session.get(WORKFLOW_STATE_KEY)


def set_workflow_state(state: CompanyFitState) -> None:
    """Persist the current workflow state for this chat session."""

    cl.user_session.set(WORKFLOW_STATE_KEY, state)


def clear_workflow_state() -> None:
    """Clear any active workflow state for this chat session."""

    cl.user_session.set(WORKFLOW_STATE_KEY, None)
