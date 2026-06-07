"""Presentation helpers for Chainlit chat messages."""


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


__all__ = [
    "build_missing_clarification_message",
    "build_missing_initial_input_message",
    "build_welcome_message",
]
