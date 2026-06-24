"""Minimal Chainlit UI for Company Fit Check."""

import asyncio
from pathlib import Path
import tempfile
import uuid

import chainlit as cl

from application.session import (
    SessionResult,
    continue_session,
    start_session,
)
from interfaces.chainlit.presenters import (
    build_missing_clarification_message,
    build_missing_initial_input_message,
    build_welcome_message,
)
from interfaces.chainlit.session import (
    clear_workflow_state,
    get_workflow_state,
    set_workflow_state,
)
from logging_utils import configure_logging, get_logger
from models.artifacts import GeneratedCsvArtifact
from models.state import (
    SESSION_STATUS_COMPLETED,
    SESSION_STATUS_FAILED,
    SESSION_STATUS_NEEDS_CLARIFICATION,
    CompanyFitState,
)

configure_logging()
logger = get_logger(__name__)


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize a new temporary chat session."""

    logger.info("Chat session started.")
    clear_workflow_state()
    await cl.Message(content=build_welcome_message()).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Route each incoming message to the initial or clarification handler."""

    logger.info(
        "Received message content_length=%s attachments=%s",
        len(message.content or ""),
        len(getattr(message, "elements", None) or []),
    )
    current_state = get_workflow_state()
    if (
        current_state
        and current_state.get("session_status") == SESSION_STATUS_NEEDS_CLARIFICATION
    ):
        logger.info("Routing incoming message to clarification handler.")
        await _handle_clarification_message(current_state, message)
        return

    logger.info("Routing incoming message to initial handler.")
    await _handle_initial_message(message)


async def _handle_initial_message(message: cl.Message) -> None:
    """Process the initial prompt plus PDF upload."""

    prompt = message.content.strip()
    pdf_path = _extract_latest_pdf_path(message)
    if not prompt or pdf_path is None:
        logger.warning(
            "Initial message missing prompt or PDF prompt_present=%s pdf_path=%s",
            bool(prompt),
            pdf_path,
        )
        await cl.Message(content=build_missing_initial_input_message()).send()
        return

    logger.info("Starting workflow from initial message pdf_path=%s", pdf_path)
    result = await asyncio.to_thread(
        start_session,
        Path(pdf_path).read_bytes(),
        prompt,
    )
    await _deliver_result(result)


async def _handle_clarification_message(
    state: CompanyFitState,
    message: cl.Message,
) -> None:
    """Handle a clarification reply for the active workflow."""

    clarification = message.content.strip()
    if not clarification:
        logger.warning("Clarification reply was empty.")
        await cl.Message(content=build_missing_clarification_message()).send()
        return

    logger.info(
        "Continuing workflow from clarification target=%s clarification_length=%s",
        state.get("clarification_target"),
        len(clarification),
    )
    result = await asyncio.to_thread(
        continue_session,
        state,
        clarification,
    )
    await _deliver_result(result)


async def _deliver_result(result: SessionResult) -> None:
    """Render the current backend result into the chat session."""

    status = result.state.get("session_status")
    logger.info("Delivering workflow result status=%s", status)
    if status == SESSION_STATUS_NEEDS_CLARIFICATION:
        set_workflow_state(result.state)
        await cl.Message(content=result.assistant_message).send()
        return

    if status == SESSION_STATUS_FAILED:
        clear_workflow_state()
        await cl.Message(content=result.assistant_message).send()
        return

    if status == SESSION_STATUS_COMPLETED:
        clear_workflow_state()
        elements = []
        if result.csv_artifact is not None:
            elements.append(_build_file_element(result.csv_artifact))

        await cl.Message(
            content=result.assistant_message,
            elements=elements,
        ).send()
        return

    set_workflow_state(result.state)
    await cl.Message(
        content=result.assistant_message or "The workflow is still running."
    ).send()


def _extract_latest_pdf_path(message: cl.Message) -> str | None:
    """Return the latest uploaded PDF path for the message."""

    elements = getattr(message, "elements", None) or []
    pdf_path = None
    for element in elements:
        element_path = _element_pdf_path(element)
        if element_path is not None:
            pdf_path = element_path
    return pdf_path


def _element_pdf_path(element: object) -> str | None:
    """Return an uploaded element path if it looks like a PDF file."""

    path = getattr(element, "path", None)
    name = getattr(element, "name", None)
    mime = getattr(element, "mime", None)
    if not isinstance(path, str):
        return None
    if mime == "application/pdf":
        return path
    if isinstance(name, str) and name.lower().endswith(".pdf"):
        return path
    if path.lower().endswith(".pdf"):
        return path
    return None


def _build_file_element(artifact: GeneratedCsvArtifact) -> cl.File:
    """Persist an artifact to a temp file and expose it as a download."""

    temp_dir = Path(tempfile.gettempdir()) / "company_fit_check_chainlit"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4()}-{artifact.filename}"
    temp_path.write_bytes(artifact.content_bytes)
    logger.info(
        "Prepared file artifact filename=%s path=%s content_type=%s",
        artifact.filename,
        temp_path,
        artifact.content_type,
    )
    return cl.File(
        name=artifact.filename,
        path=str(temp_path),
        mime=artifact.content_type,
    )
