"""Helpers for collecting workflow logs, artifacts, and tracing details."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class TextArtifact:
    """Text content recorded during a workflow action."""

    artifact_path: str
    content: str


@dataclass(slots=True)
class JsonArtifact:
    """JSON payload recorded during a workflow action."""

    artifact_path: str
    payload: Any


@dataclass(slots=True)
class ClarificationQuestion:
    """Clarification question recorded during a workflow action."""

    message: str
    target: str


@dataclass(slots=True)
class GuardrailBlockedMessage:
    """Guardrail-blocked user message recorded during a workflow action."""

    message: str
    source: str


@dataclass(slots=True)
class WorkflowRecorder:
    """In-memory observability collected while one workflow action runs."""

    span_outputs: dict[str, Any] = field(default_factory=dict)
    text_artifacts: list[TextArtifact] = field(default_factory=list)
    json_artifacts: list[JsonArtifact] = field(default_factory=list)
    clarification_questions: list[ClarificationQuestion] = field(default_factory=list)
    guardrail_blocked_messages: list[GuardrailBlockedMessage] = field(default_factory=list)


_CURRENT_RECORDER: ContextVar[WorkflowRecorder | None] = ContextVar(
    "workflow_recorder",
    default=None,
)


@contextmanager
def bind_workflow_recorder(recorder: WorkflowRecorder) -> Iterator[WorkflowRecorder]:
    """Bind one recorder so workflow actions can record observability details."""

    token = _CURRENT_RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _CURRENT_RECORDER.reset(token)


def record_span_outputs(**outputs: Any) -> None:
    """Record span outputs for the current workflow action."""

    recorder = _CURRENT_RECORDER.get()
    if recorder is not None:
        recorder.span_outputs.update(outputs)


def record_text_artifact(artifact_path: str, content: str) -> None:
    """Record a text artifact for the current workflow action."""

    recorder = _CURRENT_RECORDER.get()
    if recorder is not None:
        recorder.text_artifacts.append(TextArtifact(artifact_path, content))


def record_json_artifact(artifact_path: str, payload: Any) -> None:
    """Record a JSON artifact for the current workflow action."""

    recorder = _CURRENT_RECORDER.get()
    if recorder is not None:
        recorder.json_artifacts.append(JsonArtifact(artifact_path, payload))


def record_clarification_question(message: str, target: str) -> None:
    """Record a clarification question for the current workflow action."""

    recorder = _CURRENT_RECORDER.get()
    if recorder is not None:
        recorder.clarification_questions.append(ClarificationQuestion(message, target))


def record_guardrail_blocked_message(message: str, source: str) -> None:
    """Record a guardrail-blocked user message for the current workflow action."""

    recorder = _CURRENT_RECORDER.get()
    if recorder is not None:
        recorder.guardrail_blocked_messages.append(
            GuardrailBlockedMessage(message, source)
        )


def artifact_timestamp() -> str:
    """Return a compact timestamp for one artifact filename."""

    return str(int(perf_counter() * 1000))
