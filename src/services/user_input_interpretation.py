"""User-input interpretation into search criteria and axes."""

from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities.span import SpanType

from llm.client import create_azure_chat_model
from logging_utils import get_logger
from models.state import Axis, UserInputInterpretation
from services.mlflow_tracking import log_llm_prompt_artifact, traced_operation
from services.user_input_interpretation_validation import MAX_AXES

logger = get_logger(__name__)


def interpret_user_input(
    prompt: str,
    simplified_cv_text: str,
    clarification: str | None = None,
    previous_axes: list[Axis] | None = None,
) -> UserInputInterpretation:
    """Interpret the full user input and simplified CV into structured context."""

    if not prompt.strip():
        raise ValueError("User prompt is empty.")
    if not simplified_cv_text.strip():
        raise ValueError("Simplified CV text is empty.")

    llm = create_azure_chat_model()
    if llm is None:
        raise RuntimeError("Azure OpenAI is not configured for user-input interpretation.")

    clarification_block = (
        f"\nUser clarification / additional context:\n{clarification.strip()}\n"
        if clarification
        else ""
    )
    explicit_axes = _extract_explicit_axes(clarification) or _extract_explicit_axes(prompt)
    explicit_axes_block = ""
    if explicit_axes:
        explicit_axes_block = (
            "\nExplicit user-declared axes that must be preserved as the axes output:\n"
            + "\n".join(f"- {axis}" for axis in explicit_axes)
            + "\n"
        )

    previous_axes_block = ""
    if previous_axes:
        previous_axes_block = (
            "\nExisting axes from the previous turn. Keep their meaning stable unless "
            "the user explicitly changed them:\n"
            + "\n".join(f"- {axis.name}: {axis.description}" for axis in previous_axes)
            + "\n"
        )
    structured_llm = llm.with_structured_output(UserInputInterpretation)
    logger.info(
        "LLM call start: interpret_user_input prompt_chars=%s simplified_cv_chars=%s clarification_present=%s explicit_axes=%s previous_axes=%s",
        len(prompt.strip()),
        len(simplified_cv_text.strip()),
        bool(clarification and clarification.strip()),
        len(explicit_axes),
        len(previous_axes or []),
    )
    with traced_operation(
        "llm.interpret_user_input",
        span_type=SpanType.LLM,
        inputs={
            "prompt_chars": len(prompt.strip()),
            "simplified_cv_chars": len(simplified_cv_text.strip()),
            "clarification_present": bool(clarification and clarification.strip()),
            "previous_axes_count": len(previous_axes or []),
            "explicit_axes": explicit_axes,
        },
    ) as span:
        messages = [
            SystemMessage(
                content=(
                    "All inputs in this conversation are related to professional "
                    "employment and work preferences. Interpret them only in that "
                    "context. "
                    "Interpret a company-search prompt into search criteria and axes "
                    "using the requested structured output schema. "
                    "Use this separation rule strictly. "
                    "company_search_criteria must contain only requirements that can "
                    "be validated through official company information or other "
                    "authoritative company facts. Examples include location, company "
                    "size, industry or domain, ownership model, funding stage, and "
                    "other explicit company attributes that can be verified directly. "
                    "Axes must contain everything else from the user's prompt. "
                    "If a requirement cannot be validated directly from official "
                    "company information and instead needs interpretation, inference, "
                    "investigation, or probabilistic judgment about fit, it belongs "
                    "in axes. "
                    "If the user's prompt mixes official company filters with "
                    "non-official preferences, split them across both fields instead "
                    "of forcing everything into company_search_criteria. "
                    "Non-official preferences include role-fit judgments, skill or "
                    "stack match, work-arrangement preferences, team or culture fit, "
                    "career-transition friendliness, compensation fit, language "
                    "environment, and other preferences that cannot be verified from "
                    "official company facts alone. "
                    "Every distinct non-official requirement from the user's prompt "
                    "must appear in axes. Do not drop it, hide it inside search "
                    "criteria, or leave it implicit. "
                    "Do not merge distinct non-official requirements into one broad "
                    "axis unless the user explicitly grouped them together as a "
                    "single preference. If the user mentions multiple different "
                    "non-official requirements, return multiple axes. "
                    "If the prompt contains any non-official preference, axes must "
                    "not be empty. "
                    f"The maximum number of axes is {MAX_AXES}. If the user's "
                    "preferences suggest more than that, merge overlapping or closely "
                    "related non-official requirements into broader axes that still "
                    "preserve the user's intended meaning. "
                    "Axes must come from the user's prompt or clarification, not from "
                    "decomposing the CV into separate dimensions. "
                    "Search criteria are used only for company discovery. They are not "
                    "matching dimensions and should not be restated as axes. "
                    "The CV may be used to understand what an axis means from the "
                    "user's perspective and to help draft an axis description, but it "
                    "must not create new axes that the user did not ask for. "
                    "If the user explicitly names axes, preserve those axes and do not "
                    "invent extra ones. You may moderately refine wording while keeping "
                    "the same meaning. "
                    "If a clarification asks to add an axis, treat that as additive: "
                    "keep the existing axes unless the user explicitly removes or "
                    "replaces them. "
                    "For each axis, capture both the axis name and the intended meaning "
                    "of the axis from the user's perspective. You may use the prompt, "
                    "the clarification, the CV, and normal industry practice to draft "
                    "that meaning. If the meaning is still unclear, leave the "
                    "description empty instead of inventing a confident explanation. "
                    "Do not ask follow-up questions inside the axes output. "
                    "Unless the user explicitly revises the axes, keep prior axes "
                    "semantically stable across turns. "
                    "The company_search_criteria.undefined field is for unusual "
                    "criteria that do not fit the typed fields, such as founder "
                    "traits or other special constraints. "
                    "Axis descriptions should explain the user's perspective when that "
                    "perspective can be inferred."
                )
            ),
            HumanMessage(
                content=(
                    f"User prompt:\n{prompt.strip()}\n\n"
                    f"{previous_axes_block}"
                    f"{explicit_axes_block}"
                    f"Simplified CV:\n{simplified_cv_text.strip()}\n"
                    f"{clarification_block}"
                )
            ),
        ]
        log_llm_prompt_artifact("llm-prompt-interpret-prompt", messages)
        result = structured_llm.invoke(messages)
        if span is not None:
            span.set_outputs(
                {
                    "axes": [axis.model_dump() for axis in result.axes],
                    "company_search_criteria": result.company_search_criteria.model_dump(),
                }
            )
    logger.info(
        "LLM call end: interpret_user_input axes=%s criteria_fields=%s",
        len(result.axes),
        sum(1 for value in result.company_search_criteria.model_dump().values() if value),
    )
    return result


def _extract_explicit_axes(text: str | None) -> list[str]:
    """Extract explicitly declared axes from user text when possible."""

    if not text:
        return []

    stripped = text.strip()
    if not stripped:
        return []

    numbered_axes = [
        match.strip(" -:\t")
        for match in re.findall(
            r"(?:^|\n)\s*\d+[\.\):\-]\s*(.+?)(?=(?:\n\s*\d+[\.\):\-]\s*)|\Z)",
            stripped,
            flags=re.DOTALL,
        )
    ]
    if len(numbered_axes) >= 2:
        return [_normalize_axis_text(axis) for axis in numbered_axes if _normalize_axis_text(axis)]

    conversational_axes = _extract_conversational_axes(stripped)
    if conversational_axes:
        return conversational_axes

    lower = stripped.lower()
    marker = "axes are:"
    if marker in lower:
        remainder = stripped[lower.index(marker) + len(marker):].strip()
        parts = re.split(r",|\band\b", remainder, flags=re.IGNORECASE)
        normalized = [_normalize_axis_text(part) for part in parts]
        return [part for part in normalized if part]

    return []


def _normalize_axis_text(text: str) -> str:
    """Normalize one user-declared axis string."""

    normalized = re.sub(r"\s+", " ", text).strip(" .;,-\n\t")
    return normalized


def _extract_conversational_axes(text: str) -> list[str]:
    """Extract explicit axes phrased conversationally in one sentence."""

    patterns = [
        r"(?:please\s+)?add\s+an?\s+axis\s+for\s+(.+?)(?=$|\n)",
        r"(?:please\s+)?add\s+an?\s+axis\s+capturing\s+(.+?)(?=$|\n)",
        r"(?:please\s+)?include\s+an?\s+axis\s+for\s+(.+?)(?=$|\n)",
        r"(?:please\s+)?include\s+an?\s+axis\s+capturing\s+(.+?)(?=$|\n)",
        r"an?\s+axis\s+for\s+(.+?)(?=$|\n)",
        r"an?\s+axis\s+capturing\s+(.+?)(?=$|\n)",
        r"axis\s+capturing\s+(.+?)(?=$|\n)",
    ]

    extracted: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            normalized = _normalize_axis_text(match)
            if normalized:
                extracted.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for axis in extracted:
        lowered = axis.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(axis)
    return deduped
